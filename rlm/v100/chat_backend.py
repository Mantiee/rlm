"""An admitted CPU copy of accepted native weights, never a training candidate."""

import copy
import math
import os
import threading
import time
from contextlib import ExitStack
from pathlib import Path

from rlm.v100.common import atomic_json
from rlm.v100.competition import helper_client, managed_server
from rlm.v100.protection import file_hash
from rlm.v100.researchers import available_ram_gib
from rlm.v100.serving import assert_served_expert


def cpu_profile(accepted: dict) -> dict:
    if accepted["runtime"].get("backend", "llamacpp") not in ("llamacpp", "native"):
        raise ValueError("CPU chat requires accepted native GGUF weights")
    profile = copy.deepcopy(accepted)
    profile["server"].update(
        gpu_layers=0,
        slots=1,
        threads=max(2, min(8, (os.cpu_count() or 4) - 2)),
        context_per_slot=8192,
        batch_size=64,
        ubatch_size=32,
        flash_attention="off",
        cache_type="f16",
        draft_model="",
        draft_tokens=0,
    )
    profile["runtime"].update(
        base_url="http://127.0.0.1:8094",
        context_window=8192,
        max_output_tokens=1024,
        enable_thinking=False,
        max_timeout=600,
    )
    profile["memory"].update(chunk_tokens=512, fanout=4, retrieve_count=4)
    profile.setdefault("resources", {}).pop("mtp_validation", None)
    profile["resources"].update(
        device="cpu",
        accepted_cpu_chat=True,
        min_available_ram_gib=16,
        cpu_guard=True,
        min_free_ram_gib=4,
    )
    return profile


class AcceptedCPUChat:
    def __init__(self, root: Path, directory: Path, stop: threading.Event | None = None):
        self.root, self.directory = root, directory
        self.stack = ExitStack()
        self.identity = None
        self.client = None
        self.last_used = 0.0
        self.sha256 = None
        self.profile = None
        self.stop = stop or threading.Event()
        self.deadline = None
        self.verification = None
        self.verification_lock = threading.Lock()

    def close(self, *, cancel_verification: bool = True) -> None:
        if cancel_verification and self.verification is not None:
            self.verification["cancel"].set()
        active = self.client is not None
        self.stack.close()
        self.stack = ExitStack()
        self.client, self.identity, self.sha256, self.profile = None, None, None, None
        if active:
            atomic_json(
                self.directory / "accepted-cpu-chat-status.json",
                {
                    "available": False,
                    "state": "closed; accepted weights retained",
                },
            )

    def maintain(self) -> None:
        if self.client is not None and (
            time.monotonic() - self.last_used > 180 or available_ram_gib() < 4
        ):
            self.close(cancel_verification=False)

    def prewarm(self, accepted: dict, *, budget: float = 600) -> None:
        """Hash once in this process before chat; never load a model or extend a turn."""
        if not math.isfinite(budget) or not 0 < budget <= 600:
            raise ValueError("Prewarm budget must be greater than zero and at most 600 seconds")
        if accepted.get("resources", {}).get("accepted_cpu_chat"):
            self.verification_task(Path(accepted["server"]["model"]), time.monotonic() + budget)

    def verification_task(self, model: Path, deadline: float) -> dict:
        from rlm.v100.protection import file_identity

        identity = (model.resolve(), *file_identity(model))
        with self.verification_lock:
            current = self.verification
            if current is not None:
                if current["identity"] == identity:
                    return current
                if not current["done"].is_set():
                    current["cancel"].set()
                    raise RuntimeError("Previous accepted weight verification is stopping")
            task = {
                "identity": identity,
                "done": threading.Event(),
                "cancel": threading.Event(),
                "result": None,
                "failure": None,
            }
            self.verification = task

            def check():
                if task["cancel"].is_set() or self.stop.is_set():
                    raise RuntimeError("CPU weight verification cancelled")
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Accepted CPU weight verification exceeded its startup budget"
                    )

            last_update = [0.0]

            def progress(path, completed, total):
                check()
                now = time.monotonic()
                if now - last_update[0] >= 1 or completed == total:
                    last_update[0] = now
                    atomic_json(
                        self.directory / "accepted-cpu-chat-status.json",
                        {
                            "available": False,
                            "stage": "verifying-accepted-weights",
                            "bytes_verified": completed,
                            "bytes_total": total,
                            "updated": time.time(),
                            "model": str(path),
                        },
                    )

            def verify():
                try:
                    task["result"] = file_hash(model, check=check, progress=progress)
                except Exception as failure:
                    task["failure"] = failure
                    atomic_json(
                        self.directory / "accepted-cpu-chat-status.json",
                        {
                            "available": False,
                            "stage": "weight-verification-failed",
                            "error": str(failure),
                            "updated": time.time(),
                        },
                    )
                finally:
                    task["done"].set()

            threading.Thread(
                target=verify, daemon=True, name="accepted-cpu-weight-verification"
            ).start()
            return task

    def verify_weights(self, model: Path, deadline: float) -> str:
        existing = self.verification
        task = self.verification_task(model, deadline)
        if existing is task and not task["done"].is_set():
            raise RuntimeError("Accepted CPU weights are warming; verification is still running")
        while not task["done"].wait(0.05):
            if self.stop.is_set():
                task["cancel"].set()
                raise RuntimeError("CPU weight verification cancelled")
            if time.monotonic() >= deadline:
                task["cancel"].set()
                failure = TimeoutError(
                    "Accepted CPU weight verification exceeded its startup budget"
                )
                atomic_json(
                    self.directory / "accepted-cpu-chat-status.json",
                    {
                        "available": False,
                        "stage": "weight-verification-failed",
                        "error": str(failure),
                        "updated": time.time(),
                    },
                )
                raise failure
        if task["failure"] is not None:
            raise task["failure"]
        # Never start another synchronous full-file read if the artifact changes.
        from rlm.v100.protection import file_identity

        self.check_verification_stop(deadline)
        if (model.resolve(), *file_identity(model)) != task["identity"]:
            raise ValueError("Accepted weights changed after verification")
        return task["result"]

    def check_verification_stop(self, deadline):
        if self.stop.is_set():
            raise RuntimeError("CPU weight verification cancelled")
        if time.monotonic() >= deadline:
            raise TimeoutError("Accepted CPU weight verification exceeded its startup budget")

    def get(self, accepted: dict):
        if self.stop.is_set():
            raise RuntimeError("Mission is stopping; CPU chat cannot start")
        deadline = min(time.monotonic() + 90, self.deadline or float("inf"))
        model = Path(accepted["server"]["model"])
        stat = model.stat()
        identity = (str(model.resolve()), stat.st_size, stat.st_mtime_ns)
        if self.identity != identity:
            self.close(cancel_verification=False)
        if self.client is not None:
            self.client.request_deadline = deadline
            try:
                assert_served_expert(
                    self.client, self.profile, self.root, {"files": {"model-gguf": self.sha256}}
                )
                props = self.client.request("/props")
                if Path(props["model_path"]).resolve() != model.resolve():
                    raise ValueError("CPU chat endpoint no longer serves accepted weights")
            except Exception:
                self.close(cancel_verification=False)
                raise
            self.last_used = time.monotonic()
            return self.client
        required = max(16, stat.st_size / 2**30 * 1.5 + 4)
        if available_ram_gib() < required:
            raise ValueError(f"CPU chat needs {required:.1f} GiB available RAM; no model loaded")
        profile = cpu_profile(accepted)
        profile["resources"]["min_available_ram_gib"] = required
        profile["resources"]["max_rss_gib"] = stat.st_size / 2**30 * 1.5 + 2
        path = self.directory / "accepted-cpu-chat.json"
        atomic_json(path, profile)
        self.sha256 = self.verify_weights(model, deadline)
        profile["runtime"]["max_timeout"] = max(1, int(deadline - time.monotonic()))
        atomic_json(path, profile)
        try:
            actual = self.stack.enter_context(
                managed_server(
                    path,
                    self.root,
                    self.directory / "accepted-cpu-chat.log",
                    cancel=self.stop,
                    lifecycle_role="reader-owned",
                )
            )
            client = helper_client(actual, self.root)
            client.request_deadline = deadline
            client.timeout = 5
            assert_served_expert(client, actual, self.root, {"files": {"model-gguf": self.sha256}})
        except Exception:
            self.close(cancel_verification=False)
            raise
        client.timeout = 600
        self.client, self.identity = client, identity
        self.profile = actual
        self.last_used = time.monotonic()
        client.activity_actor = "chat-master-cpu"
        atomic_json(
            self.directory / "accepted-cpu-chat-status.json",
            {
                "available": True,
                "accepted_model_sha256": self.sha256,
                "model": str(model),
                "context": 8192,
                "threads": 2,
                "required_available_ram_gib": required,
                "scope": "Identical accepted weights, CPU only; not a promoted candidate",
            },
        )
        return client
