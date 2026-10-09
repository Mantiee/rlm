"""An admitted CPU copy of accepted native weights, never a training candidate."""

import copy
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
        threads=2,
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

    def close(self) -> None:
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
            self.close()

    def get(self, accepted: dict):
        if self.stop.is_set():
            raise RuntimeError("Mission is stopping; CPU chat cannot start")
        model = Path(accepted["server"]["model"])
        stat = model.stat()
        identity = (str(model.resolve()), stat.st_size, stat.st_mtime_ns)
        if self.identity != identity:
            self.close()
        if self.client is not None:
            try:
                assert_served_expert(
                    self.client, self.profile, self.root, {"files": {"model-gguf": self.sha256}}
                )
                props = self.client.request("/props")
                if Path(props["model_path"]).resolve() != model.resolve():
                    raise ValueError("CPU chat endpoint no longer serves accepted weights")
            except Exception:
                self.close()
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
        self.sha256 = file_hash(model)
        try:
            actual = self.stack.enter_context(
                managed_server(
                    path, self.root, self.directory / "accepted-cpu-chat.log", cancel=self.stop
                )
            )
            client = helper_client(actual, self.root)
            assert_served_expert(client, actual, self.root, {"files": {"model-gguf": self.sha256}})
        except Exception:
            self.close()
            raise
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
