"""Opt-in LAN Ollama researcher; the V100 learner remains strictly local.

Ollama has no public tokenize endpoint. Text-only Qwen byte-BPE requests use
an intentionally pessimistic UTF-8 budget plus template margin. This is an
admission estimate, never an exact token count or a quality baseline.
"""

import copy
import hashlib
import ipaddress
import json
import math
import re
import threading
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from rlm.clients.llamacpp import LlamaCppClient
from rlm.v100.common import atomic_json

MODEL = "qwen3.5:9b-q8_0"
ROLE_MODELS = {
    MODEL: ("Q8_0", 12),
    "qwen3.5:2b-q4_K_M": ("Q4_K_M", 6),
    "qwen3.5:4b-q4_K_M": ("Q4_K_M", 8),
    "qwen3.5:35b-a3b-q4_K_M": ("Q4_K_M", 21),
    "qwen3.6:35b-a3b-q4_K_M": ("Q4_K_M", 21),
}
BACKEND = "ollama-research"
HELPER_WORKLOAD_LOCK = threading.Lock()
DEDICATED_WORKLOAD_LOCKS: dict[str, threading.Lock] = {}
HELPER_PACING: dict[str, dict] = {}
FULL_METADATA = "ollama-full-v1"
STABLE_METADATA = "ollama-stable-v2"
ORDERED_METADATA = "ollama-canonical-v3"


def helper_boot_id() -> str:
    """Identify the Debian boot so persisted monotonic deadlines cannot cross boots."""
    value = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if not value:
        raise ValueError("Missing Linux boot identity for helper workload pacing")
    return value


def validate_workload(batch_tokens: int, duty_percent: int, dedicated_host: bool = False) -> None:
    if type(batch_tokens) is not int or batch_tokens not in (16, 32, 64):
        raise ValueError("Remote helper batch must be 16, 32 or 64 tokens")
    maximum = 100 if dedicated_host is True else 65
    if (
        type(dedicated_host) is not bool
        or type(duty_percent) is not int
        or not 1 <= duty_percent <= maximum
    ):
        raise ValueError(f"Remote helper active wall-time target must be 1-{maximum} percent")


def private_origin(value: str) -> str:
    parsed = urlparse(value)
    address = ipaddress.ip_address(parsed.hostname or "")
    if (
        parsed.scheme != "http"
        or address.version != 4
        or not any(
            address in ipaddress.ip_network(net)
            for net in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10")
        )
        or parsed.port is None
        or not 1 <= parsed.port <= 65535
        or parsed.path not in ("", "/")
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Remote helper requires an explicit private IPv4 HTTP origin and port")
    return value.rstrip("/")


def remote_profile(profile: dict) -> bool:
    return profile.get("resources", {}).get("device") == "remote"


def validate_remote(profile: dict) -> None:
    runtime, server, resources = profile["runtime"], profile["server"], profile["resources"]
    private_origin(runtime["base_url"])
    validate_workload(
        resources.get("helper_batch_tokens", 64),
        resources.get("helper_duty_percent", 65),
        resources.get("dedicated_host", False),
    )
    if (
        resources.get("dedicated_host")
        and private_origin(runtime["base_url"]) != "http://192.168.0.9:11435"
    ):
        raise ValueError("Dedicated capacity requires the explicitly registered 3060 Ti host")
    if (
        runtime.get("backend") != BACKEND
        or runtime.get("tool_protocol") != "json"
        or runtime["model_name"] not in ROLE_MODELS
        or type(runtime.get("enable_thinking")) is not bool
        or not re.fullmatch(r"[0-9a-f]{64}", resources.get("model_digest", ""))
        or not re.fullmatch(r"[0-9a-f]{64}", resources.get("metadata_sha256", ""))
        or resources.get("metadata_hash_scheme", FULL_METADATA)
        not in (FULL_METADATA, STABLE_METADATA, ORDERED_METADATA)
        or runtime["context_window"] not in (8192, 16384, 32768, 65536, 131072)
        or not 256 <= runtime["max_output_tokens"] <= 8192
        or server["slots"] != 1
        or server.get("model")
        or server.get("binary")
        or server.get("draft_model")
        or resources.get("max_vram_gib") != ROLE_MODELS.get(runtime["model_name"], (None, None))[1]
    ):
        raise ValueError(
            "Remote helper must be a pinned text-only declared local researcher profile"
        )


def metadata_sha(info: dict, scheme: str = FULL_METADATA) -> str:
    if scheme not in (FULL_METADATA, STABLE_METADATA, ORDERED_METADATA):
        raise ValueError("Unknown remote metadata hash scheme")
    # modified_at is a local manifest timestamp, not a model behavior field.
    # Keep every other field, including the generated modelfile, renderer,
    # parameters, tokenizer and template. Legacy hashes retain their old meaning.
    chosen = (
        {key: value for key, value in info.items() if key != "modified_at"}
        if scheme in (STABLE_METADATA, ORDERED_METADATA)
        else info
    )
    if scheme == ORDERED_METADATA:
        chosen = copy.deepcopy(chosen)
        parameters = chosen.get("parameters", "")
        if isinstance(parameters, str):
            chosen["parameters"] = "\n".join(
                sorted(
                    parameters.splitlines(),
                    key=lambda line: line.split(maxsplit=1)[0] if line.strip() else "",
                )
            )
        lines = chosen.get("modelfile", "").splitlines()
        indices, quoted = [], False
        for index, line in enumerate(lines):
            if not quoted and line.startswith("PARAMETER "):
                indices.append(index)
            if line.count('"""') % 2:
                quoted = not quoted
        values = sorted(
            (lines[index] for index in indices), key=lambda line: line.split(maxsplit=2)[1]
        )
        for index, value in zip(indices, values, strict=True):
            lines[index] = value
        chosen["modelfile"] = "\n".join(lines)
    return hashlib.sha256(json.dumps(chosen, sort_keys=True).encode()).hexdigest()


def check_guard(guard: dict, dedicated: bool = False) -> dict:
    expected = "synta-dedicated-linux-v1" if dedicated else "synta-adaptive-windows-v1"
    label = "Dedicated Linux" if dedicated else "Windows"
    age = guard.get("heartbeat_age_seconds")
    if age is None:
        stamp = guard.get("updated", 0)
        age = time.time() - stamp if type(stamp) in (int, float) else None
    problems = []
    if guard.get("contract") != expected:
        problems.append("unexpected or missing guard contract")
    if type(age) not in (int, float) or not math.isfinite(age) or not 0 <= age <= 10:
        problems.append("stale resource heartbeat: " + str(age) + " seconds")
    if guard.get("ready") is not True:
        problems.append(str(guard.get("reason", "resource admission not ready")))
    if problems:
        raise requests.ConnectionError(
            label + " adaptive guard unavailable: " + "; ".join(problems)
        )
    return guard


class OllamaResearchClient(LlamaCppClient):
    def __init__(
        self,
        *,
        base_url: str,
        model_digest: str,
        metadata_sha256: str = "",
        metadata_hash_scheme: str = FULL_METADATA,
        max_vram_gib: int = 12,
        helper_batch_tokens: int = 64,
        helper_duty_percent: int = 65,
        require_adaptive_windows_guard: bool = False,
        on_demand: bool = False,
        allow_cpu_offload: bool = False,
        dedicated_host: bool = False,
        **kwargs: Any,
    ):
        origin = private_origin(base_url)
        validate_workload(helper_batch_tokens, helper_duty_percent, dedicated_host)
        if dedicated_host and origin != "http://192.168.0.9:11435":
            raise ValueError("Dedicated capacity requires the explicitly registered 3060 Ti host")
        self.dedicated_host = dedicated_host
        if metadata_hash_scheme not in (FULL_METADATA, STABLE_METADATA, ORDERED_METADATA):
            raise ValueError("Unknown remote metadata hash scheme")
        if kwargs.get("model_name", MODEL) not in ROLE_MODELS or not re.fullmatch(
            r"[0-9a-f]{64}", model_digest
        ):
            raise ValueError("Only an explicitly pinned declared local model is permitted")
        kwargs.setdefault("model_name", MODEL)
        if (
            type(max_vram_gib) is not int
            or not 1 <= max_vram_gib <= ROLE_MODELS[kwargs["model_name"]][1]
        ):
            raise ValueError("Role VRAM allowance exceeds its explicit model budget")
        if type(on_demand) is not bool or type(allow_cpu_offload) is not bool:
            raise ValueError("Role loading/offload options must be explicit booleans")
        if allow_cpu_offload and kwargs["model_name"] not in (
            "qwen3.5:35b-a3b-q4_K_M",
            "qwen3.6:35b-a3b-q4_K_M",
        ):
            raise ValueError("CPU expert offload is admitted only for the explicit MoE role")
        self.on_demand, self.allow_cpu_offload = on_demand, allow_cpu_offload
        if type(kwargs.get("enable_thinking", False)) is not bool:
            raise ValueError("Remote thinking must be an explicit boolean")
        kwargs.setdefault("enable_thinking", False)
        # Reuse local usage/activity bookkeeping without relaxing the local
        # llama.cpp client's loopback-only contract.
        super().__init__(base_url="http://127.0.0.1:11435", **kwargs)
        self.base_url = origin
        self.model_digest = model_digest
        self.metadata_sha256 = metadata_sha256
        self.metadata_hash_scheme = metadata_hash_scheme
        self.max_vram_gib = max_vram_gib
        self.helper_batch_tokens = helper_batch_tokens
        if type(require_adaptive_windows_guard) is not bool:
            raise ValueError("Adaptive guard requirement must be boolean")
        self.require_adaptive_windows_guard = require_adaptive_windows_guard
        self.helper_duty_percent = helper_duty_percent
        self.request_lock = threading.Lock()
        self.tool_protocol = "json"

    @contextmanager
    def workload_slot(self, wait: bool = True):
        """Serialize our helper calls and wait before work, without delaying its answer."""
        if type(wait) is not bool:
            raise ValueError("Helper wait must be boolean")
        with ExitStack() as scope:
            lock = (
                DEDICATED_WORKLOAD_LOCKS.setdefault(self.base_url, threading.Lock())
                if self.dedicated_host
                else HELPER_WORKLOAD_LOCK
            )
            deadline_bound = getattr(self, "request_deadline", None)
            acquired = (
                lock.acquire(timeout=self.remaining_timeout())
                if wait and deadline_bound is not None
                else lock.acquire(blocking=wait)
            )
            if not acquired:
                raise TimeoutError("Helper busy; optional work deferred")
            scope.callback(lock.release)
            path = None
            boot = helper_boot_id()
            pacing_root = self.activity_root or getattr(self, "pacing_root", None)
            if pacing_root:
                import fcntl

                name = hashlib.sha256(self.base_url.encode()).hexdigest()[:16]
                path = Path(pacing_root) / "research/state" / f"helper-{name}.workload.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                lease = scope.enter_context(path.with_suffix(".lock").open("a"))
                try:
                    while True:
                        try:
                            fcntl.flock(
                                lease,
                                fcntl.LOCK_EX
                                | (fcntl.LOCK_NB if not wait or deadline_bound else 0),
                            )
                            break
                        except BlockingIOError:
                            if not wait:
                                raise
                            time.sleep(min(0.1, self.remaining_timeout()))
                except BlockingIOError as error:
                    raise TimeoutError("Helper lease busy; optional work deferred") from error
                state = json.loads(path.read_text()) if path.exists() else {}
            else:
                state = HELPER_PACING.get(self.base_url, {})
            if state and (
                set(state) != {"boot_id", "not_before"}
                or not isinstance(state["boot_id"], str)
                or not state["boot_id"]
                or type(state["not_before"]) not in (int, float)
                or not math.isfinite(state["not_before"])
            ):
                raise ValueError("Invalid persisted helper workload state")
            started = time.monotonic()
            deadline = state["not_before"] if state.get("boot_id") == boot else started
            if not wait and started < deadline:
                raise TimeoutError("Helper cooling down; optional work deferred")
            while time.monotonic() < deadline:
                self.remaining_timeout()
                time.sleep(max(0, min(1, deadline - time.monotonic())))
            quota = {
                "path": path,
                "boot_id": boot,
                "waited_seconds": max(0, time.monotonic() - started),
            }
            yield quota

    def preflight(self) -> None:
        if self.require_adaptive_windows_guard:
            check_guard(self.remote_request("/api/synta-resources"), self.dedicated_host)
        self.identity()

    def reserve_helper_idle(self, quota: dict, request_seconds: float) -> None:
        duty = quota.get("adaptive_duty_percent", self.helper_duty_percent)
        idle = request_seconds * (100 - duty) / duty
        state = {"boot_id": quota["boot_id"], "not_before": time.monotonic() + idle}
        if quota["path"] is not None:
            atomic_json(quota["path"], state)
        else:
            HELPER_PACING[self.base_url] = state
        quota.update(request_seconds=request_seconds, planned_idle_seconds=idle)
        if self.activity_root:
            from rlm.v100.activity import ActivityLog

            ActivityLog(self.activity_root, self.research_owner, "tester").write(
                "steps",
                "helper-workload-reservation",
                {
                    "endpoint": self.base_url,
                    "batch_tokens": self.helper_batch_tokens,
                    "active_wall_time_target_percent": duty,
                    "request_seconds": request_seconds,
                    "planned_idle_seconds_before_next_turn": idle,
                    "waited_seconds_before_this_turn": quota["waited_seconds"],
                    "scope": "Our controller calls only; not a GPU utilization, watts or peak VRAM cap",
                },
            )

    def remote_request(self, endpoint: str, data: dict | None = None) -> dict:
        if endpoint not in (
            "/api/tags",
            "/api/show",
            "/api/ps",
            "/api/chat",
            "/api/version",
            "/api/synta-resources",
        ):
            raise ValueError("Remote research transport does not expose model management")
        with requests.Session() as session:
            session.trust_env = False
            method = session.get if data is None else session.post
            arguments = {} if data is None else {"json": data}
            if data is not None:
                context = getattr(self, "activity_context", {})
                arguments["headers"] = {
                    "X-Synta-Task": re.sub(
                        r"[^A-Za-z0-9 ._:/-]", "", str(getattr(self, "activity_actor", "research"))
                    )[:100],
                    "X-Synta-Request": re.sub(
                        r"[^A-Za-z0-9_-]",
                        "",
                        str(context.get("chat_request_id", context.get("request_id", ""))),
                    )[:64],
                }
            with method(
                self.base_url + endpoint,
                timeout=(min(5, self.remaining_timeout()), self.remaining_timeout()),
                allow_redirects=False,
                stream=True,
                **arguments,
            ) as response:
                if 300 <= response.status_code < 400:
                    raise ValueError("Remote helper redirected the request")
                response.raise_for_status()
                if endpoint == "/api/chat" and data and data.get("stream"):
                    from rlm.v100.streaming import ollama_reply

                    return ollama_reply(response.iter_lines(chunk_size=128), self.stream_delta)
                body = bytearray()
                for block in response.iter_content(65536):
                    body.extend(block)
                    if len(body) > 2 * 1024**2:
                        raise ValueError("Remote helper response exceeds its byte budget")
                result = json.loads(body)
        if not isinstance(result, dict) or result.get("error"):
            raise ValueError("Remote helper returned an invalid response")
        return result

    def identity(self) -> dict:
        rows = self.remote_request("/api/tags").get("models", [])
        matches = [row for row in rows if row.get("name") == self.model_name]
        if (
            len(matches) != 1
            or matches[0].get("digest") != self.model_digest
            or matches[0].get("remote_host")
            or matches[0].get("remote_model")
        ):
            raise ValueError("Remote model missing, changed, or cloud-backed")
        info = self.remote_request("/api/show", {"model": self.model_name})
        if (
            info.get("details", {}).get("family") not in ("qwen35",)
            or info.get("details", {}).get("quantization_level") != ROLE_MODELS[self.model_name][0]
            or info.get("model_info", {}).get("tokenizer.ggml.model") != "gpt2"
            or info.get("system")
            or info.get("remote_host")
            or info.get("remote_model")
            or len(info.get("template", "").encode()) > 16384
        ):
            raise ValueError("Remote helper metadata differs from supported text byte-BPE model")
        if (
            self.metadata_sha256
            and metadata_sha(info, self.metadata_hash_scheme) != self.metadata_sha256
        ):
            raise ValueError("Remote model metadata or template changed")
        return info

    def loaded(self) -> dict:
        rows = self.remote_request("/api/ps").get("models", [])
        matches = [row for row in rows if row.get("name") == self.model_name]
        if len(matches) != 1 or matches[0].get("digest") != self.model_digest:
            raise ValueError("Pinned remote model is not loaded; start the Windows helper first")
        item = matches[0]
        if (
            item.get("context_length") != self.context_window
            or not 0 < item.get("size_vram", 0) <= self.max_vram_gib * 2**30
            or (not self.allow_cpu_offload and item.get("size_vram", 0) < item.get("size", 1))
        ):
            raise ValueError("Remote helper context, GPU residency or measured VRAM differs")
        return item

    def count_text(self, text: str, parse_special: bool = False) -> int:
        return len(text.encode("utf-8"))

    def template_args(self) -> dict:
        return {}

    def budget_prompt(self, data: dict) -> str:
        messages = data.get("messages")
        if not isinstance(messages, list) or not 1 <= len(messages) <= 64:
            raise ValueError("Remote helper needs 1-64 text messages")
        for message in messages:
            if (
                set(message) != {"role", "content"}
                or message["role"] not in ("system", "user", "assistant")
                or not isinstance(message["content"], str)
            ):
                raise ValueError("Remote helper only accepts ordinary text JSON-tool messages")
        if data.get("tools") or data.get("stream"):
            raise ValueError("Remote helper requires non-streaming JSON tools")
        encoded = json.dumps(
            {"messages": messages, "format": data.get("response_format")}, ensure_ascii=False
        )
        return encoded + " " * (2048 + 128 * len(messages))

    def http_request(self, endpoint: str, data: dict | None = None) -> dict:
        if data is None:
            raise ValueError("Remote client requires a bounded research request")
        if endpoint == "/apply-template":
            # A budgeting surrogate, not the model's actual rendered prompt.
            return {"prompt": self.budget_prompt(data), "count_mode": "utf8-upper-estimate"}
        if endpoint != "/v1/chat/completions" or data.get("model") != self.model_name:
            raise ValueError("Remote helper cannot change endpoints or models")
        prompt_budget = self.count_text(self.budget_prompt(data)) + 1
        maximum = data.get("max_tokens")
        if (
            type(maximum) is not int
            or not 1 <= maximum <= 8192
            or prompt_budget + maximum > self.context_window
        ):
            raise ValueError("Remote text/context budget exceeded; reduce retrieved material")
        options = {
            "num_ctx": self.context_window,
            "num_predict": maximum,
            "num_batch": self.helper_batch_tokens,
            "num_thread": 4,
        }
        for name in ("temperature", "seed", "top_p", "top_k"):
            if name in data:
                options[name] = data[name]
        payload = {
            "model": self.model_name,
            "messages": data["messages"],
            "stream": self.streaming_enabled(),
            "think": self.enable_thinking,
            "options": options,
            "keep_alive": -1,
        }
        shape = data.get("response_format")
        if shape:
            if shape.get("type") != "json_object":
                raise ValueError("Remote helper requires JSON-object output")
            payload["format"] = shape.get("schema", "json")
        with self.request_lock, self.workload_slot() as quota:
            if self.require_adaptive_windows_guard:
                guard = self.remote_request("/api/synta-resources")
                check_guard(guard, self.dedicated_host)
                if guard.get("throughput_enabled") is True and not self.dedicated_host:
                    duty = guard.get("duty_percent")
                    if type(duty) is int and 1 <= duty <= 100:
                        quota["adaptive_duty_percent"] = duty
                    batch = guard.get("batch_tokens")
                    if type(batch) is int and batch in (16, 32, 64, 128, 256):
                        payload["options"]["num_batch"] = batch
            self.identity()
            if not self.on_demand:
                self.loaded()
            started = time.monotonic()
            try:
                response = self.remote_request("/api/chat", payload)
            finally:
                self.reserve_helper_idle(quota, max(0, time.monotonic() - started))
            measured = self.loaded()
        incoming, outgoing = response.get("prompt_eval_count"), response.get("eval_count")
        if (
            response.get("model") != self.model_name
            or response.get("done") is not True
            or response.get("done_reason") not in ("stop", "length")
            or type(incoming) is not int
            or type(outgoing) is not int
            or not 0 < incoming <= prompt_budget
            or not 0 < outgoing <= maximum
            or incoming + outgoing > self.context_window
        ):
            raise ValueError("Remote helper returned incomplete or invalid measured usage")
        message = response.get("message", {})
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": message.get("content", ""),
                        "reasoning_content": message.get("thinking", ""),
                    },
                    "finish_reason": response["done_reason"],
                }
            ],
            "usage": {
                "prompt_tokens": incoming,
                "completion_tokens": outgoing,
                "total_tokens": incoming + outgoing,
            },
            "timings": {
                "predicted_per_second": outgoing * 1e9 / max(1, response.get("eval_duration", 0)),
                "count_mode": "ollama-measured; preflight-utf8-estimate",
                "prompt_budget": prompt_budget,
                "remote_vram_gib": measured["size_vram"] / 2**30,
                "helper_batch_tokens": self.helper_batch_tokens,
                "helper_active_wall_time_target_percent": self.helper_duty_percent,
                "helper_request_seconds": quota["request_seconds"],
                "helper_planned_idle_seconds": quota["planned_idle_seconds"],
                "helper_waited_seconds": quota["waited_seconds"],
            },
        }


def prepare_remote(profile: dict, root: Path, url: str, context: int, digest: str) -> Path:
    chosen = copy.deepcopy(profile)
    chosen["runtime"].update(
        backend=BACKEND,
        base_url=private_origin(url),
        model_name=MODEL,
        model_version="rtx3090-qwen35-q8-" + digest[:12],
        context_window=context,
        max_output_tokens=1024,
        max_timeout=300,
        enable_thinking=False,
        temperature=0.0,
        tool_protocol="json",
    )
    for name in ("top_p", "top_k"):
        chosen["runtime"].pop(name, None)
    chosen["server"].update(
        binary="", model="", draft_model="", context_per_slot=context, slots=1, gpu_layers=0
    )
    chosen["resources"] = {
        "device": "remote",
        "model_digest": digest,
        "metadata_sha256": "0" * 64,
        "metadata_hash_scheme": STABLE_METADATA,
        "max_vram_gib": 12,
        "min_available_ram_gib": 0,
        "helper_batch_tokens": 64,
        "helper_duty_percent": 65,
    }
    chosen["memory"].update(chunk_tokens=256, fanout=2, retrieve_count=2)
    validate_remote(chosen)
    client = OllamaResearchClient(
        base_url=url, model_name=MODEL, model_digest=digest, context_window=context, timeout=300
    )
    info = client.identity()
    chosen["resources"]["metadata_sha256"] = metadata_sha(info, STABLE_METADATA)
    client.loaded()
    destination = root / "research/researcher-rtx3090.json"
    if destination.exists():
        raise FileExistsError("Remote profile already exists; review it before replacing")
    snapshot = root / "research/helper-metadata" / f"show-{time.time_ns()}.json"
    atomic_json(snapshot, info)
    chosen["resources"]["metadata_snapshot"] = str(snapshot.relative_to(root))
    atomic_json(destination, chosen)
    return destination


def migrate_remote_metadata(root: Path) -> dict:
    """Explicitly migrate a legacy hash after pinned-manifest and residency checks."""
    import fcntl

    from rlm.v100.common import load_profile
    from rlm.v100.competition import helper_client
    from rlm.v100.mission import status

    directory = root / "research/mission"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "start.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if status(root)["running"]:
            raise RuntimeError("Stop the mission before migrating helper metadata")
        path = root / "research/researcher-rtx3090.json"
        original = json.loads(path.read_text())
        profile = load_profile(path, root)
        validate_remote(profile)
        client = helper_client(profile, root)
        runtime = client.remote_request("/api/version")
        if runtime.get("version") != "0.40.0":
            raise ValueError("Metadata migration requires the isolated Ollama 0.40.0 runtime")
        if client.metadata_hash_scheme == STABLE_METADATA:
            client.identity()
            client.loaded()
            return {"status": "already stable; identity verified", "profile": str(path)}
        old_hash = client.metadata_sha256
        # The user invokes this migration explicitly. Retain the pinned tag
        # manifest digest and all supported-model checks while rebinding metadata.
        client.metadata_sha256 = ""
        info = client.identity()
        client.loaded()
        new_hash = metadata_sha(info, STABLE_METADATA)
        client.metadata_hash_scheme = STABLE_METADATA
        client.metadata_sha256 = new_hash
        client.identity()
        audit = root / "research/helper-metadata" / f"migration-{time.time_ns()}"
        atomic_json(audit / "profile-before.json", original)
        atomic_json(audit / "show.json", info)
        record = {
            "status": "migrated; pinned manifest and loaded GPU context verified",
            "profile": str(path),
            "audit": str(audit),
            "model_digest": client.model_digest,
            "old_full_hash": old_hash,
            "current_full_hash": metadata_sha(info),
            "stable_hash": new_hash,
            "modified_at": info.get("modified_at"),
            "legacy_difference": "Cause cannot be established without the original show snapshot",
            "weights_changed": False,
            "mission_started": False,
        }
        atomic_json(audit / "migration.json", record)
        chosen = copy.deepcopy(original)
        chosen["resources"].update(
            metadata_hash_scheme=STABLE_METADATA,
            metadata_sha256=new_hash,
            metadata_snapshot=str((audit / "show.json").relative_to(root)),
        )
        validate_remote(chosen)
        atomic_json(path, chosen)
        return record


def selected_helper(root: Path, role: str = "fast") -> Path:
    from rlm.v100.model_roles import selected_profile
    from rlm.v100.remote_nodes import selected_node

    node = selected_node(root)
    if node is not None and role == "fast":
        return node

    selected = selected_profile(root, role)
    if selected is not None:
        return selected
    remote = root / "research/researcher-rtx3090.json"
    path = remote if remote.exists() else root / "research/researcher-cpu.toml"
    if not path.exists():
        raise ValueError("Missing prepared researcher")
    return path


def canonicalize_remote(root: Path, context_window: int | None = None) -> dict:
    """Rebind only proven parameter-order/date changes; never silently trust new templates."""
    from rlm.v100.common import load_profile
    from rlm.v100.competition import helper_client
    from rlm.v100.mission import status

    if status(root)["running"]:
        raise ValueError("Stop the mission before metadata migration")
    path = root / "research/researcher-rtx3090.json"
    original = json.loads(path.read_text())
    profile = load_profile(path, root)
    if context_window is not None:
        if type(context_window) is not int or not 4096 <= context_window <= 262144:
            raise ValueError("Invalid migration context budget")
        profile["runtime"]["context_window"] = context_window
        profile["server"]["context_per_slot"] = context_window
    client = helper_client(profile, root)
    if client.metadata_hash_scheme == ORDERED_METADATA:
        client.identity()
        client.loaded()
        return {"status": "canonical identity verified"}
    snapshot = Path(profile["resources"]["metadata_snapshot"])
    if not snapshot.is_absolute():
        snapshot = root / snapshot
    previous = json.loads(snapshot.read_text())
    if metadata_sha(previous, client.metadata_hash_scheme) != client.metadata_sha256:
        raise ValueError("Previous metadata snapshot does not match the recorded pin")
    client.metadata_sha256 = ""
    info = client.identity()
    client.loaded()
    if metadata_sha(info, ORDERED_METADATA) != metadata_sha(previous, ORDERED_METADATA):
        changed = [
            key
            for key in sorted(set(info) | set(previous))
            if key != "modified_at"
            and metadata_sha({key: info.get(key)}, ORDERED_METADATA)
            != metadata_sha({key: previous.get(key)}, ORDERED_METADATA)
        ]
        audit = root / "research/helper-metadata" / f"rejected-{time.time_ns()}"
        atomic_json(audit / "profile-before.json", original)
        atomic_json(audit / "show-before.json", previous)
        atomic_json(audit / "show-current.json", info)
        raise ValueError(
            f"Metadata changed beyond parameter order/date; fields={changed}; audit={audit}"
        )
    audit = root / "research/helper-metadata" / f"canonical-{time.time_ns()}"
    atomic_json(audit / "profile-before.json", original)
    atomic_json(audit / "show.json", info)
    original["resources"].update(
        metadata_hash_scheme=ORDERED_METADATA,
        metadata_sha256=metadata_sha(info, ORDERED_METADATA),
        metadata_snapshot=str((audit / "show.json").relative_to(root)),
    )
    if context_window is not None:
        original["runtime"]["context_window"] = context_window
        original["server"]["context_per_slot"] = context_window
    validate_remote(original)
    atomic_json(path, original)
    return {
        "status": "parameter ordering normalized; semantic fields and manifest pinned",
        "audit": str(audit),
    }
