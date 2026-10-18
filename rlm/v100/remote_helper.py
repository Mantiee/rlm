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
from rlm.v100.model_policy import metadata_contract

MODEL = "qwen3.5:9b-q8_0"
ROLE_MODELS = {
    "gemma4:e2b-it-q4_K_M": ("Q4_K_M", 7),
    "gemma4:e4b-it-q4_K_M": ("Q4_K_M", 8),
    "gemma4:26b-a4b-it-q4_K_M": ("Q4_K_M", 22),
    "qwen3.8:27b-q4_K_M": ("Q4_K_M", 22),
    MODEL: ("Q8_0", 12),
    "qwen3.5:2b-q4_K_M": ("Q4_K_M", 6),
    "qwen3.5:4b-q4_K_M": ("Q4_K_M", 8),
    "qwen3.5:35b-a3b-q4_K_M": ("Q4_K_M", 21),
    "qwen3.6:35b-a3b-q4_K_M": ("Q4_K_M", 21),
}
BACKEND = "ollama-research"
LLAMACPP_BACKEND = "llamacpp-research"
WINDOWS_3090_ORIGIN = "http://192.168.0.61:11435"
REMOTE_ORIGIN_POLICY_KEY = "remote_origin_enabled"
HELPER_WORKLOAD_LOCK = threading.Lock()
DEDICATED_WORKLOAD_LOCKS: dict[str, threading.Lock] = {}
HELPER_PACING: dict[str, dict] = {}


class OwnedGpuCooldown(requests.HTTPError):
    """An explicit pre-admission rejection that can safely return to the queue."""

    def __init__(self, message: str, response, retry_after_seconds: int):
        super().__init__(message, response=response)
        self.retry_after_seconds = retry_after_seconds


class ResourceGuardUnavailable(requests.ConnectionError):
    """A fail-closed guard rejection with a bounded, non-secret reason code."""

    def __init__(self, message: str, reason_code: str):
        super().__init__(message)
        self.reason_code = reason_code


FULL_METADATA = "ollama-full-v1"
STABLE_METADATA = "ollama-stable-v2"
ORDERED_METADATA = "ollama-canonical-v3"


def helper_boot_id() -> str:
    """Identify the Debian boot so persisted monotonic deadlines cannot cross boots."""
    value = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if not value:
        raise ValueError("Missing Linux boot identity for helper workload pacing")
    return value


def validate_workload(
    batch_tokens: int,
    duty_percent: int,
    dedicated_host: bool = False,
    throughput_mode: bool = False,
) -> None:
    if type(throughput_mode) is not bool:
        raise ValueError("Throughput mode must be an explicit boolean")
    if type(batch_tokens) is not int or batch_tokens not in (
        (16, 32, 64, 256) if throughput_mode else (16, 32, 64)
    ):
        raise ValueError("Remote helper batch must be 16, 32 or 64 tokens")
    maximum = 100 if dedicated_host is True or throughput_mode else 65
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


def origin_enabled(root: Path | None, value: str) -> bool:
    """Apply Synta's persistent per-origin policy before a remote client is built.

    The stopped Windows RTX 3090 is opt-in only. Other private origins remain
    available by default, with an explicit false value serving as an opt-out.
    """
    origin = private_origin(value)
    if root is None:
        return origin != WINDOWS_3090_ORIGIN
    try:
        from rlm.v100.mission_chat import preferences

        configured = preferences(root).get(REMOTE_ORIGIN_POLICY_KEY, {})
    except (OSError, ValueError, TypeError):
        # An unreadable policy must never silently re-enable the stopped host.
        return origin != WINDOWS_3090_ORIGIN
    if not isinstance(configured, dict):
        return origin != WINDOWS_3090_ORIGIN
    if origin == WINDOWS_3090_ORIGIN:
        return configured.get(origin) is True
    return configured.get(origin) is not False


def remote_profile(profile: dict) -> bool:
    return profile.get("resources", {}).get("device") == "remote"


def validate_remote(profile: dict) -> None:
    runtime, server, resources = profile["runtime"], profile["server"], profile["resources"]
    private_origin(runtime["base_url"])
    if runtime.get("backend") == LLAMACPP_BACKEND:
        LlamaCppResearchClient._validate_identity(resources.get("llamacpp_identity", {}))
        identity = resources["llamacpp_identity"]
        if (
            runtime.get("tool_protocol") != "native"
            or runtime.get("model_name") != identity["model"]["id"]
            or runtime.get("context_window") != identity["server"]["context"]
            or server.get("slots") != 1
            or server.get("context_per_slot") != identity["server"]["context"]
            or server.get("model")
            or server.get("binary")
            or server.get("draft_model")
            or type(resources.get("helper_duty_percent", 65)) is not int
            or not 1 <= resources.get("helper_duty_percent", 65) <= 65
        ):
            raise ValueError(
                "Remote llama.cpp researcher profile does not match its exact identity"
            )
        return
    validate_workload(
        resources.get("helper_batch_tokens", 64),
        resources.get("helper_duty_percent", 65),
        resources.get("dedicated_host", False),
        resources.get("throughput_mode", False),
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
        if guard.get("contract") != expected:
            reason_code = "resource-contract-mismatch"
        elif type(age) not in (int, float) or not math.isfinite(age):
            reason_code = "resource-heartbeat-invalid"
        elif age < 0:
            reason_code = "resource-heartbeat-from-future"
        elif age > 10:
            reason_code = "resource-heartbeat-stale"
        else:
            reason = str(guard.get("reason", "")).lower()
            if "quarantin" in reason or "cooling" in reason:
                reason_code = "resource-gpu-quarantined-or-cooling-down"
            elif "busy" in reason:
                reason_code = "resource-guard-busy"
            elif "digest" in reason and "install" in reason:
                reason_code = "model-digest-not-installed"
            elif "metadata" in reason:
                reason_code = "model-metadata-unavailable"
            elif "update" in reason:
                reason_code = "owned-update-window"
            else:
                reason_code = "resource-guard-not-ready"
        raise ResourceGuardUnavailable(
            label + " adaptive guard unavailable: " + "; ".join(problems), reason_code
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
        throughput_mode: bool = False,
        **kwargs: Any,
    ):
        origin = private_origin(base_url)
        validate_workload(helper_batch_tokens, helper_duty_percent, dedicated_host, throughput_mode)
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
            "gemma4:26b-a4b-it-q4_K_M",
            "qwen3.8:27b-q4_K_M",
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
            for attempt in range(2):
                with method(
                    self.base_url + endpoint,
                    timeout=(min(5, self.remaining_timeout()), self.remaining_timeout()),
                    allow_redirects=False,
                    stream=True,
                    **arguments,
                ) as response:
                    if 300 <= response.status_code < 400:
                        raise ValueError("Remote helper redirected the request")
                    try:
                        response.raise_for_status()
                    except requests.HTTPError as failure:
                        body = bytearray()
                        for block in response.iter_content(512):
                            body.extend(block)
                            if len(body) > 2048:
                                break
                        packet = None
                        if len(body) <= 2048:
                            try:
                                packet = json.loads(body)
                            except (ValueError, UnicodeError):
                                pass
                        detail = (
                            packet.get("error")
                            if isinstance(packet, dict) and isinstance(packet.get("error"), str)
                            else None
                        )
                        resources = packet.get("resources") if isinstance(packet, dict) else None
                        cooldown = (
                            resources.get("cooldown_seconds")
                            if isinstance(resources, dict)
                            else None
                        )
                        explicit_admission = (
                            response.status_code == 503
                            and endpoint == "/api/chat"
                            and data is not None
                            and detail is not None
                            and detail.startswith("Owned GPU cooldown:")
                            and isinstance(resources, dict)
                            and resources.get("contract") == "synta-adaptive-windows-v1"
                            and resources.get("ready") is True
                            and type(cooldown) in (int, float)
                            and math.isfinite(cooldown)
                            and 0 < cooldown <= 300
                        )
                        if explicit_admission and attempt == 0:
                            # The proxy rejects this request before forwarding it.
                            # Its published snapshot may lag by one sensor interval.
                            delay = cooldown + 2.1
                            if delay < self.remaining_timeout():
                                time.sleep(delay)
                                continue
                        if explicit_admission:
                            # Both attempts were rejected before proxy admission. Preserve
                            # that distinction so the owning worker can requeue this
                            # read-only turn instead of recording a terminal model error.
                            retry_after = max(15, min(600, math.ceil(cooldown + 2.1)))
                            raise OwnedGpuCooldown(
                                str(failure) + "; remote error: " + detail,
                                response,
                                retry_after,
                            ) from failure
                        if detail:
                            raise requests.HTTPError(
                                str(failure) + "; remote error: " + detail,
                                response=response,
                                request=failure.request,
                            ) from failure
                        raise
                    if endpoint == "/api/chat" and data and data.get("stream"):
                        from rlm.v100.streaming import ollama_reply

                        return ollama_reply(response.iter_lines(chunk_size=128), self.stream_delta)
                    body = bytearray()
                    for block in response.iter_content(65536):
                        body.extend(block)
                        if len(body) > 2 * 1024**2:
                            raise ValueError("Remote helper response exceeds its byte budget")
                    result = json.loads(body)
                    break
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
        families, tokenizers = metadata_contract(self.model_name)
        if (
            info.get("details", {}).get("family") not in families
            or info.get("details", {}).get("quantization_level") != ROLE_MODELS[self.model_name][0]
            or info.get("model_info", {}).get("tokenizer.ggml.model") not in tokenizers
            or info.get("system")
            or info.get("remote_host")
            or info.get("remote_model")
            or len(info.get("template", "").encode()) > 16384
        ):
            raise ValueError(
                "Remote helper metadata differs: "
                + json.dumps(
                    {
                        "family": info.get("details", {}).get("family"),
                        "quantization": info.get("details", {}).get("quantization_level"),
                        "tokenizer": info.get("model_info", {}).get("tokenizer.ggml.model"),
                        "custom_system": bool(info.get("system")),
                        "template_bytes": len(info.get("template", "").encode()),
                    }
                )
            )
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
        if self.model_name.startswith("gemma4:"):
            text = text.replace(" ", "\u2581")
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
                threads = guard.get("cpu_threads")
                if type(threads) is int and 1 <= threads <= 64:
                    payload["options"]["num_thread"] = threads
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


class LlamaCppResearchClient(OllamaResearchClient):
    """Opt-in native llama.cpp client with pinned runtime and model identity.

    This class is intentionally not selected by existing dispatch code. The
    remote identity is content-addressed by the server contract, never by the
    human-readable model ID returned from ``/v1/models``.
    """

    IDENTITY_SCHEMA = "synta-llamacpp-identity-v1"
    IDENTITY_FIELDS = {
        "schema",
        "runtime",
        "model",
        "server",
    }

    def __init__(self, *, base_url: str, profile_pins: dict, **kwargs: Any):
        origin = private_origin(base_url)
        if not isinstance(profile_pins, dict) or set(profile_pins) != self.IDENTITY_FIELDS:
            raise ValueError("Native llama.cpp profile pins must match the exact identity contract")
        self._validate_identity(profile_pins)
        self.profile_pins = copy.deepcopy(profile_pins)
        self.model_digest = profile_pins["model"]["sha256"]
        self.metadata_sha256 = profile_pins["model"]["template_sha256"]
        self.max_vram_gib = 0
        kwargs.setdefault("model_name", profile_pins["model"]["id"])
        kwargs.setdefault("context_window", profile_pins["server"]["context"])
        if kwargs["model_name"] != profile_pins["model"]["id"]:
            raise ValueError("Native llama.cpp model ID differs from its profile pin")
        if kwargs["context_window"] != profile_pins["server"]["context"]:
            raise ValueError("Native llama.cpp context differs from its profile pin")
        super(OllamaResearchClient, self).__init__(base_url="http://127.0.0.1:11435", **kwargs)
        self.base_url = origin
        self.request_lock = threading.Lock()
        self.tool_protocol = "native"
        self.dedicated_host = False
        self.helper_batch_tokens = 64
        self.helper_duty_percent = 65
        self.require_adaptive_windows_guard = False

    @classmethod
    def _validate_identity(cls, identity: dict) -> None:
        if set(identity) != cls.IDENTITY_FIELDS or identity.get("schema") != cls.IDENTITY_SCHEMA:
            raise ValueError("Native llama.cpp identity schema mismatch")
        expected_shapes = {
            "runtime": {"version", "binary_path", "binary_sha256"},
            "model": {
                "id",
                "path",
                "sha256",
                "template_path",
                "template_sha256",
                "tokenizer_path",
                "tokenizer_sha256",
            },
            "server": {"context", "parallel", "gpu_layers"},
        }
        for key, fields in expected_shapes.items():
            value = identity.get(key)
            if not isinstance(value, dict) or set(value) != fields:
                raise ValueError("Native llama.cpp identity fields are incomplete or unexpected")
        runtime, model, server = identity["runtime"], identity["model"], identity["server"]
        hashes = (
            runtime["binary_sha256"],
            model["sha256"],
            model["template_sha256"],
            model["tokenizer_sha256"],
        )
        if (
            not isinstance(runtime["version"], str)
            or not runtime["version"]
            or not isinstance(runtime["binary_path"], str)
            or not runtime["binary_path"]
            or not isinstance(model["id"], str)
            or not model["id"]
            or not isinstance(model["path"], str)
            or not model["path"]
            or not isinstance(model["template_path"], str)
            or not model["template_path"]
            or not isinstance(model["tokenizer_path"], str)
            or not model["tokenizer_path"]
            or any(
                not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                for value in hashes
            )
            or type(server["context"]) is not int
            or server["context"] < 1
            or type(server["parallel"]) is not int
            or server["parallel"] != 1
            or not (
                server["gpu_layers"] == "all"
                or type(server["gpu_layers"]) is int
                and server["gpu_layers"] >= 0
            )
        ):
            raise ValueError("Native llama.cpp identity values are invalid")

    def remote_request(self, endpoint: str, data: dict | None = None) -> dict:
        methods = {
            "/v1/chat/completions": "post",
            "/apply-template": "post",
            "/tokenize": "post",
            "/v1/models": "get",
            "/props": "get",
            "/api/synta-resources": "get",
            "/api/synta-identity": "get",
        }
        method_name = methods.get(endpoint)
        if method_name is None or (data is None) != (method_name == "get"):
            raise ValueError("Native llama.cpp endpoint or method is not permitted")
        with requests.Session() as session:
            session.trust_env = False
            method = getattr(session, method_name)
            arguments = {} if data is None else {"json": data}
            if data is not None:
                arguments["headers"] = {
                    "X-Synta-Task": re.sub(r"[^A-Za-z0-9 ._:/-]", "", str(self.activity_actor))[
                        :100
                    ],
                    "X-Synta-Request": re.sub(
                        r"[^A-Za-z0-9_-]",
                        "",
                        str(
                            self.activity_context.get(
                                "chat_request_id", self.activity_context.get("request_id", "")
                            )
                        ),
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
                    raise ValueError("Native llama.cpp server redirected the request")
                response.raise_for_status()
                body = bytearray()
                for block in response.iter_content(65536):
                    body.extend(block)
                    if len(body) > 2 * 1024**2:
                        raise ValueError("Native llama.cpp response exceeds its byte budget")
        result = json.loads(body)
        if not isinstance(result, dict):
            raise ValueError("Native llama.cpp returned a non-object response")
        return result

    def identity(self) -> dict:
        identity = self.remote_request("/api/synta-identity")
        self._validate_identity(identity)
        if identity != self.profile_pins:
            raise ValueError(
                "Native llama.cpp runtime, model, template, tokenizer or server pins changed"
            )
        models = self.remote_request("/v1/models").get("data")
        if not isinstance(models, list) or not any(
            isinstance(row, dict) and row.get("id") == self.model_name for row in models
        ):
            raise ValueError("Pinned llama.cpp model ID is not available")
        # Server props and guard telemetry are status only; neither can replace
        # content-addressed identity or establish residency/peak VRAM.
        props = self.remote_request("/props")
        resources = self.remote_request("/api/synta-resources")
        self.last_remote_status = {"props": props, "resources": resources}
        return identity

    def count_text(self, text: str, parse_special: bool = False) -> int:
        result = self.remote_request(
            "/tokenize", {"content": text, "add_special": False, "parse_special": parse_special}
        )
        tokens = result.get("tokens")
        if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
            raise ValueError("Native llama.cpp tokenizer returned invalid token IDs")
        return len(tokens)

    def template_args(self) -> dict:
        if self.enable_thinking is None:
            return {}
        args = {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        if not self.enable_thinking:
            args["reasoning_effort"] = "none"
        return args

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        self.identity()
        rendered = self.remote_request(
            "/apply-template", {"messages": messages, **self.template_args()}
        ).get("prompt")
        if not isinstance(rendered, str):
            raise ValueError("Native llama.cpp did not return the rendered template")
        return self.count_text(rendered, parse_special=True) + 1

    def http_request(self, endpoint: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        if endpoint == "/apply-template":
            if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
                raise ValueError("Native template rendering needs messages")
            result = self.remote_request(endpoint, data)
            if not isinstance(result.get("prompt"), str):
                raise ValueError("Native llama.cpp did not return a rendered template")
            return result
        if endpoint != "/v1/chat/completions" or not isinstance(data, dict):
            raise ValueError("Native llama.cpp client cannot change endpoints")
        if data.get("model") != self.model_name or data.get("tools") or data.get("stream"):
            raise ValueError(
                "Native researcher requires pinned model and bounded non-streaming text"
            )
        maximum = data.get("max_tokens")
        prompt_tokens = self.count_messages(data.get("messages", []))
        if (
            type(maximum) is not int
            or not 1 <= maximum <= 8192
            or prompt_tokens + maximum > self.context_window
        ):
            raise ValueError("Native rendered prompt and output exceed pinned context")
        with self.request_lock, self.workload_slot() as quota:
            self.identity()
            started = time.monotonic()
            try:
                response = self.remote_request(endpoint, data)
            finally:
                self.reserve_helper_idle(quota, max(0, time.monotonic() - started))
        usage = response.get("usage")
        if not isinstance(usage, dict):
            raise ValueError("Native llama.cpp omitted measured token usage")
        incoming, outgoing = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if (
            type(incoming) is not int
            or type(outgoing) is not int
            or incoming <= 0
            or outgoing <= 0
            or incoming > prompt_tokens
            or outgoing > maximum
            or incoming + outgoing > self.context_window
        ):
            raise ValueError("Native llama.cpp returned invalid measured token usage")
        response.setdefault("timings", {})
        response["timings"].update(
            {
                "count_mode": "llama.cpp-rendered-template-and-tokenizer",
                "prompt_budget": prompt_tokens,
            }
        )
        return response


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
    if remote.exists():
        # Guard: never return RTX3090 path when operator has disabled the Windows helper.
        # Returning it to callers that bypass helper_client would cause fail-open on Windows.
        from rlm.v100.mission_chat import preferences
        if preferences(root).get("remote_helper_enabled", True) is False:
            raise ValueError(
                "Windows RTX helper is disabled by operator; selected_helper refuses RTX3090 path"
            )
        # Load the base_url from the profile to check origin policy
        import json as _json
        try:
            _profile_url = _json.loads(remote.read_text()).get("runtime", {}).get("base_url", "")
        except (OSError, ValueError):
            _profile_url = ""
        if _profile_url and not origin_enabled(root, _profile_url):
            raise ValueError(
                "Windows RTX helper origin is disabled by resource policy; selected_helper refuses RTX3090 path"
            )
        return remote
    path = root / "research/researcher-cpu.toml"
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
