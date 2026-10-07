"""Opt-in LAN Ollama researcher; the V100 learner remains strictly local.

Ollama has no public tokenize endpoint. Text-only Qwen byte-BPE requests use
an intentionally pessimistic UTF-8 budget plus template margin. This is an
admission estimate, never an exact token count or a quality baseline.
"""

import copy
import ipaddress
import json
import re
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from rlm.clients.llamacpp import LlamaCppClient
from rlm.v100.common import atomic_json

MODEL = "qwen3.5:9b-q8_0"
BACKEND = "ollama-research"


def private_origin(value: str) -> str:
    parsed = urlparse(value)
    address = ipaddress.ip_address(parsed.hostname or "")
    if (
        parsed.scheme != "http"
        or address.version != 4
        or not any(
            address in ipaddress.ip_network(net)
            for net in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
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
    if (
        runtime.get("backend") != BACKEND
        or runtime.get("tool_protocol") != "json"
        or runtime["model_name"] != MODEL
        or runtime.get("enable_thinking") is not False
        or not re.fullmatch(r"[0-9a-f]{64}", resources.get("model_digest", ""))
        or not re.fullmatch(r"[0-9a-f]{64}", resources.get("metadata_sha256", ""))
        or runtime["context_window"] not in (8192, 16384, 32768)
        or runtime["max_output_tokens"] > 1024
        or server["slots"] != 1
        or server.get("model")
        or server.get("binary")
        or server.get("draft_model")
        or resources.get("max_vram_gib") != 12
    ):
        raise ValueError("Remote helper must be a pinned text-only local Qwen researcher profile")


def metadata_sha(info: dict) -> str:
    import hashlib

    # Include the renderer and system text; a changed server template is not
    # silently accepted as the same experiment.
    return hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()


class OllamaResearchClient(LlamaCppClient):
    def __init__(
        self,
        *,
        base_url: str,
        model_digest: str,
        metadata_sha256: str = "",
        max_vram_gib: int = 12,
        **kwargs: Any,
    ):
        origin = private_origin(base_url)
        if kwargs.get("model_name", MODEL) != MODEL or not re.fullmatch(
            r"[0-9a-f]{64}", model_digest
        ):
            raise ValueError("Only the explicitly pinned local Qwen model is permitted")
        kwargs["model_name"] = MODEL
        if kwargs.get("enable_thinking", False) is not False:
            raise ValueError("Remote helper uses bounded non-thinking research")
        kwargs["enable_thinking"] = False
        # Reuse local usage/activity bookkeeping without relaxing the local
        # llama.cpp client's loopback-only contract.
        super().__init__(base_url="http://127.0.0.1:11435", **kwargs)
        self.base_url = origin
        self.model_digest = model_digest
        self.metadata_sha256 = metadata_sha256
        self.max_vram_gib = max_vram_gib
        self.request_lock = threading.Lock()
        self.tool_protocol = "json"

    def remote_request(self, endpoint: str, data: dict | None = None) -> dict:
        if endpoint not in ("/api/tags", "/api/show", "/api/ps", "/api/chat"):
            raise ValueError("Remote research transport does not expose model management")
        with requests.Session() as session:
            session.trust_env = False
            method = session.get if data is None else session.post
            arguments = {} if data is None else {"json": data}
            with method(
                self.base_url + endpoint,
                timeout=(5, self.timeout),
                allow_redirects=False,
                stream=True,
                **arguments,
            ) as response:
                if 300 <= response.status_code < 400:
                    raise ValueError("Remote helper redirected the request")
                response.raise_for_status()
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
            info.get("details", {}).get("family") != "qwen35"
            or info.get("details", {}).get("quantization_level") != "Q8_0"
            or info.get("model_info", {}).get("tokenizer.ggml.model") != "gpt2"
            or info.get("system")
            or info.get("remote_host")
            or info.get("remote_model")
            or len(info.get("template", "").encode()) > 16384
        ):
            raise ValueError("Remote helper metadata differs from supported text byte-BPE model")
        if self.metadata_sha256 and metadata_sha(info) != self.metadata_sha256:
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
            or item.get("size_vram", 0) < item.get("size", 1)
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
            or not 1 <= maximum <= 1024
            or prompt_budget + maximum > self.context_window
        ):
            raise ValueError("Remote text/context budget exceeded; reduce retrieved material")
        options = {"num_ctx": self.context_window, "num_predict": maximum}
        for name in ("temperature", "seed", "top_p", "top_k"):
            if name in data:
                options[name] = data[name]
        payload = {
            "model": self.model_name,
            "messages": data["messages"],
            "stream": False,
            "think": False,
            "options": options,
            "keep_alive": -1,
        }
        shape = data.get("response_format")
        if shape:
            if shape.get("type") != "json_object":
                raise ValueError("Remote helper requires JSON-object output")
            payload["format"] = shape.get("schema", "json")
        with self.request_lock:
            self.identity()
            self.loaded()
            response = self.remote_request("/api/chat", payload)
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
        "max_vram_gib": 12,
        "min_available_ram_gib": 0,
    }
    chosen["memory"].update(chunk_tokens=256, fanout=2, retrieve_count=2)
    validate_remote(chosen)
    client = OllamaResearchClient(
        base_url=url, model_name=MODEL, model_digest=digest, context_window=context, timeout=300
    )
    chosen["resources"]["metadata_sha256"] = metadata_sha(client.identity())
    client.loaded()
    destination = root / "research/researcher-rtx3090.json"
    if destination.exists():
        raise FileExistsError("Remote profile already exists; review it before replacing")
    atomic_json(destination, chosen)
    return destination


def selected_helper(root: Path) -> Path:
    remote = root / "research/researcher-rtx3090.json"
    path = remote if remote.exists() else root / "research/researcher-cpu.toml"
    if not path.exists():
        raise ValueError("Missing prepared researcher")
    return path
