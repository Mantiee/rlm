import copy
import json
from pathlib import Path

import pytest

from rlm.clients.llamacpp import LlamaCppClient
from rlm.v100 import competition, mission, remote_helper
from rlm.v100.agent import native_turn
from rlm.v100.cli import server_command
from rlm.v100.common import atomic_json, load_profile
from rlm.v100.paper_agents import financial_helper_profile
from rlm.v100.tool_protocol import json_tool_turn

DIGEST = "a" * 64
URL = "http://192.168.0.61:11435"
INFO = {
    "details": {"family": "qwen35", "quantization_level": "Q8_0"},
    "model_info": {"tokenizer.ggml.model": "gpt2"},
    "template": "fixed official template",
}
LOADED = {
    "name": remote_helper.MODEL,
    "digest": DIGEST,
    "size": 10 * 2**30,
    "size_vram": 10 * 2**30,
    "context_length": 32768,
}


@pytest.fixture
def transport(monkeypatch):
    calls = []
    state = {"info": copy.deepcopy(INFO), "loaded": copy.deepcopy(LOADED), "digest": DIGEST}

    def request(self, endpoint, data=None):
        calls.append((endpoint, data))
        if endpoint == "/api/show":
            return state["info"]
        if endpoint == "/api/tags":
            return {"models": [{"name": remote_helper.MODEL, "digest": state["digest"]}]}
        if endpoint == "/api/ps":
            return {"models": [state["loaded"]]}
        assert endpoint == "/api/chat"
        return {
            "model": remote_helper.MODEL,
            "done": True,
            "done_reason": "stop",
            "message": {"content": '{"answer":"4"}', "thinking": "private reasoning"},
            "prompt_eval_count": 128,
            "eval_count": 8,
            "eval_duration": 100_000_000,
            **state.get("response", {}),
        }

    monkeypatch.setattr(remote_helper.OllamaResearchClient, "remote_request", request)
    return calls, state


def client(**kwargs):
    return remote_helper.OllamaResearchClient(
        base_url=URL,
        model_digest=DIGEST,
        metadata_sha256=remote_helper.metadata_sha(INFO),
        context_window=32768,
        **kwargs,
    )


def profile(root):
    return load_profile(Path(__file__).parents[1] / "profiles/v100.toml", root)


def prepared(root):
    path = remote_helper.prepare_remote(profile(root), root, URL, 32768, DIGEST)
    return path, load_profile(path, root)


@pytest.mark.parametrize("context", [65536, 131072])
def test_large_remote_context_still_requires_loaded_context_and_vram_budget(
    tmp_path, transport, context
):
    calls, state = transport
    state["loaded"]["context_length"] = context
    path = remote_helper.prepare_remote(profile(tmp_path), tmp_path, URL, context, DIGEST)
    chosen = load_profile(path, tmp_path)
    assert chosen["runtime"]["context_window"] == context
    state["loaded"]["size_vram"] = 13 * 2**30
    with pytest.raises(ValueError, match="GPU"):
        competition.helper_client(chosen).loaded()


@pytest.mark.parametrize(
    "url",
    [
        "http://8.8.8.8:11435",
        "http://100.78.125.51:11435",
        "http://localhost:11435",
        "http://192.168.0.61",
        "https://192.168.0.61:11435",
        URL + "/api",
        URL + "?other=host",
        "http://u:p@192.168.0.61:11435",
    ],
)
def test_remote_origin_is_explicit_private_ipv4(url):
    with pytest.raises(ValueError):
        remote_helper.private_origin(url)


def test_local_backend_remains_loopback_only():
    with pytest.raises(ValueError, match="loopback"):
        LlamaCppClient(base_url=URL)


def test_native_json_action_schema_and_measured_usage(transport, tmp_path):
    calls, _ = transport
    instance = client(activity_root=str(tmp_path), activity_actor="tester")
    result = json_tool_turn(
        instance,
        [{"role": "user", "content": "2+2?"}],
        [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
    )
    assert result["content"] == "4"
    payload = [item for endpoint, item in calls if endpoint == "/api/chat"][0]
    assert payload["think"] is False and payload["stream"] is False
    assert payload["options"]["num_ctx"] == 32768
    assert payload["format"]["oneOf"][0]["properties"]["tool"]["const"] == "calculate"
    assert "tools" not in payload
    text = "".join(
        path.read_text() for path in (tmp_path / "research/logs/activity").rglob("*.jsonl")
    )
    assert "private reasoning" not in text
    assert '"prompt_tokens": 128' in text
    assert '"remote_vram_gib": 10.0' in text
    assert instance.completion("2+2?") == '{"answer":"4"}'
    assert instance.get_last_usage().total_input_tokens == 128


def test_remote_context_admission_precedes_http(transport):
    calls, _ = transport
    with pytest.raises(ValueError, match="working context"):
        native_turn(client(), [{"role": "user", "content": "ą" * 20000}])
    assert not calls


@pytest.mark.parametrize(
    "change",
    [
        {"digest": "b" * 64},
        {"info": {**INFO, "system": "changed"}},
        {"info": {**INFO, "template": "changed"}},
        {"loaded": {**LOADED, "context_length": 8192}},
        {"loaded": {**LOADED, "size_vram": 13 * 2**30}},
        {"loaded": {**LOADED, "size_vram": 4 * 2**30}},
    ],
)
def test_changed_model_context_or_memory_blocks_research(transport, change):
    calls, state = transport
    state.update(change)
    with pytest.raises(ValueError):
        client().completion("2+2?")
    assert not any(endpoint == "/api/chat" for endpoint, _ in calls)


@pytest.mark.parametrize(
    "response",
    [
        {"done": False},
        {"prompt_eval_count": 50000},
        {"eval_count": 2000},
        {"model": "cloud:model"},
        {"done_reason": "unexpected"},
    ],
)
def test_incomplete_or_invalid_usage_not_accepted(transport, response):
    _, state = transport
    state["response"] = response
    with pytest.raises(ValueError):
        client().completion("2+2?")


def test_exhausted_output_not_accepted_as_native_turn(transport):
    _, state = transport
    state["response"] = {"done_reason": "length"}
    with pytest.raises(ValueError, match="exhausted|budget|truncated"):
        native_turn(client(), [{"role": "user", "content": "2+2?"}])


def test_remote_profile_keeps_context_and_never_spawns_or_stops_windows(
    transport, tmp_path, monkeypatch
):
    path, settings = prepared(tmp_path)
    selected = remote_helper.selected_helper(tmp_path)
    assert selected == path
    assert mission.setup_helper(settings) == settings
    snapshot = financial_helper_profile(path, tmp_path)
    assert load_profile(snapshot, tmp_path)["runtime"]["context_window"] == 32768
    assert settings["server"]["model"] == ""

    def forbidden(*args, **kwargs):
        raise AssertionError("Remote server must never launch or stop a local process")

    monkeypatch.setattr(competition.subprocess, "Popen", forbidden)
    monkeypatch.setattr(competition, "available_ram_gib", forbidden)
    with competition.managed_server(snapshot, tmp_path, tmp_path / "helper.log") as actual:
        instance = competition.helper_client(actual, tmp_path)
        assert isinstance(instance, remote_helper.OllamaResearchClient)
        assert instance.tool_protocol == "json"
        assert instance.activity_actor == "tester"
    assert URL in (tmp_path / "helper.log").read_text()
    with pytest.raises(ValueError, match="externally"):
        server_command(settings)


def test_remote_profile_failure_never_replaces_existing(transport, tmp_path):
    path, _ = prepared(tmp_path)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        prepared(tmp_path)
    assert path.read_bytes() == before
    changed = json.loads(before)
    changed["runtime"]["model_name"] = "qwen3.5:cloud"
    atomic_json(path, changed)
    with pytest.raises(ValueError, match="pinned"):
        load_profile(path, tmp_path)


def test_cpu_selected_when_remote_not_configured(tmp_path):
    path = tmp_path / "research/researcher-cpu.toml"
    path.parent.mkdir()
    path.touch()
    assert remote_helper.selected_helper(tmp_path) == path


def test_remote_transport_rejects_redirect_and_ignores_proxy(monkeypatch):
    class Response:
        status_code = 302

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url, **kwargs):
            assert url == URL + "/api/tags"
            assert self.trust_env is False and kwargs["allow_redirects"] is False
            return Response()

    monkeypatch.setattr(remote_helper.requests, "Session", Session)
    with pytest.raises(ValueError, match="redirected"):
        client().remote_request("/api/tags")
    with pytest.raises(ValueError, match="management"):
        client().remote_request("/api/pull", {"model": remote_helper.MODEL})
