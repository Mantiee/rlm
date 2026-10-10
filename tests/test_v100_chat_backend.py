import copy
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlm.v100 import (
    campaign,
    chat_backend,
    competition,
    mission,
    mission_chat,
    research_tools,
)
from rlm.v100.common import atomic_json
from tests.test_v100_continual import profile


def admitted(root, monkeypatch):
    accepted = profile(root)
    accepted["resources"] = {"accepted_cpu_chat": True, "interactive_lab": True}
    client = SimpleNamespace(
        model_name="v100", request=lambda *a: {"model_path": accepted["server"]["model"]}
    )
    events = []

    @contextmanager
    def server(path, root, log, **kwargs):
        events.append(("start", json.loads(path.read_text())))
        try:
            yield json.loads(path.read_text())
        finally:
            events.append(("stop",))

    monkeypatch.setattr(chat_backend, "managed_server", server)
    monkeypatch.setattr(chat_backend, "available_ram_gib", lambda: 24)
    monkeypatch.setattr(chat_backend, "helper_client", lambda *a: client)
    monkeypatch.setattr(
        chat_backend, "assert_served_expert", lambda *a: events.append(("verified", a[3]))
    )
    return accepted, client, events


def test_cpu_clone_preserves_weights_and_original_profile(tmp_path):
    accepted = profile(tmp_path)
    original = copy.deepcopy(accepted)
    clone = chat_backend.cpu_profile(accepted)
    assert accepted == original
    assert clone["server"]["model"] == accepted["server"]["model"]
    assert clone["server"]["gpu_layers"] == 0 and clone["server"]["draft_model"] == ""
    assert clone["runtime"]["context_window"] == 8192
    assert clone["server"]["threads"] == 2
    assert clone["resources"]["device"] == "cpu"
    competition.validate_concurrent_researcher(clone)
    accepted["runtime"]["backend"] = "isolated-architecture"
    with pytest.raises(ValueError, match="native GGUF"):
        chat_backend.cpu_profile(accepted)


def test_cpu_chat_reuses_verified_weights_and_reloads_new_accepted_version(tmp_path, monkeypatch):
    accepted, client, events = admitted(tmp_path, monkeypatch)
    manager = chat_backend.AcceptedCPUChat(tmp_path, tmp_path)
    assert manager.get(accepted) is client
    digest = manager.sha256
    assert manager.get(accepted) is client
    assert len([e for e in events if e[0] == "start"]) == 1
    Path(accepted["server"]["model"]).write_bytes(b"new accepted weights")
    assert manager.get(accepted) is client
    assert manager.sha256 != digest
    assert [e[0] for e in events] == ["start", "verified", "verified", "stop", "start", "verified"]
    manager.close()
    assert events[-1] == ("stop",)


def test_no_cpu_launch_when_ram_insufficient_or_mission_stopping(tmp_path, monkeypatch):
    accepted, _, events = admitted(tmp_path, monkeypatch)
    stop = threading.Event()
    manager = chat_backend.AcceptedCPUChat(tmp_path, tmp_path, stop)
    monkeypatch.setattr(chat_backend, "available_ram_gib", lambda: 8)
    with pytest.raises(ValueError, match="available RAM"):
        manager.get(accepted)
    stop.set()
    with pytest.raises(RuntimeError, match="stopping"):
        manager.get(accepted)
    assert events == []


@pytest.mark.parametrize("pressure", [True, False])
def test_cpu_chat_retires_on_pressure_or_idle(tmp_path, monkeypatch, pressure):
    accepted, _, events = admitted(tmp_path, monkeypatch)
    manager = chat_backend.AcceptedCPUChat(tmp_path, tmp_path)
    manager.get(accepted)
    if pressure:
        monkeypatch.setattr(chat_backend, "available_ram_gib", lambda: 3)
    else:
        manager.last_used -= 181
    manager.maintain()
    assert manager.client is None and events[-1] == ("stop",)


def test_wrong_cpu_endpoint_is_not_reused(tmp_path, monkeypatch):
    accepted, client, events = admitted(tmp_path, monkeypatch)
    manager = chat_backend.AcceptedCPUChat(tmp_path, tmp_path)
    manager.get(accepted)
    client.request = lambda *a: {"model_path": "/unaccepted-candidate.gguf"}
    with pytest.raises(ValueError, match="accepted weights"):
        manager.get(accepted)
    assert events[-1] == ("stop",)


def test_chat_selects_same_accepted_master_during_gpu_training(tmp_path, monkeypatch):
    accepted, cpu, _ = admitted(tmp_path, monkeypatch)
    cpu.timeout = 600
    cpu.sampling_args = {"max_tokens": 1024}
    cpu.activity_context = {}
    atomic_json(tmp_path / "serving-active.json", accepted)
    monkeypatch.setattr(
        mission,
        "status",
        lambda r: {"running": True, "state": {"phase": "training"}, "learning": {}},
    )
    monkeypatch.setattr(mission_chat, "load_profile", lambda *a: accepted)
    gpu = SimpleNamespace(request=lambda *a: {"model_path": "/candidate.gguf"}, model_name="v100")
    monkeypatch.setattr(competition, "helper_client", lambda *a: gpu)
    manager = chat_backend.AcceptedCPUChat(tmp_path, tmp_path)
    seen = []
    monkeypatch.setattr(
        research_tools,
        "research_turn",
        lambda c, *a: seen.append(c) or {"content": '{"answer":"accepted master", "actions":[]}'},
    )
    result = mission_chat.respond(tmp_path, tmp_path, {"message": "Sprawdź stan misji"}, manager)
    assert seen == [cpu]
    assert result["responder"]["accepted_master_on_cpu"]
    assert not result["responder"]["delegated_while_master_busy"]
    assert result["responder"]["accepted_model_sha256"] == manager.sha256
    manager.close()


def test_upgrade_keeps_current_input_before_first_learning_cycle(tmp_path, monkeypatch):
    run = tmp_path / "research/mission/run-current"
    atomic_json(run / "input-profile.json", {"accepted": True})
    monkeypatch.setattr(mission, "status", lambda r: {"run": str(run), "learning": {}})
    assert campaign.choose_profile(tmp_path) == run / "input-profile.json"


@pytest.mark.parametrize("cause", ["rss", "ram", "shutdown"])
def test_resource_guard_retires_only_owned_cpu_process(tmp_path, monkeypatch, cause):
    calls = []
    process = SimpleNamespace(
        pid=123,
        poll=lambda: None,
        terminate=lambda: calls.append(("terminate", 123)),
        wait=lambda timeout=None: calls.append(("wait", 123)),
    )
    stop = SimpleNamespace(wait=lambda seconds: False)
    cancel = threading.Event()
    if cause == "shutdown":
        cancel.set()
    monkeypatch.setattr(
        competition.psutil,
        "Process",
        lambda pid: SimpleNamespace(
            memory_info=lambda: SimpleNamespace(rss=(20 if cause == "rss" else 1) * 2**30)
        ),
    )
    monkeypatch.setattr(competition, "available_ram_gib", lambda: 2 if cause == "ram" else 24)
    competition.guard_cpu_process(process, {"min_free_ram_gib": 4, "max_rss_gib": 14}, stop, cancel)
    assert calls == [("terminate", 123), ("wait", 123)]


def test_capability_question_needs_no_inference_or_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(competition, "helper_client", lambda *a, **k: pytest.fail("LLM not needed"))
    result = mission_chat.respond(
        tmp_path, tmp_path, {"message": "co potrafisz robic i jak. w punktach"}
    )
    assert result["responder"]["model"] == "controller-capabilities"
    assert result["actions"] == []
    assert "nie gwarantuję zysku" in result["answer"]


@pytest.mark.parametrize(
    "message", ["a co z zarabianiem", "co robisz z wynikami", "jak działa ML?"]
)
def test_readonly_question_uses_conversation_route(message):
    assert mission_chat.conversation_question(message)


@pytest.mark.parametrize(
    "message",
    ["napraw dashboard", "czy możesz uruchomić test", "co zrobić? zaimplementuj test", "zmien cel"],
)
def test_explicit_work_keeps_executable_route(message):
    assert not mission_chat.conversation_question(message)


def test_conversation_is_one_real_generation_without_template_or_tools(tmp_path):
    calls = []
    client = SimpleNamespace(
        model_name="v100",
        timeout=180,
        template_args=lambda: {"chat_template_kwargs": {"enable_thinking": False}},
        request=lambda endpoint, payload: calls.append((endpoint, payload))
        or {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "Cel pozostaje ten sam; nie ma dowodu zysku."},
                }
            ]
        },
    )
    result = mission_chat.conversational_response(
        tmp_path,
        client,
        "a co z zarabianiem",
        {"running": True, "state": {"phase": "income-research"}},
        {"long": {"text": "Operator goal"}},
        {"model": "v100"},
    )
    assert len(calls) == 1 and calls[0][0] == "/v1/chat/completions"
    assert calls[0][1]["max_tokens"] == 512
    assert "tools" not in calls[0][1]
    assert "Operator goal" in calls[0][1]["messages"][0]["content"]
    assert result["applied"] == [] and client.timeout == 20


def test_conversation_rejects_truncated_generation(tmp_path):
    client = SimpleNamespace(
        model_name="v100",
        timeout=60,
        template_args=lambda: {},
        request=lambda *a: {
            "choices": [{"finish_reason": "length", "message": {"content": "partial"}}]
        },
    )
    with pytest.raises(ValueError, match="incomplete"):
        mission_chat.conversational_response(tmp_path, client, "hej", {}, {}, {"model": "v100"})


def test_question_fast_path_skips_cpu_start_and_research_context(tmp_path, monkeypatch):
    accepted = profile(tmp_path)
    atomic_json(tmp_path / "serving-active.json", accepted)
    monkeypatch.setattr(mission_chat, "load_profile", lambda *a: accepted)
    monkeypatch.setattr(
        mission, "status", lambda *a: {"running": True, "state": {}, "learning": {}}
    )
    calls = []
    client = SimpleNamespace(
        model_name="v100",
        timeout=120,
        sampling_args={},
        activity_context={},
        template_args=lambda: {},
    )

    def request(endpoint, payload=None):
        calls.append((endpoint, client.timeout))
        if endpoint == "/props":
            return {"model_path": accepted["server"]["model"]}
        return {"choices": [{"finish_reason": "stop", "message": {"content": "Odpowiedź modelu"}}]}

    client.request = request
    monkeypatch.setattr(competition, "helper_client", lambda *a: client)

    def forbidden(*a, **kw):
        pytest.fail("Question must not start a CPU model or research/tool loop")

    monkeypatch.setattr(research_tools, "research_turn", forbidden)
    manager = SimpleNamespace(get=forbidden)
    result = mission_chat.respond(tmp_path, tmp_path, {"message": "jak działa ML?"}, manager)
    assert result["answer"] == "Odpowiedź modelu"
    assert calls == [("/props", 10), ("/v1/chat/completions", 20)]


def test_connection_failures_stop_retrying_and_release_next_request(tmp_path, monkeypatch):
    import requests

    first = mission_chat.submit(tmp_path, "first")
    second = mission_chat.submit(tmp_path, "second")
    with mission_chat.connect(tmp_path) as db:
        db.execute("UPDATE requests SET attempts=1 WHERE id=?", (first,))
    seen = []

    def respond(*args):
        request = args[2]
        seen.append(request["id"])
        if request["id"] == first:
            raise requests.ConnectionError("backend unavailable")
        return {"answer": "next response", "actions": [], "applied": []}

    monkeypatch.setattr(mission_chat, "respond", respond)

    class Stop:
        def wait(self, seconds):
            return len(seen) >= 2

    mission_chat.service_loop(tmp_path, tmp_path, Stop())
    assert mission_chat.inspect(tmp_path, first)["state"] == "failed"
    assert mission_chat.inspect(tmp_path, second)["state"] == "completed"
    assert seen == [first, second]
