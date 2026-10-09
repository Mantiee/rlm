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
