import copy
import json
import multiprocessing
import threading
import time
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
from rlm.v100.serving import ServingLeaseUnavailable, endpoint_lease
from tests.test_v100_continual import profile


def _hold_candidate_endpoint(root, selected, ready, release):
    with endpoint_lease(root, selected, exclusive=True):
        ready.set()
        release.wait(10)


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


def profile_with_resources(tmp_path):
    """Give test-only profiles the optional section production profiles may omit."""
    selected = profile(tmp_path)
    selected.setdefault("resources", {})
    return selected


def test_candidate_listener_lease_excludes_chat_across_processes(tmp_path, monkeypatch):
    selected = profile_with_resources(tmp_path)
    selected["resources"].update(accepted_cpu_chat=True, interactive_lab=True)
    atomic_json(tmp_path / "serving-active.json", selected)
    monkeypatch.setattr(mission, "status", lambda root: {"state": {}})
    monkeypatch.setattr(mission_chat, "preferences", lambda root: {"remote_helper_enabled": False})
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    process = context.Process(
        target=_hold_candidate_endpoint, args=(tmp_path, selected, ready, release)
    )
    process.start()
    try:
        assert ready.wait(10), "candidate did not acquire its endpoint lease"
        with pytest.raises(ServingLeaseUnavailable):
            with endpoint_lease(tmp_path, selected, exclusive=False, timeout=0.15):
                pytest.fail("chat entered an endpoint while a candidate owned it")
        with mission_chat._chat_endpoint_leases(tmp_path, tmp_path, object()) as acquired:
            assert not acquired
        with mission_chat._chat_fallback_lease(tmp_path, tmp_path, object()):
            pass
    finally:
        release.set()
        process.join(12)
        if process.is_alive():
            process.terminate()
            process.join(5)
    assert process.exitcode == 0
    with endpoint_lease(tmp_path, selected, exclusive=False, timeout=0.5):
        pass


def test_chat_contention_skips_candidate_and_queues_without_fallback(tmp_path, monkeypatch):
    accepted = profile(tmp_path)
    atomic_json(tmp_path / "serving-active.json", accepted)
    monkeypatch.setattr(mission, "status", lambda root: {"state": {}})
    monkeypatch.setattr(mission_chat, "preferences", lambda root: {"remote_helper_enabled": False})
    identity = mission_chat.submit(tmp_path, "read only check")
    attempted = []

    @contextmanager
    def contended(*args, **kwargs):
        yield False

    def backend(root, directory, request, accepted_cpu=None, *, skip_local=False, **kwargs):
        attempted.append((accepted_cpu, skip_local))
        if not skip_local:
            pytest.fail("chat must never send a request to a candidate listener")
        raise mission_chat.BackendNotReady("accepted fallback unavailable")

    monkeypatch.setattr(mission_chat, "_chat_endpoint_leases", contended)
    monkeypatch.setattr(mission_chat, "respond", backend)
    with pytest.raises(mission_chat.BackendNotReady):
        mission_chat._respond_with_lifecycle(tmp_path, tmp_path, {"id": identity})
    assert attempted == [(None, True)]

    def unavailable(*args, **kwargs):
        raise mission_chat.BackendNotReady("accepted fallback unavailable")

    monkeypatch.setattr(mission_chat, "_respond_with_lifecycle", unavailable)

    class Stop:
        waits = 0

        def wait(self, seconds):
            self.waits += 1
            return self.waits > 1

    mission_chat.service_loop(tmp_path, tmp_path, Stop())
    result = mission_chat.inspect(tmp_path, identity)
    assert result["state"] == "queued"
    assert result["error"] == "accepted fallback unavailable"


def test_chat_contention_uses_cpu_fallback_with_skip_local(tmp_path, monkeypatch):
    seen = []
    accepted_cpu = object()
    accepted = profile_with_resources(tmp_path)
    accepted["resources"]["accepted_cpu_chat"] = True
    atomic_json(tmp_path / "serving-active.json", accepted)
    monkeypatch.setattr(mission, "status", lambda root: {"state": {}})
    monkeypatch.setattr(mission_chat, "preferences", lambda root: {"remote_helper_enabled": False})

    @contextmanager
    def contended(*args, **kwargs):
        yield False

    @contextmanager
    def cpu_lease(*args, **kwargs):
        seen.append("cpu-lease")
        yield

    def backend(root, directory, request, cpu, *, skip_local=False, **kwargs):
        seen.append((cpu, skip_local))
        if not skip_local:
            pytest.fail("chat must never send a request to a candidate listener")
        return {"answer": "delegated safely"}

    monkeypatch.setattr(mission_chat, "_chat_endpoint_leases", contended)
    monkeypatch.setattr(mission_chat, "_chat_fallback_lease", cpu_lease)
    monkeypatch.setattr(mission_chat, "respond", backend)
    result = mission_chat._respond_with_lifecycle(
        tmp_path, tmp_path, {"id": "request"}, accepted_cpu
    )
    assert result["answer"] == "delegated safely"
    assert seen == ["cpu-lease", (accepted_cpu, True)]


def test_chat_uses_the_profile_snapshot_whose_endpoint_was_leased(tmp_path, monkeypatch):
    accepted = profile(tmp_path)
    accepted["runtime"]["base_url"] = "http://127.0.0.1:8091"
    replacement = copy.deepcopy(accepted)
    replacement["runtime"]["base_url"] = "http://127.0.0.1:8092"
    atomic_json(tmp_path / "serving-active.json", accepted)
    monkeypatch.setattr(mission, "status", lambda root: {"state": {}})
    monkeypatch.setattr(mission_chat, "preferences", lambda root: {"remote_helper_enabled": False})
    seen = []
    original_leases = mission_chat._chat_endpoint_leases

    @contextmanager
    def replace_profile_after_resolution(root, directory, accepted_cpu=None, *, profiles=None):
        # Simulate a selection-file update after resolution but before response.
        # This turn must still use the immutable snapshot leased at port 8091.
        atomic_json(directory / "serving-active.json", replacement)
        with original_leases(root, directory, accepted_cpu, profiles=profiles) as acquired:
            yield acquired

    def backend(root, directory, request, accepted_cpu=None, *, lifecycle_snapshot=None, **kwargs):
        seen.append(lifecycle_snapshot["accepted"]["runtime"]["base_url"])
        return {"answer": "snapshot used"}

    monkeypatch.setattr(mission_chat, "_chat_endpoint_leases", replace_profile_after_resolution)
    monkeypatch.setattr(mission_chat, "respond", backend)
    result = mission_chat._respond_with_lifecycle(
        tmp_path, tmp_path, {"id": "race", "message": "hello"}
    )
    assert result["answer"] == "snapshot used"
    assert seen == ["http://127.0.0.1:8091"]


def test_disabled_remote_helpers_do_not_disable_accepted_cpu_fallback(tmp_path, monkeypatch):
    import requests

    from rlm.v100 import model_roles

    accepted = profile_with_resources(tmp_path)
    accepted["resources"].update(accepted_cpu_chat=True, interactive_lab=False)
    atomic_json(tmp_path / "serving-active.json", accepted)
    monkeypatch.setattr(mission, "status", lambda root: {"state": {}})
    monkeypatch.setattr(model_roles, "selected_profile", lambda *args: None)
    monkeypatch.setattr(mission_chat, "preferences", lambda root: {"remote_helper_enabled": False})
    gpu = SimpleNamespace(timeout=60, model_name="gpu", sampling_args={}, activity_context={})
    cpu = SimpleNamespace(
        timeout=60, model_name="accepted-cpu", sampling_args={}, activity_context={}
    )
    manager = SimpleNamespace(get=lambda profile: cpu, sha256="accepted-hash")
    monkeypatch.setattr(competition, "helper_client", lambda *args: gpu)
    monkeypatch.setattr(
        mission_chat,
        "assert_served_expert",
        lambda *args: (_ for _ in ()).throw(requests.ConnectionError("GPU listener transitioning")),
    )
    monkeypatch.setattr(
        mission_chat,
        "conversational_response",
        lambda root, client, *args, **kwargs: {"answer": client.model_name},
    )
    result = mission_chat.respond(tmp_path, tmp_path, {"message": "jak działa ML?"}, manager)
    assert result["answer"] == "accepted-cpu"


def test_managed_server_defaults_to_candidate_lifecycle_role(tmp_path, monkeypatch):
    selected = profile(tmp_path)
    path = tmp_path / "candidate.json"
    atomic_json(path, selected)

    @contextmanager
    def fake_server(profile_path, root, log_path, cancel=None):
        yield selected

    monkeypatch.setattr(competition, "_managed_server_unleased", fake_server)
    with competition.managed_server(path, tmp_path, tmp_path / "candidate.log"):
        with pytest.raises(ServingLeaseUnavailable):
            with endpoint_lease(tmp_path, selected, exclusive=False, timeout=0.1):
                pytest.fail("candidate server failed to exclude chat")
    with endpoint_lease(tmp_path, selected, exclusive=False, timeout=0.5):
        pass

    with competition.managed_server(
        path, tmp_path, tmp_path / "accepted.log", lifecycle_role="accepted"
    ):
        with endpoint_lease(tmp_path, selected, exclusive=True, timeout=0.1):
            pass


def test_cpu_clone_preserves_weights_and_original_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(chat_backend.os, "cpu_count", lambda: 12)
    accepted = profile(tmp_path)
    original = copy.deepcopy(accepted)
    clone = chat_backend.cpu_profile(accepted)
    assert accepted == original
    assert clone["server"]["model"] == accepted["server"]["model"]
    assert clone["server"]["gpu_layers"] == 0 and clone["server"]["draft_model"] == ""
    assert clone["runtime"]["context_window"] == 8192
    assert clone["server"]["threads"] == 8
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


@pytest.mark.parametrize("rejected_planner", [False, True, "disabled-origin"])
def test_chat_selects_same_accepted_master_during_gpu_training(
    tmp_path, monkeypatch, rejected_planner
):
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
    planner = tmp_path / "planner.json"
    disabled = copy.deepcopy(accepted)
    disabled["resources"]["device"] = "remote"
    disabled["runtime"]["base_url"] = "http://10.0.0.61:11435"
    from rlm.v100 import model_roles, remote_helper

    monkeypatch.setattr(remote_helper, "origin_enabled", lambda *a: False)

    monkeypatch.setattr(
        model_roles, "selected_profile", lambda *a: planner if rejected_planner else None
    )
    monkeypatch.setattr(
        mission_chat,
        "load_profile",
        lambda path, *a: (disabled if rejected_planner == "disabled-origin" else {"rejected": True})
        if path == planner
        else accepted,
    )
    gpu = SimpleNamespace(request=lambda *a: {"model_path": "/candidate.gguf"}, model_name="v100")

    def factory(settings, *args):
        assert settings.get("resources", {}).get("device") != "remote", (
            "Disabled optional planner must never construct a remote client"
        )
        if settings.get("rejected"):
            raise ValueError("Remote model metadata or template changed")
        return gpu

    monkeypatch.setattr(competition, "helper_client", factory)
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
        request=lambda endpoint, payload: (
            calls.append((endpoint, payload))
            or {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "Cel pozostaje ten sam; nie ma dowodu zysku."},
                    }
                ]
            }
        ),
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


@pytest.mark.parametrize("interactive_lab", [False, True])
def test_question_fast_path_skips_cpu_start_and_research_context(
    tmp_path, monkeypatch, interactive_lab
):
    accepted = profile(tmp_path)
    accepted["resources"] = {"interactive_lab": interactive_lab}
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
    monkeypatch.setattr(
        mission_chat,
        "assert_served_expert",
        lambda selected_client, selected_profile, selected_root: (
            selected_client.request("/props")["model_path"] == selected_profile["server"]["model"]
        )
        or pytest.fail("Accepted endpoint must report the selected model"),
    )

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
        stale = time.time() - 16
        db.execute("UPDATE requests SET attempts=1,updated=? WHERE id=?", (stale, first))
        db.execute("UPDATE requests SET updated=? WHERE id=?", (stale, second))
    seen = []

    def respond(*args, **kwargs):
        request = args[2]
        seen.append(request["id"])
        if request["id"] == first:
            raise requests.ConnectionError("backend unavailable")
        return {"answer": "next response", "actions": [], "applied": []}

    monkeypatch.setattr(mission_chat, "respond", respond)
    monkeypatch.setattr(
        mission_chat,
        "_respond_with_lifecycle",
        lambda root, directory, request, accepted_cpu=None: respond(
            root, directory, request, accepted_cpu
        ),
    )

    class Stop:
        checks = 0

        def wait(self, seconds):
            self.checks += 1
            return len(seen) >= 2 or self.checks > 4

    mission_chat.service_loop(tmp_path, tmp_path, Stop())
    assert mission_chat.inspect(tmp_path, first)["state"] == "failed"
    assert mission_chat.inspect(tmp_path, second)["state"] == "completed"
    assert seen == [first, second]


@pytest.mark.parametrize("operator_tools", [True, False])
def test_selected_planner_does_not_replace_operator_tool_policy(
    tmp_path, monkeypatch, operator_tools
):
    from rlm.v100 import agent, model_roles

    accepted = profile(tmp_path)
    accepted["resources"] = {"interactive_lab": operator_tools}
    atomic_json(tmp_path / "serving-active.json", accepted)
    planner_path = tmp_path / "planner.json"
    planner = copy.deepcopy(accepted)
    planner["runtime"]["backend"] = "ollama-research"
    planner["runtime"]["base_url"] = "http://10.0.0.9:11435"
    planner["resources"] = {"device": "remote", "interactive_lab": not operator_tools}
    monkeypatch.setattr(model_roles, "selected_profile", lambda *a: planner_path)
    monkeypatch.setattr(
        mission_chat, "load_profile", lambda path, *a: planner if path == planner_path else accepted
    )
    client = SimpleNamespace(
        model_name="planner", timeout=600, sampling_args={"max_tokens": 1024}, activity_context={}
    )
    monkeypatch.setattr(competition, "helper_client", lambda *a: client)
    monkeypatch.setattr(mission_chat, "backend_preflight", lambda *a: None)
    monkeypatch.setattr(mission, "status", lambda *a: {"running": True, "state": {}})
    calls = []

    def research(client, messages, schema, root):
        calls.append("tools")
        from rlm.v100.planning import read

        return {
            "content": '{"answer":"Recorded plan checked.","actions":[]}',
            "research_trace": [{"tool": "get_plan", "result": read(root)}],
        }

    def native(*a, **k):
        calls.append("native")
        return {"content": '{"answer":"No tool permissions.","actions":[]}'}

    monkeypatch.setattr(research_tools, "research_turn", research)
    monkeypatch.setattr(agent, "native_turn", native)
    result = mission_chat.respond(
        tmp_path, tmp_path, {"message": "Inspect the configured plan using get_plan"}
    )
    assert calls == (["tools"] if operator_tools else ["native"])
    assert bool(result["tool_receipts"]) == operator_tools
    assert planner["resources"]["interactive_lab"] == (not operator_tools)


def test_cpu_verification_deadline_does_not_wait_for_a_stalled_disk_read(tmp_path, monkeypatch):
    import time

    accepted, _, events = admitted(tmp_path, monkeypatch)
    manager = chat_backend.AcceptedCPUChat(tmp_path, tmp_path)
    release = threading.Event()

    def stalled(*a, **k):
        release.wait(2)
        return "a" * 64

    monkeypatch.setattr(chat_backend, "file_hash", stalled)
    manager.deadline = time.monotonic() + 0.05
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="verification exceeded"):
            manager.get(accepted)
        assert time.monotonic() - started < 1
        assert not events and manager.client is None
        state = json.loads((tmp_path / "accepted-cpu-chat-status.json").read_text())
        assert state["stage"] == "weight-verification-failed"
    finally:
        release.set()


@pytest.mark.parametrize("inherited", [0, 10, 19])
def test_cpu_child_never_raises_inherited_service_priority(monkeypatch, inherited):
    calls = []

    def setpriority(kind, pid, target):
        if target < inherited:
            raise PermissionError("Unprivileged service cannot raise priority")
        calls.append((pid, target))

    competition._lower_posix_cpu_priority(
        123,
        getpriority=lambda *a: inherited,
        setpriority=setpriority,
        process_scope=0,
    )
    assert calls == ([(123, 10)] if inherited < 10 else [])


def test_cpu_priority_is_safe_when_platform_has_no_posix_priority_api(monkeypatch):
    competition._lower_posix_cpu_priority(
        123, getpriority=None, setpriority=None, process_scope=None
    )


def test_windows_cpu_priority_lowers_only_owned_child():
    class Api:
        def __init__(self, result):
            self.result = result
            self.calls = []

        def __call__(self, *args):
            self.calls.append(args)
            return self.result

    open_process = Api(456)
    set_priority = Api(1)
    close_handle = Api(1)
    kernel = SimpleNamespace(
        OpenProcess=open_process,
        SetPriorityClass=set_priority,
        CloseHandle=close_handle,
    )
    ctypes_stub = SimpleNamespace(c_uint32=object, c_int=object, c_void_p=object)

    competition._lower_windows_cpu_priority(123, ctypes_module=ctypes_stub, kernel32=kernel)

    assert open_process.calls == [(0x0200, 0, 123)]
    assert set_priority.calls == [(456, 0x4000)]
    assert close_handle.calls == [(456,)]


def test_windows_cpu_priority_safely_skips_unopenable_process():
    class Api:
        def __init__(self, result):
            self.result = result
            self.calls = []

        def __call__(self, *args):
            self.calls.append(args)
            return self.result

    open_process = Api(None)
    set_priority = Api(1)
    close_handle = Api(1)
    kernel = SimpleNamespace(
        OpenProcess=open_process,
        SetPriorityClass=set_priority,
        CloseHandle=close_handle,
    )
    ctypes_stub = SimpleNamespace(c_uint32=object, c_int=object, c_void_p=object)

    competition._lower_windows_cpu_priority(123, ctypes_module=ctypes_stub, kernel32=kernel)

    assert open_process.calls == [(0x0200, 0, 123)]
    assert not set_priority.calls and not close_handle.calls


def test_cpu_question_has_bounded_sixty_second_generation_window(tmp_path, monkeypatch):
    accepted, cpu, _ = admitted(tmp_path, monkeypatch)
    cpu.timeout = 600
    cpu.sampling_args = {"max_tokens": 1024}
    cpu.activity_context = {}
    cpu.template_args = lambda: {}
    cpu.request = lambda *a: (
        {"choices": [{"finish_reason": "stop", "message": {"content": "437"}}]}
        if a[0] == "/v1/chat/completions"
        else {"model_path": accepted["server"]["model"]}
    )
    atomic_json(tmp_path / "serving-active.json", accepted)
    monkeypatch.setattr(
        mission, "status", lambda *a: {"running": True, "state": {"phase": "training"}}
    )
    gpu = SimpleNamespace(model_name="v100", request=lambda *a: {"model_path": "/candidate.gguf"})
    monkeypatch.setattr(competition, "helper_client", lambda *a: gpu)
    manager = chat_backend.AcceptedCPUChat(tmp_path, tmp_path)
    try:
        result = mission_chat.respond(
            tmp_path, tmp_path, {"message": "What is 19 times 23?"}, manager
        )
        assert result["answer"] == "437" and result["responder"]["accepted_master_on_cpu"]
        assert cpu.timeout == 60 and not result["actions"]
    finally:
        manager.close()


def test_lifecycle_omits_disabled_optional_planner(tmp_path, monkeypatch):
    from rlm.v100 import model_roles, remote_helper

    accepted = profile(tmp_path)
    atomic_json(tmp_path / "serving-active.json", accepted)
    planner = copy.deepcopy(accepted)
    planner.setdefault("resources", {})["device"] = "remote"
    planner["runtime"]["base_url"] = "http://10.0.0.61:11435"
    planner_path = tmp_path / "planner.json"
    atomic_json(planner_path, planner)
    monkeypatch.setattr(model_roles, "selected_profile", lambda *a: planner_path)
    monkeypatch.setattr(mission, "status", lambda *a: {"state": {}})
    monkeypatch.setattr(remote_helper, "origin_enabled", lambda *a: False)

    snapshot = mission_chat._chat_lifecycle_profiles(tmp_path, tmp_path)

    assert snapshot["planner"] is None
    assert snapshot["profiles"] == [accepted]


def test_cpu_prewarm_is_single_flight_and_requests_do_not_restart_hash(tmp_path, monkeypatch):
    import time

    accepted, _, _ = admitted(tmp_path, monkeypatch)
    manager = chat_backend.AcceptedCPUChat(tmp_path, tmp_path)
    entered, release = threading.Event(), threading.Event()
    calls = []
    original = chat_backend.file_hash

    def slow(model, **kwargs):
        calls.append(model)
        entered.set()
        release.wait(2)
        return original(model, **kwargs)

    monkeypatch.setattr(chat_backend, "file_hash", slow)
    manager.prewarm(accepted)
    assert entered.wait(1)
    task = manager.verification
    try:
        for _ in range(3):
            with pytest.raises(RuntimeError, match="warming"):
                manager.verify_weights(Path(accepted["server"]["model"]), time.monotonic() + 90)
        assert len(calls) == 1
        assert manager.verification is task
    finally:
        release.set()
    assert task["done"].wait(1)
    digest = manager.verify_weights(Path(accepted["server"]["model"]), time.monotonic() + 1)
    assert len(digest) == 64
    assert manager.verification is task


@pytest.mark.parametrize("cancel", ["close", "stop", "budget"])
def test_cpu_prewarm_has_finite_budget_and_observes_cancellation(tmp_path, monkeypatch, cancel):
    accepted, _, _ = admitted(tmp_path, monkeypatch)
    manager = chat_backend.AcceptedCPUChat(tmp_path, tmp_path)
    entered = threading.Event()

    def checking(model, *, check, progress):
        entered.set()
        while True:
            check()
            threading.Event().wait(0.005)

    monkeypatch.setattr(chat_backend, "file_hash", checking)
    manager.prewarm(accepted, budget=0.05 if cancel == "budget" else 10)
    assert entered.wait(1)
    task = manager.verification
    if cancel == "close":
        manager.close()
    elif cancel == "stop":
        manager.stop.set()
    assert task["done"].wait(1)
    assert isinstance(task["failure"], TimeoutError if cancel == "budget" else RuntimeError)
    assert manager.client is None


@pytest.mark.parametrize("budget", [0, -1, 601, float("inf"), float("nan")])
def test_prewarm_rejects_unbounded_budget(tmp_path, budget):
    manager = chat_backend.AcceptedCPUChat(tmp_path, tmp_path / "chat")
    with pytest.raises(ValueError, match="Prewarm budget"):
        manager.prewarm({}, budget=budget)
