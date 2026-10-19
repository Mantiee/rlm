import json
import threading
import time
from types import SimpleNamespace

import pytest
import requests

from rlm.v100 import chat_probe, chat_readonly, competition, mission_chat
from rlm.v100.common import atomic_json
from tests.test_v100_continual import profile
from tests.test_v100_remote_access import CONFIG, gateway, request


def admitted(tmp_path, monkeypatch, generate):
    planner = tmp_path / "planner.json"
    atomic_json(planner, profile(tmp_path))
    atomic_json(tmp_path / "research/model-roles/active.json", {"fixture": True})
    monkeypatch.setattr(chat_readonly, "selected_profile", lambda *a: planner)
    client = SimpleNamespace(
        model_name="admitted-planner",
        timeout=120,
        activity_context={},
        template_args=lambda: {},
        preflight=lambda: None,
        request=generate,
    )
    monkeypatch.setattr(competition, "helper_client", lambda *a: client)
    monkeypatch.setattr(mission_chat, "direct_facts", lambda *a: None)
    return client


def test_gateway_canary_generates_with_new_runtime_without_the_old_worker(tmp_path, monkeypatch):
    calls = []

    def generate(endpoint, payload):
        calls.append(payload)
        return {"choices": [{"finish_reason": "stop", "message": {"content": "437"}}]}

    client = admitted(tmp_path, monkeypatch, generate)
    atomic_json(tmp_path / "research/remote-access/config.json", CONFIG)
    monkeypatch.setattr(mission_chat, "submit", lambda *a: pytest.fail("Old worker not used"))
    with gateway(tmp_path) as (server, _):
        result = chat_probe.run(tmp_path, "d" * 40, 5, server.server_port)
    assert result["stage"] == "verified"
    assert result["responder"]["gateway_readonly_worker"]
    assert len(calls) == 1 and "tools" not in calls[0]
    assert calls[0]["max_tokens"] == 512
    assert client.timeout == 60


def test_pending_readonly_request_is_never_visible_to_the_old_command_worker(tmp_path, monkeypatch):
    started, release = threading.Event(), threading.Event()

    def generate(*args):
        started.set()
        assert release.wait(3)
        return {"choices": [{"finish_reason": "stop", "message": {"content": "Actual reply"}}]}

    admitted(tmp_path, monkeypatch, generate)
    with gateway(tmp_path) as (server, base):
        headers = {
            "Origin": CONFIG["origin"],
            "X-Synta-CSRF": server.csrf,
            "Content-Type": "application/json",
        }
        _, _, body = request(
            base, "/api/chat", body=b'{"message":"What is the current question?"}', **headers
        )
        row = json.loads(body)
        assert row["state"] == "queued" and row["route"] == "gateway-readonly"
        assert started.wait(2)
        with mission_chat.connect(tmp_path) as db:
            assert (
                db.execute("SELECT count(*) FROM requests WHERE state='queued'").fetchone()[0] == 0
            )
            assert (
                db.execute("SELECT state FROM requests WHERE id=?", (row["id"],)).fetchone()[0]
                == chat_readonly.RUNNING
            )
        release.set()
        deadline = time.monotonic() + 2
        while (
            mission_chat.inspect(tmp_path, row["id"])["state"] == "queued"
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        result = mission_chat.inspect(tmp_path, row["id"])
        assert result["state"] == "completed" and result["response"]["actions"] == []


def test_commands_and_disabled_helpers_keep_existing_queue(tmp_path, monkeypatch):
    admitted(tmp_path, monkeypatch, lambda *a: pytest.fail("No generation expected"))
    assert chat_readonly.submit(tmp_path, "Sprawdź i napraw dashboard") is None
    atomic_json(tmp_path / "research/user-preferences.json", {"remote_helper_enabled": False})
    assert chat_readonly.submit(tmp_path, "What is this?") is None


def test_operator_can_cancel_an_inflight_readonly_reply_without_it_overwriting_cancel(
    tmp_path, monkeypatch
):
    started, release = threading.Event(), threading.Event()

    def generate(*args):
        started.set()
        assert release.wait(3)
        return {"choices": [{"finish_reason": "stop", "message": {"content": "Late reply"}}]}

    admitted(tmp_path, monkeypatch, generate)
    identity = chat_readonly.submit(tmp_path, "What is this?")
    with gateway(tmp_path):
        assert started.wait(2)
        assert mission_chat.submit(tmp_path, "/cancel-chat " + identity) == identity
        release.set()
    result = mission_chat.inspect(tmp_path, identity)
    assert result["state"] == "cancelled" and result["response"] is None


def test_gateway_restart_recovers_only_owned_action_free_requests(tmp_path, monkeypatch):
    admitted(
        tmp_path,
        monkeypatch,
        lambda *a: {"choices": [{"finish_reason": "stop", "message": {"content": "Saved reply"}}]},
    )
    identity = chat_readonly.submit(tmp_path, "What is this?")
    command = mission_chat.submit(tmp_path, "Sprawdź dashboard")
    with mission_chat.connect(tmp_path) as db:
        db.execute("UPDATE requests SET state=? WHERE id=?", (chat_readonly.RUNNING, identity))
    with gateway(tmp_path):
        deadline = time.monotonic() + 2
        while (
            mission_chat.inspect(tmp_path, identity)["state"] == "queued"
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
    assert mission_chat.inspect(tmp_path, identity)["state"] == "completed"
    assert mission_chat.inspect(tmp_path, command)["state"] == "queued"


def test_transient_readonly_generation_retries_without_tools(tmp_path, monkeypatch):
    calls = []

    def generate(*args):
        calls.append(1)
        if len(calls) == 1:
            raise requests.ConnectionError("Temporary owned transport failure")
        return {"choices": [{"finish_reason": "stop", "message": {"content": "Saved reply"}}]}

    admitted(tmp_path, monkeypatch, generate)
    identity = chat_readonly.submit(tmp_path, "What is this?")
    with gateway(tmp_path):
        deadline = time.monotonic() + 2
        while (
            mission_chat.inspect(tmp_path, identity)["attempts"] < 1 and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        with mission_chat.connect(tmp_path) as db:
            db.execute("UPDATE requests SET retry_after=0 WHERE id=?", (identity,))
        deadline = time.monotonic() + 2
        while (
            mission_chat.inspect(tmp_path, identity)["state"] == "queued"
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
    result = mission_chat.inspect(tmp_path, identity)
    assert result["state"] == "completed" and result["attempts"] == 2
    assert result["response"]["actions"] == []


@pytest.mark.parametrize(
    "failure",
    [
        requests.ConnectionError("Windows adaptive guard unavailable: heavy game running"),
        chat_readonly.PlannerOriginDisabled("Planner origin disabled"),
    ],
)
def test_unavailable_planner_question_hands_off_to_accepted_cpu_command_worker(
    tmp_path, monkeypatch, failure
):
    admitted(tmp_path, monkeypatch, lambda *a: None)
    monkeypatch.setattr(
        chat_readonly,
        "response",
        lambda *a: (_ for _ in ()).throw(failure),
    )
    identity = chat_readonly.submit(tmp_path, "What is this?")
    stop = threading.Event()
    worker = threading.Thread(target=chat_readonly.work, args=(tmp_path, stop))
    worker.start()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        with mission_chat.connect(tmp_path) as db:
            state = db.execute("SELECT state FROM requests WHERE id=?", (identity,)).fetchone()[0]
        if state == "gateway-pending":
            break
        time.sleep(0.02)
    stop.set()
    worker.join(2)
    assert state == "gateway-pending"
    assert mission_chat.inspect(tmp_path, identity)["route"] == "gateway-commands"


def test_disabled_windows_origin_is_not_submitted_or_contacted(tmp_path, monkeypatch):
    from rlm.v100 import common
    from rlm.v100.remote_helper import WINDOWS_3090_ORIGIN

    admitted(tmp_path, monkeypatch, lambda *a: pytest.fail("Disabled Windows contacted"))
    selected = profile(tmp_path)
    selected["resources"] = {**selected.get("resources", {}), "device": "remote"}
    selected["runtime"]["base_url"] = WINDOWS_3090_ORIGIN
    # Origin routing is independent of the separately tested metadata validator.
    monkeypatch.setattr(common, "load_profile", lambda *args: selected)
    atomic_json(tmp_path / "planner.json", selected)
    atomic_json(
        tmp_path / "research/user-preferences.json",
        {"remote_origin_enabled": {WINDOWS_3090_ORIGIN: False}},
    )
    assert chat_readonly.submit(tmp_path, "What is the result?") is None
    with pytest.raises(chat_readonly.PlannerOriginDisabled):
        chat_readonly.response(tmp_path, {"id": "saved", "message": "What is the result?"})
