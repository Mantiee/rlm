import json

import pytest

from rlm.v100 import chat_probe, mission_chat
from rlm.v100.common import atomic_json
from tests.test_v100_remote_access import CONFIG, gateway


@pytest.mark.parametrize(
    "answer,actions,model,expected",
    [
        ("437", [], "accepted-fixture", "verified"),
        ("438", [], "accepted-fixture", "blocked"),
        ("437", [{"kind": "plan_short"}], "accepted-fixture", "blocked"),
        ("437", [], "controller-status", "blocked"),
    ],
)
def test_probe_uses_authenticated_gateway_and_requires_generated_action_free_reply(
    tmp_path, monkeypatch, answer, actions, model, expected
):
    atomic_json(tmp_path / "research/remote-access/config.json", CONFIG)
    original = mission_chat.submit
    calls = []

    def respond(root, message):
        calls.append(message)
        identity = original(root, message)
        result = {
            "answer": answer,
            "actions": actions,
            "applied": [],
            "responder": {"model": model},
        }
        with mission_chat.connect(root) as db:
            db.execute(
                "UPDATE requests SET state='completed',response=? WHERE id=?",
                (json.dumps(result), identity),
            )
        return identity

    monkeypatch.setattr(mission_chat, "submit", respond)
    monkeypatch.setattr(mission_chat, "direct_facts", lambda *a: None)
    with gateway(tmp_path) as (server, _):
        result = chat_probe.run(tmp_path, "a" * 40, 1, server.server_port)
        assert result["stage"] == expected
        assert len(calls) == 1 and mission_chat.conversation_question(calls[0])
        assert "Do not use tools or change anything" in calls[0]
        if expected == "verified":
            assert chat_probe.run(tmp_path, "a" * 40, 1, server.server_port) == result
            assert len(calls) == 1


@pytest.mark.parametrize("retry_fails", [False, True])
def test_failed_fixed_probe_retries_once_after_new_verified_planner_admission(
    tmp_path, monkeypatch, retry_fails
):
    from rlm.v100 import model_roles

    atomic_json(tmp_path / "research/remote-access/config.json", CONFIG)
    planner = tmp_path / "qualified-planner.json"
    planner.write_text('{"admitted": true}')
    admission = [None]
    monkeypatch.setattr(model_roles, "selected_profile", lambda *a: admission[0])
    original = mission_chat.submit
    calls = []

    def respond(root, message):
        identity = original(root, message)
        calls.append(identity)
        failed = len(calls) == 1 or retry_fails
        response = {
            "answer": "437",
            "actions": [],
            "applied": [],
            "responder": {"model": "qualified-planner"},
        }
        with mission_chat.connect(root) as db:
            db.execute(
                "UPDATE requests SET state=?,response=?,error=? WHERE id=?",
                (
                    "failed" if failed else "completed",
                    None if failed else json.dumps(response),
                    "Chat request exceeded its total time budget" if failed else None,
                    identity,
                ),
            )
        return identity

    monkeypatch.setattr(mission_chat, "submit", respond)
    monkeypatch.setattr(mission_chat, "direct_facts", lambda *a: None)
    with gateway(tmp_path) as (server, _):
        args = (tmp_path, "b" * 40, 1, server.server_port)
        assert chat_probe.run(*args)["stage"] == "blocked"
        assert chat_probe.run(*args)["stage"] == "blocked"
        assert len(calls) == 1
        admission[0] = planner
        result = chat_probe.run(*args)
        assert result["stage"] == ("blocked" if retry_fails else "verified")
        assert len(calls) == 2
        assert result["previous_attempt"]["request_id"] == calls[0]
        assert mission_chat.inspect(tmp_path, calls[0])["state"] == "failed"
        final = chat_probe.run(*args)
        assert len(calls) == 2
        assert final["previous_attempt"] == result["previous_attempt"]
