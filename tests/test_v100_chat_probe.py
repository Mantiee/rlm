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
