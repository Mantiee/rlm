import hashlib
import json

import pytest

from rlm.v100 import chat_plan, mission_chat
from rlm.v100.common import atomic_json


@pytest.mark.parametrize(
    "message", ["/plan", "Jaki jest twój cel?", "Pokaż aktualny plan", "What is your goal?"]
)
def test_explicit_readonly_plan_questions(message):
    assert chat_plan.requested(message)


@pytest.mark.parametrize(
    "message",
    [
        "/plan delete files",
        "/cel Zarabiaj",
        "Show plans and start training",
        "Jaki jest cel? Zmień go",
        "Execute a read-only check: use get_plan once, then report the recorded goal. "
        "Do not change plans, files, preferences, models or schedule work. "
        "Command verification ID: " + "a" * 40,
    ],
)
def test_plan_parser_does_not_swallow_actions_or_model_verification(message):
    assert not chat_plan.requested(message)


def test_direct_plan_reply_reads_real_records_without_inference_or_plan_changes(tmp_path):
    goal = {"text": "Measure paper strategy after costs", "suite_sha256": "a" * 64}
    goal["id"] = hashlib.sha256(json.dumps(goal, sort_keys=True).encode()).hexdigest()
    goal_path = tmp_path / "research/goal.json"
    plans_path = tmp_path / "research/plans/current.json"
    atomic_json(goal_path, goal)
    atomic_json(plans_path, {"short": {"text": "Collect depth"}, "mid": {}})
    original = {path: path.read_bytes() for path in (goal_path, plans_path)}
    result = mission_chat.direct_facts(tmp_path, "/plan")
    assert result["responder"]["inference_used"] is False
    assert result["tool_receipts"] == [
        {
            "tool": "get_plan",
            "executor": "controller-readonly",
            "result": {"short": {"text": "Collect depth"}, "mid": {}, "long": goal},
        }
    ]
    saved = mission_chat.save_direct_reply(tmp_path, "/plan", result)
    assert saved["state"] == "completed"
    assert saved["response"]["tool_receipts"] == result["tool_receipts"]
    assert not result["actions"] and not result["applied"]
    assert all(path.read_bytes() == data for path, data in original.items())


def test_missing_plan_is_reported_without_inventing_a_goal(tmp_path):
    result = chat_plan.respond(tmp_path)
    assert result["tool_receipts"][0]["result"] == {"short": {}, "mid": {}, "long": None}
    assert not (tmp_path / "research").exists()


def test_authenticated_gateway_plan_completes_without_gpu_workers(tmp_path, monkeypatch):
    from tests.test_v100_remote_access import CONFIG, gateway, request

    def forbidden(*args, **kwargs):
        pytest.fail("Explicit plan read must not queue inference")

    from rlm.v100 import chat_commands, chat_readonly

    monkeypatch.setattr(chat_commands, "submit", forbidden)
    monkeypatch.setattr(chat_readonly, "submit", forbidden)
    with gateway(tmp_path) as (server, base):
        status, _, body = request(
            base,
            "/api/chat",
            body=json.dumps({"message": "/plan"}).encode(),
            Origin=CONFIG["origin"],
            **{"X-Synta-CSRF": server.csrf, "Content-Type": "application/json"},
        )
        assert status == 200
        row = json.loads(body)
        assert row["state"] == "completed"
        assert row["response"]["responder"]["inference_used"] is False
        assert row["response"]["tool_receipts"][0]["result"]["long"] is None
