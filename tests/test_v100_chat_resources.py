import json
import time
from types import SimpleNamespace

import pytest

from rlm.v100 import chat_resources, competition, drones, mission, mission_chat, research_tools
from rlm.v100.common import atomic_json
from tests.test_v100_mission_evidence import running

MESSAGE = "zrog zeby uzywac rtx 3090 do pomocy dla siebie i dronow i cpu i ramu i dysku z windowsa - takie shared"


def action(kind, text):
    return {
        "kind": kind,
        "text": text,
        "target": "master",
        "thinking": False,
        "max_tokens": 256,
        "batch_tokens": 128,
        "enabled": False,
    }


@pytest.mark.parametrize("message", [MESSAGE, "Wykorzystaj RTX i CPU Windows do pomocy masterowi"])
def test_resource_instruction_is_bound_to_current_request(message):
    assert chat_resources.instruction(message)


@pytest.mark.parametrize(
    "message",
    [
        "Czy worker Windows CPU działa?",
        "Dodaj monitoring RTX i CPU do dashboardu",
        "Jak używać RTX i Windows CPU?",
        "Ustaw cel długoterminowy: rozwój modeli",
    ],
)
def test_other_requests_keep_their_own_scope(message):
    assert not chat_resources.instruction(message)


def test_unrelated_plans_and_stub_scripts_are_not_resource_actions():
    good = action("directive", "Delegate useful verified source work")
    bad = action("sandbox", 'print("Simulating...")')
    assert chat_resources.filter_actions([good, bad, action("plan_mid", "Trade")]) == (
        [good],
        [bad, action("plan_mid", "Trade")],
    )


def test_recovery_requeues_wrong_request_preserves_goal_and_cancels_only_pending_stub(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(mission, "status", lambda root: {"running": False})
    identity = mission_chat.submit(tmp_path, MESSAGE)
    old = {"id": "old-plan", "text": "Original research plan"}
    wrong = {"id": "a" * 32, "text": "Unrequested trading plan"}
    atomic_json(
        tmp_path / "research/plans/current.json", {"short": wrong, "mid": {"id": "later-user-plan"}}
    )
    atomic_json(tmp_path / "research/plans" / (wrong["id"] + ".json"), {"previous": old})
    goal = tmp_path / "research/goal.json"
    goal.write_text("operator-owned-goal")
    script = 'print("Simulating...")'
    job = drones.schedule(tmp_path, "A", "desktop", script, 0)
    response = {
        "answer": "Completed",
        "actions": [action("plan_short", wrong["text"]), action("sandbox", script)],
        "applied": [
            {"horizon": "short", "id": wrong["id"]},
            {"horizon": "mid", "id": "b" * 32},
            job,
        ],
    }
    with mission_chat.connect(tmp_path) as db:
        db.execute(
            "UPDATE requests SET state='completed',response=? WHERE id=?",
            (json.dumps(response), identity),
        )
    result = chat_resources.repair_completed(tmp_path)
    assert result["requeued_resource_requests"][0]["restored_plans"] == ["short"]
    plans = json.loads((tmp_path / "research/plans/current.json").read_text())
    assert plans["short"] == old and plans["mid"]["id"] == "later-user-plan"
    assert goal.read_text() == "operator-owned-goal"
    assert mission_chat.inspect(tmp_path, identity)["state"] == "queued"
    assert drones.inspect(tmp_path)[0]["state"] == "cancelled"
    assert (tmp_path / "research/state/chat-contract-repairs" / (identity + ".json")).exists()
    assert chat_resources.repair_completed(tmp_path)["requeued_resource_requests"] == []


def test_recovery_refuses_live_mission(tmp_path, monkeypatch):
    monkeypatch.setattr(mission, "status", lambda root: {"running": True})
    with pytest.raises(ValueError, match="Stop"):
        chat_resources.repair_completed(tmp_path)


def test_resource_response_ignores_dashboard_history_and_uses_host_job_receipts(
    tmp_path, monkeypatch
):
    run = running(tmp_path, monkeypatch)
    atomic_json(run / "serving-active.json", {})
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    monkeypatch.setattr(
        mission_chat,
        "load_profile",
        lambda *a: {
            "runtime": {},
            "server": {"model": str(model)},
            "resources": {"interactive_lab": True},
        },
    )
    client = SimpleNamespace(
        request=lambda *a: {"model_path": str(model)},
        model_name="v100",
        timeout=120,
        sampling_args={"max_tokens": 2048},
        activity_context={},
    )
    monkeypatch.setattr(competition, "helper_client", lambda *a: client)
    old = mission_chat.submit(tmp_path, "Edit dashboard")
    with mission_chat.connect(tmp_path) as db:
        db.execute(
            "UPDATE requests SET state='completed',response=? WHERE id=?",
            (json.dumps({"answer": "Old dashboard success"}), old),
        )
    evidence = {
        "helper": {"pinned_model_loaded": True},
        "external_compute": {"workers": [{"stale": False, "phase": "idle"}]},
    }
    monkeypatch.setattr(chat_resources, "status", lambda root: evidence)

    def reply(client, messages, schema, root):
        assert "Old dashboard success" not in str(messages)
        assert client.research_tool_names == chat_resources.RESOURCE_TOOLS
        assert client.resource_only_chat
        return {
            "content": json.dumps(
                {
                    "answer": "Updated trading HTML",
                    "actions": [action("plan_short", "Trade"), action("sandbox", 'print("done")')],
                }
            ),
            "research_trace": [
                {"tool": "propose_compute_trial", "result": {"id": "actual-job", "state": "queued"}}
            ],
        }

    monkeypatch.setattr(research_tools, "research_turn", reply)
    result = mission_chat.respond(
        tmp_path, run, {"message": MESSAGE, "deadline": time.monotonic() + 180}
    )
    assert result["actions"] == [] and result["applied"] == []
    assert "actual-job" in result["answer"] and "queued" in result["answer"]
    assert "Updated trading HTML" not in result["answer"]
    assert not (tmp_path / "research/plans/current.json").exists()
    assert result["tool_receipts"][0]["result"]["id"] == "actual-job"


def test_resource_tool_router_rejects_desktop_job_before_execution(tmp_path, monkeypatch):
    from rlm.v100 import agent

    client = SimpleNamespace(
        sampling_args={"max_tokens": 512},
        context_window=8192,
        enable_thinking=False,
        research_tool_names={"schedule_drone"},
        resource_only_chat=True,
    )

    def model(client, messages, **kwargs):
        if kwargs.get("tools"):
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call",
                        "type": "function",
                        "function": {
                            "name": "schedule_drone",
                            "arguments": json.dumps(
                                {"kind": "desktop", "payload": 'print("stub")', "interval": 0}
                            ),
                        },
                    }
                ],
            }
        return {"role": "assistant", "content": '{"answer":"done","actions":[]}'}

    monkeypatch.setattr(research_tools, "native_turn", model)
    monkeypatch.setattr(agent, "native_turn", model)
    monkeypatch.setattr(
        research_tools.ResearchTools, "execute", lambda *a: pytest.fail("Desktop reached execution")
    )
    result = research_tools.research_turn(
        client, [{"role": "user", "content": MESSAGE}], {}, tmp_path
    )
    assert all(row["result"]["status"] == "failed" for row in result["research_trace"])
    assert all("Resource request" in row["result"]["error"] for row in result["research_trace"])
