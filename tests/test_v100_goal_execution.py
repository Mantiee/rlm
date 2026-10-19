import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from rlm.v100 import competition, drones, income_work, remote_helper, research_tools, researchers
from rlm.v100.common import atomic_json
from tests.test_v100_recovery import financial_world


def test_income_job_moves_beyond_seed_rotation_to_model_selected_tools(tmp_path, monkeypatch):
    goal, _ = financial_world(tmp_path)
    atomic_json(
        tmp_path / "research/income-work/status.json", {"goal_id": goal["id"], "next_source": 159}
    )
    monkeypatch.setattr(income_work, "observe", lambda *a: pytest.fail("Fixed seed repeated"))
    client = SimpleNamespace(sampling_args={})
    monkeypatch.setattr(income_work, "accepted_master", lambda *a: client)

    def execute(chosen, branch, job, observations, root):
        assert chosen.research_tool_names is None and chosen.research_tool_steps == 6
        assert chosen.request_deadline > time.monotonic()
        assert len(job["brief"]) <= 400
        assert observations[0]["capital_contract"]["real_spendable_pln"] == 0
        assert (
            "do not wait for the operator to supply a 'Host ID'"
            in observations[0]["public_observation_contract"]
        )
        assert "observe_goal_source" in observations[0]["public_observation_contract"]
        assert "local, reproducible" in observations[0]["public_observation_contract"]
        return {"research_trace": [{"tool": "sandbox_run", "result": {"exit_code": 0}}]}

    monkeypatch.setattr(researchers, "research_task", execute)
    result = income_work.execute(tmp_path, {"id": "test", "branch": "A", "payload": goal["id"]})
    assert result["mode"] == "model-selected experiments"
    assert result["successful_tool_receipts"] == 1 and result["actual_income_pln"] is None
    assert json.loads(Path(result["result_path"]).read_text())["goal_id"] == goal["id"]


@pytest.mark.parametrize(
    "identity_error", [requests.ConnectionError("offline"), ValueError("changed model")]
)
def test_resident_transport_fallback_uses_only_accepted_master(
    tmp_path, monkeypatch, identity_error
):
    goal, _ = financial_world(tmp_path)
    monkeypatch.setattr(remote_helper, "selected_helper", lambda *a: tmp_path / "helper.json")
    monkeypatch.setattr(remote_helper, "remote_profile", lambda *a: True)
    monkeypatch.setattr(drones, "load_profile", lambda *a: {"runtime": {"max_timeout": 120}})
    remote = SimpleNamespace(identity=lambda: (_ for _ in ()).throw(identity_error))
    local = SimpleNamespace(marker="accepted")
    monkeypatch.setattr(competition, "helper_client", lambda *a: remote)
    calls = []
    monkeypatch.setattr(income_work, "accepted_master", lambda *a: calls.append(True) or local)
    monkeypatch.setattr(researchers, "research_task", lambda client, *a: {"marker": client.marker})
    job = {"id": "test", "branch": "A", "kind": "researcher", "payload": "Execute useful goal work"}
    if isinstance(identity_error, ValueError):
        with pytest.raises(ValueError, match="changed model"):
            drones.execute(tmp_path, job)
        assert not calls
    else:
        assert drones.execute(tmp_path, job)["marker"] == "accepted"
        assert calls == [True]


def test_six_tool_steps_allow_experiment_to_reach_its_fourth_operation(tmp_path, monkeypatch):
    client = SimpleNamespace(
        sampling_args={"max_tokens": 1024},
        context_window=8192,
        research_tool_steps=6,
        research_tool_names={"calculate"},
        enable_thinking=False,
    )
    calls = []

    def turn(chosen, messages, tools=None, **kwargs):
        if tools and len(calls) < 4:
            calls.append(1)
            return {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": str(len(calls)),
                        "type": "function",
                        "function": {
                            "name": "calculate",
                            "arguments": '{"expression":"100*(1-0.005)"}',
                        },
                    }
                ],
            }
        return {"role": "assistant", "content": "{}"}

    monkeypatch.setattr(research_tools, "native_turn", turn)
    result = research_tools.research_turn(client, [], {}, tmp_path)
    assert len(result["research_trace"]) == 4
    assert all(row["result"].get("status") != "failed" for row in result["research_trace"])
