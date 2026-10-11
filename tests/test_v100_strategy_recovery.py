from types import SimpleNamespace

import pytest
import requests

from rlm.v100 import (
    dashboard_health,
    income_policy,
    income_work,
    market_research,
    research_contract,
    researchers,
)
from rlm.v100.common import atomic_json
from tests.test_v100_recovery import financial_world


def test_dashboard_readiness_retries_refused_connection_and_checks_renderer(monkeypatch):
    clock = [0]
    monkeypatch.setattr(dashboard_health.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        dashboard_health.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay)
    )
    replies = [
        requests.ConnectionError("starting"),
        "wrong app",
        "mission-navigation Trade explorer",
    ]

    def get(*args, **kwargs):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(text=reply, raise_for_status=lambda: None)

    monkeypatch.setattr(dashboard_health.requests, "get", get)
    assert dashboard_health.wait(5)["dashboard_ready"]
    assert clock[0] == 2


def test_dashboard_readiness_failure_names_last_observed_error(monkeypatch):
    clock = [0]
    monkeypatch.setattr(dashboard_health.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        dashboard_health.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay)
    )
    monkeypatch.setattr(
        dashboard_health.requests,
        "get",
        lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("refused")),
    )
    with pytest.raises(RuntimeError, match="refused"):
        dashboard_health.wait(2)


def test_model_receives_previous_experiment_not_only_its_path(tmp_path, monkeypatch):
    goal, _ = financial_world(tmp_path)
    saved = tmp_path / "research/income-work/experiments/previous.json"
    atomic_json(
        saved,
        {
            "goal_id": goal["id"],
            "result": {
                "hypothesis": "First hypothesis failed",
                "suggested_test": "Next measurable test",
                "status": "advisory only",
                "research_trace": [],
            },
        },
    )
    atomic_json(
        tmp_path / "research/income-work/status.json",
        {"goal_id": goal["id"], "next_source": 159, "result_path": str(saved)},
    )
    monkeypatch.setattr(
        income_work, "accepted_master", lambda *a: SimpleNamespace(sampling_args={})
    )

    def task(client, branch, job, inputs, root):
        assert inputs[0]["previous_experiment_evidence"]["hypothesis"] == "First hypothesis failed"
        assert "Do not repeat" in inputs[0]["experiment_contract"]
        return {"research_trace": []}

    monkeypatch.setattr(researchers, "research_task", task)
    value = income_work.execute(tmp_path, {"id": "job", "branch": "A", "payload": goal["id"]})
    assert value["actual_income_pln"] is None


def test_completed_commissioning_does_not_rerun_losing_rules_hourly(tmp_path, monkeypatch):
    goal, _ = financial_world(tmp_path)
    income_policy.update(tmp_path, "Enable hypothetical capital research")
    atomic_json(
        tmp_path / "research/market-research/status.json",
        {"goal_id": goal["id"], "state": "completed", "comparisons": [], "updated": 1},
    )
    monkeypatch.setattr(
        market_research, "compare", lambda *a: pytest.fail("Fixed comparison repeated")
    )
    assert market_research.execute(tmp_path, goal["id"])["commissioning_complete"]


def test_proposal_is_not_an_executed_test(tmp_path):
    financial_world(tmp_path)
    result = research_contract.assessment(
        tmp_path,
        {
            "hypothesis": "A legal service can sell",
            "suggested_test": "Measure demand with a sample deliverable",
            "research_trace": [
                {"tool": "register_income_opportunity", "result": {"id": "proposal"}}
            ],
        },
    )
    assert result["state"] == "proposal registered - no executed test"
    assert result["proposal_receipts"] == ["proposal"]
    assert not result["work_receipts"]
