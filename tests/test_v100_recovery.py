import json
from types import SimpleNamespace

import pytest

from rlm.v100 import drones, goal_learning, goal_observer, live_status, planning, research_tools
from rlm.v100.common import atomic_json
from rlm.v100.goals import set_goal
from tests.test_v100_architectures import workload
from tests.test_v100_dashboard import dashboard


def financial_world(root):
    _, suite = workload(root)
    goal = set_goal(root, "Find lawful repeatable income using verified observations", suite)
    run = root / "research/mission/test-run"
    atomic_json(run / "input-profile.json", {"resources": {"paper_research_enabled": True}})
    atomic_json(root / "research/mission/active.json", {"run": str(run)})
    return goal, run


def test_timeout_keeps_same_run_report_and_reads_goal_independently(tmp_path, monkeypatch):
    goal, run = financial_world(tmp_path)
    report = {
        "updated_at": "old-time",
        "mission_evidence": {"run": str(run)},
        "paper": {"currency": "PLN", "branches": {"A": {"net_pnl_before_personal_tax": "12"}}},
    }
    atomic_json(tmp_path / "research/mission/latest-report.json", report)
    monkeypatch.setattr(
        dashboard.subprocess,
        "run",
        lambda command, **kwargs: (_ for _ in ()).throw(
            dashboard.subprocess.TimeoutExpired(command, 20)
        ),
    )
    state = dashboard.DashboardState(tmp_path, lambda root: {"run": str(run), "running": True})
    state.refresh()
    assert state.data["report"]["paper"] == report["paper"]
    assert state.data["report"]["mission_evidence"]["goal_learning"]["goal_id"] == goal["id"]
    assert state.data["report_freshness"]["state"] == "stale"
    assert state.data["errors"]


def test_previous_run_counters_never_carried_to_new_run(tmp_path, monkeypatch):
    goal, run = financial_world(tmp_path)
    atomic_json(
        tmp_path / "research/mission/latest-report.json",
        {"mission_evidence": {"run": "old", "optimizer_updates_observed": 999}},
    )
    monkeypatch.setattr(
        dashboard.subprocess,
        "run",
        lambda command, **kwargs: (_ for _ in ()).throw(
            dashboard.subprocess.TimeoutExpired(command, 20)
        ),
    )
    state = dashboard.DashboardState(tmp_path, lambda root: {"run": str(run), "running": True})
    state.refresh()
    evidence = state.data["report"]["mission_evidence"]
    assert "optimizer_updates_observed" not in evidence
    assert evidence["goal_learning"]["goal_id"] == goal["id"]


def test_financial_references_precommit_and_resolve_real_future_labels(tmp_path, monkeypatch):
    financial_world(tmp_path)
    clock, price = [1000.0], [100.0]
    monkeypatch.setattr(goal_learning, "time", SimpleNamespace(time=lambda: clock[0]))
    monkeypatch.setattr(goal_observer, "time", SimpleNamespace(time=lambda: clock[0]))
    monkeypatch.setattr(
        goal_learning.shutil, "disk_usage", lambda root: SimpleNamespace(free=8 * 2**30)
    )
    monkeypatch.setattr(
        research_tools,
        "download_page",
        lambda url, max_bytes: (url, json.dumps({"data": {"amount": str(price[0])}})),
    )
    for _ in range(4):
        result = goal_observer.tick(tmp_path)
        assert result["state"] == "collecting"
        assert len(result["results"]) == 2
    assert goal_learning.status(tmp_path)["pending_total"] == 8
    # No future label exists until a host observation after the committed horizon.
    assert goal_learning.records(tmp_path) == []
    clock[0] = 1301.0
    price[0] = 105.0
    goal_learning.tick(tmp_path)
    goal_learning.tick(tmp_path)
    records = goal_learning.records(tmp_path)
    assert len(records) == 8
    assert len({r["group"] for r in records}) == 8
    assert all(r["messages"][-1]["content"] == "up" for r in records)
    assert goal_learning.status(tmp_path)["resolved_total"] == 8
    with goal_learning.database(tmp_path) as db:
        forecasts = [json.loads(r[0]) for r in db.execute("SELECT data FROM forecasts")]
    assert all("Uniform controller reference" in f["specification"]["rationale"] for f in forecasts)


def test_source_failure_does_not_fabricate_forecast(tmp_path, monkeypatch):
    financial_world(tmp_path)
    monkeypatch.setattr(
        goal_learning, "observe", lambda *a, **k: (_ for _ in ()).throw(ValueError("HTTP 403"))
    )
    result = goal_observer.tick(tmp_path)
    assert all(r["state"] == "failed" for r in result["results"])
    with goal_learning.database(tmp_path) as db:
        assert db.execute("SELECT count(*) FROM forecasts").fetchone()[0] == 0


def test_upgrade_plan_recovery_preserves_goal_and_versions(tmp_path):
    goal, _ = financial_world(tmp_path)
    planning.update(tmp_path, "short", "Fix dashboard CSS layout", "user")
    with pytest.raises(ValueError, match="UI-only"):
        planning.update(tmp_path, "mid", "Fix dashboard CSS layout", "A")
    receipt = planning.realign_income_plans(tmp_path)
    assert receipt["changed"] and not receipt["long_term_goal_changed"]
    plan = planning.read(tmp_path)
    assert plan["long"] == goal and "register_income_opportunity" in plan["short"]["text"]
    assert any(
        "Fix dashboard CSS layout" in p.read_text()
        for p in (tmp_path / "research/plans").glob("*.json")
    )


def test_process_failure_is_failed_not_completed_and_tool_receipt_is_visible(tmp_path):
    receipt = drones.schedule(tmp_path, "A", "desktop", "false", 0)
    drones.finish(tmp_path, receipt["id"], {"exit_code": 1, "stderr": "failure"})
    job = drones.inspect(tmp_path)[0]
    assert job["state"] == "failed"
    assert "unverified" in live_status.agent_views([], [{**job, "state": "completed"}])[0]["state"]
    assert "precommitted" in live_status.public_summary(
        {"id": "actual-id", "state": "precommitted"}
    )
    assert "actual-id" in live_status.public_summary({"id": "actual-id", "state": "precommitted"})


def test_goal_status_read_is_not_a_write_transaction(tmp_path):
    financial_world(tmp_path)
    with goal_learning.database(tmp_path) as db:
        db.execute("PRAGMA journal_mode=WAL")
    with goal_learning.database(tmp_path) as writer:
        writer.execute("BEGIN IMMEDIATE")
        status = goal_learning.status(tmp_path)
        assert status["pending_total"] == 0


def test_known_placeholder_script_refused_before_queue(tmp_path):
    script = "def simulate_strategies():\n    # Placeholder for the logic to be executed by CPU/RTX\n    pass\nprint('Simulating multi-strategy portfolio...')\n"
    with pytest.raises(ValueError, match="Placeholder script refused"):
        drones.schedule(tmp_path, "A", "desktop", script, 0)
    assert drones.inspect(tmp_path) == []


def test_displayed_process_output_retains_secret_redaction():
    summary = live_status.public_summary({"stdout": "password=sensitive-value", "exit_code": 1})
    assert "sensitive-value" not in summary and "omitted" in summary
