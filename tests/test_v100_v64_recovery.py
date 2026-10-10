import json

import pytest

from rlm.v100 import (
    continuous,
    financial_snapshot,
    income_policy,
    mission_chat,
    progress,
    readiness,
)
from rlm.v100.common import atomic_json


def test_financial_question_in_operator_log_routes_to_measured_results(tmp_path, monkeypatch):
    monkeypatch.setattr(
        progress,
        "snapshot",
        lambda root: {
            "paper": {"executed_fills": 0},
            "paper_blockers": ["expired fees"],
            "recent_exploratory_backtests": [
                {
                    "parameters": {"rule": "momentum"},
                    "development_test": {"net_return": -0.15},
                    "report": "measured.json",
                }
            ],
        },
    )
    result = mission_chat.respond(
        tmp_path,
        tmp_path,
        {"message": "czemu nigdzie w testach ani razu nie zarobiles - czy sie myle"},
    )
    assert result["responder"]["model"] == "controller-financial-evidence"
    assert "-15.00%" in result["answer"] and "kapitału" in result["answer"]
    assert result["applied"] == []


def test_assume_capital_is_explicit_research_authorization():
    assert (
        income_policy.setting("masz zalozyc ze masz kapitaly i je zwiekszac ale to paper") is True
    )
    assert income_policy.setting("czy masz zalozyc kapital?") is None


def test_display_finance_independent_of_aggregate_no_fabricated_empty_ledger(tmp_path):
    path = tmp_path / "research/backtests/run-123/report.json"
    atomic_json(
        path,
        {
            "parameters": {"rule": "momentum"},
            "development_test": {"net_return": -0.15, "fills": 10, "trace": ["large"]},
        },
    )
    errors = []
    value = financial_snapshot.supplement(tmp_path, {}, errors)
    assert value["recent_exploratory_backtests"][0]["development_test"] == {
        "net_return": -0.15,
        "fills": 10,
    }
    assert "paper" not in value and "unknown" in value["paper_display_status"]
    assert not errors
    directory = tmp_path / "research/paper/reports/20261010-120000-all-12345678"
    atomic_json(
        directory / "report.json",
        {
            "currency": "PLN",
            "branches": {"A": {"net_pnl_before_personal_tax": "-15"}},
            "trades": [{"net_pnl": "-15"}],
            "time": "original-time",
            "period": "all",
            "expired_fee_profiles": ["old-fee"],
        },
    )
    atomic_json(tmp_path / "research/paper/latest-report.json", {"directory": str(directory)})
    value = financial_snapshot.supplement(tmp_path, {}, errors)
    assert value["paper"]["executed_fills"] == 1 and value["paper"]["updated_at"] == "original-time"
    assert value["paper_blockers"] == ["expired fees: old-fee"]


def test_loop_checkpoint_never_waits_for_full_financial_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "report", lambda root: pytest.fail("Controller blocked on audit"))
    atomic_json(tmp_path / "research/mission/active.json", {"run": "current"})
    output = tmp_path / "learning"
    continuous.save_progress(tmp_path, output, {"cycles": []})
    assert json.loads((output / "state.json").read_text()) == {"cycles": []}


def test_live_paper_display_does_not_replay_audit_and_detects_state_tamper(tmp_path, monkeypatch):
    from rlm.v100.paper import PaperBook
    from tests.test_v100_paper import Clock, configuration, quote

    clock = Clock()
    book = PaperBook(tmp_path, clock)
    book.initialize()
    book.configure(configuration(clock))
    book.ingest(quote(clock))
    monkeypatch.setattr(PaperBook, "events", lambda *args: pytest.fail("Display scanned history"))
    book.db.execute("BEGIN IMMEDIATE")
    value = financial_snapshot.live_ledger(tmp_path)
    assert value["branches"]["A"]["equity"] == "10000"
    assert value["executed_fills"] is None
    assert value["ledger_sequence"] == 3
    assert "stale quote: TEST" in value["blockers"]
    book.db.rollback()
    with book.db:
        book.db.execute("UPDATE state SET payload=? WHERE id=1", ('{"branches":{}}',))
    with pytest.raises(ValueError, match="checksum"):
        financial_snapshot.live_ledger(tmp_path)
    book.close()


def test_readiness_uses_cached_events_not_daily_journal(tmp_path, monkeypatch):
    from rlm.v100 import live_status

    monkeypatch.setattr(
        live_status, "recent_events", lambda root: pytest.fail("HTTP reparsed journal")
    )
    value = readiness.assess(tmp_path, {}, {"live_events": [{"kind": "inference-delta"}]}, {})
    assert (
        next(row for row in value["checks"] if row["name"] == "Streaming transport")["state"]
        == "observed"
    )


def test_late_reply_delivered_once_with_saved_action_receipt(tmp_path, capsys):
    identity = mission_chat.submit(tmp_path, "work")
    monitor = mission_chat.LateReplies(tmp_path)
    monitor.track(identity)
    monitor.poll()
    assert not capsys.readouterr().out

    with mission_chat.connect(tmp_path) as db:
        db.execute(
            "UPDATE requests SET state='completed',response=? WHERE id=?",
            (json.dumps({"answer": "Saved answer", "applied": [{"state": "queued"}]}), identity),
        )
    monitor.poll()
    value = capsys.readouterr().out
    assert "Saved answer" in value and identity in value and '"state": "queued"' in value
    monitor.poll()
    assert not capsys.readouterr().out


@pytest.mark.parametrize(
    "message,expected",
    [("Czy uczysz się cały czas?", "zweryfikowanych"), ("what have you learned", "verified data")],
)
def test_learning_answer_language_and_no_false_continuous_optimizer(
    tmp_path, monkeypatch, message, expected
):
    from rlm.v100 import mission_evidence

    monkeypatch.setattr(
        mission_evidence,
        "collect",
        lambda root: {
            "mission_running": True,
            "phase": "research-and-learning-loop",
            "learning": {"completed_cycles": 0},
            "optimizer_updates_observed": None,
            "accepted_weight_updates_this_run": 0,
            "reports": [],
            "goal_learning": {"forecasts": [1, 2]},
            "run": None,
        },
    )
    value = mission_chat.direct_facts(tmp_path, message)
    assert value is not None and expected in value["answer"]
    assert "Dashboard" not in value["answer"]
