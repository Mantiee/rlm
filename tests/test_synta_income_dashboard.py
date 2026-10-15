"""Regressions for misleading progress and fee status in the installed panel."""

import importlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest


@pytest.fixture
def panel(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("synta_income_learning")


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_expired_fee_and_old_decision_are_visible(panel):
    now = 1800000000
    data = {
        "last_decision_time": datetime.fromtimestamp(now - 72 * 3600, UTC).isoformat(),
        "state": {
            "instruments": {"BTC": {"fee_profile": "old"}},
            "fee_profiles": {
                "old": {"valid_until": datetime.fromtimestamp(now - 1, UTC).isoformat()}
            },
        },
    }
    issues = panel.ledger_diagnostics(data, now)
    assert any("6 h" in item for item in issues)
    assert any("BTC" in item and "wygasł" in item for item in issues)


def test_valid_fee_and_recent_decision_have_no_ledger_alarm(panel):
    now = 1800000000
    data = {
        "last_decision_time": datetime.fromtimestamp(now - 10, UTC).isoformat(),
        "state": {
            "instruments": {"BTC": {"fee_profile": "current"}},
            "fee_profiles": {
                "current": {"valid_until": datetime.fromtimestamp(now + 1000, UTC).isoformat()}
            },
        },
    }
    assert panel.ledger_diagnostics(data, now) == []


def test_missing_fee_is_reported(panel):
    issues = panel.ledger_diagnostics(
        {"state": {"instruments": {"BTC": {"fee_profile": "missing"}}}}, 1800000000
    )
    assert any("ważności opłat" in item for item in issues)


def test_dashboard_separates_guard_holds_model_markers_and_unclassified_decisions(panel, tmp_path):
    now = 1800000000
    ledger = tmp_path / "ledger.sqlite3"
    db = sqlite3.connect(ledger)
    db.executescript(
        "CREATE TABLE events(sequence INTEGER PRIMARY KEY,time TEXT,kind TEXT,branch TEXT,payload TEXT);"
        "CREATE TABLE state(payload TEXT);"
    )
    rows = [
        (
            1,
            datetime.fromtimestamp(now - 72 * 3600, UTC).isoformat(),
            "decision",
            "A",
            {"proposal": {"action": "hold"}},
        ),
        (
            2,
            datetime.fromtimestamp(now - 10, UTC).isoformat(),
            "decision",
            "A",
            {"proposal": {"action": "hold", "proposal_source": "model-proposal"}},
        ),
        (
            3,
            datetime.fromtimestamp(now, UTC).isoformat(),
            "decision",
            "B",
            {
                "proposal": {
                    "action": "hold",
                    "receipt_source": "host-quote-freshness-guard",
                }
            },
        ),
    ]
    db.executemany(
        "INSERT INTO events VALUES(?,?,?,?,?)",
        [
            (seq, time, kind, branch, json.dumps(payload))
            for seq, time, kind, branch, payload in rows
        ],
    )
    db.commit()
    db.close()

    status = panel.get_status(ledger)
    assert status["decision_sources"] == {
        "window": 3,
        "freshness_hold": 1,
        "model_proposal": 1,
        "unclassified": 1,
        "last_freshness_hold_time": rows[2][1],
        "last_model_proposal_time": rows[1][1],
        "last_unclassified_time": rows[0][1],
    }
    assert status["last_non_guard_decision_time"] == rows[1][1]
    assert "model i trening tylko na podstawie oznaczonych receiptów" in panel.HTML
    assert "Automatyczny HOLD świeżości" in panel.HTML
    assert "niepotwierdzone jako model" in panel.HTML


@pytest.mark.parametrize(
    "event_age,decision_age,state,label_fragment",
    [
        (None, None, "waiting", "BRAK POTWIERDZONEJ AKTYWNOŚCI"),
        (181, 10, "stalled", "ZASTÓJ"),
        (10, None, "stalled", "BRAK ŚWIEŻEJ DECYZJI MODELU"),
        (10, 21601, "stalled", "BRAK ŚWIEŻEJ DECYZJI MODELU"),
        (10, 10, "active", "ŚWIEŻĄ DECYZJĘ MODELU"),
    ],
)
def test_activity_banner_separates_ledger_io_from_real_model_progress(
    panel, event_age, decision_age, state, label_fragment
):
    now = 1800000000
    event_time = (
        datetime.fromtimestamp(now - event_age, UTC).isoformat() if event_age is not None else None
    )
    decision_time = (
        datetime.fromtimestamp(now - decision_age, UTC).isoformat()
        if decision_age is not None
        else None
    )
    result = panel.ledger_activity(
        {
            "last_event_time": event_time,
            "last_non_guard_decision_time": decision_time,
            "decision_sources": {"last_model_proposal_time": decision_time},
        },
        now,
    )
    assert result["state"] == state
    assert label_fragment in result["label"]
    assert 'id="activity"' in panel.HTML


def test_fresh_automatic_hold_does_not_count_as_fresh_model_activity(panel):
    now = 1800000000
    event_time = datetime.fromtimestamp(now - 1, UTC).isoformat()
    result = panel.ledger_activity(
        {
            "last_event_time": event_time,
            "last_non_guard_decision_time": None,
            "decision_sources": {
                "last_freshness_hold_time": event_time,
                "last_model_proposal_time": None,
            },
        },
        now,
    )
    assert result["state"] == "stalled"
    assert "notow" in result["detail"]


def test_recent_unclassified_decision_does_not_count_as_model_activity(panel):
    now = 1800000000
    recent = datetime.fromtimestamp(now - 1, UTC).isoformat()
    result = panel.ledger_activity(
        {
            "last_event_time": recent,
            "last_non_guard_decision_time": recent,
            "decision_sources": {
                "last_unclassified_time": recent,
                "last_model_proposal_time": None,
            },
        },
        now,
    )
    assert result["state"] == "stalled"
    assert "brak zarejestrowanej propozycji modelu" in result["detail"]


def test_fresh_guard_receipt_does_not_clear_missing_model_decision_alarm(panel):
    issues = panel.ledger_diagnostics(
        {
            "last_non_guard_decision_time": None,
            "decision_sources": {"model_proposal": 0, "freshness_hold": 100},
            "state": {"instruments": {}, "fee_profiles": {}},
        },
        1800000000,
    )
    assert any("jawnie oznaczonej jako propozycja modelu" in item for item in issues)
    assert any("Brak decyzji spoza automatycznego HOLD" in item for item in issues)


def test_refreshed_fee_with_assumed_execution_costs_remains_unknown(panel):
    now = 1800000000
    data = {
        "last_decision_time": datetime.fromtimestamp(now, UTC).isoformat(),
        "state": {
            "instruments": {"BTC": {"fee_profile": "new"}},
            "fee_profiles": {
                "new": {
                    "valid_until": datetime.fromtimestamp(now + 1000, UTC).isoformat(),
                    "cost_evidence": "FX and slippage are explicit paper assumptions",
                }
            },
        },
    }
    issues = panel.ledger_diagnostics(data, now)
    assert len(issues) == 1
    assert "FX/slippage UNKNOWN" in issues[0]


def test_stale_heartbeat_and_queued_cuda_do_not_claim_training(panel, tmp_path):
    now = 1800000000
    run = tmp_path / "research/mission/run-current"
    record(tmp_path / "research/mission/active.json", {"run": str(run)})
    record(run / "status.json", {"phase": "research-and-learning-loop"})
    record(run / "learning/heartbeat.json", {"stage": "research-A", "updated": now - 601})
    record(run / "learning/cpu-pilots/cpu-pilot-B.json", {"device": "cuda", "state": "queued"})
    result = panel.extra_status(tmp_path, now=now)
    assert result["mission"]["learning"] == "research-A"
    assert result["mission"]["learning_heartbeat_age_seconds"] == 601
    assert result["mission"]["compute_jobs"] == ["cuda:queued"]
    assert "Brak potwierdzenia" in result["gpu"]["state"]
    assert any("Heartbeat" in item for item in result["diagnostics"])


def test_missing_active_run_does_not_reuse_old_compute_receipt(panel, tmp_path):
    record(
        tmp_path / "research/mission/run-old/learning/cpu-pilots/cpu-pilot-B.json",
        {"device": "cuda", "state": "complete"},
    )
    result = panel.extra_status(tmp_path)
    assert result["mission"]["compute_jobs"] == []
    assert any("aktywnej misji" in item for item in result["diagnostics"])


def test_spot_rule_blocker_visible(panel, tmp_path):
    record(
        tmp_path / "research/paper/spot-rules-status.json",
        {"state": "blocked", "reason": "Published minimum quantity changed"},
    )
    result = panel.extra_status(tmp_path)
    assert any("Published minimum" in item for item in result["diagnostics"])


def test_fresh_learning_heartbeat_exposes_cycle_and_is_not_training(panel, tmp_path):
    now = 1800000000
    run = tmp_path / "research/mission/run-current"
    record(tmp_path / "research/mission/active.json", {"run": str(run)})
    record(run / "status.json", {"phase": "research-and-learning-loop"})
    record(
        run / "learning/heartbeat.json",
        {
            "stage": "candidate-training-and-independent-tests",
            "updated": now - 30,
            "cycle_in_progress": 4,
        },
    )

    result = panel.extra_status(tmp_path, now=now)

    assert result["mission"]["learning_heartbeat_age_seconds"] == 30
    assert result["mission"]["learning_cycle"] == 4
    assert result["mission"]["learning_heartbeat_updated"]
    assert "Heartbeat pętli (żywość, nie trening)" in panel.HTML
    assert "NIEAKTUALNY" in panel.HTML
