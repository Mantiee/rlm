"""Regressions for misleading progress and fee status in the installed panel."""

import importlib
import json
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


def test_stale_heartbeat_and_queued_cuda_do_not_claim_training(panel, tmp_path):
    now = 1800000000
    run = tmp_path / "research/mission/run-current"
    record(tmp_path / "research/mission/active.json", {"run": str(run)})
    record(run / "status.json", {"phase": "research-and-learning-loop"})
    record(run / "learning/heartbeat.json", {"stage": "research-A", "updated": now - 601})
    record(run / "learning/cpu-pilots/cpu-pilot-B.json", {"device": "cuda", "state": "queued"})
    result = panel.extra_status(tmp_path, now=now)
    assert result["mission"]["learning"] == "research-A"
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
