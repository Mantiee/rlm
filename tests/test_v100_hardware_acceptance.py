import json
import sys
from types import SimpleNamespace

import pytest

from rlm.v100 import capabilities, competition, hardware_acceptance, mission, progress


def test_required_hardware_failure_records_report_and_does_not_start_master(tmp_path, monkeypatch):
    monkeypatch.setattr(mission, "status", lambda root: {"running": False})
    monkeypatch.setattr(competition, "require_idle_gpu", lambda: None)
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    with pytest.raises(ValueError, match="sm70"):
        hardware_acceptance.run(tmp_path, tmp_path / "not-read.json")
    latest = json.loads((tmp_path / "research/hardware-acceptance/latest.json").read_text())
    report = json.loads(open(latest["report"]).read())
    assert report["required_passed"] is False
    assert report["checks"]["v100-cuda-backward"]["state"] == "failed"
    assert "master-inference" not in report["checks"]


def test_hardware_acceptance_never_runs_beside_active_mission(tmp_path, monkeypatch):
    monkeypatch.setattr(mission, "status", lambda root: {"running": True})
    with pytest.raises(ValueError, match="stopped"):
        hardware_acceptance.run(tmp_path, tmp_path / "unused.json")
    assert not (tmp_path / "research/hardware-acceptance").exists()


def test_capability_audit_before_any_mission_has_no_fake_hardware_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(mission, "status", lambda root: {"running": False})
    monkeypatch.setattr(
        progress,
        "report",
        lambda root: dict.fromkeys(
            (
                "last_learning_cycle",
                "accepted_weight_updates_this_run",
                "training_metrics",
                "paper_blockers",
                "available_audited_paper_outcomes",
                "drones",
                "external_compute",
            )
        ),
    )
    result = capabilities.audit(tmp_path)
    assert result["mission_running"] is False
    assert result["helper_active_time_target_percent"] is None
    assert result["master_architecture"]["backend"] == "llama.cpp"
