from types import SimpleNamespace

import pytest

from rlm.v100 import managed_bootstrap, mission, mission_view, remote_nodes, trade_explorer
from rlm.v100.common import atomic_json
from tests.test_v100_dashboard import dashboard


@pytest.mark.parametrize(
    "relative",
    [
        "development-quality.progress.json",
        "fresh-audit/parent.progress.json",
        "fresh-audit/candidate.progress.json",
    ],
)
def test_current_evaluation_is_read_from_current_run_only(tmp_path, relative):
    run = tmp_path / "research/mission/run-current"
    old = tmp_path / "research/mission/run-old"
    atomic_json(
        old / "learning/update-0001/generation-01/A/quality.progress.json",
        {"schema": "v100-evaluation-progress-v1", "completed": 81},
    )
    path = run / "learning/update-0001/generation-01/A" / relative
    atomic_json(
        path,
        {
            "schema": "v100-evaluation-progress-v1",
            "state": "running",
            "completed": 57,
            "passed": 46,
            "total": 81,
        },
    )
    result = mission_view.snapshot(
        tmp_path,
        {"run": str(run), "running": True, "state": {"phase": "research-and-learning-loop"}},
        {},
    )
    assert result["stage"] == "candidate evaluation"
    assert result["evaluation"]["completed"] == 57
    assert result["evaluation"]["source"] == str(path)


def test_trading_collector_failure_does_not_freeze_mission(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dashboard.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="1,100,1024,40,20")
    )
    monkeypatch.setattr(
        trade_explorer,
        "snapshot",
        lambda root: (_ for _ in ()).throw(ValueError("broken chart receipt")),
    )
    state = dashboard.DashboardState(
        tmp_path, lambda root: {"running": True, "state": {"phase": "evaluation"}}
    )
    state.data = {"collected_at": 1, "mission": {"running": False}, "trades": {"updated": 1}}
    state.refresh()
    assert state.data["collected_at"] > 1
    assert state.data["mission"]["running"] is True
    assert state.data["trades"]["stale"] is True
    assert "broken chart" in state.data["trades"]["errors"][0]


def test_stopped_controller_is_not_relabelled_as_active_evaluation(tmp_path):
    run = tmp_path / "research/mission/run-current"
    atomic_json(
        run / "learning/update-0001/generation-01/A/development-quality.progress.json",
        {
            "schema": "v100-evaluation-progress-v1",
            "state": "running",
            "completed": 0,
            "total": 81,
        },
    )
    result = mission_view.snapshot(
        tmp_path, {"run": str(run), "running": False, "state": {"phase": "stopped"}}, {}
    )
    assert result["stage"] == "stopped"
    assert result["evaluation"]["state"] == "running"  # Preserve the recorded state as evidence.
    assert result["evaluation"]["stale"] is True
    assert "Mission stopped" in result["warnings"][0]


def test_package_update_defers_entire_incomplete_cycle(tmp_path, monkeypatch):
    current = {
        "running": True,
        "learning": {
            "stage": "candidate-training-and-independent-tests",
            "cycle_in_progress": 1,
            "completed_cycles": 0,
        },
    }
    monkeypatch.setattr(mission, "status", lambda root: current)
    assert managed_bootstrap.active_learning_cycle(tmp_path)
    current["learning"]["completed_cycles"] = 1
    assert not managed_bootstrap.active_learning_cycle(tmp_path)


def test_new_dashboard_can_be_applied_while_training_package_waits(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(managed_bootstrap, "dashboard_update", lambda *a: {"stage": "ready"})
    monkeypatch.setattr(managed_bootstrap, "active_learning_cycle", lambda root: True)
    monkeypatch.setattr(
        managed_bootstrap.subprocess,
        "check_output",
        lambda args, **k: "" if args[0] == "ps" else "0.1.3+v100.89",
    )
    monkeypatch.setattr(managed_bootstrap.subprocess, "run", lambda *a, **k: calls.append(a))
    result = managed_bootstrap.linux(tmp_path, tmp_path, "a" * 40, {"version": 91})
    assert result["stage"] == "deferred-active-learning-cycle"
    assert result["dashboard"]["stage"] == "ready"
    assert not calls


def test_changed_node_digest_does_not_switch_to_another_model(tmp_path, monkeypatch):
    atomic_json(
        tmp_path / "research/owned-node/selection-lock.json",
        {"model": "gemma4:e2b-it-q4_K_M", "digest": "a" * 64},
    )
    monkeypatch.setattr(mission, "status", lambda root: {"run": str(tmp_path)})
    monkeypatch.setattr(remote_nodes, "load_profile", lambda *a: {})
    monkeypatch.setattr(
        remote_nodes,
        "request",
        lambda *a: {"models": [{"name": "gemma4:e2b-it-q4_K_M", "digest": "b" * 64}]},
    )
    result = remote_nodes.qualify(tmp_path, {})
    assert not result["admitted"]
    assert result["phase"] == "locked-model-unavailable"


def test_dashboard_receipts_do_not_claim_payment_or_optimizer_progress(tmp_path):
    result = mission_view.snapshot(
        tmp_path, {"running": True, "learning": {"completed_cycles": 0}}, {}
    )
    assert result["completed_cycles"] == 0
    assert result["accepted_updates"] is None
    assert result["activity_age_seconds"] is None
    assert result["verified_income"].startswith("unknown")


def test_dashboard_upgrade_isolated_and_rolls_back(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path

    from rlm.v100 import managed_bootstrap as updater

    root = tmp_path / "ai-v100"
    python = root / "venvs/v100-continual/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    uv = tmp_path / ".local/bin/uv"
    uv.parent.mkdir(parents=True)
    uv.touch()
    entry = root / "bin/serve-v100-dashboard.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("previous renderer")
    folder = tmp_path / "bundle"
    (folder / "tools").mkdir(parents=True)
    (folder / "tools/serve-v100-dashboard.py").write_text("new renderer")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == str(python):
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(updater.subprocess, "run", run)
    result = updater.dashboard_update(root, folder, "a" * 40)
    assert result["stage"] == "blocked"
    assert result["rollback_error"] is None
    assert entry.read_text() == "previous renderer"
    assert not (tmp_path / ".config/systemd/user/v100-dashboard.service.d/20-code.conf").exists()
    assert "--target" in calls[0] and "--no-deps" in calls[0]
    assert all("v100-mission.service" not in command for command in calls)


def test_history_does_not_reset_with_new_run_and_goal_retrieval_is_visible(tmp_path):
    old = tmp_path / "research/mission/run-previous/learning/state.json"
    atomic_json(
        old, {"cycles": [{"cycle": 1, "status": "rejected"}, {"cycle": 2, "status": "accepted"}]}
    )
    atomic_json(
        tmp_path / "research/knowledge/goal-use.json",
        {"goal_id": "goal", "updated": 100, "sources": ["https://en.wikipedia.org/wiki/Example"]},
    )
    current = {
        "run": str(tmp_path / "research/mission/run-current"),
        "learning": {"completed_cycles": 0},
    }
    view = mission_view.snapshot(tmp_path, current, {})
    assert view["completed_cycles"] == 0
    assert view["history"]["completed_cycles"] == 2
    assert view["knowledge"]["goal_use"]["goal_id"] == "goal"
    assert view["knowledge"]["goal_use"]["sources"]
