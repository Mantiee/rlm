import fcntl
import json
from pathlib import Path

import pytest

from rlm.v100 import drones
from rlm.v100 import mission_watchdog as watch
from rlm.v100.common import atomic_json
from rlm.v100.windows_resource_proxy import Pressure, budget


@pytest.fixture
def recovery(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    profile = root / "research/run/live.json"
    atomic_json(profile, {"accepted": True})
    record = {"running": False, "learning": {"live_profile": str(profile)}}
    calls = []
    monkeypatch.setattr(watch, "status", lambda root: record)
    monkeypatch.setattr(watch, "exclusive_work", lambda root: False)
    monkeypatch.setattr(watch, "load_profile", lambda path, root: json.loads(path.read_text()))
    monkeypatch.setattr(watch, "install", lambda root, path: calls.append(path) or {"mode": "test"})
    return root, record, calls


def test_stopped_mission_resumes_accepted_profile_and_backoff_prevents_restart_storm(recovery):
    root, _, calls = recovery
    assert watch.recover(root)["stage"] == "recovery-submitted"
    assert calls == [root / "research/run/live.json"]
    assert watch.recover(root)["stage"] == "recovery-backoff"
    assert len(calls) == 1


def test_running_mission_and_explicit_pause_are_untouched(recovery):
    root, record, calls = recovery
    record["running"] = True
    assert watch.recover(root)["stage"] == "mission-running"
    record["running"] = False
    atomic_json(root / "research/supervisor/pause.json", {"reason": "operator"})
    assert watch.recover(root)["stage"] == "operator-paused"
    assert not calls


def test_resume_request_survives_exclusive_training_and_recovers_afterwards(recovery, monkeypatch):
    root, _, calls = recovery
    monkeypatch.setattr(watch, "exclusive_work", lambda root: True)
    assert watch.recover(root, resume=True)["stage"] == "deferred-training"
    atomic_json(root / "research/supervisor/pause.json", {"reason": "shutdown"})
    monkeypatch.setattr(watch, "exclusive_work", lambda root: False)
    assert watch.recover(root)["stage"] == "recovery-submitted"
    assert len(calls) == 1


def test_installer_lock_excludes_recovery_and_preserves_explicit_resume(recovery):
    root, _, calls = recovery
    with (root / "research/mission-maintenance.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert watch.recover(root, resume=True)["stage"] == "maintenance-in-progress"
    assert not calls
    assert watch.recover(root)["stage"] == "recovery-submitted"


def test_recent_supervisor_heartbeat_keeps_existing_recovery(recovery):
    root, _, calls = recovery
    atomic_json(
        root / "research/supervisor/status.json", {"running": True, "updated": watch.time.time()}
    )
    assert watch.recover(root)["stage"] == "supervisor-recovering"
    assert not calls


def test_profile_escape_never_launches_a_different_model(recovery):
    root, record, calls = recovery
    record["learning"]["live_profile"] = "/tmp/foreign.json"
    assert watch.recover(root)["stage"] == "recovery-blocked"
    assert not calls


def test_exclusive_work_checks_exact_root_and_cli_arguments(tmp_path, monkeypatch):
    monkeypatch.setattr(
        watch.subprocess,
        "check_output",
        lambda *a,
        **k: f"python -m rlm.v100.cli --root {tmp_path}/foreign train file.json\nother unrelated ' text",
    )
    assert not watch.exclusive_work(tmp_path)
    monkeypatch.setattr(
        watch.subprocess,
        "check_output",
        lambda *a, **k: f"python -m rlm.v100.cli --root {tmp_path} train file.json",
    )
    assert watch.exclusive_work(tmp_path)


def test_windows_throughput_only_removes_idle_when_headroom_exists():
    idle = budget(8, 17, 53, 12, 139, 0, False, throughput=True)
    assert idle["ready"] and idle["duty_percent"] == 100 and idle["batch_tokens"] == 256
    assert budget(40, 17, 53, 12, 139, 0, False, True)["duty_percent"] == 35
    assert not budget(8, 17, 75, 12, 139, 0, False, True)["ready"]
    assert not budget(8, 17, 53, 0.5, 139, 0, False, True)["ready"]
    assert not budget(8, 17, 53, 12, 139, 0, True, True)["ready"]


def test_stale_windows_monitor_is_visible_and_cannot_admit_work(tmp_path):
    pressure = Pressure(tmp_path)
    pressure.state = {"ready": True, "updated": watch.time.time() - 60}
    assert pressure.fresh_state()["ready"] is False
    assert pressure.fresh_state()["stale"] is True


def test_telemetry_file_error_does_not_kill_live_sensors(tmp_path, monkeypatch, capsys):
    pressure = Pressure(tmp_path)
    monkeypatch.setattr(
        Path,
        "write_text",
        lambda *a, **k: (_ for _ in ()).throw(PermissionError("sharing violation")),
    )
    pressure.publish()
    assert "telemetry_write_error" in capsys.readouterr().out


def test_separate_node_and_windows_lanes_avoid_two_jobs_waiting_on_one_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(drones, "gpu_disabled", lambda root: False)
    monkeypatch.setattr(drones, "selected_node", lambda root: root / "node.json")
    assert drones.inference_lane(tmp_path, "researcher") == "debian-3060ti"
    assert drones.inference_lane(tmp_path, "critic") == "windows-helper"
    assert drones.inference_lane(tmp_path, "income") == "accepted-master"


def test_windows_live_sample_contains_measured_gpu_utilization(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import psutil

    from rlm.v100 import windows_resource_proxy as proxy

    monkeypatch.setattr(psutil, "process_iter", lambda *a: [])
    monkeypatch.setattr(psutil, "cpu_percent", lambda **k: 8)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=17 * 2**30))
    monkeypatch.setattr(proxy.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(
        proxy.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="53, 12288, 139, 0, 5")
    )
    state = proxy.Pressure(tmp_path).sample()
    assert state["gpu_utilization_percent"] == 5
    assert state["cpu_percent"] == 8 and state["ready"]
