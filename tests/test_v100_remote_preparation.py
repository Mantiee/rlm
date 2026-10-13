import json
import subprocess
from types import SimpleNamespace

from rlm.v100 import managed_bootstrap


def bundle(tmp_path):
    folder = tmp_path / "bundle"
    source = folder / "rlm/v100/source_sync.py"
    source.parent.mkdir(parents=True)
    source.write_text("verified source downloader")
    names = ("rlm/v100/managed_prepare.py", "rlm/v100/model_roles.py", "rlm/v100/remote_helper.py")
    files = {name: "a" * 64 for name in names}
    (folder / "release.json").write_text(json.dumps({"files": files}))
    snapshot = tmp_path / "research/managed-source" / ("b" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "MANAGED_SOURCE.json").write_text(json.dumps({"files": files}))
    return folder, snapshot


def test_pinned_remote_preparation_is_detached_from_training(tmp_path, monkeypatch):
    folder, snapshot = bundle(tmp_path)
    monkeypatch.setattr(managed_bootstrap.Path, "home", lambda: tmp_path / "home")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=3 if "is-active" in command else 0)

    monkeypatch.setattr(managed_bootstrap.subprocess, "run", run)
    result = managed_bootstrap.remote_preparation(tmp_path, folder, "b" * 40)
    assert result["stage"] == "queued"
    unit = (tmp_path / "home/.config/systemd/user/synta-role-qualification.service").read_text()
    assert "WorkingDirectory=" + str(snapshot) in unit
    assert "--remote-only" in unit
    assert "--no-block" in calls[-1]
    assert not any("restart" in command or "stop" in command for command in calls)


def test_changed_remote_source_does_not_start_qualification(tmp_path, monkeypatch):
    folder, snapshot = bundle(tmp_path)
    (snapshot / "MANAGED_SOURCE.json").write_text(json.dumps({"files": {}}))
    calls = []
    monkeypatch.setattr(
        managed_bootstrap.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command) or SimpleNamespace(returncode=3),
    )
    result = managed_bootstrap.remote_preparation(tmp_path, folder, "b" * 40)
    assert result["stage"] == "blocked"
    assert "differs" in result["error"]
    assert not any("start" in command for command in calls)


def test_active_learning_defers_package_but_not_remote_qualification(tmp_path, monkeypatch):
    monkeypatch.setattr(managed_bootstrap, "dashboard_update", lambda *args: {"stage": "ready"})
    monkeypatch.setattr(managed_bootstrap, "active_learning_cycle", lambda root: True)
    requested = []
    monkeypatch.setattr(
        managed_bootstrap,
        "remote_preparation",
        lambda *args: requested.append(args) or {"stage": "queued"},
    )
    monkeypatch.setattr(
        managed_bootstrap.subprocess,
        "check_output",
        lambda command, **kwargs: "" if command[0] == "ps" else "0.1.3+v100.92",
    )
    result = managed_bootstrap.linux(tmp_path, tmp_path, "b" * 40, {"version": 96})
    assert result["stage"] == "deferred-active-learning-cycle"
    assert result["remote_qualification"]["stage"] == "queued"
    assert len(requested) == 1


def test_watchdog_timeout_keeps_remote_repair_and_defers_trainer_switch(tmp_path, monkeypatch):
    watchdog = tmp_path / "rlm/v100/mission_watchdog.py"
    watchdog.parent.mkdir(parents=True)
    watchdog.write_text("pass\n")
    monkeypatch.setattr(managed_bootstrap, "dashboard_update", lambda *args: {"stage": "ready"})
    monkeypatch.setattr(managed_bootstrap, "remote_chat_update", lambda *args: {"stage": "ready"})
    monkeypatch.setattr(managed_bootstrap, "active_learning_cycle", lambda root: False)
    requested = []
    monkeypatch.setattr(
        managed_bootstrap,
        "remote_preparation",
        lambda *args: requested.append(args) or {"stage": "queued"},
    )
    monkeypatch.setattr(
        managed_bootstrap.subprocess,
        "check_output",
        lambda command, **kwargs: "" if command[0] == "ps" else "0.1.3+v100.92",
    )
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(managed_bootstrap.subprocess, "run", run)
    result = managed_bootstrap.linux(tmp_path, tmp_path, "b" * 40, {"version": 99})
    assert result["stage"] == "deferred-mission-recovery"
    assert result["remote_qualification"]["stage"] == "queued"
    assert result["remote_chat"]["stage"] == "ready"
    assert len(requested) == len(calls) == 1
    receipt = json.loads((tmp_path / "research/managed-mission-recovery.json").read_text())
    assert receipt["stage"] == "blocked"
    assert "timed out" in receipt["error"]
