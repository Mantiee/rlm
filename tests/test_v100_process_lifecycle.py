"""Exited workers must not block updates or be signalled as live missions."""

from pathlib import Path

import pytest

from rlm.v100 import mission, serving
from rlm.v100.common import atomic_json


@pytest.mark.parametrize("state", ["Z", "X", "x", "S", "R"])
def test_identity_rejects_exited_process_even_when_start_ticks_remain(monkeypatch, state):
    fields = [state] + ["0"] * 18 + ["42"]
    monkeypatch.setattr(
        Path, "read_text", lambda self: "123 (name with ) spaces) " + " ".join(fields)
    )
    if state in {"Z", "X", "x"}:
        with pytest.raises(ProcessLookupError, match="exited"):
            serving.process_identity(123)
    else:
        assert serving.process_identity(123) == "42"


def test_stop_handles_exit_after_status_without_signal(tmp_path, monkeypatch):
    live = {"running": True, "pid": 123, "process_start": "42", "run": str(tmp_path)}
    states = iter([live, {**live, "running": False}])
    monkeypatch.setattr(mission, "status", lambda root: next(states))
    monkeypatch.setattr(Path, "read_bytes", lambda self: (_ for _ in ()).throw(FileNotFoundError()))
    monkeypatch.setattr(mission.os, "killpg", lambda *args: pytest.fail("Exited process signalled"))
    assert not mission.stop(tmp_path)["running"]


@pytest.mark.parametrize("state", ["Z", "S"])
def test_updater_bootstrap_handles_old_status_but_keeps_live_identity_guard(
    tmp_path, monkeypatch, state
):
    script = Path("tools/upgrade-v100-campaign.sh").read_text()
    bootstrap = script.split("<<'PY_STOP'\n", 1)[1].split("\nPY_STOP", 1)[0]
    record = {"running": True, "pid": 123, "process_start": "42"}
    monkeypatch.setenv("AI_V100_ROOT", str(tmp_path))
    monkeypatch.setattr(mission, "status", lambda root: record)
    original = Path.read_text

    def stat(path, *args, **kwargs):
        if str(path) == "/proc/123/stat":
            return "123 (python) " + " ".join([state] + ["0"] * 18 + ["42"])
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", stat)

    def guarded_stop(root):
        raise ValueError("Mission process identity differs; no process was stopped")

    monkeypatch.setattr(mission, "stop", guarded_stop)
    if state == "S":
        with pytest.raises(ValueError, match="no process was stopped"):
            exec(compile(bootstrap, "upgrade-bootstrap", "exec"), {})
    else:
        exec(compile(bootstrap, "upgrade-bootstrap", "exec"), {})
    assert (tmp_path / "research/supervisor/pause.json").exists()


def test_zombie_status_retains_checkpoint_record(tmp_path, monkeypatch):
    run = tmp_path / "research/mission/run-test"
    run.mkdir(parents=True)
    atomic_json(run.parent / "active.json", {"pid": 123, "process_start": "42", "run": str(run)})
    atomic_json(run / "status.json", {"phase": "failed"})
    monkeypatch.setattr(
        mission, "process_identity", lambda pid: (_ for _ in ()).throw(ProcessLookupError())
    )
    result = mission.status(tmp_path)
    assert not result["running"]
    assert result["run"] == str(run)
    assert result["state"]["phase"] == "failed"
