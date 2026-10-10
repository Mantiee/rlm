import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlm.v100 import compute_worker
from rlm.v100.experiments import SharedLab
from rlm.v100.paper import PaperBook


def test_financial_chat_snapshot_does_not_run_audit_or_reuse_previous_run(tmp_path, monkeypatch):
    from rlm.v100 import mission, progress

    monkeypatch.setattr(
        mission,
        "status",
        lambda root: {
            "run": "new-run",
            "running": True,
            "state": {"phase": "research"},
        },
    )
    monkeypatch.setattr(progress, "report", lambda root: pytest.fail("Unexpected ledger audit"))
    path = tmp_path / "research/mission/latest-report.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"mission_evidence": {"run": "old-run"}, "paper": {"executed_fills": 5}})
    )
    result = progress.snapshot(tmp_path)
    assert "paper" not in result and result["phase"] == "research"
    path.write_text(
        json.dumps(
            {
                "mission_evidence": {"run": "new-run"},
                "updated_at": "original timestamp",
                "paper": {"executed_fills": 2},
            }
        )
    )
    result = progress.snapshot(tmp_path)
    assert result["paper"]["executed_fills"] == 2
    assert result["updated_at"] == "original timestamp"


def test_readonly_reports_work_during_writer_and_cannot_modify(tmp_path):
    writer = PaperBook(tmp_path)
    writer.initialize()
    writer.db.execute("BEGIN IMMEDIATE")
    reader = PaperBook(tmp_path, read_only=True)
    try:
        assert reader.state()["currency"] == "PLN"
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reader.db.execute("DELETE FROM state")
    finally:
        reader.close()
        writer.db.rollback()
        writer.close()
    shared_path = tmp_path / "lab.sqlite3"
    shared = SharedLab(shared_path)
    shared.append("A", "worker-result", {"hypothesis": "test"})
    shared.db.execute("BEGIN IMMEDIATE")
    readonly = SharedLab(shared_path, read_only=True)
    try:
        assert readonly.recent()[0]["payload"]["hypothesis"] == "test"
    finally:
        readonly.close()
        shared.db.rollback()
        shared.close()


@pytest.mark.parametrize(
    "cpu,ram,foreground,ready",
    [
        (41, 12, False, False),
        (10, 5, False, False),
        (10, 12, True, False),
        (20, 12, False, True),
    ],
)
def test_windows_worker_yields_to_host_pressure(monkeypatch, cpu, ram, foreground, ready):
    import psutil

    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=ram * 2**30))
    monkeypatch.setattr(psutil, "cpu_percent", lambda interval: cpu)
    monkeypatch.setattr(psutil, "process_iter", lambda fields: [])
    monkeypatch.setattr(compute_worker, "foreground_busy", lambda: foreground)
    assert compute_worker.available()[0] is ready


def test_installer_restricts_owned_processes_and_leaves_board_settings():
    source = (Path(__file__).parents[1] / "tools/install-synta-low-load.ps1").read_text()
    assert "GetFullPath($process.ExecutablePath) -eq $python" in source
    assert "compute_worker.py" in source and "compute_kernel.py" in source
    assert "Disable-ScheduledTask" in source and "Synta-RTX3090-Helper" in source
    assert "Get-Content -LiteralPath $path -Tail 20 -Wait" in source
    assert "nvidia-smi" not in source


def test_running_job_stops_on_cpu_pressure(tmp_path, monkeypatch):
    import hashlib

    import psutil

    kernel = tmp_path / "compute_kernel.py"
    kernel.write_text("# controlled test")
    path = tmp_path / "jobs" / ("a" * 24 + ".json")
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "kernel_sha256": hashlib.sha256(kernel.read_bytes()).hexdigest(),
                "worker_sha256": hashlib.sha256(
                    Path(compute_worker.__file__).read_bytes()
                ).hexdigest(),
            }
        )
    )
    lease = {"worker": "test", "nonce": "token"}
    compute_worker.atomic(tmp_path / "claims" / path.stem / "lease.json", lease)
    process = SimpleNamespace(pid=123, returncode=None)
    process.poll = lambda: process.returncode
    process.kill = lambda: setattr(process, "returncode", -9)
    process.wait = lambda: process.returncode
    monkeypatch.setattr(compute_worker.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(compute_worker, "limit_child", lambda pid: None)
    monkeypatch.setattr(compute_worker, "available", lambda: (False, "Host CPU busy"))
    monkeypatch.setattr(
        psutil,
        "Process",
        lambda pid: SimpleNamespace(
            memory_info=lambda: SimpleNamespace(rss=1024), children=lambda recursive: []
        ),
    )
    with pytest.raises(RuntimeError, match="Host CPU busy"):
        compute_worker.execute(tmp_path, path, lease, kernel)
    assert process.returncode == -9
