import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlm.v100 import code_lab, desktop
from rlm.v100.common import atomic_json
from rlm.v100.protection import file_hash


def setup(root, monkeypatch):
    folder = root / "research/desktop"
    folder.mkdir(parents=True)
    (folder / "source.iso").write_bytes(b"old ISO")
    (folder / "work.qcow2").write_bytes(b"private persistent work")
    atomic_json(
        root / "research/self-code-source.json",
        {"source": str(root / "source"), "revision": "b" * 40},
    )
    value = {"runtime": {}, "source_revision": "a" * 40}
    atomic_json(folder / "manifest.json", value)
    monkeypatch.setattr(desktop, "binary", lambda *a: ["xorriso"])
    monkeypatch.setattr(desktop.shutil, "disk_usage", lambda p: SimpleNamespace(free=3 * 2**30))

    def snapshot(source, target):
        target.mkdir()
        (target / "controller.py").write_text("new source")

    monkeypatch.setattr(code_lab, "snapshot_source", snapshot)

    def build(command, **kwargs):
        source = Path(command[-1])
        assert (source / "V100_SOURCE_REVISION").read_text().strip() == "b" * 40
        Path(command[command.index("-o") + 1]).write_bytes(b"new readonly source ISO")

    monkeypatch.setattr(desktop.subprocess, "run", build)
    return folder, value


def test_source_refresh_keeps_guest_disk_and_updates_pinned_iso(tmp_path, monkeypatch):
    folder, value = setup(tmp_path, monkeypatch)
    result = desktop.refresh_source(tmp_path, value)
    assert result["source_revision"] == "b" * 40
    assert result["source_iso_sha256"] == file_hash(folder / "source.iso")
    assert (folder / "work.qcow2").read_bytes() == b"private persistent work"
    monkeypatch.setattr(
        desktop.subprocess, "run", lambda *a, **k: pytest.fail("Unnecessary rebuild")
    )
    assert desktop.refresh_source(tmp_path, result) == result
    (folder / "source.iso").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="ISO changed"):
        desktop.refresh_source(tmp_path, result)


def test_failed_iso_build_keeps_old_manifest_and_iso(tmp_path, monkeypatch):
    folder, value = setup(tmp_path, monkeypatch)

    def failed(*a, **k):
        raise subprocess.CalledProcessError(1, ["xorriso"])

    monkeypatch.setattr(desktop.subprocess, "run", failed)
    with pytest.raises(subprocess.CalledProcessError):
        desktop.refresh_source(tmp_path, value)
    assert (folder / "source.iso").read_bytes() == b"old ISO"
    assert json.loads((folder / "manifest.json").read_text()) == value


def test_source_iso_never_replaced_under_running_guest(tmp_path, monkeypatch):
    folder, value = setup(tmp_path, monkeypatch)
    atomic_json(folder / "status.json", {"running": True, "pid": 123})
    monkeypatch.setattr(desktop.psutil, "pid_exists", lambda pid: True)
    with pytest.raises(ValueError, match="Stop the owned guest"):
        desktop.refresh_source(tmp_path, value)
    assert (folder / "source.iso").read_bytes() == b"old ISO"


def test_health_reports_specific_failed_guest_components(tmp_path, monkeypatch):
    def result(*a, **k):
        return {
            "exit_code": 1,
            "stdout": "CHECK_cloud_init=failed\nCHECK_source=ok\nCHECK_display=failed\nSOURCE_REVISION="
            + "a" * 40
            + "\napt download failed\n",
            "stderr": "",
        }

    monkeypatch.setattr(desktop, "run", result)
    value = desktop.health(tmp_path)
    assert not value["ready"]
    assert value["checks"]["cloud_init"] == "failed"
    assert value["loaded_source_revision"] == "a" * 40
    assert "apt download failed" in value["diagnostic"]


def test_gui_readiness_requires_current_readonly_source_revision(tmp_path, monkeypatch):
    atomic_json(tmp_path / "research/desktop/manifest.json", {"source_revision": "b" * 40})
    monkeypatch.setattr(
        desktop,
        "run",
        lambda *a, **k: {
            "exit_code": 0,
            "stdout": "SOURCE_REVISION=" + "a" * 40 + "\nGUEST_READY\n",
            "stderr": "",
        },
    )
    result = desktop.health(tmp_path)
    assert not result["ready"] and result["expected_source_revision"] == "b" * 40
