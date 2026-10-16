"""Real filesystem recovery, corruption and path-boundary checks."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from rlm.v100 import package_recovery as recovery


@pytest.fixture
def installation(tmp_path, monkeypatch):
    root = tmp_path / "owned"
    site = root / "venvs/runtime/site-packages"
    (site / "rlm").mkdir(parents=True)
    (site / "rlm/code.py").write_text("OLD")
    meta = site / "rlms-103.dist-info"
    meta.mkdir()
    (meta / "METADATA").write_text("Version: 103")
    dist = SimpleNamespace(
        version="103",
        files=[Path("rlms-103.dist-info/METADATA")],
        locate_file=lambda path: site / path,
    )
    monkeypatch.setattr(recovery.importlib.metadata, "distribution", lambda name: dist)
    return root, site


def test_restore_recovers_old_code_and_metadata_and_keeps_failed_files(installation):
    root, site = installation
    destination = root / "research/backup"
    recovery.snapshot(root, destination)
    (site / "rlm/code.py").write_text("FAILED NEW")
    meta = site / "rlms-104.dist-info"
    meta.mkdir()
    (meta / "METADATA").write_text("Version: 104")
    result = recovery.restore(root, destination)
    assert result["restored_version"] == "103"
    assert (site / "rlm/code.py").read_text() == "OLD"
    assert (site / "rlms-103.dist-info/METADATA").read_text() == "Version: 103"
    assert not meta.exists()
    assert (destination / "displaced-runtime/rlm/code.py").read_text() == "FAILED NEW"


def test_corrupt_backup_does_not_touch_running_installation(installation):
    root, site = installation
    destination = root / "research/backup"
    recovery.snapshot(root, destination)
    (destination / "rlm/code.py").write_text("CORRUPTED")
    with pytest.raises(ValueError, match="integrity"):
        recovery.restore(root, destination)
    assert (site / "rlm/code.py").read_text() == "OLD"


def test_foreign_backup_path_is_refused(installation):
    root, _ = installation
    with pytest.raises(ValueError, match="owned research"):
        recovery.snapshot(root, root.parent / "foreign")


def test_existing_snapshot_is_never_overwritten(installation):
    root, _ = installation
    destination = root / "research/backup"
    recovery.snapshot(root, destination)
    with pytest.raises(ValueError, match="new backup"):
        recovery.snapshot(root, destination)
