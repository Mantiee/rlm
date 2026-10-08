import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlm.v100 import (
    benchmark_worker,
    code_lab,
    hardware_acceptance,
    mission,
    sandbox_python,
    supervisor,
)
from rlm.v100.common import atomic_json


def test_nltk_private_root_is_created_before_authorization_without_changing_parent(
    tmp_path, monkeypatch
):
    tmp_path.chmod(0o775)
    calls = []

    def download(package, download_dir, quiet):
        resources = Path(download_dir)
        assert resources.is_dir() and resources.stat().st_mode & 0o077 == 0
        calls.append(package)
        return True

    monkeypatch.setitem(sys.modules, "nltk", SimpleNamespace(download=download))
    benchmark_worker.prepare_nltk(tmp_path)
    assert len(calls) == 3 and tmp_path.stat().st_mode & 0o777 == 0o775


@pytest.mark.parametrize("unsafe", ["writable", "link"])
def test_nltk_never_downloads_into_a_preexisting_unsafe_data_root(tmp_path, monkeypatch, unsafe):
    resources = tmp_path / "nltk"
    if unsafe == "writable":
        resources.mkdir(mode=0o775)
        resources.chmod(0o775)
    else:
        outside = tmp_path / "outside"
        outside.mkdir(mode=0o700)
        resources.symlink_to(outside, target_is_directory=True)
    monkeypatch.setitem(
        sys.modules,
        "nltk",
        SimpleNamespace(download=lambda *a, **k: pytest.fail("unsafe download")),
    )
    with pytest.raises(ValueError, match="private directory"):
        benchmark_worker.prepare_nltk(tmp_path)


def test_actual_nltk_enforcement_accepts_precreated_private_root_under_group_writable_parent(
    tmp_path, monkeypatch
):
    nltk = pytest.importorskip("nltk")
    from nltk.downloader import _authorize_data_dir

    assert nltk.pathsec.ENFORCE
    tmp_path.chmod(0o775)
    original = list(nltk.data.path)

    def offline_download(package, download_dir, quiet):
        _authorize_data_dir(download_dir)
        target = Path(download_dir) / (package + ".txt")
        nltk.pathsec.validate_path(target, context="deployment-regression")
        target.write_text("known fixture")
        assert nltk.data.find(package + ".txt")
        return True

    monkeypatch.setattr(nltk, "download", offline_download)
    try:
        benchmark_worker.prepare_nltk(tmp_path)
        assert nltk.pathsec.ENFORCE
    finally:
        nltk.data.path[:] = original


def test_systemd_parses_the_generated_service_instead_of_only_matching_its_text(
    tmp_path, monkeypatch
):
    analyzer = shutil.which("systemd-analyze")
    if not analyzer:
        pytest.skip("systemd unit parser not available")
    actual_run = subprocess.run
    monkeypatch.setattr(supervisor.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(
        supervisor.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="yes\n")
    )
    profile = tmp_path / "profile.json"
    profile.write_text("{}")
    supervisor.install(tmp_path, profile)
    unit = tmp_path / "research/supervisor/v100-mission.service"
    verified = actual_run(
        [analyzer, "verify", "--man=no", str(unit)], capture_output=True, text=True, timeout=30
    )
    if verified.stderr.strip() == "Failed to setup working directory: No such file or directory":
        pytest.skip("The restricted local runtime cannot initialize the systemd unit parser")
    assert verified.returncode == 0, verified.stderr
    unit.write_text(
        unit.read_text().replace(f"WorkingDirectory={tmp_path}", f'WorkingDirectory="{tmp_path}"')
    )
    broken = actual_run(
        [analyzer, "verify", "--man=no", str(unit)], capture_output=True, text=True, timeout=30
    )
    assert broken.returncode != 0 and "WorkingDirectory" in broken.stderr


def test_supervisor_first_start_keeps_new_setup_then_resumes_its_own_accepted_weights(
    tmp_path, monkeypatch
):
    folder = tmp_path / "research/supervisor"
    profile = tmp_path / "prepared.json"
    old = tmp_path / "old-live.json"
    accepted = tmp_path / "new-live.json"
    for path in (profile, old, accepted):
        path.write_text("{}")
    calls = []

    def status(root):
        return {
            "running": False,
            "run": "new-run" if calls else "old-run",
            "learning": {"live_profile": str(accepted if calls else old)},
        }

    def start(root, selected, **kwargs):
        calls.append(selected)
        if len(calls) == 2:
            atomic_json(folder / "pause.json", {})
        return {"run": "new-run"}

    monkeypatch.setattr(mission, "status", status)
    monkeypatch.setattr(mission, "start", start)
    monkeypatch.setattr(mission, "stop", lambda root: None)
    monkeypatch.setattr(supervisor.time, "sleep", lambda seconds: None)
    ticks = iter(range(0, 2000, 100))
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: next(ticks))
    supervisor.loop(tmp_path, profile)
    assert calls == [profile, accepted]


def test_uv_python_chain_mounts_intermediate_bin_without_exposing_project_root(tmp_path):
    base = tmp_path / "python/base/bin/python3"
    train = tmp_path / "venvs/train/bin/python"
    current = tmp_path / "venvs/continual/bin/python"
    for path in (base, train, current):
        path.parent.mkdir(parents=True)
    base.write_text("interpreter fixture")
    train.symlink_to(base)
    current.symlink_to(os.path.relpath(train, current.parent))
    mounts = sandbox_python.runtime_mounts(current)
    assert train.parent in mounts and base.parent in mounts
    assert current.parent.parent in mounts
    assert tmp_path not in mounts and tmp_path / "venvs" not in mounts


def test_sandbox_probe_failure_preserves_actual_child_error_and_log(tmp_path, monkeypatch):
    monkeypatch.setattr(code_lab, "sandbox_command", lambda *args: ["isolated-fixture"])

    def run(args, stdout, **kwargs):
        stdout.write("bwrap: exec Python: No such file or directory\n")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(hardware_acceptance.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="exec Python: No such file"):
        hardware_acceptance.sandbox_probe(tmp_path)
    assert "No such file" in (tmp_path / "sandbox-probe.log").read_text()
