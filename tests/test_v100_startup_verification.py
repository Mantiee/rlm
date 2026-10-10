import hashlib
import json
import os
import socket
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path

import pytest

from rlm.v100 import competition, protection, serving
from rlm.v100.common import atomic_json
from tests.test_v100_continual import profile


def test_unchanged_model_admission_does_not_reread_weights(tmp_path, monkeypatch):
    settings = profile(tmp_path)
    monkeypatch.setattr(serving, "process_identity", lambda pid: "verified-process")
    serving.write_receipt(settings, tmp_path)
    original_open = Path.open

    def forbid_artifact_reads(path, *args, **kwargs):
        if path in (Path(settings["server"]["model"]), Path(settings["server"]["binary"])):
            raise AssertionError("Verified unchanged artifact was read again")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", forbid_artifact_reads)

    class Client:
        def request(self, endpoint):
            assert endpoint == "/props"
            return {"model_path": settings["server"]["model"]}

    for _ in range(3):
        serving.assert_served_expert(Client(), settings, tmp_path)
    changed = deepcopy(settings)
    changed["server"]["threads"] += 1
    with pytest.raises(ValueError, match="not the model"):
        serving.assert_served_expert(Client(), changed, tmp_path)
    with pytest.raises(ValueError, match="not the model"):
        serving.assert_served_expert(Client(), settings, tmp_path, {"files": {"model-gguf": "bad"}})


def test_hash_cache_detects_same_size_change_with_restored_mtime(tmp_path):
    model = tmp_path / "weights"
    model.write_bytes(b"old")
    before = model.stat()
    assert protection.file_hash(model) == hashlib.sha256(b"old").hexdigest()
    model.write_bytes(b"new")
    os.utime(model, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert protection.file_hash(model) == hashlib.sha256(b"new").hexdigest()


def test_hash_cache_detects_replaced_file_and_new_symlink_target(tmp_path):
    original, other, alias = (tmp_path / name for name in ("original", "other", "alias"))
    original.write_bytes(b"old")
    other.write_bytes(b"new")
    alias.symlink_to(original)
    assert protection.file_hash(alias) == hashlib.sha256(b"old").hexdigest()
    alias.unlink()
    alias.symlink_to(other)
    assert protection.file_hash(alias) == hashlib.sha256(b"new").hexdigest()
    os.replace(other, original)
    assert protection.file_hash(original) == hashlib.sha256(b"new").hexdigest()


def test_hash_refuses_mutation_during_read_and_does_not_cache_result(tmp_path):
    model = tmp_path / "weights"
    model.write_bytes(b"old")

    def change(path, completed, total):
        if completed:
            path.write_bytes(b"new")

    with pytest.raises(ValueError, match="changed during hashing"):
        protection.file_hash(model, progress=change)
    assert protection.file_hash(model) == hashlib.sha256(b"new").hexdigest()


def test_cancellation_checked_before_cache_hit_and_between_reads(tmp_path):
    model = tmp_path / "weights"
    model.write_bytes(b"old")
    checks = []

    def cancel():
        checks.append(True)
        raise TimeoutError("deadline")

    with pytest.raises(TimeoutError, match="deadline"):
        protection.file_hash(model, check=cancel)
    protection.file_hash(model)
    with pytest.raises(TimeoutError, match="deadline"):
        protection.file_hash(model, check=cancel)
    assert len(checks) == 2


def test_receipt_rejects_restored_mtime_and_wrong_process(tmp_path, monkeypatch):
    settings = profile(tmp_path)
    monkeypatch.setattr(serving, "process_identity", lambda pid: "verified-process")
    serving.write_receipt(settings, tmp_path)
    model = Path(settings["server"]["model"])
    before = model.stat()
    model.write_bytes(b"changed weights!")
    os.utime(model, ns=(before.st_atime_ns, before.st_mtime_ns))

    class Client:
        def request(self, endpoint):
            return {"model_path": settings["server"]["model"]}

    with pytest.raises(ValueError):
        serving.assert_served_expert(Client(), settings, tmp_path)
    monkeypatch.setattr(serving, "process_identity", lambda pid: "different-process")
    with pytest.raises(ValueError, match="different process"):
        serving.assert_served_expert(Client(), settings, tmp_path)


def test_blocked_verification_times_out_without_yield_or_late_receipt(tmp_path, monkeypatch):
    settings = profile(tmp_path)
    binary = Path(settings["server"]["binary"])
    binary.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(30)\n")
    binary.chmod(0o755)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    settings["runtime"].update(base_url=f"http://127.0.0.1:{port}", max_timeout=0.3)
    snapshot = tmp_path / "profile.json"
    atomic_json(snapshot, settings)
    release, finished = threading.Event(), threading.Event()
    processes = []
    launch = competition.subprocess.Popen

    def capture_process(*args, **kwargs):
        process = launch(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(competition.subprocess, "Popen", capture_process)
    receipt_writer = serving.write_receipt

    def blocked(*args, **kwargs):
        try:
            release.wait(5)
            receipt_writer(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(serving, "write_receipt", blocked)
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="verification timed out"):
            with competition.managed_server(snapshot, tmp_path, tmp_path / "server.log"):
                pytest.fail("Unverified server admitted")
        assert time.monotonic() - started < 3
        status = json.loads((tmp_path / "server.startup.json").read_text())
        assert status["stage"] == "failed"
        assert len(processes) == 1 and processes[0].poll() is not None
    finally:
        release.set()
        assert finished.wait(3)
    assert not serving.receipt_path(settings, tmp_path).exists()
    assert json.loads((tmp_path / "server.startup.json").read_text())["stage"] == "failed"


def test_targeted_updater_retains_configuration_and_does_not_reconfigure_windows():
    script = (Path(__file__).parents[1] / "tools/update-synta-startup.sh").read_text()
    assert "'profile.json', profile" in script
    assert "load_profile(source, root)" in script
    assert "startup-update" in script
    assert "remote_helper_enabled" not in script
    assert "prepare_desktop" not in script
    assert "update-synta-navigation" not in script
