import subprocess
from types import SimpleNamespace

from rlm.v100 import mission_view


def test_renderer_metadata_does_not_replace_trainer_identity(tmp_path, monkeypatch):
    python = tmp_path / "venvs/v100-continual/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    calls = []
    monkeypatch.setattr(mission_view, "version", lambda name: "0.1.3+v100.99")

    def run(command, **kwargs):
        calls.append(command)
        assert "-I" in command
        assert kwargs["timeout"] == 3
        return SimpleNamespace(stdout="0.1.3+v100.92\n")

    monkeypatch.setattr(mission_view.subprocess, "run", run)
    view = mission_view.snapshot(tmp_path, {}, {})
    assert view["installed_package"] == "0.1.3+v100.92"
    assert view["renderer_package"] == "0.1.3+v100.99"
    assert view["training_package_probe"]["version"] == "0.1.3+v100.92"
    mission_view.snapshot(tmp_path, {}, {})
    assert len(calls) == 1


def test_failed_probe_is_unknown_instead_of_renderer_version(tmp_path, monkeypatch):
    python = tmp_path / "venvs/v100-continual/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()

    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 3)

    monkeypatch.setattr(mission_view.subprocess, "run", timeout)
    value = mission_view.training_package(tmp_path)
    assert value["version"] is None
    assert value["error"]


def test_invalid_identity_is_not_reported_as_installed(tmp_path, monkeypatch):
    python = tmp_path / "venvs/v100-continual/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(
        mission_view.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="unexpected")
    )
    assert mission_view.training_package(tmp_path)["version"] is None
