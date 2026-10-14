"""The bounded supervisor verification never leaves a valid unit unstartable."""

import subprocess
from types import SimpleNamespace

import pytest

from rlm.v100 import supervisor


def _run_with_verifier(tmp_path, monkeypatch, *, verify, live_active=True):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == "systemd-analyze":
            if verify == "timeout":
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            return SimpleNamespace(
                returncode=1 if verify == "rejected" else 0,
                stderr="Unit has an invalid directive" if verify == "rejected" else "",
                stdout="",
            )
        if command[:3] == ["systemctl", "--user", "is-active"]:
            return SimpleNamespace(returncode=0 if live_active else 3, stderr="", stdout="")
        if command[0] == "loginctl":
            return SimpleNamespace(returncode=0, stderr="", stdout="yes\\n")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(supervisor.shutil, "which", lambda _: "/usr/bin/existing")
    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    root = tmp_path / "ai-v100"
    root.mkdir()
    profile = root / "research" / "accepted-profile.json"
    profile.parent.mkdir()
    profile.write_text("{}")
    return root, profile, calls


def test_timeout_does_not_abort_unit_if_real_systemd_activation_succeeds(
    tmp_path, monkeypatch
):
    root, profile, calls = _run_with_verifier(tmp_path, monkeypatch, verify="timeout")
    receipt = supervisor.install(root, profile)
    assert receipt["mode"] == "user service"
    assert receipt["static_unit_check"].startswith("timed-out")
    assert receipt["live_unit_active"] is True
    assert ["systemctl", "--user", "enable", "--now", "v100-mission.service"] in calls
    assert ["systemctl", "--user", "is-active", "--quiet", "v100-mission.service"] in calls


def test_timeout_does_not_claim_success_when_owned_service_fails(
    tmp_path, monkeypatch
):
    root, profile, _ = _run_with_verifier(
        tmp_path, monkeypatch, verify="timeout", live_active=False
    )
    with pytest.raises(RuntimeError, match="did not activate"):
        supervisor.install(root, profile)


def test_static_unit_rejection_still_fails_before_service_stop(tmp_path, monkeypatch):
    root, profile, calls = _run_with_verifier(tmp_path, monkeypatch, verify="rejected")
    with pytest.raises(RuntimeError, match="Invalid owned service unit"):
        supervisor.install(root, profile)
    assert not any(call[:3] == ["systemctl", "--user", "stop"] for call in calls)
