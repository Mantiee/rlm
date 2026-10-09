"""Startup policy contracts; Windows Scheduled Task execution needs Windows QA."""

from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def test_cpu_startup_preserves_authenticated_mailbox_and_ongoing_jobs():
    source = (TOOLS / "install-owned-compute-autostart.ps1").read_text()
    assert "worker-settings.json" in source and "--mailbox" in source
    assert "Synta-Owned-CPU-Worker" in source and "-RunLevel Limited" in source
    assert "-AtLogOn -User $user" in source and "-LogonType Interactive" in source
    assert "-MultipleInstances IgnoreNew" in source
    assert "$_.CommandLine.Contains($root)" in source
    for dangerous in (
        "Stop-Process",
        "New-SmbShare",
        "New-PSDrive",
        "ConvertTo-SecureString",
        "-Password",
    ):
        assert dangerous not in source
    assert "if (-not $existing)" in source
    assert "cpu-worker.stdout.log" in source and "cpu-worker.stderr.log" in source


def test_combined_startup_includes_cpu_and_separate_visible_monitor():
    source = (TOOLS / "install-rtx3090-autostart.ps1").read_text()
    assert "'install-owned-compute-autostart.ps1') -Revision $Revision" in source
    monitor = source.split("$monitorAction =", 1)[1].split("$monitorPrincipal =", 1)[0]
    assert "-WindowStyle Hidden" not in monitor
    assert "Synta-Helper-Monitor" in source
    assert "-ActiveTimePercent 50 -BatchTokens 16 -Context 32768" in source
    assert "'guardian-owner.json'" in source
    assert "-ActiveTimePercent 50" in (TOOLS / "start-windows-lab.ps1").read_text()


def test_monitor_uses_actual_lan_binding_and_startup_does_not_resolve_packages_unconditionally():
    monitor = (TOOLS / "watch-rtx3090-helper.ps1").read_text()
    assert "'http://192.168.0.61:11435'" in monitor
    assert "$Endpoint.TrimEnd('/') + '/api/ps'" in monitor
    worker = (TOOLS / "start-owned-compute-worker.ps1").read_text()
    assert "m.version('torch').split('+')[0] == '2.6.0'" in worker
    assert "m.version('safetensors') == '0.5.3'" in worker
    assert "m.version('psutil') == '7.0.0'" in worker
    assert "worker-settings.json" in worker
