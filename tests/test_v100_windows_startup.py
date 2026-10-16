"""Startup policy contracts; Windows Scheduled Task execution needs Windows QA."""

import json
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


def test_combined_startup_includes_cpu_and_separate_hidden_monitor():
    source = (TOOLS / "install-rtx3090-autostart.ps1").read_text()
    assert "'install-owned-compute-autostart.ps1') -Revision $Revision" in source
    monitor = source.split("$monitorAction =", 1)[1].split("$monitorPrincipal =", 1)[0]
    assert "-WindowStyle Hidden" in monitor
    assert "Synta-Helper-Monitor" in source
    assert (
        "-ActiveTimePercent 65 -BatchTokens 64 -Context 32768"
        in (TOOLS / "autostart-watchdog.ps1").read_text()
    )
    assert "ResourceProxyPython" in (TOOLS / "autostart-watchdog.ps1").read_text()
    assert "guardian-owner.json" in (TOOLS / "autostart-watchdog.ps1").read_text()
    assert "-ActiveTimePercent 50" in (TOOLS / "start-windows-lab.ps1").read_text()


def test_rtx_watchdog_has_durable_fail_closed_crash_loop_breaker():
    installer = (TOOLS / "install-rtx3090-autostart.ps1").read_text()
    watchdog = (TOOLS / "autostart-watchdog.ps1").read_text()
    launcher = (TOOLS / "start-rtx3090-helper.ps1").read_text()
    monitor = (TOOLS / "watch-rtx3090-helper.ps1").read_text()

    assert "'autostart-watchdog.ps1'" in installer
    assert "-RestartCount 0" in installer
    assert "-RestartCount 20" not in installer
    assert "watchdog-state.json" in watchdog
    assert "[IO.File]::Replace" in watchdog and ".tmp" in watchdog
    assert "Global\\Synta-RTX3090-AutostartWatchdog-$sid" in watchdog
    assert "WaitOne(0)" in watchdog
    assert "if (-not $mutexOwned)" in watchdog
    assert "attempt_in_progress = $true" in watchdog
    assert "attempt_paused_for_game = $false" in watchdog
    assert "Save-State\n        $gameSeenDuringAttempt" in watchdog
    assert "Record-Failure 'Watchdog restarted while" in watchdog
    interrupted = watchdog.split("if ($state.attempt_in_progress)", 1)[1].split(
        "Start-Sleep -Seconds 2", 1
    )[0]
    assert "-ManualReview" in interrupted
    failure = watchdog.split("function Record-Failure", 1)[1].split(
        "function Read-Owned-Guardian", 1
    )[0]
    assert "[switch]$ManualReview" in failure
    assert "if ($ManualReview)" in failure
    assert "$script:state.consecutive_failures = 3" in failure
    assert "$script:state.retry_after_utc = $null" in failure
    assert "GPU autostart locked pending manual crash review" in watchdog
    assert "Record-Failure 'Previously qualified owned helper exited" in watchdog
    assert "consecutive_failures = [Math]::Min(3" in watchdog
    assert "if ($script:state.consecutive_failures -eq 1) { 1 } else { 5 }" in watchdog
    assert "CIRCUIT OPEN" in watchdog
    assert "RESET_AFTER_REVIEW" in watchdog
    assert "stable_seconds -ge 600" in watchdog
    assert "helper-manual-stop.flag" in watchdog

    # The game gate must occur before interrupted-attempt accounting.
    assert watchdog.index("Get-Process -Name 'League of Legends'") < watchdog.index(
        "if ($state.attempt_in_progress)"
    )
    game_gate = watchdog.split("# League is a pause", 1)[1].split(
        "if ($state.attempt_in_progress)", 1
    )[0]
    assert "Record-Failure" not in game_gate
    paused_attempt = watchdog.split("if ($state.attempt_paused_for_game)", 1)[1].split(
        "if ($state.attempt_in_progress)", 1
    )[0]
    assert "$state.attempt_in_progress = $false" in paused_attempt
    assert "Record-Failure" not in paused_attempt
    assert "attempt_paused_for_game = $true" in game_gate
    assert "Save-State" in game_gate
    assert "paused-for-game during failed qualification: no failure count added." in watchdog

    assert "helper-manual-stop.flag" in launcher
    assert "CIRCUIT OPEN" in monitor and "watchdog-reset.txt" in monitor


def test_qualified_rtx_previous_boot_state_source_contract(tmp_path):
    """Source contract for reboot recovery, using only isolated temporary state."""
    state_path = tmp_path / "watchdog-state.json"
    previous_state = {
        "schema_version": 1,
        "qualified": True,
        "qualified_boot_identity": "2026-10-15T01:00:00.0000000Z",
        "consecutive_failures": 0,
        "circuit_open": False,
    }
    current_boot_identity = "2026-10-15T02:00:00.0000000Z"
    state_path.write_text(json.dumps(previous_state), encoding="utf-8")
    state_text = state_path.read_text(encoding="utf-8")
    watchdog = (TOOLS / "autostart-watchdog.ps1").read_text()

    # No live helper state is consulted: the fixture is created under tmp_path.
    assert json.loads(state_text)["qualified_boot_identity"] == ("2026-10-15T01:00:00.0000000Z")
    assert json.loads(state_text)["qualified"] is True
    assert json.loads(state_text)["qualified_boot_identity"] != current_boot_identity
    assert "function Get-BootIdentity" in watchdog
    assert "Get-CimInstance -ClassName Win32_OperatingSystem -Property LastBootUpTime" in watchdog
    startup = watchdog.split("$state = Load-State", 1)[1].split("while ($true)", 1)[0]
    assert "$currentBootIdentity = Get-BootIdentity" in startup
    assert (
        "if (($state.qualified -or $state.attempt_in_progress) -and $bootIdentityChanged)"
        in startup
    )
    assert "[string]::IsNullOrWhiteSpace([string]$state.qualified_boot_identity)" in startup
    assert "$state.qualified_boot_identity -cne $currentBootIdentity" in startup
    assert (
        'Record-Failure ("Windows boot identity changed while an RTX helper was qualified or starting'
        in startup
    )
    assert ") -ManualReview" in startup
    assert "Windows boot identity changed while an RTX helper was qualified or starting" in startup
    assert watchdog.index('Record-Failure ("Windows boot identity changed') < watchdog.index(
        "while ($true)"
    )
    assert watchdog.index("while ($true)") < watchdog.index("$launchProcess = Start-Process")

    failure = watchdog.split("function Record-Failure", 1)[1].split(
        "function Read-Owned-Guardian", 1
    )[0]
    manual_review = failure.split("if ($ManualReview)", 1)[1].split("} else {", 1)[0]
    assert "$script:state.consecutive_failures = 3" in manual_review
    assert "$script:state.circuit_open = $script:state.consecutive_failures -ge 3" in failure
    assert "if ($script:state.circuit_open)" in failure
    assert "$script:state.retry_after_utc = $null" in failure
    assert "$state.qualified_boot_identity = $currentBootIdentity" in watchdog
    assert "$script:state.qualified_boot_identity = $script:currentBootIdentity" in watchdog


def test_rtx_reboot_during_unqualified_attempt_with_league_running_requires_review():
    """Pure policy regression: a reboot cannot be disguised as a League pause."""
    watchdog = (TOOLS / "autostart-watchdog.ps1").read_text()
    previous_boot = "2026-10-15T01:00:00.0000000Z"
    current_boot = "2026-10-15T02:00:00.0000000Z"
    state = {
        "qualified": False,
        "attempt_in_progress": True,
        "attempt_paused_for_game": False,
        "qualified_boot_identity": previous_boot,
    }
    league_running = True

    # Mirror the pure startup decision: boot recovery precedes gameplay pause.
    outstanding_attempt = state["qualified"] or state["attempt_in_progress"]
    boot_changed = state["qualified_boot_identity"] != current_boot
    outcome = (
        "manual-review"
        if outstanding_attempt and boot_changed
        else "paused-for-game"
        if league_running
        else "continue"
    )
    assert outcome == "manual-review"

    startup = watchdog.split("$state = Load-State", 1)[1].split("while ($true)", 1)[0]
    recovery = startup.split("$bootIdentityChanged =", 1)[1]
    assert "$state.attempt_in_progress" in recovery
    assert watchdog.index('Record-Failure ("Windows boot identity changed') < watchdog.index(
        "# League is a pause"
    )
    assert "-ManualReview" in recovery


def test_rtx_manual_stop_blocks_installer_refresh_and_direct_launchers():
    installer = (TOOLS / "install-rtx3090-autostart.ps1").read_text()
    refresh = (TOOLS / "refresh-synta-windows.ps1").read_text()
    launcher = (TOOLS / "start-rtx3090-helper.ps1").read_text()
    prepare = (TOOLS / "prepare-managed-windows.ps1").read_text()
    benchmark = (TOOLS / "benchmark-synta-windows.ps1").read_text()
    qualify = (TOOLS / "qualify-synta-models-windows.ps1").read_text()
    resume = (TOOLS / "resume-rtx3090-helper.ps1").read_text()

    installer_guard = installer.split("$manualStopPath =", 1)[1].split(
        "New-Item -ItemType Directory", 1
    )[0]
    assert "Test-Path -LiteralPath $manualStopPath" in installer_guard
    assert "throw 'RTX helper has an operator manual-stop flag." in installer_guard
    assert "Remove-Item -LiteralPath (Join-Path $root 'helper-manual-stop.flag')" not in installer
    assert installer.index("$manualStopPath =") < installer.index(
        "Start-ScheduledTask -TaskName 'Synta-RTX3090-Helper'"
    )

    refresh_guard = refresh.split("$manualStopPath =", 1)[1].split("$receipt =", 1)[0]
    assert "Test-Path -LiteralPath $manualStopPath" in refresh_guard
    assert "throw 'RTX helper operator manual-stop flag is present" in refresh_guard
    assert refresh.index("$manualStopPath =") < refresh.index("Wait-SyntaMaintenance")
    assert refresh.index("$manualStopPath =") < refresh.index("Stop-ScheduledTask")
    assert "Remove-Item -LiteralPath $manualStopPath" not in refresh

    launcher_guard = launcher.split("if (Test-Path -LiteralPath $ManualStopPath)", 1)[1].split(
        "if (Get-NetTCPConnection", 1
    )[0]
    assert "throw 'RTX helper operator manual-stop flag is present" in launcher_guard
    assert "Remove-Item -LiteralPath $ManualStopPath" not in launcher
    assert launcher.index("if (Test-Path -LiteralPath $ManualStopPath)") < launcher.index(
        "nvidia-smi --query-gpu"
    )
    assert launcher.index("if (Test-Path -LiteralPath $ManualStopPath)") < launcher.index(
        "Start-Process -FilePath $OllamaPath"
    )
    worker_source = launcher.split("$WorkerScript =", 1)[1].split("'@ | Set-Content", 1)[0]
    assert "$manualStopPath = Join-Path $HelperRoot 'helper-manual-stop.flag'" in worker_source
    assert "function Stop-IsolatedGpuProcesses" in worker_source
    assert "phase='paused-for-manual-stop'" in worker_source
    assert "Stop-IsolatedGpuProcesses" in worker_source
    assert "-not (Test-Path -LiteralPath $manualStopPath)" in worker_source

    prepare_guard = prepare.split("$manualStopPath =", 1)[1].split("$python =", 1)[0]
    assert "Test-Path -LiteralPath $manualStopPath" in prepare_guard
    assert "model pulls or benchmark work" in prepare_guard
    benchmark_guard = benchmark.split("$manualStopPath =", 1)[1].split("$python =", 1)[0]
    assert "Test-Path -LiteralPath $manualStopPath" in benchmark_guard
    assert "native GPU benchmark refused" in benchmark_guard
    assert benchmark.index("$manualStopPath =") < benchmark.index("& $python $script")

    qualify_guard = qualify.split("$manualStopPath =", 1)[1].split("$python =", 1)[0]
    assert "Test-Path -LiteralPath $manualStopPath" in qualify_guard
    assert "model qualification refused before native GPU work" in qualify_guard
    assert qualify.index("$manualStopPath =") < qualify.index("Get-ChildItem -LiteralPath $runtime")

    assert "[Parameter(Mandatory = $true)]" in resume
    assert "$ReviewedCause" in resume and "Read-Host" in resume
    assert "RESUME RTX 3090" in resume
    assert "Remove-Item -LiteralPath $manualStopPath -ErrorAction Stop" in resume

    installer_names = installer.split("foreach ($name in @(", 1)[1].split(")", 1)[0]
    assert "'resume-rtx3090-helper.ps1'" in installer_names


def test_monitor_uses_actual_lan_binding_and_startup_does_not_resolve_packages_unconditionally():
    monitor = (TOOLS / "watch-rtx3090-helper.ps1").read_text()
    assert "'http://192.168.0.61:11435'" in monitor
    assert "$Endpoint.TrimEnd('/') + '/api/ps'" in monitor
    worker = (TOOLS / "start-owned-compute-worker.ps1").read_text()
    assert "m.version('torch').split('+')[0] == '2.6.0'" in worker
    assert "m.version('safetensors') == '0.5.3'" in worker
    assert "m.version('psutil') == '7.0.0'" in worker
    assert "worker-settings.json" in worker
