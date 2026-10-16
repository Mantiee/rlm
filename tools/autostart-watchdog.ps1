# Persistent fail-closed supervisor for the isolated RTX helper.
# This script never changes global GPU settings or stops unowned processes.
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$logPath = Join-Path $root 'logs/autostart.log'
$statePath = Join-Path $root 'watchdog-state.json'
$resetPath = Join-Path $root 'watchdog-reset.txt'
$manualStopPath = Join-Path $root 'helper-manual-stop.flag'
$launcher = Join-Path $root 'start-rtx3090-helper.ps1'

New-Item -ItemType Directory -Force -Path (Join-Path $root 'logs') | Out-Null

function Write-Log([string]$Message) {
    "$(Get-Date -Format o) $Message" | Add-Content -LiteralPath $logPath
}

# A SID-scoped global mutex serializes scheduled and manual runs across logon
# sessions. Any mutex creation/acquisition error fails closed before the loop.
try {
    $sid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    $mutexName = "Global\Synta-RTX3090-AutostartWatchdog-$sid"
    $watchdogMutex = [System.Threading.Mutex]::new($false, $mutexName)
    try {
        $mutexOwned = $watchdogMutex.WaitOne(0)
    } catch [System.Threading.AbandonedMutexException] {
        $mutexOwned = $true
        Write-Log 'Acquired abandoned watchdog mutex; durable state recovery runs before any launch.'
    }
    if (-not $mutexOwned) {
        Write-Log 'Another watchdog instance owns the per-user mutex; this instance exits without starting work.'
        exit 0
    }
} catch {
    Write-Log "Cannot acquire per-user watchdog mutex; fail-closed without launch: $($_.Exception.Message)"
    exit 21
}

function New-State([bool]$Open = $false, [string]$ErrorText = '') {
    [pscustomobject]@{
        schema_version = 1
        consecutive_failures = 0
        circuit_open = $Open
        retry_after_utc = $null
        attempt_in_progress = $false
        attempt_paused_for_game = $false
        attempt_started_utc = $null
        qualified = $false
        qualified_pid = $null
        qualified_started = $null
        qualified_boot_identity = $null
        stable_seconds = 0
        last_error = $ErrorText
        last_failure_utc = $null
        last_success_utc = $null
        last_reset_utc = $null
    }
}

function Get-BootIdentity {
    # Win32_OperatingSystem.LastBootUpTime is stable for the full Windows boot,
    # unlike a process start time or a wall-clock/uptime estimate. If CIM cannot
    # provide it, the caller must fail closed before any GPU launch.
    $operatingSystem = Get-CimInstance -ClassName Win32_OperatingSystem -Property LastBootUpTime -ErrorAction Stop
    if ($null -eq $operatingSystem.LastBootUpTime) {
        throw 'Windows did not provide LastBootUpTime.'
    }
    return $operatingSystem.LastBootUpTime.ToUniversalTime().ToString('o')
}

function Save-State {
    $temp = $statePath + '.' + [guid]::NewGuid().ToString('N') + '.tmp'
    $json = $script:state | ConvertTo-Json -Depth 5
    [IO.File]::WriteAllText($temp, $json, (New-Object System.Text.UTF8Encoding -ArgumentList $false))
    try {
        if (Test-Path -LiteralPath $statePath) {
            [IO.File]::Replace($temp, $statePath, $null)
        } else {
            [IO.File]::Move($temp, $statePath)
        }
    } finally {
        Remove-Item -LiteralPath $temp -ErrorAction SilentlyContinue
    }
}

function Load-State {
    if (-not (Test-Path -LiteralPath $statePath)) { return (New-State) }
    try {
        $loaded = Get-Content -Raw -LiteralPath $statePath | ConvertFrom-Json
        if ($loaded.schema_version -ne 1 -or
            $loaded.consecutive_failures -notin @(0, 1, 2, 3) -or
            $null -eq $loaded.circuit_open -or
            $null -eq $loaded.attempt_in_progress -or
            $null -eq $loaded.attempt_paused_for_game -or
            $null -eq $loaded.qualified -or
            $null -eq $loaded.stable_seconds) {
            throw 'Watchdog state has an unsupported or incomplete schema.'
        }
        if ($loaded.consecutive_failures -ge 3) { $loaded.circuit_open = $true }
        return $loaded
    } catch {
        $backup = $statePath + '.corrupt-' + [DateTimeOffset]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
        Copy-Item -LiteralPath $statePath -Destination $backup -ErrorAction SilentlyContinue
        return (New-State $true ('Invalid watchdog state: ' + $_.Exception.Message))
    }
}

function Reset-Open-Circuit {
    if (-not (Test-Path -LiteralPath $resetPath)) { return $false }
    $token = (Get-Content -Raw -LiteralPath $resetPath).Trim()
    if ($token -cne 'RESET_AFTER_REVIEW') {
        Write-Log 'Reset request ignored: expected exact token RESET_AFTER_REVIEW.'
        Remove-Item -LiteralPath $resetPath -ErrorAction SilentlyContinue
        return $false
    }
    Remove-Item -LiteralPath $resetPath -ErrorAction Stop
    $script:state = New-State
    $script:state.last_reset_utc = [DateTimeOffset]::UtcNow.ToString('o')
    $script:state.qualified_boot_identity = $script:currentBootIdentity
    Save-State
    Write-Log 'Crash-loop circuit manually reset after operator review.'
    return $true
}

function Record-Failure([string]$Reason, [switch]$ManualReview) {
    # A system/watchdog restart during GPU qualification is a stronger failure
    # signal than an ordinary launcher exit. Do not automatically retry a GPU
    # workload after a machine restart; require the operator to inspect the
    # crash cause and explicitly reset the circuit.
    if ($ManualReview) {
        $script:state.consecutive_failures = 3
    } else {
        $script:state.consecutive_failures = [Math]::Min(3, [int]$script:state.consecutive_failures + 1)
    }
    $script:state.circuit_open = $script:state.consecutive_failures -ge 3
    $script:state.attempt_in_progress = $false
    $script:state.attempt_paused_for_game = $false
    $script:state.attempt_started_utc = $null
    $script:state.qualified = $false
    $script:state.qualified_pid = $null
    $script:state.qualified_started = $null
    $script:state.stable_seconds = 0
    $script:state.last_error = $Reason
    $script:state.last_failure_utc = [DateTimeOffset]::UtcNow.ToString('o')
    if ($script:state.circuit_open) {
        $script:state.retry_after_utc = $null
    } else {
        $minutes = if ($script:state.consecutive_failures -eq 1) { 1 } else { 5 }
        $script:state.retry_after_utc = [DateTimeOffset]::UtcNow.AddMinutes($minutes).ToString('o')
    }
    Save-State
    Write-Log ("Owned helper failure {0}/3: {1}" -f $script:state.consecutive_failures, $Reason)
    if ($script:state.circuit_open) {
        Write-Log ("CIRCUIT OPEN. Review the cause, then reset explicitly with: Set-Content -NoNewline -Path '{0}' -Value RESET_AFTER_REVIEW" -f $resetPath)
    }
}

function Read-Owned-Guardian {
    $ownerFile = Join-Path $root 'guardian-owner.json'
    if (-not (Test-Path -LiteralPath $ownerFile)) { return $null }
    try {
        $owner = Get-Content -Raw -LiteralPath $ownerFile | ConvertFrom-Json
        $process = Get-Process -Id ([int]$owner.pid) -ErrorAction SilentlyContinue
        if ($process -and
            $process.StartTime.ToUniversalTime().ToString('o') -eq [string]$owner.started -and
            [string]::Equals($process.Path, [string]$owner.path, [StringComparison]::OrdinalIgnoreCase)) {
            return $owner
        }
    } catch { return $null }
    return $null
}

$state = Load-State
try {
    $currentBootIdentity = Get-BootIdentity
    $bootIdentityChanged =
        ([string]::IsNullOrWhiteSpace([string]$state.qualified_boot_identity) -or
         [string]$state.qualified_boot_identity -cne $currentBootIdentity)
    if (($state.qualified -or $state.attempt_in_progress) -and $bootIdentityChanged) {
        # A reboot while either qualified or still attempting startup can be a
        # system crash (including BugCheck), not a routine worker exit. Detect
        # this before the League pause branch so gameplay cannot mask a reboot
        # during an unqualified GPU startup. Require explicit review before retry.
        $previousBootIdentity = if ($state.qualified_boot_identity) { [string]$state.qualified_boot_identity } else { 'missing' }
        Record-Failure ("Windows boot identity changed while an RTX helper was qualified or starting (previous={0}, current={1}); GPU autostart locked pending manual crash review." -f $previousBootIdentity, $currentBootIdentity) -ManualReview
    } else {
        # Migrate older unqualified schema-v1 state and stamp the current boot.
        $state.qualified_boot_identity = $currentBootIdentity
        Save-State
    }
} catch {
    Write-Log "Cannot verify/persist Windows boot identity; fail-closed without launch: $($_.Exception.Message)"
    exit 20
}

while ($true) {
    try {
        if (Reset-Open-Circuit) { Start-Sleep -Seconds 2; continue }
        if ($state.circuit_open) { Start-Sleep -Seconds 30; continue }

        # The explicit stop flag persists across watchdog/task restarts. The
        # installer or a deliberate manual launch is required to resume.
        if (Test-Path -LiteralPath $manualStopPath) { Start-Sleep -Seconds 30; continue }

        # League is a pause, checked before pending attempts or failure logic.
        if (Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue) {
            if ($state.attempt_in_progress -and -not $state.attempt_paused_for_game) {
                # Persist interruption so a later watchdog start cannot count
                # this abandoned qualification as a crash after League exits.
                $state.attempt_paused_for_game = $true
                Save-State
            }
            Write-Log 'paused-for-game: no launch and no failure count.'
            Start-Sleep -Seconds 30
            continue
        }

        if ($state.attempt_paused_for_game) {
            $state.attempt_in_progress = $false
            $state.attempt_paused_for_game = $false
            $state.attempt_started_utc = $null
            Save-State
            Write-Log 'Discarded interrupted qualification marked by League pause; no failure count added.'
            Start-Sleep -Seconds 2
            continue
        }

        if ($state.attempt_in_progress) {
            Record-Failure 'Watchdog restarted while an owned helper qualification was in progress; GPU autostart locked pending manual crash review.' -ManualReview
            Start-Sleep -Seconds 2
            continue
        }

        $owner = Read-Owned-Guardian
        if ($state.qualified) {
            $sameOwner = $owner -and ([int]$owner.pid -eq [int]$state.qualified_pid) -and ([string]$owner.started -eq [string]$state.qualified_started)
            if ($sameOwner) {
                $listener = @(Get-NetTCPConnection -State Listen -LocalPort 11435 -ErrorAction SilentlyContinue).Count -gt 0
                if ($listener) {
                    $state.stable_seconds = [int]$state.stable_seconds + 30
                    if ($state.stable_seconds -ge 600 -and $state.consecutive_failures -gt 0) {
                        $state.consecutive_failures = 0
                        $state.retry_after_utc = $null
                        $state.last_error = ''
                        Write-Log 'Qualified helper remained stable for 10 active minutes; consecutive failure count cleared.'
                    }
                    Save-State
                }
                Start-Sleep -Seconds 30
                continue
            }
            Record-Failure 'Previously qualified owned helper exited or its process identity changed unexpectedly.'
            Start-Sleep -Seconds 2
            continue
        }

        if ($owner) {
            # An unqualified but identity-verified worker may still be warming;
            # never create a competing GPU worker while it is alive.
            Start-Sleep -Seconds 30
            continue
        }

        $retryAfter = $null
        if ($state.retry_after_utc) {
            try { $retryAfter = [DateTimeOffset]::Parse([string]$state.retry_after_utc).ToUniversalTime() }
            catch { $state = New-State $true 'Invalid retry_after_utc; manual review required.'; Save-State; continue }
        }
        if ($retryAfter -and [DateTimeOffset]::UtcNow -lt $retryAfter) { Start-Sleep -Seconds 30; continue }

        $listener = @(Get-NetTCPConnection -State Listen -LocalPort 11435 -ErrorAction SilentlyContinue).Count -gt 0
        if ($listener) {
            Write-Log 'Blocked: port 11435 is occupied without a verified owned guardian; no process stopped or failure counted.'
            Start-Sleep -Seconds 30
            continue
        }

        $configPath = Join-Path $root 'adaptive-settings.json'
        if (-not (Test-Path -LiteralPath $configPath)) {
            Write-Log 'Blocked: adaptive-settings.json is missing; no GPU start attempted.'
            Start-Sleep -Seconds 30
            continue
        }
        $config = Get-Content -Raw -LiteralPath $configPath | ConvertFrom-Json
        $python = Join-Path $env:USERPROFILE 'ai-owned-compute\venv\Scripts\python.exe'
        if ($config.adaptive -and -not (Test-Path -LiteralPath $python)) {
            Write-Log 'Blocked: isolated CPU Python runtime is unavailable; no GPU start attempted.'
            Start-Sleep -Seconds 30
            continue
        }

        $state.attempt_in_progress = $true
        $state.attempt_paused_for_game = $false
        $state.attempt_started_utc = [DateTimeOffset]::UtcNow.ToString('o')
        $state.qualified_boot_identity = $currentBootIdentity
        Save-State
        $gameSeenDuringAttempt = $false
        try {
            $arguments = '-NoProfile -ExecutionPolicy Bypass -File "{0}" -ActiveTimePercent 65 -BatchTokens 64 -Context 32768' -f $launcher
            if ($config.adaptive) { $arguments += ' -ResourceProxyPython "{0}"' -f $python }
            $stdout = Join-Path $root 'logs/launcher.stdout.log'
            $stderr = Join-Path $root 'logs/launcher.stderr.log'
            $launchProcess = Start-Process -FilePath (Join-Path $PSHOME 'powershell.exe') -ArgumentList $arguments -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
            while (-not $launchProcess.HasExited) {
                if (Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue) {
                    $gameSeenDuringAttempt = $true
                    if (-not $state.attempt_paused_for_game) {
                        $state.attempt_paused_for_game = $true
                        Save-State
                    }
                }
                Start-Sleep -Seconds 3
                $launchProcess.Refresh()
            }

            if ($launchProcess.ExitCode -ne 0) {
                if ($gameSeenDuringAttempt -or (Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue)) {
                    $state.attempt_in_progress = $false
                    $state.attempt_paused_for_game = $false
                    $state.attempt_started_utc = $null
                    Save-State
                    Write-Log 'paused-for-game during failed qualification: no failure count added.'
                } else {
                    $tail = if (Test-Path -LiteralPath $stderr) { (Get-Content -Tail 12 -LiteralPath $stderr) -join ' ' } else { 'no launcher stderr' }
                    throw ("Launcher exited {0}: {1}" -f $launchProcess.ExitCode, $tail)
                }
            } else {
                $receiptPath = Join-Path $root 'helper-ready.json'
                $attemptTime = [DateTimeOffset]::Parse([string]$state.attempt_started_utc).UtcDateTime
                if (-not (Test-Path -LiteralPath $receiptPath) -or (Get-Item -LiteralPath $receiptPath).LastWriteTimeUtc -lt $attemptTime) {
                    throw 'Launcher returned without a fresh helper-ready receipt.'
                }
                $receipt = Get-Content -Raw -LiteralPath $receiptPath | ConvertFrom-Json
                $owner = Read-Owned-Guardian
                $listener = @(Get-NetTCPConnection -State Listen -LocalPort 11435 -ErrorAction SilentlyContinue).Count -gt 0
                if (-not $receipt.infrastructure_smoke_passed -or -not $owner -or -not $listener) {
                    throw 'Launcher did not leave a qualified owned process and listener.'
                }
                $state.attempt_in_progress = $false
                $state.attempt_paused_for_game = $false
                $state.attempt_started_utc = $null
                $state.qualified = $true
                $state.qualified_pid = [int]$owner.pid
                $state.qualified_started = [string]$owner.started
                $state.qualified_boot_identity = $currentBootIdentity
                $state.stable_seconds = 0
                $state.last_success_utc = [DateTimeOffset]::UtcNow.ToString('o')
                Save-State
                Write-Log 'Owned helper qualified; starting 10 active-minute stability window.'
            }
        } catch {
            if ($gameSeenDuringAttempt -or (Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue)) {
                $state.attempt_in_progress = $false
                $state.attempt_paused_for_game = $false
                $state.attempt_started_utc = $null
                Save-State
                Write-Log 'paused-for-game during failed qualification: no failure count added.'
            } else {
                Record-Failure $_.Exception.Message
            }
        }
    } catch {
        Write-Log "Watchdog error; fail-closed, no launch this interval: $($_.Exception.Message)"
    }
    Start-Sleep -Seconds 30
}
