# Update only owned Synta processes; preserve desktop applications and GPU settings.
param([Parameter(Mandatory=$true)][string]$Revision)
$ErrorActionPreference = 'Stop'
if ($Revision -notmatch '^[0-9a-f]{40}$') { throw 'Use the exact published source revision.' }
$helper = Join-Path $env:USERPROFILE 'ai-v100-helper'
$owned = Join-Path $env:USERPROFILE 'ai-owned-compute'
$retained = Join-Path $owned 'worker-settings.json'
$oldLock = $null
$oldNonce = $null
if (Test-Path -LiteralPath $retained) {
    $settings = Get-Content -Raw -LiteralPath $retained | ConvertFrom-Json
    if ($settings.worker -match '^[a-zA-Z0-9_-]{1,48}$' -and $settings.mailbox) {
        $oldLock = Join-Path $settings.mailbox ('workers\' + $settings.worker + '.lock')
        $owner = Join-Path $oldLock 'owner.json'
        if (Test-Path -LiteralPath $owner) {
            $oldNonce = (Get-Content -Raw -LiteralPath $owner | ConvertFrom-Json).nonce
        }
    }
}
# A duty cycle cannot bound instantaneous RTX power. Retire only the isolated helper.
if (Get-ScheduledTask -TaskName 'Synta-RTX3090-Helper' -ErrorAction SilentlyContinue) {
    Stop-ScheduledTask -TaskName 'Synta-RTX3090-Helper'
    Disable-ScheduledTask -TaskName 'Synta-RTX3090-Helper' | Out-Null
}
$stopper = Join-Path $helper 'start-rtx3090-helper.ps1'
if (Test-Path -LiteralPath $stopper) { & $stopper -Stop }
if (Get-ScheduledTask -TaskName 'Synta-Owned-CPU-Worker' -ErrorAction SilentlyContinue) {
    Stop-ScheduledTask -TaskName 'Synta-Owned-CPU-Worker'
}
# Old workers have no update signal. Verify exact isolated executable and source root.
$python = [IO.Path]::GetFullPath((Join-Path $owned 'venv\Scripts\python.exe'))
$processes = Get-CimInstance Win32_Process
$retiredWorker = $false
foreach ($process in $processes) {
    if ($process.ExecutablePath -and
        [IO.Path]::GetFullPath($process.ExecutablePath) -eq $python -and
        $process.CommandLine -and $process.CommandLine.Contains($owned + '\') -and
        ($process.CommandLine.Contains('compute_worker.py') -or $process.CommandLine.Contains('compute_kernel.py'))) {
        Stop-Process -Id $process.ProcessId -ErrorAction SilentlyContinue
        Wait-Process -Id $process.ProcessId -Timeout 10 -ErrorAction SilentlyContinue
        if (Get-Process -Id $process.ProcessId -ErrorAction SilentlyContinue) {
            throw 'Owned worker still running; update stopped to avoid duplicate compute.'
        }
        if ($process.CommandLine.Contains('compute_worker.py')) { $retiredWorker = $true }
    }
}
if ($retiredWorker -and $oldLock -and $oldNonce) {
    $owner = Join-Path $oldLock 'owner.json'
    if (Test-Path -LiteralPath $owner) {
        $currentNonce = (Get-Content -Raw -LiteralPath $owner | ConvertFrom-Json).nonce
        if ($currentNonce -eq $oldNonce) {
            # Retain the verified old worker lock as evidence; do not touch job claims.
            Move-Item -LiteralPath $oldLock -Destination ($oldLock + '-retired-' + [guid]::NewGuid().ToString('N'))
        }
    }
}
$installer = Join-Path $env:TEMP ('synta-cpu-' + $Revision + '.ps1')
Invoke-WebRequest -UseBasicParsing "https://raw.githubusercontent.com/Mantiee/rlm/$Revision/tools/install-owned-compute-autostart.ps1" -OutFile $installer
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $installer -Revision $Revision
if ($LASTEXITCODE -ne 0) { throw 'CPU autostart install failed; see logs.' }
$logs = Join-Path $helper 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$stdout = Join-Path $logs 'cpu-worker.stdout.log'
if (-not (Test-Path -LiteralPath $stdout)) { New-Item -ItemType File -Path $stdout | Out-Null }
$watcher = Join-Path $owned 'watch-low-load.ps1'
@'
$path = Join-Path $env:USERPROFILE 'ai-v100-helper\logs\cpu-worker.stdout.log'
Write-Host 'SYNTA WINDOWS CPU | 2 threads | idle priority | one job | RTX disabled'
Write-Host 'Yields at host CPU >40%, free RAM <6 GiB or foreground browser/video/game.'
Write-Host 'Close this console to stop viewing logs; the scheduled CPU worker stays running.'
Get-Content -LiteralPath $path -Tail 20 -Wait
'@ | Set-Content -Encoding UTF8 -LiteralPath $watcher
Start-Process powershell.exe -ArgumentList @('-NoProfile', '-NoExit', '-ExecutionPolicy', 'Bypass', '-File', ('"' + $watcher + '"'))
Write-Host 'Synta Windows CPU updated and console opened. Board power, clocks and other apps unchanged.'
