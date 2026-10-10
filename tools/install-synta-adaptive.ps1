# Explicit operator opt-in: update isolated CPU + RTX, retaining global hardware settings.
param([Parameter(Mandatory=$true)][string]$Revision)
$ErrorActionPreference = 'Stop'
if ($Revision -notmatch '^[0-9a-f]{40}$') { throw 'Use an exact published revision.' }
$root = Join-Path $env:USERPROFILE 'ai-v100-helper'
New-Item -ItemType Directory -Force -Path $root | Out-Null
foreach ($name in @('install-synta-low-load.ps1','install-rtx3090-autostart.ps1')) {
    $target = Join-Path $root $name
    Invoke-WebRequest -UseBasicParsing "https://raw.githubusercontent.com/Mantiee/rlm/$Revision/tools/$name" -OutFile $target
}
# Low-load updater safely retires only verified owned workers/old leases.
& (Join-Path $root 'install-synta-low-load.ps1') -Revision $Revision
# Wait for dependencies, not for an arbitrary fixed delay. An unavailable share is explicit.
$python = Join-Path $env:USERPROFILE 'ai-owned-compute\venv\Scripts\python.exe'
$deadline = (Get-Date).AddMinutes(3)
$readyFile = Join-Path $env:USERPROFILE 'ai-owned-compute\runtime-ready.json'
while ($true) {
    if ((Test-Path -LiteralPath $python) -and (Test-Path -LiteralPath $readyFile)) {
        try { $ready = Get-Content -Raw -LiteralPath $readyFile | ConvertFrom-Json } catch { $ready = $null }
        if ($ready -and $ready.revision -eq $Revision) { break }
    }
    if ((Get-Date) -gt $deadline) { throw 'CPU runtime not ready. See cpu-worker.stderr.log and authenticated mailbox.' }
    Start-Sleep -Seconds 3
}
& (Join-Path $root 'install-rtx3090-autostart.ps1') -Revision $Revision -Adaptive
Write-Host 'Adaptive Synta installed: browser allowed, one RTX request, up to 65% active-time target, 2 idle CPU threads / 4 GiB.'
Write-Host 'CPU/RAM pressure, games, GPU heat, free VRAM and measured board watts can interrupt only Synta work.'
Write-Host 'Sampling does not cap instantaneous board power or guarantee hardware stability. Global GPU limits and clocks unchanged.'
