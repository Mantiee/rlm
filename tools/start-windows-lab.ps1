# One operator console for the isolated RTX helper, optional owned CPU worker and logs.
param(
    [Parameter(Mandatory=$true)][string]$Revision,
    [string]$Mailbox = '\\192.168.0.68\dane\tests\v100-owned-compute',
    [switch]$NoWatch
)
$ErrorActionPreference = 'Stop'
if ($Revision -notmatch '^[0-9a-f]{40}$') { throw 'Use an exact published revision.' }
$root = Join-Path $env:USERPROFILE 'ai-v100-helper'
New-Item -ItemType Directory -Force -Path $root | Out-Null
$source = Join-Path $root $Revision
New-Item -ItemType Directory -Force -Path $source | Out-Null
$logs = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
foreach ($name in @('start-rtx3090-helper.ps1','watch-rtx3090-helper.ps1','start-owned-compute-worker.ps1')) {
    Invoke-WebRequest -UseBasicParsing "https://raw.githubusercontent.com/Mantiee/rlm/$Revision/tools/$name" -OutFile (Join-Path $source $name) -ErrorAction Stop
}
$helper = Join-Path $source 'start-rtx3090-helper.ps1'
& powershell -NoProfile -ExecutionPolicy Bypass -File $helper -Stop
if ($LASTEXITCODE -ne 0) { throw 'Owned RTX helper did not stop; no replacement started.' }
& powershell -NoProfile -ExecutionPolicy Bypass -File $helper -Context 32768 -BatchTokens 16 -ActiveTimePercent 15 -DebugLogs
if ($LASTEXITCODE -ne 0) { throw 'RTX helper acceptance failed; inspect the displayed diagnostic.' }

# A shared-folder mapping needs existing operator SMB access. No credentials,
# new share, drive mapping or firewall rule is created for the CPU worker.
$waiter = Join-Path $source 'wait-owned-compute.ps1'
@'
param([string]$Revision, [string]$Mailbox, [string]$Source)
$ErrorActionPreference = 'Stop'
$deadline = (Get-Date).AddHours(2)
while (-not (Test-Path -LiteralPath (Join-Path $Mailbox 'jobs'))) {
    Write-Output ('Owned CPU worker waiting for authenticated mailbox: ' + $Mailbox)
    if ((Get-Date) -ge $deadline) { throw 'Mailbox unavailable. Configure the correct Samba share; no credentials changed.' }
    Start-Sleep -Seconds 30
}
$launcher = Join-Path $Source 'start-owned-compute-worker.ps1'
& powershell -NoProfile -ExecutionPolicy Bypass -File $launcher -Revision $Revision -Mailbox $Mailbox
if ($LASTEXITCODE -ne 0) { throw 'Owned CPU worker failed; inspect cpu-worker.stderr.log.' }
'@ | Set-Content -LiteralPath $waiter -Encoding UTF8
$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and
    ($_.CommandLine.Contains('wait-owned-compute.ps1') -or $_.CommandLine.Contains('compute_worker.py')) -and
    ($_.CommandLine.Contains($root) -or $_.CommandLine.Contains((Join-Path $env:USERPROFILE 'ai-owned-compute')))
}
if (-not $existing) {
    $worker = Start-Process powershell -WindowStyle Hidden -PassThru -ArgumentList @(
        '-NoProfile','-ExecutionPolicy','Bypass','-File',('"' + $waiter + '"'),
        '-Revision',$Revision,'-Mailbox',('"' + $Mailbox + '"'),'-Source',('"' + $source + '"')
    ) -RedirectStandardOutput (Join-Path $logs 'cpu-worker.stdout.log') -RedirectStandardError (Join-Path $logs 'cpu-worker.stderr.log')
    Write-Host ('Owned CPU worker launcher PID: ' + $worker.Id)
} else {
    Write-Host 'Existing owned CPU worker retained; no duplicate started.'
}
Write-Host 'Helper request active-time target: 15%. This is not a hard GPU peak or power cap.'
Write-Host 'League game guard pauses the isolated helper; previous black-screen cause remains unresolved.'
if (-not $NoWatch) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $source 'watch-rtx3090-helper.ps1')
}
