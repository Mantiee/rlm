param([Parameter(Mandatory=$true)][string]$Revision,[Parameter(Mandatory=$true)][string]$Bundle)
$ErrorActionPreference = 'Stop'
if ($Revision -notmatch '^[0-9a-f]{40}$') { throw 'Immutable revision required.' }
$root = Join-Path $env:USERPROFILE 'ai-v100-helper'
$receipt = Join-Path $root 'logs\managed-helper.json'
$manifest = Get-Content -Raw -LiteralPath (Join-Path $Bundle 'release.json') | ConvertFrom-Json
$expected = $manifest.files.'rlm/v100/windows_resource_proxy.py'
if ($expected -notmatch '^[0-9a-f]{64}$') { throw 'Release inventory omits the proxy; existing helper retained.' }
# Validate every declared artifact before maintenance or stopping any owned process.
foreach ($entry in $manifest.files.PSObject.Properties) {
    $artifact = Join-Path $Bundle $entry.Name
    if (-not (Test-Path -LiteralPath $artifact -PathType Leaf)) { throw "Incomplete release: $($entry.Name); existing helper retained." }
    if ((Get-FileHash -LiteralPath $artifact -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value) {
        throw "Release checksum mismatch: $($entry.Name); existing helper retained."
    }
}
& (Join-Path $Bundle 'tools\install-synta-background-tasks.ps1')
$launcher = Join-Path $root 'start-rtx3090-helper.ps1'
$launcherCurrent = (Test-Path -LiteralPath $launcher -PathType Leaf) -and
    ((Get-FileHash -LiteralPath $launcher -Algorithm SHA256).Hash.ToLowerInvariant() -eq $manifest.files.'tools/start-rtx3090-helper.ps1')
$origin = 'http://192.168.0.61:11435'
$now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
$guard = $null
$age = 1e99
try { $guard = Invoke-RestMethod "$origin/api/synta-resources" -TimeoutSec 5 } catch { Write-Warning $_.Exception.Message }
$age = if ($guard -and $null -ne $guard.heartbeat_age_seconds) { [double]$guard.heartbeat_age_seconds } elseif ($guard) { $now - $guard.updated } else { 1e99 }
if ($launcherCurrent -and $guard -and $guard.proxy_code_sha256 -eq $expected -and $guard.throughput_enabled -and $age -ge 0 -and $age -lt 10) {
    @{stage='verified';revision=$Revision;updated=$now;resources=$guard} | ConvertTo-Json -Depth 7 | Set-Content -Encoding UTF8 $receipt
    return
}
# Bounded atomic maintenance admission reaches an idle gap even under repeated work.
. (Join-Path $Bundle 'tools\windows-maintenance.ps1')
if ($guard -and $guard.proxy_code_sha256) {
    try { Wait-SyntaMaintenance -Endpoint "$origin/api/synta-maintenance" -WaitSeconds 60 | Out-Null }
    catch {
        @{stage='deferred-drain';error=$_.Exception.Message;revision=$Revision;updated=$now} | ConvertTo-Json | Set-Content -Encoding UTF8 $receipt
        return
    }
}
$task = Get-ScheduledTask -TaskName 'Synta-RTX3090-Helper' -ErrorAction SilentlyContinue
if ($task) {
    if ($task.Actions.Count -ne 1 -or (-not $task.Actions[0].Arguments.Contains((Join-Path $root 'autostart-watchdog.ps1')) -and -not $task.Actions[0].Arguments.Contains((Join-Path $root 'tools\background-tasks\Synta-RTX3090-Helper.vbs')))) {
        throw 'Helper task identity differs; no process stopped.'
    }
    Stop-ScheduledTask -TaskName 'Synta-RTX3090-Helper'
}
$runtime = [IO.Path]::GetFullPath((Join-Path $root 'runtime\ollama-0.40.0')).TrimEnd('\') + '\'
$proxy = Join-Path $root 'windows_resource_proxy.py'
$worker = Join-Path $root 'serve-worker.ps1'
$members = @(Get-CimInstance Win32_Process | Where-Object {
    ($_.ExecutablePath -and $_.ExecutablePath.StartsWith($runtime,[StringComparison]::OrdinalIgnoreCase)) -or
    ($_.Name -ieq 'python.exe' -and $_.CommandLine -and $_.CommandLine.Contains($proxy)) -or
    ($_.Name -ieq 'powershell.exe' -and $_.CommandLine -and $_.CommandLine.Contains($worker))
})
foreach ($member in $members) {
    $process = Get-Process -Id $member.ProcessId -ErrorAction SilentlyContinue
    if (-not $process) { continue }
    if ($process.Path -ine $member.ExecutablePath -or [Math]::Abs(($process.StartTime.ToUniversalTime() - $member.CreationDate.ToUniversalTime()).TotalMilliseconds) -ge 1) {
        throw 'Owned process identity changed; reused PID not stopped.'
    }
    Stop-Process -InputObject $process -Force
    if (-not $process.WaitForExit(10000)) { throw 'Owned service did not stop.' }
}
Copy-Item -Force -LiteralPath (Join-Path $Bundle 'rlm\v100\windows_resource_proxy.py') -Destination $proxy
@{enabled=$true;revision=$Revision;scope='More throughput under measured headroom; games and pressure still pause owned work'} |
    ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $root 'throughput-settings.json')
& (Join-Path $Bundle 'tools\install-rtx3090-autostart.ps1') -Revision $Revision -Adaptive
& (Join-Path $Bundle 'tools\install-synta-background-tasks.ps1')
$deadline = (Get-Date).AddMinutes(3)
do {
    try {
        $guard = Invoke-RestMethod "$origin/api/synta-resources" -TimeoutSec 5
        $now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
        $age = if ($null -ne $guard.heartbeat_age_seconds) { [double]$guard.heartbeat_age_seconds } else { $now - $guard.updated }
        if ($guard.proxy_code_sha256 -eq $expected -and $guard.throughput_enabled -and $age -ge 0 -and $age -lt 10) {
            @{stage='verified';revision=$Revision;updated=$now;resources=$guard} | ConvertTo-Json -Depth 7 | Set-Content -Encoding UTF8 $receipt
            Write-Host 'Owned Windows proxy refreshed; fresh heartbeat verified. CPU worker and other applications retained.'
            return
        }
    } catch { Write-Warning $_.Exception.Message }
    Start-Sleep -Seconds 3
} while ((Get-Date) -lt $deadline)
throw 'Windows proxy did not return a fresh matching heartbeat; inspect autostart.log and resource-proxy.stderr.log.'

