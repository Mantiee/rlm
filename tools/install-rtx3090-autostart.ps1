param(
    [Parameter(Mandatory=$true)][string]$Revision
)
# Current-user logon startup, with no saved password and no global resource changes.
$ErrorActionPreference = 'Stop'
$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run PowerShell as administrator to register the elevated helper task.'
}
if ($Revision -notmatch '^[0-9a-f]{40}$') { throw 'Use an exact source revision.' }
$root = Join-Path $env:USERPROFILE 'ai-v100-helper'
New-Item -ItemType Directory -Force -Path $root | Out-Null
foreach ($name in @('watch-rtx3090-helper.ps1', 'install-owned-compute-autostart.ps1')) {
    $target = Join-Path $root $name
    Invoke-WebRequest -UseBasicParsing "https://raw.githubusercontent.com/Mantiee/rlm/$Revision/tools/$name" -OutFile ($target + '.new')
    Move-Item -Force -LiteralPath ($target + '.new') -Destination $target
}
& (Join-Path $root 'install-owned-compute-autostart.ps1') -Revision $Revision
$launcher = Join-Path $root 'start-rtx3090-helper.ps1'
Invoke-WebRequest -UseBasicParsing "https://raw.githubusercontent.com/Mantiee/rlm/$Revision/tools/start-rtx3090-helper.ps1" -OutFile ($launcher + '.new')
Move-Item -Force -LiteralPath ($launcher + '.new') -Destination $launcher
$watchdog = Join-Path $root 'autostart-watchdog.ps1'
@'
$ErrorActionPreference = 'Stop'
$root = Join-Path $env:USERPROFILE 'ai-v100-helper'
New-Item -ItemType Directory -Force -Path (Join-Path $root 'logs') | Out-Null
while ($true) {
    try {
        $ownedGuardian = $false
        $ownerFile = Join-Path $root 'guardian-owner.json'
        if (Test-Path -LiteralPath $ownerFile) {
            $owner = Get-Content -Raw -LiteralPath $ownerFile | ConvertFrom-Json
            $process = Get-Process -Id $owner.pid -ErrorAction SilentlyContinue
            $ownedGuardian = $process -and $process.StartTime.ToUniversalTime().ToString('o') -eq $owner.started -and $process.Path -eq $owner.path
        }
        if (-not $ownedGuardian -and -not (Get-NetTCPConnection -State Listen -LocalPort 11435 -ErrorAction SilentlyContinue)) {
            & (Join-Path $root 'start-rtx3090-helper.ps1') -ActiveTimePercent 50 -BatchTokens 16 -Context 32768 *>> (Join-Path $root 'logs/autostart.log')
        }
    } catch {
        "$(Get-Date -Format o) $($_.Exception.Message)" | Add-Content (Join-Path $root 'logs/autostart.log')
    }
    Start-Sleep -Seconds 30
}
'@ | Set-Content -Encoding UTF8 -LiteralPath $watchdog
$exe = Join-Path $PSHOME 'powershell.exe'
$action = New-ScheduledTaskAction -Execute $exe -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$watchdog`""
$user = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
$identity = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable -RestartCount 20 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName 'Synta-RTX3090-Helper' -Action $action -Trigger $trigger -Principal $identity -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName 'Synta-RTX3090-Helper'
# Separate visible read-only monitor. Closing its console does not stop workers.
$monitor = Join-Path $root 'watch-rtx3090-helper.ps1'
$monitorAction = New-ScheduledTaskAction -Execute $exe -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$monitor`""
$monitorPrincipal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName 'Synta-Helper-Monitor' -Action $monitorAction -Trigger $trigger -Principal $monitorPrincipal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName 'Synta-Helper-Monitor'

Write-Host 'Installed: Synta-RTX3090-Helper. Starts after this user signs in; existing guardian/game guard preserved.'
Write-Host "Log: $root\logs\autostart.log"
Write-Host 'Startup pacing target: 50%; context 32768; batch 16. Debian research pacing is a separate setting. No hard board power/temperature limit.'
