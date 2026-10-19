param(
    [Parameter(Mandatory=$true)][string]$Revision,
    [switch]$Adaptive
)
# Current-user logon startup, with no saved password and no global resource changes.
$ErrorActionPreference = 'Stop'
$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run PowerShell as administrator to register the elevated helper task.'
}
if ($Revision -notmatch '^[0-9a-f]{40}$') { throw 'Use an exact source revision.' }
$root = Join-Path $env:USERPROFILE 'ai-v100-helper'
$manualStopPath = Join-Path $root 'helper-manual-stop.flag'
if (Test-Path -LiteralPath $manualStopPath) {
    throw 'RTX helper has an operator manual-stop flag. Installer refused to clear it or start GPU work; resume only through a separate reviewed operator action.'
}
New-Item -ItemType Directory -Force -Path $root | Out-Null

foreach ($name in @(
    'watch-rtx3090-helper.ps1',
    'install-owned-compute-autostart.ps1',
    'start-rtx3090-helper.ps1',
    'autostart-watchdog.ps1',
    'resume-rtx3090-helper.ps1'
)) {
    $target = Join-Path $root $name
    Invoke-WebRequest -UseBasicParsing "https://raw.githubusercontent.com/Mantiee/rlm/$Revision/tools/$name" -OutFile ($target + '.new')
    Move-Item -Force -LiteralPath ($target + '.new') -Destination $target
}
& (Join-Path $root 'install-owned-compute-autostart.ps1') -Revision $Revision
if ($Adaptive) {
    Invoke-WebRequest -UseBasicParsing "https://raw.githubusercontent.com/Mantiee/rlm/$Revision/rlm/v100/windows_resource_proxy.py" -OutFile (Join-Path $root 'windows_resource_proxy.py')
}
@{adaptive=[bool]$Adaptive} | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $root 'adaptive-settings.json')

$watchdog = Join-Path $root 'autostart-watchdog.ps1'
$exe = Join-Path $PSHOME 'powershell.exe'
$action = New-ScheduledTaskAction -Execute $exe -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$watchdog`""
$user = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
$identity = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Highest
# The watchdog owns persistent backoff/circuit logic; Task Scheduler must not
# create a second, unbounded retry loop around it.
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable -RestartCount 0 -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName 'Synta-RTX3090-Helper' -Action $action -Trigger $trigger -Principal $identity -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName 'Synta-RTX3090-Helper'
# Separate hidden read-only monitor; it cannot take focus from the user's session.
$monitor = Join-Path $root 'watch-rtx3090-helper.ps1'
$monitorAction = New-ScheduledTaskAction -Execute $exe -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$monitor`""
$monitorPrincipal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName 'Synta-Helper-Monitor' -Action $monitorAction -Trigger $trigger -Principal $monitorPrincipal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName 'Synta-Helper-Monitor'

Write-Host 'Installed: Synta-RTX3090-Helper. Starts after this user signs in; persistent circuit breaker and League pause enabled.'
Write-Host "Log: $root\logs\autostart.log"
Write-Host 'Startup pacing target: 65%; context 32768; batch up to 64. Sampling is not a hard board power/temperature limit.'
