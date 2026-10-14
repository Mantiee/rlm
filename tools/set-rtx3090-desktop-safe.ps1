# Safe defaults for the user's daily Windows desktop.
# GPU use by Synta remains disabled; this does not limit games or other applications.
param([switch]$InspectOnly)
$ErrorActionPreference = 'Stop'
$root = Join-Path $env:USERPROFILE 'ai-v100-helper'
$taskName = 'Synta-RTX3090-Helper'
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($InspectOnly) {
    Write-Host "Synta RTX task: $(if ($task) { $task.State } else { 'not installed' })"
    $cpu = Get-ScheduledTask -TaskName 'Synta-Owned-CPU-Worker' -ErrorAction SilentlyContinue
    Write-Host "Synta CPU task: $(if ($cpu) { $cpu.State } else { 'not installed' })"
    return
}
$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Administrator PowerShell is required.'
}
if ($task) {
    Disable-ScheduledTask -TaskName $taskName | Out-Null
    Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
}
$launcher = Join-Path $root 'start-rtx3090-helper.ps1'
if (Test-Path -LiteralPath $launcher) {
    & $launcher -Stop
}
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($task -and $task.State -ne 'Disabled') { throw 'Synta GPU autostart is not disabled.' }
Write-Host 'Desktop Safe: Synta RTX 3090 helper disabled; independent CPU worker unchanged.'
Write-Host 'No global power limits, NVIDIA settings, drivers, or non-Synta applications changed.'
