param([Parameter(Mandatory=$true)][string]$Revision)
$ErrorActionPreference = 'Stop'
if ($Revision -notmatch '^[0-9a-f]{40}$') { throw 'Exact GitHub revision required.' }
$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Elevated PowerShell required for the scoped peer firewall rule and updater task.' }
$root = Join-Path $env:USERPROFILE 'ai-v100-helper'
$python = Join-Path $env:USERPROFILE 'ai-owned-compute\venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Existing owned Python runtime required before updater registration.' }
$dir = Join-Path $root 'tools\managed-bootstrap'
New-Item -ItemType Directory -Force -Path $dir | Out-Null
foreach ($name in @('managed_bootstrap','inference_benchmark')) {
    Invoke-WebRequest -UseBasicParsing "https://raw.githubusercontent.com/Mantiee/rlm/$Revision/rlm/v100/$name.py" -OutFile (Join-Path $dir ($name + '.py'))
}
$runner = Join-Path $dir 'managed_bootstrap.py'
& $python -m py_compile $runner
if ($LASTEXITCODE -ne 0) { throw 'Bootstrap Python failed syntax validation; no updater task started.' }
$action = New-ScheduledTaskAction -Execute $python -Argument "-u `"$runner`" --root `"$root`""
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Hours 6)
$user = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$identity = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 5) -StartWhenAvailable
Register-ScheduledTask -TaskName 'Synta-Managed-Update' -Action $action -Trigger $trigger -Principal $identity -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName 'Synta-Managed-Update'
Write-Host 'Automatic current-user updater registered. Inspect ai-v100-helper\logs\managed-update.json. No ChatGPT remote shell is created.'
