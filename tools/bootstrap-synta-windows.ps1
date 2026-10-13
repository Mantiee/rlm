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
foreach ($name in @('managed_bootstrap','inference_benchmark','release_selector')) {
    Invoke-WebRequest -UseBasicParsing "https://raw.githubusercontent.com/Mantiee/rlm/$Revision/rlm/v100/$name.py" -OutFile (Join-Path $dir ($name + '.py'))
}
$runner = Join-Path $dir 'managed_bootstrap.py'
& $python -m py_compile $runner
if ($LASTEXITCODE -ne 0) { throw 'Bootstrap Python failed syntax validation; no updater task started.' }
$log = Join-Path $root 'logs\managed-update-run.log'
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
$wrapper = Join-Path $dir 'run-managed-update.ps1'
@'
$ErrorActionPreference = 'Continue'
$root = Join-Path $env:USERPROFILE 'ai-v100-helper'
$python = Join-Path $env:USERPROFILE 'ai-owned-compute\venv\Scripts\python.exe'
$runner = Join-Path $root 'tools\managed-bootstrap\managed_bootstrap.py'
$log = Join-Path $root 'logs\managed-update-run.log'
& $python -u $runner --root $root *>> $log
exit $LASTEXITCODE
'@ | Set-Content -Encoding UTF8 $wrapper
$action = New-ScheduledTaskAction -Execute (Join-Path $PSHOME 'powershell.exe') -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$wrapper`""
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 15)
$user = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$identity = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 5) -StartWhenAvailable
Register-ScheduledTask -TaskName 'Synta-Managed-Update' -Action $action -Trigger $trigger -Principal $identity -Settings $settings -Force | Out-Null
# Repair serving in this console, independently of a previous model downloader.
$bundle = Join-Path $root ("tools\managed-releases\" + $Revision)
# Windows PowerShell 5.1 turns native stderr into error records. Capture the
# complete traceback, restore the caller policy, then inspect the process exit.
$previousErrorPolicy = $ErrorActionPreference
try {
    $ErrorActionPreference = 'Continue'
    & $python -c 'import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); from managed_bootstrap import release,materialize; rev,manifest=release(sys.argv[3]); materialize(Path(sys.argv[2]),rev,manifest)' $dir $bundle $Revision *>> $log
    $acquisitionExit = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $previousErrorPolicy
}
if ($acquisitionExit -ne 0) {
    Get-Content -LiteralPath $log -Tail 30
    throw "Release acquisition failed before helper changes. Inspect $log"
}
& (Join-Path $bundle 'tools\refresh-synta-windows.ps1') -Revision $Revision -Bundle $bundle
& (Join-Path $bundle 'tools\install-synta-background-tasks.ps1')
Start-ScheduledTask -TaskName 'Synta-Managed-Update'
Write-Host "Updater stdout/stderr: $log"
if (Test-Path -LiteralPath $log) { Get-Content -LiteralPath $log -Tail 8 }
Write-Host 'Automatic current-user updater registered. Inspect ai-v100-helper\logs\managed-update.json. No ChatGPT remote shell is created.'

