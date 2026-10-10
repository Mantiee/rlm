# Current-user CPU worker recovery. Uses existing authenticated SMB access only.
param(
    [Parameter(Mandatory=$true)][string]$Revision,
    [string]$Mailbox = '\\192.168.0.68\dane\tests\v100-owned-compute',
    [string]$WorkerName = 'windows-cpu'
)
$ErrorActionPreference = 'Stop'
if ($Revision -notmatch '^[0-9a-f]{40}$') { throw 'Use an exact source revision.' }
if ($WorkerName -notmatch '^[a-zA-Z0-9_-]{1,48}$') { throw 'Invalid worker name.' }
if ($Mailbox.Contains('"') -or $Mailbox.Contains("`n") -or $Mailbox.Contains("`r")) { throw 'Invalid mailbox path.' }
$root = Join-Path $env:USERPROFILE 'ai-owned-compute'
New-Item -ItemType Directory -Force -Path $root | Out-Null
$config = Join-Path $root 'autostart-settings.json'
if (-not $PSBoundParameters.ContainsKey('Mailbox')) {
    foreach ($candidate in @($config, (Join-Path $root 'worker-settings.json'))) {
        if (Test-Path -LiteralPath $candidate) {
            $previous = Get-Content -Raw -LiteralPath $candidate | ConvertFrom-Json
            $Mailbox = $previous.mailbox
            break
        }
    }
    # Recover the original mailbox from a running owned worker on older installs.
    if (-not (Test-Path -LiteralPath $config) -and -not (Test-Path -LiteralPath (Join-Path $root 'worker-settings.json'))) {
        $running = Get-CimInstance Win32_Process | Where-Object {
            $_.CommandLine -and $_.CommandLine.Contains($root) -and $_.CommandLine.Contains('compute_worker.py')
        } | Select-Object -First 1
        if ($running -and $running.CommandLine -match '--mailbox\s+(?:"([^"\r\n]+)"|(\S+))') {
            $Mailbox = if ($Matches[1]) { $Matches[1] } else { $Matches[2] }
        }
    }
}
if (-not $Mailbox -or $Mailbox.Contains('"') -or $Mailbox.Contains("`n") -or $Mailbox.Contains("`r")) { throw 'Invalid retained mailbox path.' }
$source = Join-Path $root ('launchers\' + $Revision)
New-Item -ItemType Directory -Force -Path $source | Out-Null
$launcher = Join-Path $source 'start-owned-compute-worker.ps1'
Invoke-WebRequest -UseBasicParsing "https://raw.githubusercontent.com/Mantiee/rlm/$Revision/tools/start-owned-compute-worker.ps1" -OutFile ($launcher + '.new')
Move-Item -Force -LiteralPath ($launcher + '.new') -Destination $launcher
@{revision=$Revision; mailbox=$Mailbox; worker=$WorkerName; launcher=$launcher} | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath $config
$watchdog = Join-Path $root 'cpu-autostart-watchdog.ps1'
@'
$ErrorActionPreference = 'Stop'
$root = Join-Path $env:USERPROFILE 'ai-owned-compute'
$logs = Join-Path $env:USERPROFILE 'ai-v100-helper\logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
while ($true) {
    try {
        $config = Get-Content -Raw -LiteralPath (Join-Path $root 'autostart-settings.json') | ConvertFrom-Json
        if (-not (Test-Path -LiteralPath (Join-Path $config.mailbox 'jobs'))) {
            throw 'Authenticated compute mailbox unavailable; no credentials or share changed.'
        }
        $existing = Get-CimInstance Win32_Process | Where-Object {
            $_.CommandLine -and $_.CommandLine.Contains($root) -and $_.CommandLine.Contains('compute_worker.py')
        }
        if (-not $existing) {
            $arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$($config.launcher)`" -Revision $($config.revision) -Mailbox `"$($config.mailbox)`" -WorkerName $($config.worker)"
            $launcher = Start-Process -FilePath (Join-Path $PSHOME 'powershell.exe') -ArgumentList $arguments -PassThru -Wait -WindowStyle Hidden `
                -RedirectStandardOutput (Join-Path $logs 'cpu-worker.stdout.log') `
                -RedirectStandardError (Join-Path $logs 'cpu-worker.stderr.log')
            if ($launcher.ExitCode -ne 0) { throw "CPU launcher exited $($launcher.ExitCode). Full native traceback: cpu-native.stderr.log." }
        }
    } catch {
        "$(Get-Date -Format o) $($_.Exception.Message)" | Add-Content -LiteralPath (Join-Path $logs 'cpu-autostart.log')
        # A failed launch is not an idle/ready worker. Show the diagnostic in the
        # same visible stream; do not repeat only the startup banner forever.
        $lastError = Get-Content -LiteralPath (Join-Path $logs 'cpu-native.stderr.log') -Tail 100 -ErrorAction SilentlyContinue
        "$(Get-Date -Format o) BLOCKED: $($_.Exception.Message)`n$($lastError -join "`n")" | Add-Content -LiteralPath (Join-Path $logs 'cpu-worker.stdout.log')
        Start-Sleep -Seconds 90
    }
    Start-Sleep -Seconds 30
}
'@ | Set-Content -Encoding UTF8 -LiteralPath $watchdog
$exe = Join-Path $PSHOME 'powershell.exe'
$action = New-ScheduledTaskAction -Execute $exe -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$watchdog`""
$user = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable -RestartCount 20 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName 'Synta-Owned-CPU-Worker' -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName 'Synta-Owned-CPU-Worker'
Write-Host 'Installed: Synta-Owned-CPU-Worker. Starts at this user logon; existing workers/jobs retained until they exit.'
Write-Host "Mailbox: $Mailbox. CPU budget: 2 threads, 4 GiB child RAM. Logs: $env:USERPROFILE\ai-v100-helper\logs"
