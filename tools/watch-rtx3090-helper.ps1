# Read-only helper and hardware summary. Ctrl+C closes this view, not Ollama.
param([int]$Seconds = 3, [string]$Endpoint = 'http://192.168.0.61:11435')
$ErrorActionPreference = 'Continue'
$dir = Join-Path $env:USERPROFILE 'ai-v100-helper'
while ($true) {
    Clear-Host
    Write-Host ('RTX HELPER | ' + (Get-Date -Format 'HH:mm:ss'))
    & nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.free,power.draw,temperature.gpu --format=csv
    $os = Get-CimInstance Win32_OperatingSystem
    $cpu = Get-CimInstance Win32_Processor
    Write-Host ('CPU total: {0:N0}% | RAM available: {1:N1} GiB' -f (($cpu | Measure-Object LoadPercentage -Average).Average), ($os.FreePhysicalMemory / 1MB))
    Write-Host 'GPU values are for the entire board. Helper pacing is not a hard utilization/power limit.'
    $breakerPath = Join-Path $dir 'watchdog-state.json'
    if (Test-Path -LiteralPath $breakerPath) {
        try {
            $breaker = Get-Content -Raw -LiteralPath $breakerPath | ConvertFrom-Json
            if ($breaker.circuit_open) {
                Write-Host ('CIRCUIT OPEN | failures={0} | {1}' -f $breaker.consecutive_failures, $breaker.last_error) -ForegroundColor Red
                Write-Host ("After reviewing the cause, reset with: Set-Content -NoNewline -Path '{0}' -Value RESET_AFTER_REVIEW" -f (Join-Path $dir 'watchdog-reset.txt')) -ForegroundColor Yellow
            } elseif ($breaker.retry_after_utc) {
                Write-Host ('Retry backoff until {0} | failures={1}/3' -f $breaker.retry_after_utc, $breaker.consecutive_failures) -ForegroundColor Yellow
            }
        } catch { Write-Host 'WATCHDOG STATE INVALID: helper is fail-closed; inspect watchdog-state.json.' -ForegroundColor Red }
    }
    try {
        $loaded = Invoke-RestMethod -Uri ($Endpoint.TrimEnd('/') + '/api/ps') -TimeoutSec 2
        $loaded.models | Select-Object name,context_length,@{Name='VRAM_GiB';Expression={[math]::Round($_.size_vram/1GB,2)}} | Format-Table -AutoSize
    } catch { Write-Host 'Helper unavailable or paused by game guard.' }
    try {
        $guard = Invoke-RestMethod -Uri ($Endpoint.TrimEnd('/') + '/api/synta-resources') -TimeoutSec 3
        $age = if ($null -ne $guard.heartbeat_age_seconds) { [double]$guard.heartbeat_age_seconds } else { [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - $guard.updated }
        if ($age -lt 0 -or $age -gt 10 -or $guard.stale) {
            Write-Host 'RESOURCE MONITOR STALE: admission disabled; inspect managed-helper.json.' -ForegroundColor Red
        } else {
            Write-Host 'FRESH ADMISSION (browser allowed):'
            $guard | ConvertTo-Json -Depth 6
        }
    } catch { Write-Host 'RESOURCE MONITOR UNAVAILABLE' -ForegroundColor Red }
    $log = Join-Path $dir 'logs\server.stderr.log'
    if (Test-Path -LiteralPath $log) { Get-Content -LiteralPath $log -Tail 8 }
    $cpuLog = Join-Path $dir 'logs\cpu-native.stdout.log'
    if (Test-Path -LiteralPath $cpuLog) {
        Write-Host 'OWNED CPU WORKER:'
        Get-Content -LiteralPath $cpuLog -Tail 3
    }
    $cpuError = Join-Path $dir 'logs\cpu-native.stderr.log'
    if ((Test-Path -LiteralPath $cpuError) -and (Get-Item -LiteralPath $cpuError).Length -gt 0) {
        Get-Content -LiteralPath $cpuError -Tail 2
    }
    Start-Sleep -Seconds ([math]::Max(2,$Seconds))
}
