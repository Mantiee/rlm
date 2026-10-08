# Read-only helper and hardware summary. Ctrl+C closes this view, not Ollama.
param([int]$Seconds = 3)
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
    try {
        $loaded = Invoke-RestMethod -Uri 'http://127.0.0.1:11435/api/ps' -TimeoutSec 2
        $loaded.models | Select-Object name,context_length,@{Name='VRAM_GiB';Expression={[math]::Round($_.size_vram/1GB,2)}} | Format-Table -AutoSize
    } catch { Write-Host 'Helper unavailable or paused by game guard.' }
    $log = Join-Path $dir 'logs\server.stderr.log'
    if (Test-Path -LiteralPath $log) { Get-Content -LiteralPath $log -Tail 8 }
    Start-Sleep -Seconds ([math]::Max(2,$Seconds))
}
