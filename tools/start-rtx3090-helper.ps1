param(
    [string]$WindowsIp = '192.168.0.61',
    [string]$DebianIp = '192.168.0.68',
    [int]$Port = 11435,
    [int]$Context = 32768,
    [switch]$Stop
)

# Windows PowerShell 5.1 compatible. Only the new worker receives environment
# changes. No setx, driver updates, existing Ollama shutdown or paid/cloud calls.
$ErrorActionPreference = 'Stop'
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run PowerShell as administrator: the scoped LAN firewall rule requires it.'
}
foreach ($address in @($WindowsIp, $DebianIp)) {
    if ($address -notmatch '^(192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})$') {
        throw 'Use private IPv4 LAN addresses.'
    }
    $null = [Net.IPAddress]::Parse($address)
}
if ($Port -ne 11435 -or $Context -notin @(8192, 16384, 32768)) {
    throw 'Use the isolated port 11435 and context 8192, 16384 or 32768.'
}
$Root = Join-Path $env:USERPROFILE 'ai-v100-helper'
$Ollama = (Get-Command ollama -ErrorAction Stop).Source
$Model = 'qwen3.5:9b-q8_0'
$Origin = "http://${WindowsIp}:$Port"
$Rule = "v100-helper-$Port-from-$DebianIp"
$BlockRule = "$Rule-block-other-addresses"
$OwnerPath = Join-Path $Root 'server-owner.json'
if ($Stop) {
    if (Test-Path $OwnerPath) {
        $Owner = Get-Content -Raw $OwnerPath | ConvertFrom-Json
        $Owned = Get-Process -Id $Owner.pid -ErrorAction SilentlyContinue
        if ($Owned) {
            if ($Owned.StartTime.ToUniversalTime().ToString('o') -ne $Owner.started -or $Owned.Path -ne $Ollama) {
                throw 'Helper process identity changed; no process stopped.'
            }
            Stop-Process -Id $Owned.Id
        }
    }
    Remove-NetFirewallRule -Name $Rule -ErrorAction SilentlyContinue
    Remove-NetFirewallRule -Name $BlockRule -ErrorAction SilentlyContinue
    Write-Host 'Isolated helper stopped. Model files retained.'
    return
}
if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) {
    throw "Port $Port is already in use. Existing process left intact."
}
$GpuRows = @(& nvidia-smi --query-gpu=name,memory.free --format=csv,noheader,nounits)
if ($LASTEXITCODE -ne 0 -or $GpuRows.Count -ne 1 -or $GpuRows[0] -notmatch '3090') {
    throw 'Expected one RTX 3090; no process started.'
}
$FreeMiB = [int](($GpuRows[0] -split ',')[-1].Trim())
if ($FreeMiB -lt 14336) { throw 'Need at least 14 GiB free before the helper smoke test.' }
$drive = Get-PSDrive -Name ([IO.Path]::GetPathRoot($Root).Substring(0, 1))
if ($drive.Free -lt 20GB) { throw 'Need 20 GiB free disk for the isolated model download.' }
New-Item -ItemType Directory -Path $Root -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $Root 'models') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $Root 'logs') -Force | Out-Null

$ExistingRule = Get-NetFirewallRule -Name $Rule -ErrorAction SilentlyContinue
if ($ExistingRule -or (Get-NetFirewallRule -Name $BlockRule -ErrorAction SilentlyContinue)) { throw "Helper firewall rule already exists; inspect it before restarting." }
# Explicit port-specific blocks also protect against a pre-existing broad
# application allow rule. Keep the Windows self-test address accessible.
function Ip-Number([string]$Address) {
    $bytes = [Net.IPAddress]::Parse($Address).GetAddressBytes()
    return [uint64]$bytes[0] * 16777216 + [uint64]$bytes[1] * 65536 + [uint64]$bytes[2] * 256 + [uint64]$bytes[3]
}
function Number-Ip([uint64]$Number) {
    return "$([int][Math]::Floor($Number / 16777216)).$([int][Math]::Floor(($Number % 16777216) / 65536)).$([int][Math]::Floor(($Number % 65536) / 256)).$($Number % 256)"
}
$Allowed = @( (Ip-Number $WindowsIp), (Ip-Number $DebianIp) ) | Sort-Object -Unique
$Ranges = @()
$First = [uint64]0
foreach ($number in $Allowed) {
    if ($number -gt $First) { $Ranges += "$(Number-Ip $First)-$(Number-Ip ($number - 1))" }
    $First = $number + 1
}
if ($First -lt 4294967295) { $Ranges += "$(Number-Ip $First)-255.255.255.255" }
$Worker = $null
Remove-Item -LiteralPath $OwnerPath -ErrorAction SilentlyContinue
try {
New-NetFirewallRule -Name $BlockRule -DisplayName $BlockRule -Direction Inbound -Action Block `
    -Protocol TCP -LocalPort $Port -LocalAddress $WindowsIp -RemoteAddress $Ranges -Profile Any | Out-Null
New-NetFirewallRule -Name $Rule -DisplayName $Rule -Direction Inbound -Action Allow `
    -Protocol TCP -LocalPort $Port -LocalAddress $WindowsIp -RemoteAddress $DebianIp `
    -Program $Ollama -Profile Any | Out-Null

$WorkerScript = Join-Path $Root 'serve-worker.ps1'
@'
param([string]$OllamaPath, [string]$HelperRoot)
$ErrorActionPreference = 'Stop'
$server = Start-Process -FilePath $OllamaPath -ArgumentList 'serve' -PassThru `
    -RedirectStandardOutput (Join-Path $HelperRoot 'logs/server.stdout.log') `
    -RedirectStandardError (Join-Path $HelperRoot 'logs/server.stderr.log')
@{ pid = $server.Id; started = $server.StartTime.ToUniversalTime().ToString('o'); path = $OllamaPath } |
    ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $HelperRoot 'server-owner.json')
$server.WaitForExit()
'@ | Set-Content -Encoding UTF8 $WorkerScript

$Info = New-Object Diagnostics.ProcessStartInfo
$Info.FileName = Join-Path $PSHOME 'powershell.exe'
$Info.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$WorkerScript`" -OllamaPath `"$Ollama`" -HelperRoot `"$Root`""
$Info.UseShellExecute = $false
$Info.CreateNoWindow = $true
$Settings = @{
    OLLAMA_HOST = "${WindowsIp}:$Port"
    OLLAMA_MODELS = (Join-Path $Root 'models')
    OLLAMA_CONTEXT_LENGTH = "$Context"
    OLLAMA_NUM_PARALLEL = '1'
    OLLAMA_MAX_LOADED_MODELS = '1'
    OLLAMA_MAX_QUEUE = '2'
    OLLAMA_FLASH_ATTENTION = '1'
    OLLAMA_KV_CACHE_TYPE = 'q8_0'
    OLLAMA_NO_CLOUD = '1'
    OLLAMA_DEBUG_LOG_REQUESTS = '0'
    OLLAMA_KEEP_ALIVE = '-1'
    # Reserve the other half as a scheduler hint, not a hard VRAM partition.
    OLLAMA_GPU_OVERHEAD = "$([int64][Math]::Max(0, ($FreeMiB - 12288)) * 1048576)"
}
foreach ($key in $Settings.Keys) { $Info.EnvironmentVariables[$key] = $Settings[$key] }
$Worker = [Diagnostics.Process]::Start($Info)
$Ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        if ($Worker.HasExited) { throw "Helper startup failed. Inspect $Root\logs." }
        try {
            $null = Invoke-RestMethod "$Origin/api/version" -TimeoutSec 2
            $Ready = $true
            break
        } catch { Start-Sleep -Seconds 1 }
    }
    if (-not $Ready) { throw 'Helper startup timed out.' }
    Write-Host "Downloading $Model (about 10 GB) into $Root\models ..."
    # The isolated CLI prints the normal download progress bar. Its environment
    # points at 11435, not the existing desktop Ollama instance.
    $PullInfo = New-Object Diagnostics.ProcessStartInfo
    $PullInfo.FileName = $Ollama
    $PullInfo.Arguments = "pull $Model"
    $PullInfo.UseShellExecute = $false
    foreach ($key in $Settings.Keys) { $PullInfo.EnvironmentVariables[$key] = $Settings[$key] }
    $PullProcess = [Diagnostics.Process]::Start($PullInfo)
    $PullProcess.WaitForExit()
    if ($PullProcess.ExitCode -ne 0) { throw 'Model download did not complete.' }
    $schema = @{ type = 'object'; properties = @{ result = @{ type = 'integer' } }; required = @('result'); additionalProperties = $false }
    $Started = Get-Date
    $Request = @{
        model = $Model; stream = $false; think = $false; keep_alive = -1; format = $schema
        options = @{ num_ctx = $Context; num_predict = 128; temperature = 0; seed = 42 }
        messages = @(@{ role = 'user'; content = 'What is 17 * (6013 - 5347) - 319 - 367 - 113 - 79? Return JSON with result.' })
    } | ConvertTo-Json -Depth 10
    $Response = Invoke-RestMethod "$Origin/api/chat" -Method Post -ContentType 'application/json' -Body $Request -TimeoutSec 600
    $Answer = $Response.message.content | ConvertFrom-Json
    if ($Answer.result -ne 10444 -or -not $Response.done -or $Response.done_reason -ne 'stop') { throw 'JSON/math smoke test failed.' }
    $Loaded = @((Invoke-RestMethod "$Origin/api/ps" -TimeoutSec 10).models | Where-Object { $_.name -eq $Model })
    if ($Loaded.Count -ne 1 -or $Loaded[0].context_length -ne $Context) { throw 'Unexpected loaded model or context.' }
    $Item = $Loaded[0]
    if ($Item.size_vram -le 0 -or $Item.size_vram -lt $Item.size -or $Item.size_vram -gt 12GB) {
        throw 'Helper is not fully on GPU within 12 GiB. Inspect logs; next try can use a smaller context.'
    }
    $Receipt = @{
        model = $Model; digest = $Item.digest; context = $Context; result = $Answer.result
        origin = $Origin; debian = $DebianIp; vram_gib = [Math]::Round($Item.size_vram / 1GB, 2)
        seconds = [Math]::Round(((Get-Date) - $Started).TotalSeconds, 2)
        generation_tps = [Math]::Round($Response.eval_count * 1e9 / [Math]::Max(1, $Response.eval_duration), 2)
        note = 'GPU memory measurement after load, not a hard or transient peak limit'
    }
    $Receipt | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $Root 'helper-ready.json')
    Write-Host 'RTX HELPER READY. Paste the following result back:'
    $Receipt | ConvertTo-Json
    & nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.free --format=csv
} catch {
    # Stop only the helper created by this invocation, never the original app.
    if (Test-Path $OwnerPath) {
        $Owner = Get-Content -Raw $OwnerPath | ConvertFrom-Json
        $Owned = Get-Process -Id $Owner.pid -ErrorAction SilentlyContinue
        if ($Owned -and $Owned.StartTime.ToUniversalTime().ToString('o') -eq $Owner.started -and $Owned.Path -eq $Ollama) {
            Stop-Process -Id $Owned.Id
        }
    }
    if ($Worker -and -not $Worker.HasExited) { $Worker.Kill() }
    Remove-NetFirewallRule -Name $Rule -ErrorAction SilentlyContinue
    Remove-NetFirewallRule -Name $BlockRule -ErrorAction SilentlyContinue
    throw
}
