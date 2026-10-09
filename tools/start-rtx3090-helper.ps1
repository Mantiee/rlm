param(
    [string]$WindowsIp = '192.168.0.61',
    [string]$DebianIp = '192.168.0.68',
    [int]$Port = 11435,
    [int]$Context = 32768,
    [ValidateSet(16, 32, 64)]
    [int]$BatchTokens = 16,
    [ValidateRange(1,65)]
    [int]$ActiveTimePercent = 30,
    [switch]$DebugLogs,
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
if ($Port -ne 11435 -or $Context -notin @(8192, 16384, 32768, 65536, 131072)) {
    throw 'Use the isolated port 11435 and context 8192, 16384, 32768, 65536 or 131072.'
}
$Root = Join-Path $env:USERPROFILE 'ai-v100-helper'
$RuntimeVersion = '0.40.0'
$RuntimeRoot = Join-Path $Root "runtime\ollama-$RuntimeVersion"
$Ollama = Join-Path $RuntimeRoot 'ollama.exe'
$Model = 'qwen3.5:9b-q8_0'
$Origin = "http://${WindowsIp}:$Port"
$Rule = "v100-helper-$Port-from-$DebianIp"
$BlockRule = "$Rule-block-other-addresses"
$OwnerPath = Join-Path $Root 'server-owner.json'
$GuardianPath = Join-Path $Root 'guardian-owner.json'
function Stop-HelperProcesses([string]$RuntimePath) {
    # A dead parent can leave llama-server.exe alive. This directory is reserved
    # for this single helper, so include its orphan workers as well as Ollama.
    # Never select a process merely by name or by a possibly reused parent PID.
    $Prefix = [IO.Path]::GetFullPath($RuntimePath).TrimEnd('\') + '\'
    $Members = @(Get-CimInstance Win32_Process -Filter "Name='ollama.exe' OR Name='llama-server.exe'" |
        Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($Prefix, [StringComparison]::OrdinalIgnoreCase) } |
        Sort-Object CreationDate)
    foreach ($Member in $Members) {
        $Candidate = Get-Process -Id $Member.ProcessId -ErrorAction SilentlyContinue
        if (-not $Candidate) { continue }
        $SamePath = [string]::Equals($Candidate.Path, $Member.ExecutablePath, [StringComparison]::OrdinalIgnoreCase)
        $SameStart = [Math]::Abs(($Candidate.StartTime.ToUniversalTime() - $Member.CreationDate.ToUniversalTime()).TotalMilliseconds) -lt 1
        if (-not $SamePath -or -not $SameStart) { throw 'Helper process identity changed during cleanup; no reused PID stopped.' }
        Write-Host "Stopping isolated helper PID $($Candidate.Id): $($Candidate.Path)"
        Stop-Process -InputObject $Candidate -Force -ErrorAction Stop
        if (-not $Candidate.WaitForExit(5000)) { throw 'Helper process did not exit within 5 seconds.' }
    }
}
if ($Stop) {
    if (Test-Path $GuardianPath) {
        $Guardian = Get-Content -Raw $GuardianPath | ConvertFrom-Json
        $GuardProcess = Get-Process -Id $Guardian.pid -ErrorAction SilentlyContinue
        if ($GuardProcess) {
            if ($GuardProcess.StartTime.ToUniversalTime().ToString('o') -ne $Guardian.started -or $GuardProcess.Path -ne $Guardian.path) {
                throw 'Guardian identity changed; nothing stopped.'
            }
            Stop-Process -InputObject $GuardProcess -Force
        }
        Remove-Item -LiteralPath $GuardianPath -ErrorAction SilentlyContinue
    }
    if (Test-Path $OwnerPath) {
        $Owner = Get-Content -Raw $OwnerPath | ConvertFrom-Json
        $Owned = Get-Process -Id $Owner.pid -ErrorAction SilentlyContinue
        if ($Owned) {
            if ($Owned.StartTime.ToUniversalTime().ToString('o') -ne $Owner.started -or $Owned.Path -ne $Ollama) {
                throw 'Helper process identity changed; no process stopped.'
            }
        }
    }
    Stop-HelperProcesses $RuntimeRoot
    Remove-NetFirewallRule -Name $Rule -ErrorAction SilentlyContinue
    Remove-NetFirewallRule -Name $BlockRule -ErrorAction SilentlyContinue
    Write-Host 'Isolated helper stopped. Model files retained.'
    return
}
if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) {
    throw "Port $Port is already in use. Existing process left intact."
}
if (Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue) {
    throw 'League is running. Start the isolated helper after the game; guardian will pause it during later games.'
}
$GpuRows = @(& nvidia-smi --query-gpu=name,memory.free --format=csv,noheader,nounits)
if ($LASTEXITCODE -ne 0 -or $GpuRows.Count -ne 1 -or $GpuRows[0] -notmatch '3090') {
    throw 'Expected one RTX 3090; no process started.'
}
$FreeMiB = [int](($GpuRows[0] -split ',')[-1].Trim())
if ($FreeMiB -lt 14336) { throw 'Need at least 14 GiB free before the helper smoke test.' }
$drive = Get-PSDrive -Name ([IO.Path]::GetPathRoot($Root).Substring(0, 1))
$RequiredFree = 2GB
if (-not (Test-Path (Join-Path $RuntimeRoot 'runtime-ready.json'))) { $RequiredFree += 12GB }
if (-not (Test-Path (Join-Path $Root 'models\manifests\registry.ollama.ai\library\qwen3.5\9b-q8_0'))) { $RequiredFree += 14GB }
if ($drive.Free -lt $RequiredFree) { throw "Need $($RequiredFree / 1GB) GiB free disk for uncached downloads and working space." }
New-Item -ItemType Directory -Path $Root -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $Root 'models') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $Root 'logs') -Force | Out-Null

function Confirm-HelperRule([string]$Name, [string]$Action, [string[]]$Remote, [string]$Program) {
    $existing = @(Get-NetFirewallRule -Name $Name -ErrorAction SilentlyContinue)
    if ($existing.Count -eq 0) { return $false }
    if ($existing.Count -ne 1) { throw "Ambiguous helper firewall rule: $Name" }
    $item = $existing[0]
    $ports = $item | Get-NetFirewallPortFilter
    $addresses = $item | Get-NetFirewallAddressFilter
    $app = $item | Get-NetFirewallApplicationFilter
    $expected = ($Remote | Sort-Object) -join ','
    $actual = (@($addresses.RemoteAddress) | Sort-Object) -join ','
    if ($item.Direction -ne 'Inbound' -or $item.Action -ne $Action -or $item.Enabled -ne 'True' -or
        [string]$ports.Protocol -notin @('TCP','6') -or [string]$ports.LocalPort -ne [string]$Port -or
        [string]$addresses.LocalAddress -ne $WindowsIp -or $actual -ne $expected -or
        ($Program -and $app.Program -ne $Program)) {
        throw "Existing helper firewall scope differs: $Name. Nothing replaced."
    }
    Write-Host "Reusing verified helper firewall rule: $Name"
    return $true
}

# The current model manifest rejects the user's installed Ollama 0.21.2.
# Use the official standalone runtime, including optional NVIDIA MLX libraries,
# in a version-specific directory. Never run the global installer or change PATH.
$RuntimeReceipt = Join-Path $RuntimeRoot 'runtime-ready.json'
$Assets = @(
    @{ name = 'ollama-windows-amd64.zip'; sha256 = '3623e256762ca89bd6fa99b0cc4106401919ce9df926411673e632e3ea287bb5' },
    @{ name = 'ollama-windows-amd64-mlx.zip'; sha256 = 'c0ea38f2e42e298bf023f6ce1e48dcf9dc2b757d1e72f281dc0e2092ca6388d1' }
)
if (Test-Path $RuntimeReceipt) {
    $Installed = Get-Content -Raw $RuntimeReceipt | ConvertFrom-Json
    if ($Installed.version -ne $RuntimeVersion -or -not (Test-Path $Ollama) -or
        (Get-FileHash -Algorithm SHA256 $Ollama).Hash.ToLowerInvariant() -ne $Installed.executable_sha256) {
        throw 'Isolated runtime changed or is incomplete. Inspect its directory before replacing it.'
    }
} else {
    $Downloads = Join-Path $Root 'downloads'
    New-Item -ItemType Directory -Path $Downloads -Force | Out-Null
    New-Item -ItemType Directory -Path $RuntimeRoot -Force | Out-Null
    foreach ($Asset in $Assets) {
        $Archive = Join-Path $Downloads "v$RuntimeVersion-$($Asset.name)"
        if (-not (Test-Path $Archive)) {
            $Partial = "$Archive.partial"
            $DownloadUrl = "https://github.com/ollama/ollama/releases/download/v$RuntimeVersion/$($Asset.name)"
            Write-Host "Downloading isolated Ollama ${RuntimeVersion}: $($Asset.name) ..."
            Invoke-WebRequest -UseBasicParsing -Uri $DownloadUrl -OutFile $Partial -TimeoutSec 7200
            if ((Get-FileHash -Algorithm SHA256 $Partial).Hash.ToLowerInvariant() -ne $Asset.sha256) {
                throw "Runtime download checksum mismatch: $($Asset.name). Nothing started."
            }
            Move-Item -LiteralPath $Partial -Destination $Archive
        }
        if ((Get-FileHash -Algorithm SHA256 $Archive).Hash.ToLowerInvariant() -ne $Asset.sha256) {
            throw "Runtime cache checksum mismatch: $Archive. Nothing started."
        }
        Write-Host "Extracting $($Asset.name) ..."
        Expand-Archive -LiteralPath $Archive -DestinationPath $RuntimeRoot -Force
    }
    if (-not (Test-Path $Ollama)) { throw 'Standalone archive did not contain ollama.exe.' }
    @{ version = $RuntimeVersion; executable_sha256 = (Get-FileHash -Algorithm SHA256 $Ollama).Hash.ToLowerInvariant(); assets = $Assets } |
        ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $RuntimeReceipt
}
Write-Host "Using isolated Ollama ${RuntimeVersion}: $Ollama"
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
$CreatedAllowRule = $false
$CreatedBlockRule = $false
Remove-Item -LiteralPath $OwnerPath -ErrorAction SilentlyContinue
try {
if (-not (Confirm-HelperRule $BlockRule 'Block' $Ranges '')) {
New-NetFirewallRule -Name $BlockRule -DisplayName $BlockRule -Direction Inbound -Action Block `
    -Protocol TCP -LocalPort $Port -LocalAddress $WindowsIp -RemoteAddress $Ranges -Profile Any | Out-Null
$CreatedBlockRule = $true
}
if (-not (Confirm-HelperRule $Rule 'Allow' @($DebianIp) $Ollama)) {
New-NetFirewallRule -Name $Rule -DisplayName $Rule -Direction Inbound -Action Allow `
    -Protocol TCP -LocalPort $Port -LocalAddress $WindowsIp -RemoteAddress $DebianIp `
    -Program $Ollama -Profile Any | Out-Null
$CreatedAllowRule = $true
}

$WorkerScript = Join-Path $Root 'serve-worker.ps1'
@'
param([string]$OllamaPath, [string]$HelperRoot)
$ErrorActionPreference = 'Stop'
$prefix = [IO.Path]::GetFullPath((Split-Path $OllamaPath)).TrimEnd('\') + '\'
$restarting = $false
while ($true) {
    while (Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue) {
        @{time=(Get-Date).ToUniversalTime().ToString('o');phase='paused-for-game'} | ConvertTo-Json -Compress | Add-Content (Join-Path $HelperRoot 'logs/guardian.jsonl')
        Start-Sleep -Seconds 15
        $restarting = $true
    }
    if ($restarting) { Start-Sleep -Seconds 60 }
    if (Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue) { continue }
    $server = Start-Process -FilePath $OllamaPath -ArgumentList 'serve' -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $HelperRoot 'logs/server.stdout.log') `
    -RedirectStandardError (Join-Path $HelperRoot 'logs/server.stderr.log')
@{ pid = $server.Id; started = $server.StartTime.ToUniversalTime().ToString('o'); path = $OllamaPath } |
    ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $HelperRoot 'server-owner.json')
    if ($restarting) {
        Start-Sleep -Seconds 5
        $warm = @{model='qwen3.5:9b-q8_0';stream=$false;think=$false;keep_alive=-1;messages=@(@{role='user';content='Return only OK.'});options=@{num_ctx=[int]$env:OLLAMA_CONTEXT_LENGTH;num_batch=[int]$env:HELPER_BATCH_TOKENS;num_predict=8;num_thread=4}} | ConvertTo-Json -Depth 8
        $warmJob = Start-Job -ArgumentList "http://$env:OLLAMA_HOST/api/chat",$warm -ScriptBlock {
            param($url,$body)
            Invoke-RestMethod $url -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 120
        }
        while ($warmJob.State -eq 'Running' -and -not (Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue)) { Start-Sleep -Seconds 3 }
        if ($warmJob.State -eq 'Running') { Stop-Job $warmJob }
        $null = Receive-Job $warmJob -ErrorAction SilentlyContinue
        Remove-Job $warmJob -Force
    }
    @{time=(Get-Date).ToUniversalTime().ToString('o');phase='serving';pid=$server.Id} | ConvertTo-Json -Compress | Add-Content (Join-Path $HelperRoot 'logs/guardian.jsonl')
    while (-not $server.HasExited) {
        if (Get-Process -Name 'League of Legends' -ErrorAction SilentlyContinue) {
            Get-CimInstance Win32_Process -Filter "Name='ollama.exe' OR Name='llama-server.exe'" | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase) } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
            $restarting = $true
            break
        }
        Start-Sleep -Seconds 3
    }
    if (-not $restarting) { break }
}
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
    OLLAMA_DEBUG = '0'
    OLLAMA_KEEP_ALIVE = '-1'
    HELPER_BATCH_TOKENS = "$BatchTokens"
    # Reserve the other half as a scheduler hint, not a hard VRAM partition.
    OLLAMA_GPU_OVERHEAD = "$([int64][Math]::Max(0, ($FreeMiB - 12288)) * 1048576)"
}
if ($DebugLogs) { $Settings['OLLAMA_DEBUG'] = '1' }
foreach ($key in $Settings.Keys) { $Info.EnvironmentVariables[$key] = $Settings[$key] }
$Worker = [Diagnostics.Process]::Start($Info)
@{pid=$Worker.Id;started=$Worker.StartTime.ToUniversalTime().ToString('o');path=$Worker.MainModule.FileName} | ConvertTo-Json | Set-Content -Encoding UTF8 $GuardianPath
$Ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        if ($Worker.HasExited) { throw "Helper startup failed. Inspect $Root\logs." }
        try {
            $ServerVersion = Invoke-RestMethod "$Origin/api/version" -TimeoutSec 2
            if ($ServerVersion.version -ne $RuntimeVersion) { throw 'Unexpected helper server version.' }
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
    # Infrastructure health is separate from the earlier arithmetic quality case.
    # A model cannot qualify financial findings by merely agreeing with itself.
    $Started = Get-Date
    $Request = @{
        model = $Model; stream = $false; think = $false; keep_alive = -1; format = $schema
        options = @{ num_ctx = $Context; num_predict = 256; num_batch = $BatchTokens; num_thread = 4; temperature = 0; seed = 42 }
        messages = @(@{ role = 'user'; content = 'What is 2 + 2? Return JSON with the integer result.' })
    } | ConvertTo-Json -Depth 10
    $Response = Invoke-RestMethod "$Origin/api/chat" -Method Post -ContentType 'application/json' -Body $Request -TimeoutSec 600
    $Diagnostic = $Response | ConvertTo-Json -Depth 20 | ConvertFrom-Json
    $ThinkingChars = ([string]$Response.message.thinking).Length
    $Diagnostic.message | Add-Member -NotePropertyName thinking_chars -NotePropertyValue $ThinkingChars -Force
    $Diagnostic.message.PSObject.Properties.Remove('thinking')
    $Diagnostic | ConvertTo-Json -Depth 20 | Set-Content -Encoding UTF8 (Join-Path $Root 'logs/smoke-json.response.json')
    Write-Host 'JSON SMOKE RESPONSE:'
    $Diagnostic | ConvertTo-Json -Depth 20
    $Answer = $Response.message.content | ConvertFrom-Json
    if (($Answer.result -isnot [int] -and $Answer.result -isnot [long]) -or $Answer.result -ne 4 -or -not $Response.done -or $Response.done_reason -ne 'stop') {
        throw "JSON transport smoke failed. Actual response saved in $Root\logs\smoke-json.response.json."
    }
    $SmokeSeconds = ((Get-Date) - $Started).TotalSeconds
    # Pace only our startup requests. The updated Debian controller separately
    # reserves idle time between all its helper turns; no board-wide power change.
    $SmokeIdleMs = [int][Math]::Ceiling(($Response.total_duration / 1e9) * (100-$ActiveTimePercent) / $ActiveTimePercent * 1000)
    if ($SmokeIdleMs -gt 0) { Start-Sleep -Milliseconds $SmokeIdleMs }

    # Preserve the original harder question as an explicitly recorded quality
    # diagnostic, rather than claiming that the easier transport test replaces it.
    $MathRequest = @{
        model = $Model; stream = $false; think = $false; keep_alive = -1; format = $schema
        options = @{ num_ctx = $Context; num_predict = 1024; num_batch = $BatchTokens; num_thread = 4; temperature = 0; seed = 42 }
        messages = @(@{ role = 'user'; content = 'What is 17 * (6013 - 5347) - 319 - 367 - 113 - 79? Return JSON with result.' })
    } | ConvertTo-Json -Depth 10
    $MathResponse = Invoke-RestMethod "$Origin/api/chat" -Method Post -ContentType 'application/json' -Body $MathRequest -TimeoutSec 600
    $MathPassed = $false
    $MathError = $null
    try {
        $MathAnswer = $MathResponse.message.content | ConvertFrom-Json
        $MathPassed = $MathAnswer.result -eq 10444 -and $MathResponse.done -and $MathResponse.done_reason -eq 'stop'
    } catch { $MathError = $_.Exception.Message }
    $MathDiagnostic = @{
        expected = 10444; content = $MathResponse.message.content; passed = [bool]$MathPassed
        finish_reason = $MathResponse.done_reason; parse_error = $MathError
        prompt_tokens = $MathResponse.prompt_eval_count; output_tokens = $MathResponse.eval_count
        output_budget = 1024; thinking_requested = $false
        thinking_chars = ([string]$MathResponse.message.thinking).Length
        scope = 'Direct arithmetic quality diagnostic; not the infrastructure gate or a financial benchmark'
    }
    $MathDiagnostic | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 (Join-Path $Root 'logs/smoke-math.diagnostic.json')
    Write-Host 'DIRECT MATH DIAGNOSTIC:'
    $MathDiagnostic | ConvertTo-Json -Depth 10
    $Loaded = @((Invoke-RestMethod "$Origin/api/ps" -TimeoutSec 10).models | Where-Object { $_.name -eq $Model })
    if ($Loaded.Count -ne 1 -or $Loaded[0].context_length -ne $Context) { throw 'Unexpected loaded model or context.' }
    $Item = $Loaded[0]
    $Loaded | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 (Join-Path $Root 'logs/smoke-gpu.loaded.json')
    Write-Host 'GPU MODEL MEASUREMENT:'
    $Loaded | ConvertTo-Json -Depth 10
    if ($Item.size_vram -le 0 -or $Item.size_vram -lt $Item.size -or $Item.size_vram -gt 12GB) {
        throw 'Helper is not fully on GPU within 12 GiB. Inspect logs; next try can use a smaller context.'
    }
    $Receipt = @{
        model = $Model; digest = $Item.digest; context = $Context; result = $Answer.result
        infrastructure_smoke_passed = $true; direct_math_passed = [bool]$MathPassed
        ollama_version = $RuntimeVersion; runtime = $Ollama
        origin = $Origin; debian = $DebianIp; vram_gib = [Math]::Round($Item.size_vram / 1GB, 2)
        seconds = [Math]::Round($SmokeSeconds, 2)
        generation_tps = [Math]::Round($Response.eval_count * 1e9 / [Math]::Max(1, $Response.eval_duration), 2)
        note = 'GPU memory measurement after load, not a hard or transient peak limit'
        batch_tokens = $BatchTokens; startup_request_active_time_target_percent = $ActiveTimePercent
        game_guard = 'Pause isolated helper during League of Legends; resume after game and cooldown'
        workload_note = 'Controller update required for research pacing; no hard GPU utilization or board power cap'
        debug_logs = [bool]$DebugLogs
        server_log = (Join-Path $Root 'logs/server.stderr.log')
    }
    $Receipt | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $Root 'helper-ready.json')
    Write-Host 'RTX HELPER READY. Paste the following result back:'
    $Receipt | ConvertTo-Json
    & nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.free --format=csv
} catch {
    # Stop the owned launcher first so it cannot spawn another helper during
    # cleanup, then stop verified native workers including orphan GPU runners.
    $LaunchError = $_
    if ($Worker -and -not $Worker.HasExited) { $Worker.Kill() }
    try { Stop-HelperProcesses $RuntimeRoot }
    catch { Write-Warning "Helper cleanup failed: $($_.Exception.Message)" }
    if ($CreatedAllowRule) { Remove-NetFirewallRule -Name $Rule -ErrorAction SilentlyContinue }
    if ($CreatedBlockRule) { Remove-NetFirewallRule -Name $BlockRule -ErrorAction SilentlyContinue }
    throw $LaunchError
}
