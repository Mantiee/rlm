param([Parameter(Mandatory=$true)][string]$Revision,[Parameter(Mandatory=$true)][string]$Bundle)
$ErrorActionPreference = 'Stop'
if ($Revision -notmatch '^[0-9a-f]{40}$') { throw 'Exact GitHub revision required.' }
$parseErrors = $null
$tokens = $null
[System.Management.Automation.Language.Parser]::ParseFile($PSCommandPath, [ref]$tokens, [ref]$parseErrors) | Out-Null
if ($parseErrors.Count) { throw 'Preparation script failed PowerShell syntax validation.' }
$root = Join-Path $env:USERPROFILE 'ai-v100-helper'
$python = Join-Path $env:USERPROFILE 'ai-owned-compute\venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Existing owned compute Python required.' }
New-Item -ItemType Directory -Force -Path (Join-Path $root 'logs') | Out-Null
try {
    & $python (Join-Path $Bundle 'rlm\v100\source_sync.py') --root $root --revision $Revision
    if ($LASTEXITCODE -ne 0) { throw 'Public source snapshot acquisition failed; remaining preparations continue.' }
} catch { Write-Warning $_.Exception.Message }
# The existing proxy remains the only admission controller. Tailscale is a scoped forwarding path.
$network = @{state='not-configured';error='Tailscale executable unavailable'}
try {
$tailscale = Join-Path $env:ProgramFiles 'Tailscale\tailscale.exe'
if (Test-Path -LiteralPath $tailscale) {
    $state = (& $tailscale status --json | ConvertFrom-Json)
    $peers = @($state.Peer.PSObject.Properties.Value | Where-Object { $_.HostName -ieq 'debian1' })
    $selfIp = @($state.Self.TailscaleIPs | Where-Object { $_ -match '^100\.' })
    if ($peers.Count -ne 1 -or $selfIp.Count -ne 1) { throw 'Expected one authenticated debian1 tailnet peer and one local tailnet IPv4.' }
    $owner = @($peers[0].TailscaleIPs | Where-Object { $_ -match '^100\.' })
    if ($owner.Count -ne 1) { throw 'Ambiguous Debian tailnet address; no forwarding rule created.' }
    $rule = 'synta-owned-tailnet-helper-11437'
    $existing = @(Get-NetFirewallRule -Name $rule -ErrorAction SilentlyContinue)
    if ($existing.Count -eq 0) {
        New-NetFirewallRule -Name $rule -DisplayName $rule -Direction Inbound -Action Allow -Protocol TCP `
            -LocalAddress $selfIp[0] -LocalPort 11437 -RemoteAddress $owner[0] -Profile Any | Out-Null
    } else {
        $address = $existing[0] | Get-NetFirewallAddressFilter
        $port = $existing[0] | Get-NetFirewallPortFilter
        if ($existing.Count -ne 1 -or $existing[0].Action -ne 'Allow' -or $address.LocalAddress -ne $selfIp[0] -or $address.RemoteAddress -ne $owner[0] -or $port.LocalPort -ne '11437') { throw 'Existing tailnet rule differs; no broad replacement made.' }
    }
    $routes = & netsh.exe interface portproxy show v4tov4
    $matched = @($routes | Where-Object { $_ -match ('^\s*' + [regex]::Escape($selfIp[0]) + '\s+11437\s+') })
    if ($matched.Count -gt 0 -and $matched[0] -notmatch '\s192\.168\.0\.61\s+11435\s*$') { throw 'Tailnet forwarding port already belongs to another route.' }
    Start-Service iphlpsvc
    & netsh.exe interface portproxy add v4tov4 listenaddress=$($selfIp[0]) listenport=11437 connectaddress=192.168.0.61 connectport=11435
    if ($LASTEXITCODE -ne 0) { throw 'Scoped tailnet forwarding failed.' }
    $network = @{state='configured';endpoint="http://$($selfIp[0]):11437";owner=$owner[0];revision=$Revision}
}
} catch {
    $network = @{state='blocked';error=$_.Exception.Message;revision=$Revision}
    Write-Warning $_.Exception.Message
}
$network | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $root 'logs\tailnet-helper.json')
# Download only public local model tags. The preserved 9B baseline is never pulled/rebound.
$models = @('qwen3.5:2b-q4_K_M','qwen3.5:4b-q4_K_M','qwen3.6:35b-a3b-q4_K_M')
$failures = @()
$available = @()
try {
    $available = @((Invoke-RestMethod 'http://127.0.0.1:11436/api/tags' -TimeoutSec 10).models.name)
} catch { Write-Warning $_.Exception.Message }
foreach ($model in $models) {
    try {
        if ($available -contains $model) { Write-Host "Retaining existing model identity: $model"; continue }
        $drive = Get-PSDrive -Name ([IO.Path]::GetPathRoot($root).Substring(0,1))
        if ($drive.Free -lt 32GB) { throw 'Model preparation needs at least 32 GiB free disk; existing models retained.' }
        Write-Host "Preparing owned local model: $model"
        Invoke-RestMethod 'http://127.0.0.1:11436/api/pull' -Method Post -ContentType 'application/json' `
            -Body (@{model=$model;stream=$false} | ConvertTo-Json) -TimeoutSec 3600 | ConvertTo-Json -Depth 4
    } catch {
        $failures += @{model=$model;error=$_.Exception.Message}
        Write-Warning $_.Exception.Message
    }
}
$inventory = @()
try { $inventory = @((Invoke-RestMethod 'http://127.0.0.1:11436/api/tags' -TimeoutSec 10).models) }
catch { $failures += @{model='inventory';error=$_.Exception.Message} }
$stage = 'prepared'
if ($failures.Count -gt 0 -or $network.state -ne 'configured') { $stage = 'partially-blocked' }
@{stage=$stage;models=$inventory;errors=$failures;network=$network;revision=$Revision} | ConvertTo-Json -Depth 7 |
    Set-Content -Encoding UTF8 (Join-Path $root 'logs\managed-models.json')
$benchmark = Join-Path $Bundle 'tools\benchmark-synta-windows.ps1'
$nativeReceipt = Join-Path $root 'logs\managed-native.json'
$previous = $null
if (Test-Path -LiteralPath $nativeReceipt) {
    try { $previous = Get-Content -Raw -LiteralPath $nativeReceipt | ConvertFrom-Json }
    catch { Write-Warning 'Previous native receipt malformed; new attempt will use checked artifacts.' }
}
$now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
if ($null -eq $previous -or ($previous.revision -ne $Revision) -or
    ($previous.state -ne 'measured' -and ($now - $previous.updated) -ge 3600)) {
    try {
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $benchmark -Revision $Revision -Bundle $Bundle
        if ($LASTEXITCODE -ne 0) { throw 'Native benchmark deferred or blocked; inspect performance logs.' }
        @{state='measured';revision=$Revision;updated=$now} | ConvertTo-Json | Set-Content -Encoding UTF8 $nativeReceipt
    } catch {
        @{state='blocked';error=$_.Exception.Message;revision=$Revision;updated=$now} | ConvertTo-Json | Set-Content -Encoding UTF8 $nativeReceipt
        Write-Warning $_.Exception.Message
    }
}
$task = Get-ScheduledTask -TaskName 'Synta-Managed-Update' -ErrorAction SilentlyContinue
$runner = Join-Path $root 'tools\managed-bootstrap\managed_bootstrap.py'
if ($task -and $task.Actions.Count -eq 1 -and $task.Actions[0].Execute -eq $python -and $task.Actions[0].Arguments.Contains($runner)) {
    $trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(15) -RepetitionInterval (New-TimeSpan -Minutes 15)
    Set-ScheduledTask -TaskName 'Synta-Managed-Update' -Trigger $trigger | Out-Null
}
Write-Host "Owned model preparation: $stage. Debian independently qualifies roles; blocked steps remain recorded."
