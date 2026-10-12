# Repair the running owned proxy rule. Does not restart workers, alter VPN or disable firewall.
$ErrorActionPreference = 'Stop'
$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Run PowerShell as administrator.' }
$root = Join-Path $env:USERPROFILE 'ai-v100-helper'
$proxyScript = Join-Path $root 'windows_resource_proxy.py'
$listeners = @(Get-NetTCPConnection -State Listen -LocalAddress '192.168.0.61' -LocalPort 11435)
if ($listeners.Count -ne 1) { throw 'Expected one existing owned listener at 192.168.0.61:11435; nothing changed.' }
$proxyProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$($listeners[0].OwningProcess)"
if (-not $proxyProcess.ExecutablePath -or -not $proxyProcess.CommandLine -or
    $proxyProcess.CommandLine.IndexOf($proxyScript, [StringComparison]::OrdinalIgnoreCase) -lt 0 -or
    [IO.Path]::GetFileName($proxyProcess.ExecutablePath) -ine 'python.exe') {
    throw 'Listener is not the expected owned Python proxy; nothing changed.'
}
$actualPython = $proxyProcess.ExecutablePath
$ownedPython = [IO.Path]::GetFullPath((Join-Path $env:USERPROFILE 'ai-owned-compute')) + '\'
if (-not [IO.Path]::GetFullPath($actualPython).StartsWith($ownedPython, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'TCP owner is outside the isolated Synta Python installation; nothing changed.'
}
$name = 'v100-helper-proxy-11435-from-192.168.0.68-actual-python'
$existing = @(Get-NetFirewallRule -Name $name -ErrorAction SilentlyContinue)
if ($existing.Count -gt 1) { throw 'Ambiguous existing rule; nothing changed.' }
if ($existing.Count -eq 1) {
    $rule = $existing[0]
    $address = $rule | Get-NetFirewallAddressFilter
    $port = $rule | Get-NetFirewallPortFilter
    $app = $rule | Get-NetFirewallApplicationFilter
    if ($rule.Direction -ne 'Inbound' -or $rule.Action -ne 'Allow' -or $rule.Enabled -ne 'True' -or
        [string]$address.LocalAddress -ne '192.168.0.61' -or [string]$address.RemoteAddress -ne '192.168.0.68' -or
        [string]$port.LocalPort -ne '11435' -or [string]$port.Protocol -notin @('TCP','6') -or
        $app.Program -ine $actualPython) { throw 'Existing repair rule differs; nothing replaced.' }
    Write-Host 'Reusing the existing verified actual-interpreter rule.'
} else {
    New-NetFirewallRule -Name $name -DisplayName $name -Direction Inbound -Action Allow `
        -Protocol TCP -LocalPort 11435 -LocalAddress '192.168.0.61' -RemoteAddress '192.168.0.68' `
        -Program $actualPython -Profile Any | Out-Null
}
$receipt = @{process_id=$proxyProcess.ProcessId;program=$actualPython;local_address='192.168.0.61';remote_address='192.168.0.68';port=11435;rule=$name;updated=(Get-Date).ToUniversalTime().ToString('o')}
New-Item -ItemType Directory -Force -Path (Join-Path $root 'logs') | Out-Null
$receipt | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $root 'logs\firewall-repair.json')
$receipt | ConvertTo-Json
Write-Host 'Scoped actual-interpreter rule ready. Existing deny rules, VPN, workers and GPU policy retained. Verify TCP/API access from Debian.'
