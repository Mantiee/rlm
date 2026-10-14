$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'windows-maintenance.ps1')
Add-Type @'
using System;
public class SyntaMaintenanceTestResponse { public int StatusCode; }
public class SyntaMaintenanceTestFailure : Exception {
    public SyntaMaintenanceTestResponse Response;
    public SyntaMaintenanceTestFailure(int status) : base("Simulated owned endpoint error") {
        Response = new SyntaMaintenanceTestResponse { StatusCode = status };
    }
}
'@
$script:calls = 0
$script:mode = 'busy-then-ready'
function Invoke-RestMethod {
    param($Uri,$Method,$TimeoutSec)
    $script:calls += 1
    if ($script:mode -eq 'busy-then-ready' -and $script:calls -le 2) {
        throw (New-Object SyntaMaintenanceTestFailure 503)
    }
    if ($script:mode -eq 'forbidden') { throw (New-Object SyntaMaintenanceTestFailure 403) }
    if ($script:mode -eq 'busy') { throw (New-Object SyntaMaintenanceTestFailure 503) }
    if ($script:mode -eq 'bad-lease') { return @{ready=$true;maintenance=$false} }
    return @{ready=$true;maintenance=$true}
}
$reply = Wait-SyntaMaintenance -Endpoint 'http://owned.test/api/synta-maintenance' -WaitSeconds 4
if ($script:calls -ne 3 -or -not $reply.maintenance) { throw 'Busy helper was not safely drained.' }
foreach ($mode in @('forbidden','bad-lease','busy')) {
    $script:mode=$mode
    $script:calls=0
    $failed=$false
    $timer=[Diagnostics.Stopwatch]::StartNew()
    try { Wait-SyntaMaintenance -Endpoint 'http://owned.test/api/synta-maintenance' -WaitSeconds 1 | Out-Null }
    catch { $failed=$true }
    if (-not $failed) { throw "Invalid maintenance accepted: $mode" }
    if ($mode -ne 'busy' -and $script:calls -ne 1) { throw 'Non-transient failure retried.' }
    if ($mode -eq 'busy' -and ($script:calls -gt 4 -or $timer.Elapsed.TotalSeconds -gt 5)) { throw 'Busy drain was not bounded.' }
}
Write-Output 'Owned maintenance: busy retry, deadline, forbidden response and invalid lease passed.'
