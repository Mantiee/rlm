[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateLength(15, 500)]
    [string]$ReviewedCause
)

# This is the only source-controlled action that clears the RTX manual-stop
# lock. It never changes League, other processes, or global GPU settings.
$ErrorActionPreference = 'Stop'
$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run elevated only after reviewing the crash cause; the helper remains stopped.'
}

$root = Join-Path $env:USERPROFILE 'ai-v100-helper'
$manualStopPath = Join-Path $root 'helper-manual-stop.flag'
if (-not (Test-Path -LiteralPath $manualStopPath)) {
    throw 'RTX helper is not manually stopped; no resume action was needed.'
}

$confirmation = Read-Host 'After reviewing the Windows crash evidence, type RESUME RTX 3090 to clear the manual stop'
if ($confirmation -cne 'RESUME RTX 3090') {
    throw 'Resume confirmation did not match; the helper remains stopped.'
}

$logDirectory = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$receiptPath = Join-Path $logDirectory 'manual-resume.jsonl'
$receipt = [pscustomobject]@{
    utc = [DateTimeOffset]::UtcNow.ToString('o')
    reviewed_cause = $ReviewedCause.Trim()
    operator = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    action = 'cleared helper-manual-stop.flag after interactive confirmation'
}
Add-Content -LiteralPath $receiptPath -Value ($receipt | ConvertTo-Json -Compress) -Encoding UTF8
Remove-Item -LiteralPath $manualStopPath -ErrorAction Stop
Write-Host 'Manual stop cleared after explicit review. An already-running watchdog may resume the helper on its next check.'
Write-Host 'If watchdog-state.json has an open circuit, that separate breaker remains open until a reviewed reset.'
