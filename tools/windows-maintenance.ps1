# Shared bounded drain; callers verify this file against the release inventory first.
function Wait-SyntaMaintenance {
    param([Parameter(Mandatory=$true)][string]$Endpoint,[ValidateRange(1,60)][int]$WaitSeconds=60)
    $timer = [Diagnostics.Stopwatch]::StartNew()
    do {
        try {
            $reply = Invoke-RestMethod $Endpoint -Method Post -TimeoutSec 5
            if ($reply.ready -ne $true -or $reply.maintenance -ne $true) {
                throw 'Owned helper did not confirm its maintenance lease; no process stopped.'
            }
            return $reply
        } catch {
            $status = 0
            if ($_.Exception.Response -and $null -ne $_.Exception.Response.StatusCode) {
                $status = [int]$_.Exception.Response.StatusCode
            }
            if ($status -ne 503 -or $timer.Elapsed.TotalSeconds -ge $WaitSeconds) { throw }
            # A busy old proxy has no pending-drain API. Retry its atomic claim
            # until it admits maintenance; never stop an in-flight inference.
            Start-Sleep -Milliseconds 500
        }
    } while ($true)
}
