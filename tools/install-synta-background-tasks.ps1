$ErrorActionPreference = 'Stop'
$root = Join-Path $env:USERPROFILE 'ai-v100-helper'
$cpuRoot = Join-Path $env:USERPROFILE 'ai-owned-compute'
$directory = Join-Path $root 'tools\background-tasks'
New-Item -ItemType Directory -Force -Path $directory | Out-Null
$hostExe = Join-Path $env:WINDIR 'System32\wscript.exe'
if (-not (Test-Path -LiteralPath $hostExe)) { throw 'Windowless Windows script host is unavailable.' }
$changes = @()
foreach ($name in @('Synta-Managed-Update','Synta-RTX3090-Helper','Synta-Owned-CPU-Worker','Synta-Helper-Monitor')) {
    $task = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
    if (-not $task) { continue }
    if ($task.Actions.Count -ne 1) { throw "Unexpected action count: $name" }
    $action = $task.Actions[0]
    $wrapper = Join-Path $directory ($name + '.vbs')
    if ($action.Execute -ieq $hostExe -and $action.Arguments.Contains($wrapper)) { continue }
    $ownedRoot = if ($name -eq 'Synta-Owned-CPU-Worker') { $cpuRoot } else { $root }
    if ([IO.Path]::GetFileName($action.Execute) -ine 'powershell.exe' -or -not $action.Arguments.Contains($ownedRoot)) {
        throw "Task is outside the owned PowerShell scope: $name"
    }
    $command = '"' + $action.Execute + '" ' + $action.Arguments
    $escaped = $command.Replace('"','""')
    @('Set shell = CreateObject("WScript.Shell")', ('code = shell.Run("' + $escaped + '", 0, True)'), 'WScript.Quit code') |
        Set-Content -Encoding Unicode -LiteralPath $wrapper
    $hidden = New-ScheduledTaskAction -Execute $hostExe -Argument ('//B //Nologo "' + $wrapper + '"')
    $changes += [pscustomobject]@{Name=$name;Action=$hidden}
}
foreach ($change in $changes) { Set-ScheduledTask -TaskName $change.Name -Action $change.Action | Out-Null }
Write-Host 'Owned background tasks launch without a console window. Existing running tasks continue.'
