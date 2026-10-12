param(
    [Parameter(Mandatory=$true)][string]$Revision,
    [ValidateSet('fast','planner')][string]$Role='fast',
    [ValidateSet(32768,131072,262144,524288)][int]$Context=32768
)
$ErrorActionPreference = 'Stop'
if ($Revision -notmatch '^[0-9a-f]{40}$') { throw 'Exact published revision required.' }
$root = Join-Path $env:USERPROFILE 'ai-v100-helper'
$python = Join-Path $env:USERPROFILE 'ai-owned-compute\venv\Scripts\python.exe'
$runtime = Join-Path $root 'native-b10683-cuda124'
$binary = @(Get-ChildItem -LiteralPath $runtime -Filter 'llama-server.exe' -Recurse)
if ($binary.Count -ne 1 -or -not (Test-Path -LiteralPath $python)) { throw 'Existing isolated native benchmark runtime and Python required.' }
$folder = Join-Path $root ('performance\candidate-' + $Role + '-' + (Get-Date -Format 'yyyyMMdd-HHmmss-ffff'))
New-Item -ItemType Directory -Path $folder | Out-Null
foreach ($name in @('model_candidates','inference_benchmark')) {
    Invoke-WebRequest -UseBasicParsing "https://raw.githubusercontent.com/Mantiee/rlm/$Revision/rlm/v100/$name.py" -OutFile (Join-Path $folder ($name + '.py'))
}
$dllDirectories = @(Get-ChildItem -LiteralPath $runtime -Filter '*.dll' -Recurse | Select-Object -ExpandProperty DirectoryName -Unique)
$oldPath = $env:PATH
try {
    $env:PATH = ($dllDirectories -join ';') + ';' + $oldPath
    & $python (Join-Path $folder 'model_candidates.py') --root $root --binary $binary[0].FullName --role $Role --context $Context --output (Join-Path $folder 'results') --port 8097
    if ($LASTEXITCODE -ne 0) { throw "Candidate qualification did not complete. Inspect $folder. Production helper retained." }
} finally { $env:PATH = $oldPath }
Write-Host "Results: $folder\results\results.json"
Write-Host 'Candidate downloaded and compared. Production promotion requires independent task-quality evidence; no model or weights silently replaced.'
