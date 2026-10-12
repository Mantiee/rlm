param([Parameter(Mandatory=$true)][string]$Revision,[string]$Bundle="")
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
if ($Revision -notmatch '^[0-9a-f]{40}$') { throw 'Use an exact published revision.' }
$root = Join-Path $env:USERPROFILE 'ai-v100-helper'
$python = Join-Path $env:USERPROFILE 'ai-owned-compute\venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Existing Synta isolated Python runtime required.' }
$manifestPath = Join-Path $root 'models\manifests\registry.ollama.ai\library\qwen3.5\9b-q8_0'
if (-not (Test-Path -LiteralPath $manifestPath)) { throw 'Existing Synta qwen3.5:9b-q8_0 manifest required; no model is downloaded.' }
$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
$layers = @($manifest.layers | Where-Object { $_.mediaType -eq 'application/vnd.ollama.image.model' })
if ($layers.Count -ne 1 -or $layers[0].digest -notmatch '^sha256:[0-9a-f]{64}$') { throw 'Ambiguous or invalid GGUF model manifest.' }
$modelHash = $layers[0].digest.Substring(7)
$model = Join-Path $root ('models\blobs\sha256-' + $modelHash)
if (-not (Test-Path -LiteralPath $model)) { throw 'Local model blob unavailable.' }
& $python -c 'import psutil'
if ($LASTEXITCODE -ne 0) { throw 'Existing Windows psutil runtime is unavailable.' }
# Official pinned CUDA 12.4 binaries in a separate directory. No driver, service or scheduler changes.
$runtime = Join-Path $root 'native-b10683-cuda124'
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
$assets = @(
    @{Name='llama-b10683-bin-win-cuda-12.4-x64.zip';Hash='e4681eade030b4075bf89c7449cadb08de1af315ad26353b1e9c39bf43ee9e07'},
    @{Name='cudart-llama-bin-win-cuda-12.4-x64.zip';Hash='8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6'}
)
foreach ($asset in $assets) {
    $zip = Join-Path $runtime $asset.Name
    if (-not (Test-Path -LiteralPath $zip)) {
        Invoke-WebRequest -UseBasicParsing ('https://github.com/ggml-org/llama.cpp/releases/download/b10683/' + $asset.Name) -OutFile ($zip + '.new')
        if ((Get-FileHash -Algorithm SHA256 -LiteralPath ($zip + '.new')).Hash.ToLowerInvariant() -ne $asset.Hash) { throw 'Native runtime checksum mismatch; nothing launched.' }
        Move-Item -LiteralPath ($zip + '.new') -Destination $zip
    }
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $zip).Hash.ToLowerInvariant() -ne $asset.Hash) { throw 'Cached native runtime checksum mismatch.' }
    Expand-Archive -LiteralPath $zip -DestinationPath $runtime -Force
}
$binary = @(Get-ChildItem -LiteralPath $runtime -Filter 'llama-server.exe' -Recurse)
if ($binary.Count -ne 1) { throw 'Expected one isolated native server executable.' }
# CUDA DLLs may be packaged in a separate subdirectory; scope PATH to this PowerShell only.
$dllDirectories = @(Get-ChildItem -LiteralPath $runtime -Filter '*.dll' -Recurse | Select-Object -ExpandProperty DirectoryName -Unique)
$oldPath = $env:PATH
$folder = Join-Path $root ('performance\' + (Get-Date -Format 'yyyyMMdd-HHmmss-ffff'))
New-Item -ItemType Directory -Path $folder | Out-Null
$script = Join-Path $folder 'inference_benchmark.py'
if ($Bundle) {
    Copy-Item -LiteralPath (Join-Path $Bundle 'rlm\v100\inference_benchmark.py') -Destination $script
} else {
    Invoke-WebRequest -UseBasicParsing "https://raw.githubusercontent.com/Mantiee/rlm/$Revision/rlm/v100/inference_benchmark.py" -OutFile $script
}
$profile = @{server=@{binary=$binary[0].FullName;model=$model;model_sha256=$modelHash;context_per_slot=32768;slots=1;threads=4;batch_size=128;ubatch_size=64;flash_attention='on';cache_type='q8_0';jinja=$true;gpu_layers=999;draft_model='';spec_type='none'}}
$profilePath = Join-Path $folder 'profile.json'
$profile | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $profilePath -Encoding UTF8
try {
    $env:PATH = ($dllDirectories -join ';') + ';' + $oldPath
    Write-Host 'Comparing the existing model: baseline, n-gram and CPU offload. Browser allowed; host pressure interrupts only this owned experiment.'
    & $python $script --profile $profilePath --output (Join-Path $folder 'results') --port 8097
    if ($LASTEXITCODE -ne 0) { throw "Benchmark blocked or failed. Inspect $folder. Existing helper stays unchanged." }
} finally { $env:PATH = $oldPath }
Write-Host "Results: $folder\results\results.json"
Write-Host 'No production helper changed. A native winner is a measured proposal; Ollama does not inherit native n-gram flags.'
