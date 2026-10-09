# Separate CPU-only worker. Does not change Ollama, board power limits or PATH.
param(
    [Parameter(Mandatory=$true)][string]$Revision,
    [Parameter(Mandatory=$true)][string]$Mailbox,
    [string]$WorkerName = 'windows-cpu',
    [switch]$Once
)
$ErrorActionPreference = 'Stop'
if ($Revision -notmatch '^[0-9a-f]{40}$') { throw 'Use the exact published source revision.' }
if ($WorkerName -notmatch '^[a-zA-Z0-9_-]{1,48}$') { throw 'Invalid worker name.' }
if (-not (Test-Path -LiteralPath $Mailbox)) { throw 'Connect the authenticated shared compute folder first.' }
$dir = Join-Path $env:USERPROFILE 'ai-owned-compute'
New-Item -ItemType Directory -Force -Path $dir | Out-Null
@{revision=$Revision; mailbox=$Mailbox; worker=$WorkerName} | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $dir 'worker-settings.json')
$uvDir = Join-Path $dir 'uv-0.8.22'
$uv = Join-Path $uvDir 'uv.exe'
if (-not (Test-Path -LiteralPath $uv)) {
    $zip = Join-Path $dir 'uv-0.8.22.zip'
    Invoke-WebRequest -UseBasicParsing 'https://github.com/astral-sh/uv/releases/download/0.8.22/uv-x86_64-pc-windows-msvc.zip' -OutFile $zip
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $zip).Hash.ToLowerInvariant() -ne '5049375aa2a5162f132b2c1cb992e25d42d47d934cab8c174dbe6f60973dcc12') {
        throw 'uv archive checksum mismatch.'
    }
    Expand-Archive -LiteralPath $zip -DestinationPath $uvDir -Force
}
$priorPythonDir = $env:UV_PYTHON_INSTALL_DIR
$priorCache = $env:UV_CACHE_DIR
try {
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $dir 'python'
    $env:UV_CACHE_DIR = Join-Path $dir 'cache'
    $venv = Join-Path $dir 'venv'
    $python = Join-Path $venv 'Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python)) {
        & $uv --no-config venv --python 3.11 $venv
        if ($LASTEXITCODE -ne 0) { throw 'Isolated CPU venv creation failed.' }
    }
    # Avoid network package resolution at every logon when exact dependencies exist.
    & $python -c "import importlib.metadata as m; assert m.version('torch').split('+')[0] == '2.6.0'; assert m.version('safetensors') == '0.5.3'; assert m.version('psutil') == '7.0.0'" 2>$null
    if ($LASTEXITCODE -ne 0) {
        & $uv --no-config pip install --python $python --index-url 'https://download.pytorch.org/whl/cpu' 'torch==2.6.0'
        if ($LASTEXITCODE -ne 0) { throw 'CPU Torch install failed.' }
        & $uv --no-config pip install --python $python --index-url 'https://pypi.org/simple' 'safetensors==0.5.3' 'psutil==7.0.0'
        if ($LASTEXITCODE -ne 0) { throw 'CPU worker dependency install failed.' }
    }
    $source = Join-Path $dir $Revision
    New-Item -ItemType Directory -Force -Path $source | Out-Null
    foreach ($name in @('compute_worker.py','compute_kernel.py')) {
        $target = Join-Path $source $name
        if (-not (Test-Path -LiteralPath $target)) {
            $temporary = $target + '.download'
            Invoke-WebRequest -UseBasicParsing "https://raw.githubusercontent.com/Mantiee/rlm/$Revision/rlm/v100/$name" -OutFile $temporary
            Move-Item -LiteralPath $temporary -Destination $target
        }
    }
    Write-Host 'Owned CPU worker: max 2 threads / 4 GiB child RAM. RTX remains unchanged.'
    Write-Host 'One shared job at a time. Host CPU/RAM pressure or League gameplay pauses experiments.'
    $worker = Join-Path $source 'compute_worker.py'
    if ($Once) {
        & $python -u $worker --mailbox $Mailbox --name $WorkerName --once
    } else {
        & $python -u $worker --mailbox $Mailbox --name $WorkerName
    }
    if ($LASTEXITCODE -ne 0) { throw 'Owned CPU worker stopped with an error.' }
} finally {
    $env:UV_PYTHON_INSTALL_DIR = $priorPythonDir
    $env:UV_CACHE_DIR = $priorCache
}
