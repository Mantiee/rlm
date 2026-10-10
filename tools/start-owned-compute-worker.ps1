# Separate CPU-only worker. Does not change Ollama, board power limits or PATH.
param(
    [Parameter(Mandatory=$true)][string]$Revision,
    [Parameter(Mandatory=$true)][string]$Mailbox,
    [string]$WorkerName = 'windows-cpu',
    [switch]$Once
)
$ErrorActionPreference = 'Stop'
function Invoke-OwnedNative([string]$File, [string[]]$NativeArguments) {
    # Windows PowerShell 5.1 treats native stderr as an ErrorRecord. Preserve ALL
    # stderr (including tracebacks) and use the actual exit code, not its first line.
    $ErrorActionPreference = 'Continue'
    & $File @NativeArguments
    if ($LASTEXITCODE -ne 0) { throw "Native command exited $LASTEXITCODE : $File" }
}
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
        Invoke-OwnedNative $uv @('--no-config','venv','--python','3.11',$venv)
        if ($LASTEXITCODE -ne 0) { throw 'Isolated CPU venv creation failed.' }
    }
    # Avoid network package resolution at every logon when exact dependencies exist.
    $savedErrors = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $python -c "import importlib.metadata as m; assert m.version('torch').split('+')[0] == '2.6.0'; assert m.version('safetensors') == '0.5.3'; assert m.version('psutil') == '7.0.0'" 2>$null
    $checkExit = $LASTEXITCODE
    $ErrorActionPreference = $savedErrors
    if ($checkExit -ne 0) {
        Invoke-OwnedNative $uv @('--no-config','pip','install','--python',$python,'--index-url','https://download.pytorch.org/whl/cpu','torch==2.6.0')
        if ($LASTEXITCODE -ne 0) { throw 'CPU Torch install failed.' }
        Invoke-OwnedNative $uv @('--no-config','pip','install','--python',$python,'--index-url','https://pypi.org/simple','safetensors==0.5.3','psutil==7.0.0')
        if ($LASTEXITCODE -ne 0) { throw 'CPU worker dependency install failed.' }
    }
    @{revision=$Revision;python=$python;ready=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $dir 'runtime-ready.json')
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
    Write-Host 'One job; idle priority and 2 CPU affinity slots on Windows. RAM is monitored every 2 s.'
    Write-Host 'Browser allowed. Host CPU >65%, free RAM <6 GiB or heavy game suspends owned CPU work; it resumes when pressure clears.'
    $worker = Join-Path $source 'compute_worker.py'
    $logs = Join-Path $env:USERPROFILE 'ai-v100-helper\logs'
    New-Item -ItemType Directory -Force -Path $logs | Out-Null
    $arguments = "-u `"$worker`" --mailbox `"$Mailbox`" --name $WorkerName"
    if ($Once) { $arguments += ' --once' }
    $child = Start-Process -FilePath $python -ArgumentList $arguments -PassThru -Wait -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logs 'cpu-native.stdout.log') `
        -RedirectStandardError (Join-Path $logs 'cpu-native.stderr.log')
    if ($child.ExitCode -ne 0) {
        Get-Content -LiteralPath (Join-Path $logs 'cpu-native.stderr.log') -Tail 100 | Write-Host
        throw "Owned CPU worker exited $($child.ExitCode). Full traceback: $logs\cpu-native.stderr.log"
    }

} finally {
    $env:UV_PYTHON_INSTALL_DIR = $priorPythonDir
    $env:UV_CACHE_DIR = $priorCache
}
