#!/usr/bin/env bash
# Dedicated Debian RTX 3060 Ti services. No controller or operator home is exposed.
set -euo pipefail
REV="${1:?Pass an exact published revision}"
[[ "$REV" =~ ^[a-f0-9]{40}$ ]] || exit 1
[[ "$EUID" -eq 0 ]] || { echo 'Run this installer with sudo on 192.168.0.9 only.' >&2; exit 1; }
[[ "$(uname -m)" == x86_64 ]] || { echo 'The pinned runtime requires x86_64.' >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo 'Working existing NVIDIA driver required; no driver replaced.' >&2; exit 1; }
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
[[ "$GPU" == *'RTX 3060 Ti'* && "$GPU" != *$'\n'* ]] || { echo 'Expected one RTX 3060 Ti; no service changed.' >&2; exit 1; }
ip -4 -o addr show | python3 -c 'import sys; assert any("192.168.0.9/" in line for line in sys.stdin), "Run on the explicitly owned 192.168.0.9 host"'
BASE=/opt/synta3060
STATE=/var/lib/synta3060
if [[ "${2:-}" == --update ]]; then
  python3 - <<'PY'
import json,urllib.request
try:
    with urllib.request.urlopen('http://127.0.0.1:11435/api/synta-resources',timeout=3) as r: value=json.load(r)
except OSError:
    value={}
if value.get('generating') or value.get('cpu_active',0):
    print('Owned node busy; update deferred without cancelling jobs',flush=True)
    raise SystemExit(75)
PY
fi
mkdir -p "$BASE/releases/$REV" "$BASE/downloads" "$STATE/work" "$STATE/ollama" "$STATE/research"
python3 - "$BASE" <<'PY'
import shutil,sys
assert shutil.disk_usage(sys.argv[1]).free >= 16*2**30, 'Need 16 GiB free in the dedicated installation volume'
PY
apt-get update
apt-get install -y --no-install-recommends python3-venv curl ca-certificates zstd
SOURCE="$BASE/releases/$REV"
curl -fL --retry 3 "https://raw.githubusercontent.com/Mantiee/rlm/$REV/tools/managed-release.json" -o "$SOURCE/manifest.json"
python3 - "$REV" "$SOURCE" "$0" <<'PY'
import hashlib,json,sys,urllib.request
from pathlib import Path
revision,folder,script=sys.argv[1],Path(sys.argv[2]),Path(sys.argv[3])
m=json.loads((folder/'manifest.json').read_text())
assert m.get('node3060_installer')=='tools/bootstrap-synta-3060.sh'
assert hashlib.sha256(script.read_bytes()).hexdigest()==m['files']['tools/bootstrap-synta-3060.sh'], 'Installer bytes differ from pinned manifest'
for path in ('rlm/v100/owned_node.py','rlm/v100/owned_node_update.py','rlm/v100/release_selector.py','rlm/v100/compute_kernel.py','rlm/v100/compute_worker.py','rlm/v100/source_sync.py'):
    req=urllib.request.Request(f'https://raw.githubusercontent.com/Mantiee/rlm/{revision}/{path}',headers={'User-Agent':'SyntaOwned3060/1.0'})
    with urllib.request.urlopen(req,timeout=30) as r: data=r.read(2*2**20+1)
    assert len(data)<=2*2**20 and hashlib.sha256(data).hexdigest()==m['files'][path], 'Source checksum differs'
    (folder/Path(path).name).write_bytes(data)
PY
if ! id -u synta3060 >/dev/null 2>&1; then
  useradd --system --home-dir "$STATE/work" --shell /usr/sbin/nologin synta3060
fi
SUPPLEMENTAL=''
for GROUP in video render; do
  if getent group "$GROUP" >/dev/null; then SUPPLEMENTAL="$SUPPLEMENTAL $GROUP"; fi
done
if [[ ! -x "$BASE/venv/bin/python" ]]; then python3 -m venv "$BASE/venv"; fi
PY="$BASE/venv/bin/python"
"$PY" -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0'
"$PY" -m pip install --index-url https://pypi.org/simple 'safetensors==0.5.3' 'psutil==7.0.0'
"$PY" -m py_compile "$SOURCE/owned_node.py" "$SOURCE/owned_node_update.py" "$SOURCE/compute_kernel.py"
RUNTIME="$BASE/runtime/ollama-0.40.0"
ARCHIVE="$BASE/downloads/ollama-linux-amd64-0.40.0.tar.zst"
HASH=c94aa4156b3d13e64ebc2efe5ea53f015384c882be776e6695cfb37fb180d5ad
if [[ ! -x "$RUNTIME/bin/ollama" ]]; then
  curl -fL --retry 3 -C - 'https://github.com/ollama/ollama/releases/download/v0.40.0/ollama-linux-amd64.tar.zst' -o "$ARCHIVE"
  printf '%s  %s\n' "$HASH" "$ARCHIVE" | sha256sum -c -
  mkdir -p "$RUNTIME"
  tar --zstd -xf "$ARCHIVE" -C "$RUNTIME" --no-same-owner --no-same-permissions
fi
[[ -x "$RUNTIME/bin/ollama" ]] || { echo 'Pinned runtime lacks its executable' >&2; exit 1; }
"$PY" "$SOURCE/source_sync.py" --root "$STATE" --revision "$REV" || echo 'Source snapshot unavailable; fixed worker services still proceed.' >&2
chown synta3060:synta3060 "$STATE/work" "$STATE/ollama"
chmod 0750 "$STATE/work" "$STATE/ollama"
chmod -R go-w "$SOURCE" "$RUNTIME"
modprobe nvidia_uvm
cat > /etc/systemd/system/synta3060.slice <<'UNIT'
[Unit]
Description=Dedicated owned compute with OS RAM reserve
[Slice]
MemoryMax=90%
MemorySwapMax=0
CPUAccounting=yes
MemoryAccounting=yes
UNIT
cat > /etc/systemd/system/synta3060-ollama.service <<UNIT
[Unit]
Description=Isolated owned RTX 3060 Ti model server
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=0
[Service]
User=synta3060
Group=synta3060
SupplementaryGroups=$SUPPLEMENTAL
Slice=synta3060.slice
Environment=HOME=$STATE/ollama
Environment=OLLAMA_MODELS=$STATE/ollama/models
Environment=OLLAMA_HOST=127.0.0.1:11436
Environment=OLLAMA_NUM_PARALLEL=1
Environment=OLLAMA_MAX_LOADED_MODELS=1
Environment=OLLAMA_CONTEXT_LENGTH=32768
Environment=OLLAMA_FLASH_ATTENTION=1
Environment=OLLAMA_KV_CACHE_TYPE=q8_0
ExecStart=$RUNTIME/bin/ollama serve
Restart=always
RestartSec=10
NoNewPrivileges=yes
CapabilityBoundingSet=
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectControlGroups=yes
ProtectClock=yes
RestrictSUIDSGID=yes
LockPersonality=yes
UMask=0077
ReadWritePaths=$STATE/ollama
DevicePolicy=closed
DeviceAllow=char-nvidia* rw
RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX
MemoryMax=50%
MemorySwapMax=0
TasksMax=2048
[Install]
WantedBy=multi-user.target
UNIT
cat > /etc/systemd/system/synta3060-node.service <<UNIT
[Unit]
Description=Dedicated owned GPU/CPU endpoint, no remote shell
After=network-online.target synta3060-ollama.service
Wants=network-online.target synta3060-ollama.service
StartLimitIntervalSec=0
[Service]
User=synta3060
Group=synta3060
SupplementaryGroups=$SUPPLEMENTAL
Slice=synta3060.slice
WorkingDirectory=$STATE/work
Environment=HOME=$STATE/work
Environment=PYTHONDONTWRITEBYTECODE=1
ExecStart=$PY -u $BASE/current/owned_node.py --root $STATE/work --source $BASE/current --bind 192.168.0.9 --owner 192.168.0.68
Restart=always
RestartSec=10
NoNewPrivileges=yes
CapabilityBoundingSet=
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectControlGroups=yes
ProtectClock=yes
RestrictSUIDSGID=yes
LockPersonality=yes
UMask=0077
ReadWritePaths=$STATE/work
DevicePolicy=closed
DeviceAllow=char-nvidia* rw
RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX
MemoryMax=40%
MemorySwapMax=0
TasksMax=2048
[Install]
WantedBy=multi-user.target
UNIT
cat > /etc/systemd/system/synta3060-update.service <<UNIT
[Unit]
Description=Authorized GitHub updates for dedicated worker
After=network-online.target
Wants=network-online.target
[Service]
Type=oneshot
ExecStart=$PY -u $BASE/current/owned_node_update.py
Nice=10
TimeoutStartSec=3h
UNIT
cat > /etc/systemd/system/synta3060-update.timer <<'UNIT'
[Unit]
Description=Check shared Synta release after boot and every 15 minutes
[Timer]
OnBootSec=2min
OnUnitInactiveSec=15min
Persistent=yes
[Install]
WantedBy=timers.target
UNIT
PREVIOUS="$(readlink "$BASE/current" || true)"
restore_services() {
  local failure=${1:-$?}
  trap - ERR
  rm -f "$STATE/installing"
  if [[ -n "$PREVIOUS" && -d "$PREVIOUS" ]]; then
    ln -sfn "$PREVIOUS" "$BASE/current"
    systemctl restart synta3060-ollama.service synta3060-node.service || true
  fi
  echo 'Owned worker installation failed; inspect journal. No driver or unrelated service changed.' >&2
  exit "$failure"
}
trap restore_services ERR
if [[ -n "$PREVIOUS" ]]; then
  if "$PY" - <<'PY'
import json,urllib.request,urllib.error
req=urllib.request.Request('http://127.0.0.1:11435/api/synta-maintenance',data=b'{}',headers={'Content-Type':'application/json'})
try:
    with urllib.request.urlopen(req,timeout=3) as r:
        assert json.load(r)['state']=='drained'
except urllib.error.HTTPError as error:
    if error.code==503: raise SystemExit(75)
    raise
except urllib.error.URLError:
    # A stopped endpoint has no surviving node tasks; Ollama may still be generating.
    with urllib.request.urlopen('http://127.0.0.1:11436/api/ps',timeout=3) as r: value=json.load(r)
    if value.get('models'): raise SystemExit(75)
PY
  then
    :
  else
    FAILURE=$?
    if [[ "$FAILURE" -eq 75 ]]; then
      trap - ERR
      echo 'New owned jobs arrived; update deferred before restart.'
      exit 75
    fi
    restore_services "$FAILURE"
  fi
fi
touch "$STATE/installing"
ln -sfn "$SOURCE" "$BASE/current"
systemctl daemon-reload
systemctl enable --now synta3060-ollama.service synta3060-node.service synta3060-update.timer
systemctl restart synta3060-ollama.service synta3060-node.service
"$PY" - <<'PY'
import json,time,urllib.request
urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
for attempt in range(60):
    try:
        with urllib.request.urlopen('http://127.0.0.1:11436/api/version',timeout=2) as r: json.load(r)
        break
    except OSError:
        if attempt==59: raise
        time.sleep(1)
with urllib.request.urlopen('http://127.0.0.1:11436/api/tags',timeout=5) as r: existing={row['name'] for row in json.load(r)['models']}
for model in ('gemma4:e2b-it-q4_K_M','gemma4:e4b-it-q4_K_M'):
    if model in existing:
        print(json.dumps({'retained':model,'scope':'Existing digest retained; no tag repull'}),flush=True)
        continue
    data=json.dumps({'model':model,'stream':False}).encode()
    req=urllib.request.Request('http://127.0.0.1:11436/api/pull',data=data,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=3600) as r:
        result=json.load(r)
        if result.get('error'): raise RuntimeError(result['error'])
    print(json.dumps({'downloaded':model}),flush=True)
errors=[]
for model in ('gemma4:e2b-it-q4_K_M','gemma4:e4b-it-q4_K_M'):
    try:
        data=json.dumps({'model':model,'stream':False,'think':False,'messages':[{'role':'user','content':'Return the integer result of 2+2.'}],'options':{'num_ctx':32768,'num_predict':32},'keep_alive':-1}).encode()
        req=urllib.request.Request('http://127.0.0.1:11436/api/chat',data=data,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=180) as r:
            result=json.load(r)
            if result.get('error') or not result.get('done'): raise RuntimeError('Warmup did not complete')
        with urllib.request.urlopen('http://127.0.0.1:11436/api/ps',timeout=5) as r: loaded=json.load(r)['models']
        rows=[v for v in loaded if v.get('name')==model]
        assert len(rows)==1 and rows[0]['context_length']==32768 and 0<rows[0]['size_vram']<=8*2**30 and rows[0]['size_vram']>=rows[0]['size'], 'Full GPU residency was not observed'
        print(json.dumps({'gpu_verified':True,'model':model,'loaded':rows[0]}),flush=True)
        break
    except (OSError,ValueError,KeyError,AssertionError,RuntimeError) as error:
        errors.append(f'{model}: {error}')
else:
    raise SystemExit('Owned CPU service available but GPU qualification blocked: '+'; '.join(errors))
print('Dedicated GPU residency verified; controller still checks task quality before routing. No trained weights or income claimed.',flush=True)
PY
rm -f "$STATE/installing"
"$PY" - "$STATE" "$REV" <<'PY'
import json,os,sys,time
from pathlib import Path
path=Path(sys.argv[1])/'installation.json'
temporary=path.with_suffix('.new')
temporary.write_text(json.dumps({'stage':'prepared','revision':sys.argv[2],'updated':time.time()}))
os.replace(temporary,path)
PY
systemctl status synta3060-node.service --no-pager -l
