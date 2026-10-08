#!/usr/bin/env bash
# Pinned isolated update; do not interrupt an active calibration or speed sweep.
set -euo pipefail
REV="${1:?Pass the published 40-character release commit}"
[[ "$REV" =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid revision' >&2; exit 1; }
shift
source "$HOME/ai-v100/env.sh"
ROOT="$AI_V100_ROOT"
PY="$ROOT/venvs/v100-continual/bin/python"
export UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=5
export HF_HUB_DOWNLOAD_TIMEOUT=600 HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DISABLE_XET=1
mkdir -p "$ROOT/research/logs" "$ROOT/bin"
exec > >(tee "$ROOT/research/logs/upgrade-campaign-$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1
"$PY" <<'PY'
import os, subprocess
from pathlib import Path
root = Path(os.environ['AI_V100_ROOT'])
for line in subprocess.check_output(['ps','-eo','pid=,args='], text=True).splitlines():
    if 'rlm.v100.cli' in line and any(' '+mode+' ' in line+' ' for mode in ('mission-prepare','train','optimize-mtp','calibrate-training','test-mtp')):
        raise RuntimeError('An exclusive experiment still runs. Finish it or Ctrl+C in its existing console first. No files changed: '+line)
PY
# This bootstrap also works with older packages that count zombies as running.
"$PY" <<'PY_STOP'
import os, time
from pathlib import Path
from rlm.v100.common import atomic_json
from rlm.v100.mission import status, stop
root = Path(os.environ['AI_V100_ROOT'])
atomic_json(root / 'research/supervisor/pause.json', {'reason': 'operator requested upgrade'})
def live(record):
    if not record.get('running'):
        return False
    try:
        fields = Path(f"/proc/{record['pid']}/stat").read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return False
    if fields[19] != record['process_start']:
        return False
    return fields[0] not in {'Z', 'X', 'x'}
record = status(root)
if live(record):
    try:
        print(stop(root), flush=True)
    except (ValueError, ProcessLookupError, FileNotFoundError):
        if live(status(root)):
            raise
else:
    print('Owned mission has exited; no process was signalled. Checkpoints retained.', flush=True)
deadline = time.monotonic() + 180
while live(status(root)):
    if time.monotonic() >= deadline:
        raise RuntimeError('Mission still stopping. No packages changed.')
    time.sleep(2)
PY_STOP
uv --no-config pip install --python "$PY" --no-deps --reinstall-package rlms \
  "rlms @ git+https://github.com/Mantiee/rlm.git@$REV"
uv --no-config pip check --python "$PY"
uv --no-config pip freeze --python "$PY" > "$ROOT/research/requirements.continual.v10035.txt"
LAB="$ROOT/bin/v100-continual"
if [[ -e "$LAB" ]]; then cp -p "$LAB" "$LAB.backup-$(date -u +%Y%m%dT%H%M%SZ)"; fi
TMP="$(mktemp "$ROOT/bin/.v100-continual.XXXXXX")"
trap 'rm -f "$TMP"' EXIT
cat > "$TMP" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
source "$HOME/ai-v100/env.sh"
exec "$AI_V100_ROOT/venvs/v100-continual/bin/python" -u -m rlm.v100.cli --root "$AI_V100_ROOT" "$@"
SH
bash -n "$TMP"
chmod 755 "$TMP"
mv "$TMP" "$LAB"
"$LAB" --help >/dev/null
if [[ -d /srv/samba/dane/tests ]]; then
  "$LAB" compute-configure --mailbox /srv/samba/dane/tests/v100-owned-compute
fi
"$LAB" campaign-prepare "$@"
PROFILE="$ROOT/research/campaign/current.json"
ACCEPT_PROFILE="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["profile"])' "$PROFILE")"
"$LAB" --profile "$ACCEPT_PROFILE" hardware-acceptance
"$LAB" supervisor-start
"$LAB" mission-status
printf '\nGotowe. Jedna konsola: ~/ai-v100/bin/v100-continual chat\nRaport: ~/ai-v100/bin/v100-continual mission-report\nLogi: ~/ai-v100/bin/v100-continual mission-watch\nCtrl+C w czacie/logach nie zatrzymuje misji.\n'
