#!/usr/bin/env bash
# Replace repeated startup artifact reads; retain the current mission configuration.
set -euo pipefail
REV="${1:?Pass the pinned 40-character commit}"
[[ "$REV" =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid revision' >&2; exit 1; }
source "$HOME/ai-v100/env.sh"
PY="$AI_V100_ROOT/venvs/v100-continual/bin/python"
export SYNTA_STARTUP_REV="$REV"

"$PY" <<'PY'
import os
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
from rlm.v100.common import atomic_json, load_profile
from rlm.v100.mission import status, stop

installed = version('rlms')
if installed not in tuple(f'0.1.3+v100.{number}' for number in range(67, 71)):
    raise SystemExit(f'Targeted startup update requires v67-v70; installed: {installed}')
subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'v100-mission.service'], check=True)
root = Path(os.environ['AI_V100_ROOT'])
record = status(root)
if not record.get('run'):
    raise SystemExit('No mission profile to preserve; no service changed')
for line in subprocess.check_output(['ps', '-eo', 'pid=,args='], text=True).splitlines():
    if 'rlm.v100.cli' in line and any(' '+mode+' ' in line+' ' for mode in
        ('train', 'mission-prepare', 'optimize-mtp', 'calibrate-training', 'test-mtp')):
        raise SystemExit('Exclusive experiment still runs; no service changed. '+line)
source = Path(record['learning']['live_profile']) if record.get('learning', {}).get('live_profile') else Path(record['run']) / 'input-profile.json'
profile = load_profile(source, root)
folder = root / 'research/startup-update' / str(time.time_ns())
folder.mkdir(parents=True)
atomic_json(folder / 'before.json', record)
atomic_json(folder / 'profile.json', profile)
atomic_json(root / 'research/startup-update/current.json', {'profile': str(folder / 'profile.json')})
print('Preserved profile:', folder / 'profile.json', flush=True)
print(stop(root), flush=True)
subprocess.run(['systemctl', '--user', 'stop', 'v100-mission.service'], check=True, timeout=90)
if status(root)['running']:
    raise SystemExit('Mission still running; package not changed')
PY

export UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=5
uv --no-config pip install --python "$PY" --no-deps --reinstall-package rlms \
  "rlms @ git+https://github.com/Mantiee/rlm.git@$REV"
uv --no-config pip check --python "$PY"
"$PY" <<'PY'
import json
import os
from importlib.metadata import version
from pathlib import Path
from rlm.v100.supervisor import install

root = Path(os.environ['AI_V100_ROOT'])
if version('rlms') != '0.1.3+v100.70':
    raise SystemExit('Pinned revision is not the v70 startup fix; mission remains stopped')
path = Path(json.loads((root / 'research/startup-update/current.json').read_text())['profile'])
print(json.dumps(install(root, path), indent=2), flush=True)
PY
"$AI_V100_ROOT/bin/v100-continual" mission-status
systemctl --user try-restart v100-dashboard.service
echo 'Startup verification updated. Current goals, weights, Windows policy and archived runs retained.'
