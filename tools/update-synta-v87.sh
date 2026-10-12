#!/usr/bin/env bash
# Install continuous goal work; preserve configuration.
set -euo pipefail
REV="${1:?Pass the pinned 40-character commit}"
[[ "$REV" =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid revision' >&2; exit 1; }
source "$HOME/ai-v100/env.sh"
PY="$AI_V100_ROOT/venvs/v100-continual/bin/python"
export SYNTA_GOAL_WORK_REV="$REV"
export PATH="$HOME/.local/bin:$PATH"
UV="$(command -v uv || true)"
if [[ -z "$UV" || ! -x "$UV" ]]; then
  echo 'uv executable unavailable; preflight failed before any service change.' >&2
  exit 127
fi
"$UV" --version
for COMMAND in git curl systemctl; do
  command -v "$COMMAND" >/dev/null || { echo "Required executable unavailable: $COMMAND; no service changed." >&2; exit 127; }
done
[[ -x "$PY" ]] || { echo 'Existing Synta Python unavailable; no service changed.' >&2; exit 127; }
RESTORE_MARKER="$(mktemp "$AI_V100_ROOT/research/.update-v87-restore.XXXXXX")"
export SYNTA_RESTORE_MARKER="$RESTORE_MARKER"
restore_after_failure() {
  local failure=$?
  trap - ERR
  echo "Update failed with exit $failure. Checking preserved mission recovery." >&2
  if [[ -s "$RESTORE_MARKER" ]]; then
    if ! "$PY" <<'RECOVER'
import json
import os
from pathlib import Path
from rlm.v100.mission import status
from rlm.v100.supervisor import install
root = Path(os.environ['AI_V100_ROOT']).resolve()
profile = Path(Path(os.environ['SYNTA_RESTORE_MARKER']).read_text().strip()).resolve()
if not profile.is_relative_to(root / 'research') or not profile.is_file():
    raise SystemExit('Preserved recovery profile unavailable; inspect update journal')
if status(root).get('running'):
    print('Mission is already running; recovery did not restart it', flush=True)
else:
    print(json.dumps(install(root, profile), indent=2), flush=True)
    print('Preserved profile submitted to supervisor. Verify mission-status.', flush=True)
RECOVER
    then
      echo 'Automatic mission recovery failed; inspect journal. Package rollback was not claimed.' >&2
    fi
  fi
  rm -f "$RESTORE_MARKER"
  exit "$failure"
}
trap restore_after_failure ERR

"$PY" <<'PY'
import json
import os
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
from rlm.v100.common import atomic_json, load_profile
from rlm.v100.mission import status, stop

installed = version('rlms')
if installed not in tuple(f'0.1.3+v100.{number}' for number in range(78, 88)):
    raise SystemExit(f'Targeted goal work update requires v78-v87; installed: {installed}')
service = subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'v100-mission.service'])
root = Path(os.environ['AI_V100_ROOT'])
record = status(root)
if not record.get('run'):
    raise SystemExit('No mission profile to preserve; no service changed')
for line in subprocess.check_output(['ps', '-eo', 'pid=,args='], text=True).splitlines():
    if 'rlm.v100.cli' in line and any(' '+mode+' ' in line+' ' for mode in
        ('train', 'mission-prepare', 'optimize-mtp', 'calibrate-training', 'test-mtp')):
        raise SystemExit('Exclusive experiment still runs; no service changed. '+line)
if service.returncode == 0 and record.get('running'):
    source = Path(record['learning']['live_profile']) if record.get('learning', {}).get('live_profile') else Path(record['run']) / 'input-profile.json'
else:
    checkpoint = root / 'research/availability-recovery-update/current.json'
    if not checkpoint.is_file():
        raise SystemExit('Stopped mission has no preserved updater profile; no service changed')
    source = Path(json.loads(checkpoint.read_text())['profile']).resolve()
    if not source.is_relative_to(root.resolve() / 'research/availability-recovery-update'):
        raise SystemExit('Preserved profile escaped updater folder; no service changed')
    print('Recovering previously stopped update from:', source, flush=True)
profile = load_profile(source, root)
folder = root / 'research/availability-recovery-update' / str(time.time_ns())
folder.mkdir(parents=True)
atomic_json(folder / 'before.json', record)
atomic_json(folder / 'profile.json', profile)
atomic_json(root / 'research/availability-recovery-update/current.json', {'profile': str(folder / 'profile.json')})
print('Preserved profile:', folder / 'profile.json', flush=True)
Path(os.environ['SYNTA_RESTORE_MARKER']).write_text(str(folder / 'profile.json'))
print(stop(root), flush=True)
subprocess.run(['systemctl', '--user', 'stop', 'v100-mission.service'], check=True, timeout=90)
if status(root)['running']:
    raise SystemExit('Mission still running; package not changed')
PY

export UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=5
"$UV" --no-config pip install --python "$PY" --no-deps --reinstall-package rlms \
  "rlms @ git+https://github.com/Mantiee/rlm.git@$REV"
"$UV" --no-config pip check --python "$PY"
"$PY" <<'PY'
import json
import os
from importlib.metadata import version
from pathlib import Path
from rlm.v100.supervisor import install
from rlm.v100.mission import status
import time

root = Path(os.environ['AI_V100_ROOT'])
if version('rlms') != '0.1.3+v100.87':
    raise SystemExit('Pinned revision is not the v87 updater recovery update; mission remains stopped')
path = Path(json.loads((root / 'research/availability-recovery-update/current.json').read_text())['profile'])
print(json.dumps(install(root, path), indent=2), flush=True)
deadline = time.monotonic() + 30
while time.monotonic() < deadline:
    record = status(root)
    if record.get('running'):
        print(json.dumps(record, indent=2), flush=True)
        break
    time.sleep(0.5)
else:
    raise SystemExit('Supervisor installed, but mission did not start within 30 seconds; inspect v100-mission.service journal')
PY
curl -fL --retry 3 "https://raw.githubusercontent.com/Mantiee/rlm/$REV/tools/serve-v100-dashboard.py" -o "$AI_V100_ROOT/bin/serve-v100-dashboard.py.new"
"$PY" -m py_compile "$AI_V100_ROOT/bin/serve-v100-dashboard.py.new"
mv "$AI_V100_ROOT/bin/serve-v100-dashboard.py.new" "$AI_V100_ROOT/bin/serve-v100-dashboard.py"
systemctl --user restart v100-dashboard.service
systemctl --user try-restart synta-remote.service
systemctl --user try-restart synta-owned-nodes.service
if ! "$PY" <<'PY'
import json
from rlm.v100.dashboard_health import wait
import os
from pathlib import Path
print(json.dumps(wait(60, root=Path(os.environ['AI_V100_ROOT'])), indent=2))
PY
then
  systemctl --user status v100-dashboard.service --no-pager -l || true
  journalctl --user -u v100-dashboard.service -n 30 --no-pager || true
  exit 1
fi
echo 'v87 installed: mission restarted with its preserved profile. Goals, weights, policy, balances and archives retained. Reload dashboard once.'

curl -fL --retry 3 "https://raw.githubusercontent.com/Mantiee/rlm/$REV/tools/qualify-synta-models-debian.sh" -o "$AI_V100_ROOT/qualify-synta-models-debian.sh"

trap - ERR
rm -f "$RESTORE_MARKER"
