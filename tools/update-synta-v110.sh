#!/usr/bin/env bash
# Install the v110 managed runtime; preserve profiles, weights and resource policy.
set -Eeuo pipefail
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
exec 9>"$AI_V100_ROOT/research/mission-maintenance.lock"
flock -n -x 9 || { echo "Another owned maintenance task is active; update deferred."; exit 75; }
for COMMAND in git curl systemctl flock sed; do
  command -v "$COMMAND" >/dev/null || { echo "Required executable unavailable: $COMMAND; no service changed." >&2; exit 127; }
done
[[ -x "$PY" ]] || { echo 'Existing Synta Python unavailable; no service changed.' >&2; exit 127; }
DASHBOARD_EXEC="$(systemctl --user show v100-dashboard.service -p ExecStart --value)"
DASHBOARD_BIND="$(printf '%s' "$DASHBOARD_EXEC" | sed -nE 's/.*--bind[[:space:]]+([^ ;]+).*/\1/p')"
[[ "$DASHBOARD_BIND" =~ ^[A-Za-z0-9.:%_-]+$ && "$DASHBOARD_BIND" != 0.0.0.0 && "$DASHBOARD_BIND" != :: && "$DASHBOARD_BIND" != 127.* && "$DASHBOARD_BIND" != ::1 && "$DASHBOARD_BIND" != localhost ]] || {
  echo 'Could not safely derive the dashboard bind address; no service changed.' >&2
  exit 1
}
if [[ "$DASHBOARD_BIND" == *:* ]]; then DASHBOARD_BIND="[$DASHBOARD_BIND]"; fi
RESTORE_MARKER="$(mktemp "$AI_V100_ROOT/research/.update-v110-restore.XXXXXX")"
export SYNTA_RESTORE_MARKER="$RESTORE_MARKER"
PACKAGE_CHANGED=0
MERGED_DASHBOARD_CHANGED=0
MERGED_DASHBOARD_DROPIN="$HOME/.config/systemd/user/synta-income-learning.service.d/zz-merged-dashboard.conf"
PACKAGE_BACKUP="$AI_V100_ROOT/research/v110-package-recovery/$(date +%s%N)/package"
RECOVERY_HELPER="$(dirname "$0")/../rlm/v100/package_recovery.py"
export SYNTA_PACKAGE_BACKUP="$PACKAGE_BACKUP" SYNTA_RECOVERY_HELPER="$RECOVERY_HELPER"
[[ -f "$RECOVERY_HELPER" ]] || { echo 'Verified package recovery helper missing; no service changed.' >&2; exit 1; }
restore_after_failure() {
  local failure=$?
  trap - EXIT
  echo "Update failed with exit $failure. Checking preserved mission recovery." >&2
  if [[ "$MERGED_DASHBOARD_CHANGED" -eq 1 ]]; then
    rm -f "$MERGED_DASHBOARD_DROPIN"
    systemctl --user daemon-reload || true
    systemctl --user restart synta-income-learning.service || true
  fi
  if [[ "$PACKAGE_CHANGED" -eq 1 ]]; then
    if ! systemctl --user stop v100-mission.service; then
      echo 'Cannot stop the new runtime safely; package rollback deferred.' >&2
      exit "$failure"
    fi
    if ! "$PY" "$RECOVERY_HELPER" restore --root "$AI_V100_ROOT" --destination "$PACKAGE_BACKUP"; then
      echo 'Package rollback failed. Mission remains stopped; inspect preserved snapshot.' >&2
      exit "$failure"
    fi
    systemctl --user try-restart v100-dashboard.service || true
  fi
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
trap restore_after_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

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
if installed not in tuple(f'0.1.3+v100.{number}' for number in range(78, 111)):
    raise SystemExit(f'Targeted goal work update requires v78-v110; installed: {installed}')
service = subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'v100-mission.service'])
if service.returncode != 0:
    raise SystemExit('Mission service is not active; no service changed')
root = Path(os.environ['AI_V100_ROOT'])
record = status(root)
learning=record.get('learning', {})
cycle, completed = learning.get('cycle_in_progress'), learning.get('completed_cycles')
training_stages = {
    'candidate-training-and-independent-tests',
    'scratch-architecture-training-and-tests',
    'alternative-architecture-training-and-tests',
}
if (
    record.get('running') is True
    and type(cycle) is int
    and type(completed) is int
    and cycle > completed
    and learning.get('stage') in training_stages
):
    raise SystemExit(f'Active training cycle {cycle} retained; package update deferred')
if not record.get('run'):
    raise SystemExit('No mission profile to preserve; no service changed')
for line in subprocess.check_output(['ps', '-eo', 'pid=,args='], text=True).splitlines():
    if 'rlm.v100.cli' in line and any(' '+mode+' ' in line+' ' for mode in
        ('train', 'mission-prepare', 'optimize-mtp', 'calibrate-training', 'test-mtp')):
        raise SystemExit('Exclusive experiment still runs; no service changed. '+line)
from rlm.v100.common import load_profile
paths=[]
if record.get('learning', {}).get('live_profile'):
    paths.append(Path(record['learning']['live_profile']))
paths.append(Path(record['run']) / 'input-profile.json')
checkpoint=root/'research/availability-recovery-update/current.json'
if checkpoint.is_file(): paths.append(Path(json.loads(checkpoint.read_text())['profile']))
source=None
for candidate in paths:
    candidate=candidate.resolve()
    if candidate.is_relative_to(root.resolve()/'research') and candidate.is_file():
        try: load_profile(candidate,root)
        except (OSError,ValueError,KeyError): continue
        source=candidate
        break
if source is None: raise SystemExit('No valid preserved profile; no service changed')
profile = load_profile(source, root)
folder = root / 'research/availability-recovery-update' / str(time.time_ns())
folder.mkdir(parents=True)
atomic_json(folder / 'before.json', record)
atomic_json(folder / 'profile.json', profile)
atomic_json(root / 'research/availability-recovery-update/current.json', {'profile': str(folder / 'profile.json')})
print('Preserved profile:', folder / 'profile.json', flush=True)
Path(os.environ['SYNTA_RESTORE_MARKER']).write_text(str(folder / 'profile.json'))
subprocess.run([str(root/'venvs/v100-continual/bin/python'), os.environ['SYNTA_RECOVERY_HELPER'], 'snapshot', '--root', str(root), '--destination', os.environ['SYNTA_PACKAGE_BACKUP']], check=True)
print(stop(root), flush=True)
subprocess.run(['systemctl', '--user', 'stop', 'v100-mission.service'], check=True, timeout=90)
if status(root)['running']:
    raise SystemExit('Mission still running; package not changed')
PY

export UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=5
PACKAGE_CHANGED=1
"$UV" --no-config pip install --python "$PY" --no-deps --reinstall-package rlms \
  "rlms @ git+https://github.com/Mantiee/rlm.git@$REV"
"$UV" --no-config pip check --python "$PY"
"$PY" -m py_compile "$(dirname "$RECOVERY_HELPER")/dashboard_redirect.py"
mkdir -p "$(dirname "$MERGED_DASHBOARD_DROPIN")"
MERGED_DASHBOARD_TMP="${MERGED_DASHBOARD_DROPIN}.new"
cat > "$MERGED_DASHBOARD_TMP" <<EOF
[Service]
ExecStart=
ExecStart=$PY -m rlm.v100.dashboard_redirect --host 0.0.0.0 --port 8766 --target http://$DASHBOARD_BIND:8765/?v=110
ReadWritePaths=
EOF
chmod 600 "$MERGED_DASHBOARD_TMP"
mv "$MERGED_DASHBOARD_TMP" "$MERGED_DASHBOARD_DROPIN"
MERGED_DASHBOARD_CHANGED=1
systemctl --user daemon-reload
systemctl --user restart synta-income-learning.service
systemctl --user is-active --quiet synta-income-learning.service
"$PY" <<'PY'
import json
import os
from importlib.metadata import version
from pathlib import Path
from rlm.v100.supervisor import install
from rlm.v100.mission import status
import time

root = Path(os.environ['AI_V100_ROOT'])
if version('rlms') != '0.1.3+v100.110':
    raise SystemExit('Pinned revision is not the v110 managed runtime update; mission remains stopped')
path = Path(json.loads((root / 'research/availability-recovery-update/current.json').read_text())['profile'])
from rlm.v100.common import atomic_json, load_profile
profile=load_profile(path,root)
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
systemctl --user restart v100-dashboard.service
systemctl --user is-active --quiet v100-dashboard.service
echo 'v110 installed: preserved-profile mission started. Verify observer, checkpoints and live receipts separately.'
MERGED_DASHBOARD_CHANGED=0
trap - EXIT
rm -f "$RESTORE_MARKER"
flock -u 9
