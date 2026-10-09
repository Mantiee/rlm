#!/usr/bin/env bash
# Update the existing mission setup only; do not rerun calibration/benchmarks.
set -euo pipefail
REV="${1:?Pass the pinned 40-character commit}"
[[ "$REV" =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid revision' >&2; exit 1; }
source "$HOME/ai-v100/env.sh"
PY="$AI_V100_ROOT/venvs/v100-continual/bin/python"
export UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=5
export RETENTION_REV="$REV"

"$PY" <<'PY'
import os, subprocess, time
from pathlib import Path
from rlm.v100.common import atomic_json, load_profile
from rlm.v100.mission import status, stop

root = Path(os.environ['AI_V100_ROOT'])
for line in subprocess.check_output(['ps', '-eo', 'pid=,args='], text=True).splitlines():
    if 'rlm.v100.cli' in line and any(' '+mode+' ' in line+' ' for mode in
        ('train', 'mission-prepare', 'optimize-mtp', 'calibrate-training', 'test-mtp')):
        raise RuntimeError('Exclusive experiment still runs; no files changed. '+line)
record = status(root)
source = Path(record['learning']['live_profile']) if record.get('learning', {}).get('live_profile') else (
    Path(record['run']) / 'input-profile.json' if record.get('run') else None)
if source is None or not source.exists():
    import json
    source = Path(json.loads((root / 'research/campaign/current.json').read_text())['profile'])
profile = load_profile(source, root)
folder = root / 'research/retention-setup' / str(time.time_ns())
folder.mkdir(parents=True)
atomic_json(folder / 'previous-profile.json', profile)
profile.setdefault('resources', {})['retention_experiments'] = True
atomic_json(folder / 'profile.json', profile)
atomic_json(root / 'research/retention-setup/current.json', {'profile': str(folder / 'profile.json')})
atomic_json(root / 'research/supervisor/pause.json', {'reason': 'operator retention upgrade'})
if record['running']:
    print(stop(root), flush=True)
deadline = time.monotonic() + 180
while status(root)['running']:
    if time.monotonic() >= deadline:
        raise RuntimeError('Mission still stopping; package not changed. Checkpoints retained.')
    time.sleep(2)
subprocess.run(['systemctl', '--user', 'stop', 'v100-mission.service'], check=False)
print('Previous profile and checkpoints retained.', flush=True)
PY

uv --no-config pip install --python "$PY" --no-deps --reinstall-package rlms \
  "rlms @ git+https://github.com/Mantiee/rlm.git@$REV"
uv --no-config pip check --python "$PY"
uv --no-config pip freeze --python "$PY" > "$AI_V100_ROOT/research/requirements.continual.v10051.txt"

"$PY" <<'PY'
import json, os
from pathlib import Path
from rlm.v100.common import atomic_json
from rlm.v100.self_code import prepare
from rlm.v100.desktop import prepare as prepare_desktop
from rlm.v100.supervisor import install
from rlm.v100.chat_resources import repair_completed

root = Path(os.environ['AI_V100_ROOT'])
prepare(root)
if (root / 'research/desktop/manifest.json').exists():
    prepare_desktop(root)
path = Path(json.loads((root / 'research/retention-setup/current.json').read_text())['profile'])
profile = json.loads(path.read_text())
preferences_path = root / 'research/user-preferences.json'
preferences = json.loads(preferences_path.read_text()) if preferences_path.exists() else {'directive': '', 'alerts': False}
# Explicit operator request: display reasoning returned by owned local models.
preferences['system_name'] = 'Synta'
preferences['capture_local_model_trace'] = True
preferences['stream_local_model_trace'] = True
preferences['retention'] = {
    'research': 'RETENTION_RESEARCH.md in the readonly own-source mount',
    'experiments': 'Bounded replay/KL, L2, empirical diagonal Fisher EWC, delta-A orthogonality, A-GEM and standard LoRA rank growth are exposed to A/B planning. Historical modes require prior verified TRAINING references. Frozen-column/embedding growth primitives are tiny-model experiments, not serving Gemma modifications.',
    'scope': 'Do not change the operator long-term goal. Prefer measured learning/retention/time trade-offs; never bypass ancestor, official or fresh audit gates.'
}
preferences['goal_learning'] = {
    'instructions': 'Read get_plan and follow user directions or discover goal-relevant hypotheses yourself in any domain. Use observe_goal_source to archive fresh signals and a numeric public outcome, then predict_goal_pattern to precommit probability, threshold and horizon. The resident CPU/network observer checks future outcomes; both successes and failures become host-verified training candidates. Goal-linked development gates require sufficient independent validation groups. A goal score is not causal or profit proof.',
    'evidence': 'research/goal-learning/ledger.sqlite3 and mission_evidence.goal_learning',
    'scope': 'The operator alone authorizes long-term goal changes; no fixed asset, source or pattern list.'
}
atomic_json(preferences_path, preferences)
print('CHAT RESOURCE RECOVERY:', json.dumps(repair_completed(root), ensure_ascii=False), flush=True)
print(json.dumps(install(root, path), indent=2), flush=True)
print('RETENTION AND GOAL-LINKED FORECAST LEARNING AVAILABLE. Existing setup preserved; no calibration sweep.', flush=True)
PY
"$AI_V100_ROOT/bin/v100-continual" mission-status
# Install the viewer from the SAME revision, retaining guest edits and publication receipts.
VIEWER_SETUP="$AI_V100_ROOT/bin/update-dashboard-$REV.sh"
curl -fL --retry 3 "https://raw.githubusercontent.com/Mantiee/rlm/$REV/tools/start-v100-dashboard.sh" -o "$VIEWER_SETUP"
bash "$VIEWER_SETUP" "$REV"
"$PY" <<'PY'
import os, time
from pathlib import Path
from rlm.v100.dashboard_editor import repair, status
root = Path(os.environ['AI_V100_ROOT'])
for attempt in range(6):
    try:
        print('DASHBOARD LAYOUT:', repair(root), flush=True)
        print('PUBLICATION RECEIPT:', status(root), flush=True)
        break
    except (OSError, ValueError, RuntimeError) as error:
        if attempt == 5:
            print('Dashboard guest edit deferred:', str(error), flush=True)
        else:
            time.sleep(5)
PY
