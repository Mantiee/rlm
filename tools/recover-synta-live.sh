#!/usr/bin/env bash
set -euo pipefail
REV="${1:?Pass exact published revision}"
[[ "$REV" =~ ^[0-9a-f]{40}$ ]] || exit 1
source "$HOME/ai-v100/env.sh"
PY="$AI_V100_ROOT/venvs/v100-continual/bin/python"
INSTALLER="$AI_V100_ROOT/update-synta-v87.sh"
curl -fL --retry 3 "https://raw.githubusercontent.com/Mantiee/rlm/$REV/tools/update-synta-v87.sh" -o "$INSTALLER"
"$PY" -c 'import rlm.v100.mission, rlm.v100.supervisor'
# Stop only the updater maintenance task; the accepted mission is resumed below if needed.
systemctl --user stop synta-managed-update.service || true
"$PY" <<'PY'
import json, os, subprocess
from pathlib import Path
from rlm.v100.common import load_profile
from rlm.v100.mission import status
from rlm.v100.supervisor import install
root=Path(os.environ['AI_V100_ROOT']).resolve()
current=status(root)
if current.get('running'):
    print('Accepted mission is running; no recovery restart requested',flush=True)
else:
    processes=subprocess.check_output(['ps','-eo','args='],text=True)
    if any('rlm.v100.cli' in line and any(' '+mode+' ' in line+' ' for mode in
        ('train','mission-prepare','optimize-mtp','calibrate-training','test-mtp')) for line in processes.splitlines()):
        raise SystemExit('Exclusive training remains active; no model process interrupted. Retry after its saved checkpoint.')
    paths=[]
    if current.get('learning',{}).get('live_profile'): paths.append(Path(current['learning']['live_profile']))
    if current.get('run'): paths.append(Path(current['run'])/'input-profile.json')
    native=root/'research/managed-update/native-qualification.json'
    if native.exists():
        value=json.loads(native.read_text())
        if value.get('profile'):paths.append(Path(value['profile']))
    backup=root/'research/availability-recovery-update/current.json'
    if backup.exists(): paths.append(Path(json.loads(backup.read_text())['profile']))
    selected=None
    for candidate in paths:
        candidate=candidate.resolve()
        if candidate.is_relative_to(root/'research') and candidate.is_file():
            load_profile(candidate,root)
            selected=candidate
            break
    if selected is None: raise SystemExit('No owned preserved profile available; no replacement model guessed')
    print(json.dumps(install(root,selected),indent=2),flush=True)
PY
bash "$INSTALLER" "$REV"
# Retire only the old standalone import units. The managed importer resumes
# their archived parts and committed index with the new checkpoint code.
for UNIT in synta-wikipedia-en.service synta-wikipedia-en-v2.service synta-wikipedia-en-v3.service; do
  if systemctl --user is-active --quiet "$UNIT"; then
    systemctl --user stop "$UNIT"
  fi
done
systemctl --user start --no-block synta-managed-update.service
"$AI_V100_ROOT/bin/v100-continual" mission-status
