#!/usr/bin/env bash
# Use only the existing isolated installation; no driver or system service changes.
set -euo pipefail
REV="${1:?Pass the published 40-character release commit}"
if [[ ! "$REV" =~ ^[0-9a-f]{40}$ ]]; then
    echo 'Invalid release revision' >&2
    exit 1
fi
shift
source "$HOME/ai-v100/env.sh"
ROOT="$AI_V100_ROOT"
PY="$ROOT/venvs/v100-continual/bin/python"
LAB="$ROOT/bin/v100-continual"
export UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=5
export HF_HUB_DOWNLOAD_TIMEOUT=600 HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DISABLE_XET=1
mkdir -p "$ROOT/research/logs"
exec > >(tee "$ROOT/research/logs/upgrade-v10022-$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1
"$LAB" mission-stop
"$PY" <<'PY'
import os, time
from pathlib import Path
from rlm.v100.mission import status
root = Path(os.environ['AI_V100_ROOT'])
deadline = time.monotonic() + 180
while status(root)['running']:
    if time.monotonic() >= deadline:
        raise RuntimeError('Mission still stopping. No packages changed; inspect mission-status.')
    time.sleep(2)
print('Owned mission stopped; checkpoints retained.', flush=True)
PY
uv --no-config pip install --python "$PY" --no-deps --reinstall-package rlms \
    "rlms @ git+https://github.com/Mantiee/rlm.git@$REV"
uv --no-config pip check --python "$PY"
uv --no-config pip freeze --python "$PY" > "$ROOT/research/requirements.continual.v10022.txt"
"$LAB" mission-prepare "$@"
"$PY" <<'PY'
import json, os
from pathlib import Path
from rlm.v100.common import atomic_json, load_profile
from rlm.v100.mission import start
root = Path(os.environ['AI_V100_ROOT'])
receipt = json.loads((root/'research/prepared-mission.json').read_text())
profile = load_profile(Path(receipt['profile']), root)
profile.setdefault('resources', {})['mission_max_context'] = 131072
profile['server']['flash_attention'] = 'on'
path = root/'research/preparation/start-profile.json'
atomic_json(path, profile)
print(json.dumps(start(root, path, max_context=131072, flash_attention='on'), indent=2), flush=True)
PY
"$LAB" mission-status
printf '\nFollow progress: ~/ai-v100/bin/v100-continual mission-watch\nChat: ~/ai-v100/bin/v100-continual chat\nReport: ~/ai-v100/bin/v100-continual mission-report\n'
