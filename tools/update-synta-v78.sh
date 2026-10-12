#!/usr/bin/env bash
# Repair dashboard collection and health checks without interrupting model work.
set -euo pipefail
REV="${1:?Pass the pinned 40-character commit}"
[[ "$REV" =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid revision' >&2; exit 1; }
source "$HOME/ai-v100/env.sh"
PY="$AI_V100_ROOT/venvs/v100-continual/bin/python"
"$PY" <<'PY'
from importlib.metadata import version
if version('rlms') not in ('0.1.3+v100.77', '0.1.3+v100.78'):
    raise SystemExit('Targeted dashboard recovery requires installed v77-v78; no service changed')
PY
FOLDER="$AI_V100_ROOT/research/dashboard-recovery/$(date +%s%N)"
mkdir -p "$FOLDER"
cp "$AI_V100_ROOT/bin/serve-v100-dashboard.py" "$FOLDER/serve-v100-dashboard.before.py"
curl -fL --retry 3 "https://raw.githubusercontent.com/Mantiee/rlm/$REV/tools/serve-v100-dashboard.py" -o "$FOLDER/serve-v100-dashboard.py"
"$PY" -m py_compile "$FOLDER/serve-v100-dashboard.py"
export UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=5
uv --no-config pip install --python "$PY" --no-deps --reinstall-package rlms \
  "rlms @ git+https://github.com/Mantiee/rlm.git@$REV"
uv --no-config pip check --python "$PY"
"$PY" <<'PY'
from importlib.metadata import version
if version('rlms') != '0.1.3+v100.78':
    raise SystemExit('Pinned revision is not v78; dashboard launcher not replaced')
PY
cp "$FOLDER/serve-v100-dashboard.py" "$AI_V100_ROOT/bin/serve-v100-dashboard.py.new"
mv "$AI_V100_ROOT/bin/serve-v100-dashboard.py.new" "$AI_V100_ROOT/bin/serve-v100-dashboard.py"
systemctl --user restart v100-dashboard.service
systemctl --user try-restart synta-remote.service
if ! "$PY" <<'PY'
import json
import os
from pathlib import Path
from rlm.v100.dashboard_health import wait
print(json.dumps(wait(60, root=Path(os.environ['AI_V100_ROOT'])), indent=2))
PY
then
  systemctl --user status v100-dashboard.service --no-pager -l || true
  journalctl --user -u v100-dashboard.service -n 30 --no-pager || true
  exit 1
fi
echo 'Dashboard renderer and fresh collector verified at the configured private address. Reload browser once.'
echo 'Mission and training were not restarted. Inference benchmark fixed but not rerun; existing inference settings retained.'
