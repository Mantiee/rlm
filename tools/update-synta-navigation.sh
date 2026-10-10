#!/usr/bin/env bash
# v67/v68 -> v69: production changes are limited to the remote gateway.
set -euo pipefail
REV="${1:?Pass the pinned 40-character v69 commit}"
[[ "$REV" =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid revision' >&2; exit 1; }
source "$HOME/ai-v100/env.sh"
PY="$AI_V100_ROOT/venvs/v100-continual/bin/python"
"$PY" <<'PY'
from importlib.metadata import version

installed = version('rlms')
if installed not in ('0.1.3+v100.67', '0.1.3+v100.68', '0.1.3+v100.69'):
    raise SystemExit(f'Targeted navigation update requires v67, v68 or v69; installed: {installed}. Use the full retention upgrader for older versions.')
PY
systemctl --user is-active --quiet synta-remote.service
export UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=5
uv --no-config pip install --python "$PY" --no-deps --reinstall-package rlms \
  "rlms @ git+https://github.com/Mantiee/rlm.git@$REV"
uv --no-config pip check --python "$PY"
systemctl --user restart synta-remote.service
systemctl --user is-active --quiet synta-remote.service
"$PY" <<'PY'
import json
import os
from pathlib import Path
from rlm.v100.remote_access import verify_navigation

print(json.dumps(verify_navigation(Path(os.environ['AI_V100_ROOT'])), indent=2))
PY
echo 'Remote gateway updated. Mission, dashboard, Tailscale routing and Windows tasks were not restarted.'
echo 'Reload the Tailscale HTTPS dashboard to see Dashboard / Master chat navigation.'
