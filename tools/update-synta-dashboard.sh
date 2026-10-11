#!/usr/bin/env bash
# Update only the host dashboard/gateway; keep the mission and its weights running.
set -euo pipefail
REV="${1:?Pass the pinned 40-character v76 commit}"
[[ "$REV" =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid revision' >&2; exit 1; }
source "$HOME/ai-v100/env.sh"
PY="$AI_V100_ROOT/venvs/v100-continual/bin/python"
"$PY" <<'PY'
from importlib.metadata import version
if version('rlms') not in ('0.1.3+v100.74', '0.1.3+v100.75', '0.1.3+v100.76'):
    raise SystemExit('This targeted dashboard update requires v74-v76.')
PY
systemctl --user is-active --quiet v100-dashboard.service
export UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=5
uv --no-config pip install --python "$PY" --no-deps --reinstall-package rlms \
  "rlms @ git+https://github.com/Mantiee/rlm.git@$REV"
uv --no-config pip check --python "$PY"
"$PY" <<'PY'
from importlib.metadata import version
from rlm.v100.dashboard_layout import BASE_TEMPLATE, validate_template
assert version('rlms') == '0.1.3+v100.76'
validate_template(BASE_TEMPLATE)
PY
systemctl --user restart v100-dashboard.service
systemctl --user try-restart synta-remote.service
systemctl --user is-active --quiet v100-dashboard.service
if ! "$PY" <<'PY'
import json
from rlm.v100.dashboard_health import wait
print(json.dumps(wait(60), indent=2))
PY
then
  systemctl --user status v100-dashboard.service --no-pager -l || true
  journalctl --user -u v100-dashboard.service -n 30 --no-pager || true
  exit 1
fi
printf '%s\n' 'Dashboard v76 updated. Reload the HTTPS dashboard once.' \
  'Mission was not restarted. Goals, weights, paper balances, archived results and Windows policy retained.'
