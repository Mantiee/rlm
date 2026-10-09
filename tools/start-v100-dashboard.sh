#!/usr/bin/env bash
# Install only the read-only LAN viewer. Do not stop or update the mission.
set -euo pipefail
REV="${1:?Pass the pinned 40-character commit}"
[[ "$REV" =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid revision' >&2; exit 1; }
source "$HOME/ai-v100/env.sh"
PY="$AI_V100_ROOT/venvs/v100-continual/bin/python"
TARGET="$AI_V100_ROOT/bin/serve-v100-dashboard.py"
mkdir -p "$AI_V100_ROOT/bin" "$HOME/.config/systemd/user"
TMP="$(mktemp "$AI_V100_ROOT/bin/.dashboard.XXXXXX")"
trap 'rm -f "$TMP"' EXIT
curl -fL --retry 3 \
  "https://raw.githubusercontent.com/Mantiee/rlm/$REV/tools/serve-v100-dashboard.py" \
  -o "$TMP"
"$PY" "$TMP" --help >/dev/null
if [[ -f "$TARGET" ]]; then
  cp -p "$TARGET" "$TARGET.backup-$(date -u +%Y%m%dT%H%M%SZ)"
fi
chmod 700 "$TMP"
mv "$TMP" "$TARGET"
"$PY" <<'PY'
import json
import os
import sys
from pathlib import Path
from rlm.v100.common import atomic_json

root = Path(os.environ['AI_V100_ROOT']).resolve()
script = root / 'bin/serve-v100-dashboard.py'
if any(char in str(root) + sys.executable for char in ('\n', '\r', '"', '%', '\\')):
    raise ValueError('Unsupported service path')
unit = Path.home() / '.config/systemd/user/v100-dashboard.service'
if unit.exists():
    unit.with_suffix('.service.backup').write_bytes(unit.read_bytes())
unit.write_text(
    '[Unit]\nDescription=Synta read-only LAN dashboard\nAfter=network.target\n'
    '[Service]\nType=simple\n'
    f'ExecStart="{sys.executable}" -u "{script}" --root "{root}" --bind 192.168.0.68 --port 8765\n'
    f'WorkingDirectory={root}\n'
    'Environment=PYTHONNOUSERSITE=1\n'
    'Restart=on-failure\nRestartSec=10\nNice=15\nCPUQuota=25%\n'
    'MemoryMax=512M\nTasksMax=64\nUMask=0077\n'
    '[Install]\nWantedBy=default.target\n'
)
path = root / 'research/user-preferences.json'
preferences = json.loads(path.read_text()) if path.exists() else {'directive': '', 'alerts': False}
if not isinstance(preferences, dict):
    raise ValueError('Invalid user preferences; no goal changes applied')
preferences['dashboard'] = {
    'url': 'http://192.168.0.68:8765',
    'editable_guest_file': '/workspace/dashboard/index.html',
    'edit_action': 'write_dashboard',
    'instructions': 'The operator authorizes dashboard HTML/CSS edits from chat. Use read_dashboard, write_dashboard and dashboard_status to edit this file and verify publication. Keep all existing component IDs. Do not add scripts, external resources or change the data API. The host validates and publishes the layout automatically; receipts are in research/dashboard/layout-status.json. A queued edit is not proof of publication.',
    'data_refresh_seconds': 20,
    'scope': 'Presentation permission only. Long-term goal, training gates, weights and financial records are unchanged. Endpoint is configured, not proof of reachability.'
}
atomic_json(path, preferences)
PY
systemctl --user daemon-reload
systemctl --user enable v100-dashboard.service
systemctl --user restart v100-dashboard.service
for attempt in {1..12}; do
  if curl --noproxy '*' -fsS --connect-timeout 2 --max-time 3 \
      http://192.168.0.68:8765/api/status >/dev/null; then
    printf '\nDASHBOARD READY: http://192.168.0.68:8765\nMission was not restarted.\n'
    exit 0
  fi
  sleep 2
done
journalctl --user -u v100-dashboard.service -n 20 --no-pager
echo 'Dashboard did not become reachable; mission was not changed.' >&2
exit 1
