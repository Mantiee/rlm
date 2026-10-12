#!/usr/bin/env bash
# Register the explicitly owned node without stopping mission or training.
set -euo pipefail
source "$HOME/ai-v100/env.sh"
ROOT="$AI_V100_ROOT"
PY="$ROOT/venvs/v100-continual/bin/python"
"$PY" - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
from rlm.v100.common import atomic_json
root=Path(sys.argv[1]).resolve()
atomic_json(root/'research/owned-node.json',{'endpoint':'http://192.168.0.9:11435','name':'debian-3060ti','dedicated':True,'scope':'Explicit operator-owned host; no arbitrary remote shell'})
config=root/'research/compute-config.json'
mailbox=Path(json.loads(config.read_text())['mailbox']) if config.exists() else root/'research/owned-compute/mailbox'
for name in ('jobs','claims','results','workers','closed'): (mailbox/name).mkdir(parents=True,exist_ok=True)
if not config.exists(): atomic_json(config,{'schema':'v100-compute-config-v1','mailbox':str(mailbox),'scope':'Dedicated host mailbox; CPU results transported over operator-owned LAN and revalidated locally'})
preferences=root/'research/user-preferences.json'
value=json.loads(preferences.read_text()) if preferences.exists() else {}
value.update(automatic_goal_compute=True,remote_helper_enabled=True)
atomic_json(preferences,value)
PY
mkdir -p "$HOME/.config/systemd/user"
mkdir -p "$ROOT/bin"
cat > "$ROOT/bin/synta-owned-nodes" <<'RUNNER'
#!/usr/bin/env bash
set -euo pipefail
source "$HOME/ai-v100/env.sh"
PY="$AI_V100_ROOT/venvs/v100-continual/bin/python"
"$PY" - <<'VERSION'
import re,sys
from importlib.metadata import version
value=version('rlms')
match=re.fullmatch(r'0\.1\.3\+v100\.(\d+)',value)
if not match or int(match[1])<87:
    print('Waiting for deferred v87 package update; existing mission/train unchanged',flush=True)
    sys.exit(75)
VERSION
exec "$PY" -u -m rlm.v100.remote_nodes --root "$AI_V100_ROOT"
RUNNER
chmod 0755 "$ROOT/bin/synta-owned-nodes"
cat > "$HOME/.config/systemd/user/synta-owned-nodes.service" <<UNIT
[Unit]
Description=Reconnect owned Debian RTX 3060 Ti and audit fixed CPU jobs
[Service]
ExecStart=$ROOT/bin/synta-owned-nodes
Restart=always
RestartSec=30
Nice=5
MemoryMax=512M
TasksMax=64
[Install]
WantedBy=default.target
UNIT
systemctl --user daemon-reload
systemctl --user enable --now synta-owned-nodes.service
systemctl --user start --no-block synta-managed-update.service
echo 'Owned node registered. v87 package installation waits for exclusive training; node reconnects automatically afterward.'
systemctl --user status synta-owned-nodes.service --no-pager -l || true
