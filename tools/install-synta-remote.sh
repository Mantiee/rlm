#!/usr/bin/env bash
# Private access to the existing dashboard, master queue and SSH. No router ports.
set -euo pipefail
source "$HOME/ai-v100/env.sh"
PY="$AI_V100_ROOT/venvs/v100-continual/bin/python"
if ! command -v tailscale >/dev/null; then
  source /etc/os-release
  [[ "$ID" == debian && "$VERSION_CODENAME" =~ ^(bookworm|trixie)$ ]] || {
    echo 'Install Tailscale for this OS using https://tailscale.com/download, then rerun.' >&2
    exit 1
  }
  REMOTE_TMP="$(mktemp -d)"
  trap 'rm -rf "$REMOTE_TMP"' EXIT
  curl -fsSL --retry 3 "https://pkgs.tailscale.com/stable/debian/$VERSION_CODENAME.noarmor.gpg" -o "$REMOTE_TMP/key.gpg"
  curl -fsSL --retry 3 "https://pkgs.tailscale.com/stable/debian/$VERSION_CODENAME.tailscale-keyring.list" -o "$REMOTE_TMP/tailscale.list"
  sudo install -m 644 "$REMOTE_TMP/key.gpg" /usr/share/keyrings/tailscale-archive-keyring.gpg
  sudo install -m 644 "$REMOTE_TMP/tailscale.list" /etc/apt/sources.list.d/tailscale.list
  sudo apt-get update
  sudo apt-get install -y tailscale
fi
sudo systemctl enable --now tailscaled
if ! tailscale status --json | "$PY" -c 'import json,sys; sys.exit(json.load(sys.stdin).get("BackendState") != "Running")'; then
  echo 'Open the Tailscale sign-in link printed below. Use the same account on your phone/laptop.'
  sudo tailscale up
fi
"$PY" <<'PY'
import getpass
import json
import subprocess
import sys
from pathlib import Path
import os

from rlm.v100.common import atomic_json
from rlm.v100.remote_access import validate_config

root = Path(os.environ['AI_V100_ROOT']).resolve()
status = json.loads(subprocess.check_output(['tailscale', 'status', '--json'], text=True))
node = status['Self']
owner = status['User'][str(node['UserID'])]['LoginName']
host = node['DNSName'].rstrip('.')
origin = 'https://' + host
config = validate_config({'owner_login': owner, 'origin': origin, 'dashboard': 'http://192.168.0.68:8765'})
serve = json.loads(subprocess.check_output(['sudo', 'tailscale', 'serve', 'status', '--json'], text=True))
handlers = serve.get('Web', {}).get(host + ':443', {}).get('Handlers', {})
ours = {'/': {'Proxy': 'http://127.0.0.1:8786'}}
if handlers and handlers != ours:
    raise RuntimeError('Existing HTTPS service on this device is preserved. Use a separate Tailscale device or move that service before configuring Synta.')
if serve.get('TCP', {}).get('443') and not handlers:
    raise RuntimeError('Existing TCP listener on Tailscale port 443 is preserved; move it explicitly before configuring Synta.')
if any(serve.get('AllowFunnel', {}).values()):
    raise RuntimeError('Public Funnel is configured on this device. Disable it explicitly before enabling owner-only Synta access.')
config_path = root / 'research/remote-access/config.json'
atomic_json(config_path, config)
config_path.chmod(0o600)
if any(c in str(root) + sys.executable for c in ('\n', '\r', '"', '%', '\\')):
    raise ValueError('Unsupported service path')
unit = Path.home() / '.config/systemd/user/synta-remote.service'
unit.parent.mkdir(parents=True, exist_ok=True)
if unit.exists():
    unit.with_suffix('.service.backup').write_bytes(unit.read_bytes())
unit.write_text(
    '[Unit]\nDescription=Synta owner-only private remote gateway\nAfter=network.target\n'
    '[Service]\nType=simple\n'
    f'ExecStart="{sys.executable}" -u -m rlm.v100.remote_access --root "{root}" --config "{config_path}"\n'
    'Environment=PYTHONNOUSERSITE=1\nRestart=on-failure\nRestartSec=5\nNice=15\nCPUQuota=25%\nMemoryMax=256M\nTasksMax=32\nUMask=0077\n'
    '[Install]\nWantedBy=default.target\n'
)
subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
subprocess.run(['systemctl', '--user', 'enable', '--now', 'synta-remote.service'], check=True)
subprocess.run(['systemctl', '--user', 'restart', 'synta-remote.service'], check=True)
# Keep this user's services running after logout/reboot, without a password in a unit.
subprocess.run(['sudo', 'loginctl', 'enable-linger', getpass.getuser()], check=True)
print('Private dashboard:', origin, flush=True)
print('Private master chat:', origin + '/chat', flush=True)
print('SSH with existing Debian credentials: ssh ' + getpass.getuser() + '@' + host, flush=True)
print('Only the configured Tailscale owner can read or command the web gateway. Other home devices need their own Tailscale installation.', flush=True)
PY
# Serve is private to this tailnet. Never use Funnel for the master command API.
sudo tailscale serve --bg --https=443 http://127.0.0.1:8786
echo 'Install Tailscale on your phone/laptop and sign in to the same account. Open the HTTPS URLs above.'
echo 'Existing SSH authentication is unchanged. No Windows GPU, power limit or clock setting was changed.'
"$PY" -m rlm.v100.remote_connect --root "$AI_V100_ROOT"
