#!/usr/bin/env bash
# One-time controller login and opt-in private reports/code mirror.
set -euo pipefail
REV="${1:?Pass immutable source revision}"
[[ "$REV" =~ ^[a-f0-9]{40}$ ]] || exit 1
ROOT="$HOME/ai-v100"
PY="$ROOT/venvs/v100-continual/bin/python"
[[ -x "$PY" ]] || { echo 'Existing Synta Python required.' >&2; exit 1; }
if ! command -v gh >/dev/null; then
  sudo apt-get update
  sudo apt-get install -y gh
fi
if ! gh auth status --hostname github.com >/dev/null 2>&1; then
  gh auth login --hostname github.com --git-protocol https --web </dev/tty
fi
[[ "$(gh api user --jq .login)" == Mantiee ]] || { echo 'Use the Mantiee GitHub account for this owned repository.' >&2; exit 1; }
umask 077
FOLDER="$ROOT/tools/private-sync"
mkdir -p "$FOLDER" "$ROOT/research/private-sync" "$HOME/.config/systemd/user"
curl -fL --retry 3 "https://raw.githubusercontent.com/Mantiee/rlm/$REV/tools/managed-release.json" -o "$FOLDER/manifest.json"
"$PY" - "$REV" "$FOLDER" <<'PY'
import hashlib,json,sys,urllib.request
from pathlib import Path
revision,folder=sys.argv[1],Path(sys.argv[2])
manifest=json.loads((folder/'manifest.json').read_text())
for name in ('rlm/v100/private_sync.py','rlm/v100/release_selector.py','tools/audit-synta-hosts.py'):
    with urllib.request.urlopen(f'https://raw.githubusercontent.com/Mantiee/rlm/{revision}/{name}',timeout=30) as reply: body=reply.read(2**20)
    assert hashlib.sha256(body).hexdigest()==manifest['files'][name], 'Private sync artifact checksum differs'
    path=folder/Path(name).name
    temporary=path.with_suffix('.new');temporary.write_bytes(body);temporary.replace(path)
PY
if ! gh repo view Mantiee/synta-control --json visibility >/dev/null 2>&1; then
  gh repo create Mantiee/synta-control --private --add-readme --description 'Synta private code mirror, approved releases and owned-host receipts'
  gh api --method PUT repos/Mantiee/synta-control/actions/permissions -F enabled=false >/dev/null
fi
[[ "$(gh repo view Mantiee/synta-control --json visibility --jq .visibility)" == PRIVATE ]] || { echo 'Refusing telemetry to a non-private repository.' >&2; exit 1; }
GH="$(command -v gh)"
cat > "$HOME/.config/systemd/user/synta-private-sync.service" <<EOF
[Unit]
Description=Synta private GitHub source and host receipt synchronization
[Service]
Type=oneshot
ExecStart=$PY -u $FOLDER/private_sync.py --root $ROOT
Environment="PATH=$(dirname "$GH"):/usr/local/bin:/usr/bin:/bin"
Nice=10
NoNewPrivileges=true
PrivateTmp=true
TimeoutStartSec=5min
EOF
cat > "$HOME/.config/systemd/user/synta-private-sync.timer" <<'EOF'
[Unit]
Description=Synchronize private Synta receipts and source every five minutes
[Timer]
OnBootSec=2min
OnUnitInactiveSec=5min
Persistent=true
[Install]
WantedBy=timers.target
EOF
sudo loginctl enable-linger "$(id -un)"
systemctl --user daemon-reload
systemctl --user enable --now synta-private-sync.timer
systemctl --user start --no-block synta-private-sync.service
printf '%s\n' 'Private sync registered. Status: ~/ai-v100/research/private-sync/status.json' 'Grant ChatGPT GitHub app access to Mantiee/synta-control once so this assistant can read and edit it.'
