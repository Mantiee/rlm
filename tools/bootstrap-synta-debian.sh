#!/usr/bin/env bash
set -euo pipefail
REV="${1:?Pass the exact source revision}"
[[ "$REV" =~ ^[0-9a-f]{40}$ ]] || exit 1
ROOT="$HOME/ai-v100"
PY="$ROOT/venvs/v100-continual/bin/python"
FOLDER="$ROOT/tools/managed-bootstrap"
[[ -x "$PY" && "$ROOT" != *" "* ]] || { echo "Existing Python and a path without spaces required" >&2; exit 1; }
mkdir -p "$FOLDER" "$HOME/.config/systemd/user"
STAGE="$(mktemp -d "$FOLDER/.bootstrap.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
curl -fL --retry 3 "https://raw.githubusercontent.com/Mantiee/rlm/$REV/tools/managed-release.json" -o "$STAGE/manifest.json"
for MODULE in inference_benchmark release_selector managed_bootstrap; do
  curl -fL --retry 3 "https://raw.githubusercontent.com/Mantiee/rlm/$REV/rlm/v100/$MODULE.py" -o "$STAGE/$MODULE.py"
done
"$PY" - "$STAGE" "$FOLDER" <<'PYCODE'
import hashlib,json,os,sys
from pathlib import Path
stage,folder=map(Path,sys.argv[1:])
manifest=json.loads((stage/'manifest.json').read_text())
names=('inference_benchmark','release_selector','managed_bootstrap')
for module in names:
    path=stage/(module+'.py')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['files']['rlm/v100/'+path.name], 'Bootstrap artifact checksum differs'
    compile(path.read_bytes(),str(path),'exec')
for module in names:
    os.replace(stage/(module+'.py'),folder/(module+'.py'))
PYCODE
cat > "$HOME/.config/systemd/user/synta-managed-update.service" <<EOF
[Unit]
Description=Synta managed updates and role preparation
[Service]
Type=oneshot
ExecStart=$PY -u $FOLDER/managed_bootstrap.py --root $ROOT
Nice=10
IOSchedulingClass=idle
TimeoutStartSec=5h
EOF
cat > "$HOME/.config/systemd/user/synta-managed-update.timer" <<'EOF'
[Unit]
Description=Check authorized Synta GitHub release with retry after a deferred training window
[Timer]
OnBootSec=5min
OnUnitInactiveSec=15min
Persistent=true
[Install]
WantedBy=timers.target
EOF
curl -fL --retry 3 "https://raw.githubusercontent.com/Mantiee/rlm/$REV/tools/install-synta-watchdog.sh" -o "$ROOT/install-synta-watchdog.sh"
bash "$ROOT/install-synta-watchdog.sh" "$REV"
systemctl --user daemon-reload
systemctl --user enable --now synta-managed-update.timer
systemctl --user reset-failed synta-managed-update.service
systemctl --user start --no-block synta-managed-update.service
journalctl --user -u synta-managed-update.service -n 20 --no-pager
