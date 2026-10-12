#!/usr/bin/env bash
set -euo pipefail
REV="${1:?Pass immutable revision}"
[[ "$REV" =~ ^[0-9a-f]{40}$ ]] || exit 1
ROOT="$HOME/ai-v100"
PY="$ROOT/venvs/v100-continual/bin/python"
FOLDER="$ROOT/tools/managed-bootstrap"
mkdir -p "$FOLDER" "$HOME/.config/systemd/user"
curl -fL --retry 3 "https://raw.githubusercontent.com/Mantiee/rlm/$REV/rlm/v100/mission_watchdog.py" -o "$FOLDER/mission_watchdog.py.new"
"$PY" -m py_compile "$FOLDER/mission_watchdog.py.new"
mv "$FOLDER/mission_watchdog.py.new" "$FOLDER/mission_watchdog.py"
cat > "$HOME/.config/systemd/user/synta-mission-watchdog.service" <<EOF
[Unit]
Description=Synta independent mission availability recovery
[Service]
Type=oneshot
ExecStart=$PY -u $FOLDER/mission_watchdog.py --root $ROOT
TimeoutStartSec=120
EOF
cat > "$HOME/.config/systemd/user/synta-mission-watchdog.timer" <<'EOF'
[Unit]
Description=Check Synta mission independently of downloads
[Timer]
OnBootSec=1min
OnUnitInactiveSec=1min
[Install]
WantedBy=timers.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now synta-mission-watchdog.timer
if [[ "${2:-}" == "--resume" ]]; then
  "$PY" -u "$FOLDER/mission_watchdog.py" --root "$ROOT" --resume
else
  systemctl --user start --no-block synta-mission-watchdog.service
fi
