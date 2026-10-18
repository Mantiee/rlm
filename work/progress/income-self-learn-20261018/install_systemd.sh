#!/usr/bin/env bash
# Install Synta income-self-learn systemd user units on Debian1
# Run after release.sh deploy
# Usage: ./install_systemd.sh

set -euo pipefail

UNITS_DIR="${HOME}/.config/systemd/user"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${UNITS_DIR}"

for unit in synta-gig-worker synta-earnings-api synta-model-updater; do
  src="${SCRIPT_DIR}/systemd/${unit}.service"
  dst="${UNITS_DIR}/${unit}.service"
  echo "[INSTALL] ${unit}.service"
  cp "${src}" "${dst}"
  systemctl --user daemon-reload
  systemctl --user enable "${unit}.service"
  systemctl --user restart "${unit}.service"
  sleep 2
  systemctl --user is-active "${unit}.service" && echo "  OK: ${unit} running" || echo "  WARN: ${unit} not active — check: journalctl --user -u ${unit}"
done

echo "[INSTALL] Done. Check status:"
echo "  systemctl --user status synta-gig-worker synta-earnings-api synta-model-updater"
echo "  journalctl --user -u synta-earnings-api -n 20"
