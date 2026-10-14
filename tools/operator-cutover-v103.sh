#!/usr/bin/env bash
# One-time, operator-approved handoff from live v102 to the vetted v103 installer.
# This explicitly interrupts the current training cycle and can lose
# non-checkpointed cycle progress; never runs as an unattended updater.
set -Eeuo pipefail
if [[ "${1:-}" != "--accept-interrupt-current-cycle" || "$#" != 1 ]]; then
  echo "Usage: $0 --accept-interrupt-current-cycle" >&2
  echo "WARNING: incomplete progress from the current training cycle may be lost." >&2
  exit 2
fi
if [[ "$(id -u)" -eq 0 ]]; then
  echo "Run as the owner's normal Debian user, not root." >&2
  exit 2
fi
ROOT="$HOME/ai-v100"
ENV="$ROOT/env.sh"
[[ -f "$ENV" ]] || { echo "Synta env.sh is missing; no service changed." >&2; exit 1; }
# shellcheck disable=SC1090
source "$ENV"
[[ "${AI_V100_ROOT:-}" == "$ROOT" ]] || {
  echo "Unexpected AI_V100_ROOT; no service changed." >&2
  exit 1
}
export PATH="$HOME/.local/bin:$PATH"
PY="$ROOT/venvs/v100-continual/bin/python"
REV="366ebfcdb963b65713062f76a9e0eba9778518b3"
INSTALLER="$ROOT/tools/managed-releases/$REV/tools/update-synta-v103.sh"
WATCHDOG="$ROOT/tools/managed-bootstrap/mission_watchdog.py"
for tool in curl uv git systemctl flock; do
  command -v "$tool" >/dev/null || { echo "Missing $tool; no service changed." >&2; exit 1; }
done
[[ -x "$PY" && -s "$INSTALLER" && -s "$WATCHDOG" ]] || {
  echo "Required staged, previously verified release or Python is missing. No service changed." >&2
  exit 1
}
"$PY" - "$ROOT" "$REV" <<'PYCODE'
import json
import sys
from importlib.metadata import version
from pathlib import Path
from rlm.v100.mission import status
from rlm.v100.mission_watchdog import preserved_profile
root, revision = Path(sys.argv[1]).resolve(), sys.argv[2]
manifest = root / "tools/managed-releases" / revision / "release.json"
data = json.loads(manifest.read_text())
if data.get("revision") != revision or data.get("version") != 103:
    raise SystemExit("Staged release identity/version mismatch; no service changed")
installed = version("rlms")
if installed not in ("0.1.3+v100.102", "0.1.3+v100.103"):
    raise SystemExit(f"Unexpected installed version {installed}; no service changed")
active = status(root)
if not active.get("running"):
    raise SystemExit("Mission is not running; use recovery diagnostics instead")
preserved = preserved_profile(root, active)
print(json.dumps({
    "installed": installed,
    "target": "0.1.3+v100.103",
    "preserved_profile": str(preserved),
    "active_run": active.get("run"),
    "learning_stage": active.get("learning", {}).get("stage"),
    "warning": "Current cycle may lose incomplete optimizer progress"
}, ensure_ascii=False), flush=True)
PYCODE
# Check the release source before stopping a working mission.
curl -fLsS --connect-timeout 8 --max-time 20 \
  "https://raw.githubusercontent.com/Mantiee/rlm/$REV/tools/managed-release.json" \
  -o /dev/null
# Fail fast if another owned maintenance action holds the lease.
exec 8>"$ROOT/research/mission-maintenance.lock"
flock -n 8 || { echo "Maintenance busy; no service changed." >&2; exit 75; }
flock -u 8
exec 8>&-
NEEDS_RECOVERY=0
recover_on_failure() {
  rc=$?
  trap - EXIT
  if [[ "$rc" -ne 0 && "$NEEDS_RECOVERY" -eq 1 ]]; then
    echo "Cutover failed; attempting automatic preserved-profile recovery." >&2
    systemctl --user start v100-mission.service || true
    "$PY" "$WATCHDOG" --root "$ROOT" --resume || true
    echo "Check mission status; an unsuccessful recovery must not be called a success." >&2
  fi
  exit "$rc"
}
trap recover_on_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
NEEDS_RECOVERY=1
echo "Stopping only the owned Debian mission for the explicit v103 handoff..."
"$PY" - "$ROOT" <<'PYCODE'
from pathlib import Path
from rlm.v100.mission import stop
import sys
print(stop(Path(sys.argv[1])), flush=True)
PYCODE
systemctl --user stop v100-mission.service
echo "Starting the already staged and verified v103 installer..."
bash "$INSTALLER" "$REV"
echo "Checking real installed package and running mission..."
"$PY" - "$ROOT" <<'PYCODE'
import sys
from importlib.metadata import version
from pathlib import Path
from rlm.v100.mission import status
installed = version("rlms")
active = status(Path(sys.argv[1]))
print({"installed": installed, "running": active.get("running"), "run": active.get("run")}, flush=True)
if installed != "0.1.3+v100.103" or active.get("running") is not True:
    raise SystemExit("Cutover verification failed; do not report success")
PYCODE
NEEDS_RECOVERY=0
systemctl --user start --no-block synta-managed-update.service || true
echo "v103 running. The updater will refresh receipts; profit is NOT implied."
