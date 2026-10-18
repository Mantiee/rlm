#!/usr/bin/env bash
# Deploy mtp_gate.py fix to Debian1
# Fixes: temperature noise bug causing spurious answers_exact_match=False in MTP benchmark
# SHA: f7c345b1 (replace temperature=N with temperature=0.0 in optimize())

set -euo pipefail

HOST="marek@192.168.0.68"
REMOTE_REPO="/home/marek/ai-v100/rlm"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SHA_LOCAL="$(sha256sum "${SCRIPT_DIR}/mtp_gate.py" | awk '{print $1}')"

echo "[MTP-GATE] Deploying mtp_gate.py fix to ${HOST}"
echo "[MTP-GATE] Local SHA: ${SHA_LOCAL:0:8}"

# Backup remote
ssh "${HOST}" "
  set -e
  BACKUP=/home/marek/ai-v100/research/deploy/rollback/mtp-gate-\$(date -u +%Y%m%dT%H%M%SZ)
  mkdir -p \"\${BACKUP}\"
  [ -f '${REMOTE_REPO}/rlm/v100/mtp_gate.py' ] && cp '${REMOTE_REPO}/rlm/v100/mtp_gate.py' \"\${BACKUP}/\" || true
  echo \"Backed up to \${BACKUP}\"
"

# Upload
scp "${SCRIPT_DIR}/mtp_gate.py" "${HOST}:${REMOTE_REPO}/rlm/v100/mtp_gate.py"

# Verify SHA
REMOTE_SHA=$(ssh "${HOST}" "sha256sum '${REMOTE_REPO}/rlm/v100/mtp_gate.py'" | awk '{print $1}')
if [ "${REMOTE_SHA}" != "${SHA_LOCAL}" ]; then
  echo "[ERROR] SHA mismatch! Local: ${SHA_LOCAL:0:8}, Remote: ${REMOTE_SHA:0:8}"
  exit 1
fi

echo "[MTP-GATE] SHA verified OK: ${REMOTE_SHA:0:8}"

# Verify syntax
ssh "${HOST}" "python3 -c 'import ast; ast.parse(open(\"${REMOTE_REPO}/rlm/v100/mtp_gate.py\").read()); print(\"syntax OK\")'"

echo "[MTP-GATE] Deploy complete."
