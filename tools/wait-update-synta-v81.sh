#!/usr/bin/env bash
set -euo pipefail
REV="${1:?Pass the exact pinned revision}"
[[ "$REV" =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid revision' >&2; exit 1; }
ROOT="$HOME/ai-v100"
SCRIPT="$ROOT/update-synta-v81.sh"
curl -fL --retry 3 "https://raw.githubusercontent.com/Mantiee/rlm/$REV/tools/update-synta-v81.sh" -o "$SCRIPT"
DEADLINE=$((SECONDS + 86400))
while ps -eo args= | awk '/rlm[.]v100[.]cli/ && / (train|mission-prepare|optimize-mtp|calibrate-training|test-mtp)( |$)/ {found=1} END {exit !found}'; do
    if (( SECONDS >= DEADLINE )); then
        echo 'Update wait expired after 24 hours; active experiment untouched.' >&2
        exit 1
    fi
    echo 'Update pending: exclusive experiment still active. No service changed.'
    sleep 30
done
bash "$SCRIPT" "$REV"
