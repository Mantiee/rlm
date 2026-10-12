#!/usr/bin/env bash
set -euo pipefail
source "$HOME/ai-v100/env.sh"
ROLE="${1:-fast}"
CONTEXT="${2:-32768}"
OUTPUT="$AI_V100_ROOT/research/performance/candidate-$ROLE-$(date +%s%N)"
"$AI_V100_ROOT/venvs/v100-continual/bin/python" -m rlm.v100.model_candidates \
  --root "$AI_V100_ROOT" \
  --binary "$AI_V100_ROOT/src/llama-v0.6.0/build-v100/bin/llama-server" \
  --role "$ROLE" --context "$CONTEXT" --output "$OUTPUT" --port 8097
