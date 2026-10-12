#!/usr/bin/env bash
set -euo pipefail
REV="${1:?Pass immutable revision}"
[[ "$REV" =~ ^[0-9a-f]{40}$ ]] || exit 1
ROOT="$HOME/ai-v100"
mkdir -p "$ROOT"
# Explicit operator recovery resumes the preserved mission. No training job is killed.
curl -fL --retry 3 "https://raw.githubusercontent.com/Mantiee/rlm/$REV/tools/install-synta-watchdog.sh" -o "$ROOT/install-synta-watchdog.sh"
bash "$ROOT/install-synta-watchdog.sh" "$REV" --resume
curl -fL --retry 3 "https://raw.githubusercontent.com/Mantiee/rlm/$REV/tools/bootstrap-synta-debian.sh" -o "$ROOT/bootstrap-synta-debian.sh"
bash "$ROOT/bootstrap-synta-debian.sh" "$REV"
"$ROOT/bin/v100-continual" mission-status
journalctl --user -u synta-mission-watchdog.service -n 15 --no-pager

curl -fL --retry 3 "https://raw.githubusercontent.com/Mantiee/rlm/$REV/tools/attach-synta-3060.sh" -o "$HOME/ai-v100/attach-synta-3060.sh"
bash "$HOME/ai-v100/attach-synta-3060.sh"
