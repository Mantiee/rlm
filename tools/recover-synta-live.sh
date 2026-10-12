#!/usr/bin/env bash
set -euo pipefail
REV="${1:?Pass immutable revision}"
[[ "$REV" =~ ^[0-9a-f]{40}$ ]] || exit 1
ROOT="$HOME/ai-v100"
mkdir -p "$ROOT"
# Explicit operator recovery resumes the preserved mission. No training job is killed.
curl -fL --retry 3 "https://raw.githubusercontent.com/Mantiee/rlm/$REV/tools/install-synta-watchdog.sh" -o "$ROOT/install-synta-watchdog.sh"
bash "$ROOT/install-synta-watchdog.sh" "$REV" --resume
"$ROOT/venvs/v100-continual/bin/python" - "$ROOT" <<'PY'
import subprocess,sys
from pathlib import Path
root=Path(sys.argv[1])
transactions=[p for p in (root/'research').glob('.update-v*-restore.*') if p.stat().st_size]
group=subprocess.run(['systemctl','--user','show','synta-managed-update.service','-p','ControlGroup','--value'],capture_output=True,text=True,timeout=10)
base=Path('/sys/fs/cgroup').resolve()
path=(base/group.stdout.strip().lstrip('/')).resolve()
exclusive=False
if group.returncode==0 and group.stdout.strip() and path.is_relative_to(base) and path!=base:
    for file in path.rglob('cgroup.procs'):
        for pid in file.read_text().split():
            try: args=(Path('/proc')/pid/'cmdline').read_bytes().split(b'\0')
            except FileNotFoundError: continue
            if b'rlm.v100.cli' in args and any(mode in args for mode in (b'train',b'mission-prepare',b'optimize-mtp',b'calibrate-training',b'test-mtp')):
                exclusive=True
    if not transactions and not exclusive:
        subprocess.run(['systemctl','--user','stop','synta-managed-update.service'],check=True,timeout=120)
    else:
        print('Active installer or updater-owned experiment retained; package replacement will retry.',flush=True)
PY
curl -fL --retry 3 "https://raw.githubusercontent.com/Mantiee/rlm/$REV/tools/bootstrap-synta-debian.sh" -o "$ROOT/bootstrap-synta-debian.sh"
bash "$ROOT/bootstrap-synta-debian.sh" "$REV"
"$ROOT/bin/v100-continual" mission-status
journalctl --user -u synta-mission-watchdog.service -n 15 --no-pager

curl -fL --retry 3 "https://raw.githubusercontent.com/Mantiee/rlm/$REV/tools/attach-synta-3060.sh" -o "$HOME/ai-v100/attach-synta-3060.sh"
bash "$HOME/ai-v100/attach-synta-3060.sh"
