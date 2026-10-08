#!/usr/bin/env bash
# Separate CPU-only worker. No host/global packages or GPU changes.
set -euo pipefail
REV="${1:?Pass an exact published source revision}"
MAILBOX="${2:?Pass the authenticated shared compute folder}"
NAME="${3:-linux-cpu}"
[[ "$REV" =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid source revision' >&2; exit 1; }
[[ "$NAME" =~ ^[a-zA-Z0-9_-]{1,48}$ ]] || { echo 'Invalid worker name' >&2; exit 1; }
[[ -d "$MAILBOX" ]] || { echo 'Connect the authenticated shared folder first' >&2; exit 1; }
ROOT="$HOME/ai-owned-compute"
mkdir -p "$ROOT/$REV"
command -v uv >/dev/null || { echo 'This isolated worker requires uv' >&2; exit 1; }
if [[ ! -x "$ROOT/venv/bin/python" ]]; then uv --no-config venv --python 3.11 "$ROOT/venv"; fi
PY="$ROOT/venv/bin/python"
uv --no-config pip install --python "$PY" --index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0'
uv --no-config pip install --python "$PY" --index-url https://pypi.org/simple 'safetensors==0.5.3' 'psutil==7.0.0'
for name in compute_worker.py compute_kernel.py; do
    target="$ROOT/$REV/$name"
    if [[ ! -e "$target" ]]; then
        curl -fL --retry 3 "https://raw.githubusercontent.com/Mantiee/rlm/$REV/rlm/v100/$name" -o "$target.download"
        mv "$target.download" "$target"
    fi
done
exec "$PY" -u "$ROOT/$REV/compute_worker.py" --mailbox "$MAILBOX" --name "$NAME"
