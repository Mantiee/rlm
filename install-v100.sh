#!/usr/bin/env bash
set -euo pipefail
ROOT="${AI_V100_ROOT:-$HOME/ai-v100}"
SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=5 UV_LINK_MODE=copy
export PYTHONNOUSERSITE=1
test -f "$ROOT/env.sh" || { echo "Missing $ROOT/env.sh"; exit 1; }
source "$ROOT/env.sh"
mkdir -p "$ROOT/research/logs" "$ROOT/bin" "$ROOT/venvs"
PY="$ROOT/venvs/train/bin/python"
test -x "$PY" || { echo "Missing working train Python: $PY"; exit 1; }
if [[ ! -x "$ROOT/venvs/memory-lab/bin/python" ]]; then
    uv --no-config venv --python "$PY" "$ROOT/venvs/memory-lab"
fi
uv --no-config pip install --python "$ROOT/venvs/memory-lab/bin/python" "$SOURCE"
if [[ ! -f "$ROOT/research/v100.toml" ]]; then
    cp "$SOURCE/profiles/v100.toml" "$ROOT/research/v100.toml"
fi
cat > "$ROOT/bin/v100-lab" <<'WRAPPER'
#!/usr/bin/env bash
set -euo pipefail
ROOT="${AI_V100_ROOT:-$HOME/ai-v100}"
source "$ROOT/env.sh"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONNOUSERSITE=1
PY="$ROOT/venvs/memory-lab/bin/python"
SKIP=0
for arg in "$@"; do
    if [[ "$SKIP" == "1" ]]; then SKIP=0; continue; fi
    case "$arg" in
        --profile|--root) SKIP=1; continue ;;
        --*) continue ;;
    esac
    if [[ "$arg" == "train" || "$arg" == "export-model" ]]; then
        PY="$ROOT/venvs/memory-lab-train/bin/python"
        test -x "$PY" || { echo "Run install-v100.sh --training first"; exit 1; }
    fi
    break
done
exec "$PY" -m rlm.v100.cli --root "$ROOT" "$@"
WRAPPER
chmod +x "$ROOT/bin/v100-lab"
uv --no-config pip freeze --python "$ROOT/venvs/memory-lab/bin/python" > "$ROOT/research/requirements.controller.txt"
if [[ "${1:-}" == "--training" ]]; then
    if [[ ! -x "$ROOT/venvs/memory-lab-train/bin/python" ]]; then
        uv --no-config venv --python "$PY" "$ROOT/venvs/memory-lab-train"
    fi
    uv --no-config pip freeze --python "$PY" > "$ROOT/research/requirements.train-source.txt"
    "$PY" - "$ROOT" <<'PYCODE'
import sys
from pathlib import Path
root = Path(sys.argv[1])
lines = (root / "research/requirements.train-source.txt").read_text().splitlines()
lines = [line for line in lines if not line.lower().startswith(("torch==", "torchvision==", "rlms", "-e "))]
(root / "research/requirements.train-copy.txt").write_text("\n".join(lines) + "\n")
PYCODE
    uv --no-config pip install --python "$ROOT/venvs/memory-lab-train/bin/python" \
        --index-url https://pypi.org/simple -r "$ROOT/research/requirements.train-copy.txt" \
        "torch @ https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp311-cp311-linux_x86_64.whl" \
        "torchvision @ https://download.pytorch.org/whl/cu124/torchvision-0.21.0%2Bcu124-cp311-cp311-linux_x86_64.whl" "$SOURCE"
    uv --no-config pip check --python "$ROOT/venvs/memory-lab-train/bin/python"
    uv --no-config pip freeze --python "$ROOT/venvs/memory-lab-train/bin/python" > "$ROOT/research/requirements.training.txt"
fi
"$ROOT/bin/v100-lab" doctor
