#!/usr/bin/env bash
set -euo pipefail
ROOT="${AI_V100_ROOT:-$HOME/ai-v100}"
SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=5 UV_LINK_MODE=copy PYTHONNOUSERSITE=1
test -f "$ROOT/env.sh" || { echo "Missing $ROOT/env.sh"; exit 1; }
source "$ROOT/env.sh"
TRAIN_PY="$ROOT/venvs/train/bin/python"
LAB_PY="$ROOT/venvs/v100-continual/bin/python"
test -x "$TRAIN_PY" || { echo "Missing working train Python"; exit 1; }
mkdir -p "$ROOT/research/logs" "$ROOT/research/state" "$ROOT/bin"
if [[ ! -x "$LAB_PY" ]]; then
    uv --no-config venv --python "$TRAIN_PY" "$ROOT/venvs/v100-continual"
fi
uv --no-config pip freeze --python "$TRAIN_PY" > "$ROOT/research/requirements.continual-source.txt"
"$TRAIN_PY" - "$ROOT" <<'PY'
import re
import sys
from pathlib import Path
root = Path(sys.argv[1])
lines = (root / "research/requirements.continual-source.txt").read_text().splitlines()
lines = [line for line in lines if not re.match(r"^(torch|torchvision|rlms)(?:[=@ ]|$)|^-e ", line, re.I)]
(root / "research/requirements.continual-copy.txt").write_text("\n".join(lines) + "\n")
PY
uv --no-config pip install --python "$LAB_PY" --index-url https://pypi.org/simple \
    -r "$ROOT/research/requirements.continual-copy.txt" \
    "torch @ https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp311-cp311-linux_x86_64.whl" \
    "torchvision @ https://download.pytorch.org/whl/cu124/torchvision-0.21.0%2Bcu124-cp311-cp311-linux_x86_64.whl" \
    "tomli-w==1.2.0" "$SOURCE"
uv --no-config pip check --python "$LAB_PY"
uv --no-config pip freeze --python "$LAB_PY" > "$ROOT/research/requirements.continual.txt"
"$LAB_PY" - "$ROOT" "$SOURCE" <<'PY'
import os
import sqlite3
import sys
import tempfile
import tomllib
from pathlib import Path
import tomli_w
root, source = map(Path, sys.argv[1:])
destination = root / "research/v100-continual.toml"
if not destination.exists():
    original = root / "research/v100.toml"
    profile = tomllib.loads((original if original.exists() else source / "profiles/v100.toml").read_text())
    old_database = Path(profile["memory"]["database"]).expanduser()
    if not old_database.is_absolute():
        old_database = root / old_database
    new_database = root / "research/state/continual-memory.sqlite3"
    if old_database.exists() and not new_database.exists():
        descriptor, temporary = tempfile.mkstemp(prefix=".memory-copy-", dir=new_database.parent)
        os.close(descriptor)
        try:
            with sqlite3.connect(old_database.resolve().as_uri() + "?mode=ro", uri=True) as original_db:
                with sqlite3.connect(temporary) as copied_db:
                    original_db.backup(copied_db)
            os.link(temporary, new_database)
        finally:
            os.unlink(temporary)
    profile["runtime"]["base_url"] = "http://127.0.0.1:8089"
    profile["memory"].update(database=str(new_database), retrieval="lexical",
                             encoder_path="models/memory-encoder", encoder_device="cpu")
    profile["training"].update(output="research/checkpoints/continual-round-001",
        split_ledger="research/state/continual-splits.sqlite3", teacher_adapter="",
        distillation_weight=0.1, distillation_temperature=1.0)
    # Exclusive creation: never overwrite a user's experiment profile.
    with destination.open("x") as handle:
        handle.write(tomli_w.dumps(profile))
        handle.flush()
        os.fsync(handle.fileno())
PY
cat > "$ROOT/bin/v100-continual" <<'WRAPPER'
#!/usr/bin/env bash
set -euo pipefail
ROOT="${AI_V100_ROOT:-$HOME/ai-v100}"
source "$ROOT/env.sh"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONNOUSERSITE=1
exec "$ROOT/venvs/v100-continual/bin/python" -m rlm.v100.cli \
    --root "$ROOT" --profile "$ROOT/research/v100-continual.toml" "$@"
WRAPPER
chmod +x "$ROOT/bin/v100-continual"
"$ROOT/bin/v100-continual" doctor
echo "Isolated install ready. Original v100-lab, venvs and model unchanged."
echo "New server port: 8089. Read CONTINUAL_LEARNING.md before training or breeding."
