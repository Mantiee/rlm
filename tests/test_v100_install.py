import os
import shlex
import sqlite3
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest


def test_continual_install_preserves_working_environment_and_copies_sqlite(tmp_path):
    pytest.importorskip("tomli_w")
    root = tmp_path / "ai-v100"
    root.mkdir()
    (root / "env.sh").write_text(f"export AI_V100_ROOT={shlex.quote(str(root))}\n")
    (root / "venvs/train/bin").mkdir(parents=True)
    interpreter = root / "venvs/train/bin/python"
    interpreter.write_text(f'#!/bin/bash\nexec {shlex.quote(sys.executable)} "$@"\n')
    interpreter.chmod(0o755)
    (root / "bin").mkdir()
    original_wrapper = root / "bin/v100-lab"
    original_wrapper.write_text("working wrapper\n")
    (root / "research/state").mkdir(parents=True)
    repo = Path(__file__).parents[1]
    original_profile = root / "research/v100.toml"
    original_profile.write_bytes((repo / "profiles/v100.toml").read_bytes())
    database = root / "research/state/memory.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE originals(text)")
        db.execute("INSERT INTO originals VALUES('remember this')")
    untouched = {
        path: path.read_bytes()
        for path in (interpreter, original_wrapper, original_profile, database)
    }
    mocks = tmp_path / "mocks"
    mocks.mkdir()
    uv = mocks / "uv"
    uv.write_text("""#!/bin/bash
set -euo pipefail
printf '%s\\n' "$*" >> "$AI_V100_ROOT/research/uv-calls.txt"
if [[ "$2" == "venv" ]]; then
  destination="${@: -1}"
  mkdir -p "$destination/bin"
  cp "$AI_V100_ROOT/venvs/train/bin/python" "$destination/bin/python"
fi
if [[ "$2" == "pip" && "$3" == "freeze" ]]; then echo 'torch==2.6.0+cu124'; fi
""")
    uv.chmod(0o755)
    gpu = mocks / "nvidia-smi"
    gpu.write_text("#!/bin/bash\necho 'mock V100'\n")
    gpu.chmod(0o755)
    env = {**os.environ, "AI_V100_ROOT": str(root), "PATH": str(mocks) + ":" + os.environ["PATH"]}
    subprocess.run(
        ["bash", str(repo / "install-continual-v100.sh")],
        env=env,
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    assert all(path.read_bytes() == data for path, data in untouched.items())
    profile = tomllib.loads((root / "research/v100-continual.toml").read_text())
    assert profile["runtime"]["base_url"].endswith(":8089")
    assert profile["training"]["distillation_weight"] == 0.1
    with sqlite3.connect(profile["memory"]["database"]) as db:
        assert db.execute("SELECT * FROM originals").fetchall() == [("remember this",)]
    calls = (root / "research/uv-calls.txt").read_text().splitlines()
    installs = [line for line in calls if " pip install " in " " + line + " "]
    assert installs and all(
        str(root / "venvs/v100-continual/bin/python") in line for line in installs
    )
    # Reinstallation must preserve an edited profile and the new memory too.
    preserved = root / "research/v100-continual.toml"
    preserved.write_text(
        preserved.read_text().replace("distillation_weight = 0.1", "distillation_weight = 0.2")
    )
    before = preserved.read_bytes()
    subprocess.run(
        ["bash", str(repo / "install-continual-v100.sh")],
        env=env,
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    assert preserved.read_bytes() == before
