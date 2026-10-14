"""The managed channel must advance the installed trainer package on code changes."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_managed_v103_is_a_genuine_pinned_new_package():
    manifest = json.loads((ROOT / "tools/managed-release.json").read_text())
    config = (ROOT / "pyproject.toml").read_text()
    assert manifest["version"] == 103
    assert 'version = "0.1.3+v100.103"' in config
    installer = manifest["linux_installer"]
    assert installer == "tools/update-synta-v103.sh"
    content = (ROOT / installer).read_bytes()
    assert manifest["files"][installer] == hashlib.sha256(content).hexdigest()


def test_v103_cutover_preserves_checkpoint_and_refuses_active_learning():
    script = ROOT / "tools/update-synta-v103.sh"
    content = script.read_text()
    assert "0.1.3+v100.103" in content
    assert "range(78, 104)" in content
    assert "active candidate training/evaluation retained".lower() in content.lower()
    assert "Preserved recovery profile unavailable" in content
    assert "No valid preserved profile" in content
    assert "from rlm.v100.supervisor import install" in content
    assert "no service changed" in content.lower()
    assert "No mission profile to preserve" in content
    assert "flock -n -x 9" in content
    bash = shutil.which("bash")
    assert bash, "Bash is required to validate the Linux installer"
    parsed = subprocess.run([bash, "-n", str(script)], capture_output=True, text=True)
    assert parsed.returncode == 0, parsed.stderr
