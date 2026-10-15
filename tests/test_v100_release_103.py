"""The managed channel must advance the installed trainer package on code changes."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_managed_channel_is_a_genuine_pinned_new_package():
    manifest = json.loads((ROOT / "tools/managed-release.json").read_text())
    config = (ROOT / "pyproject.toml").read_text()
    assert manifest["version"] == 110
    assert 'version = "0.1.3+v100.110"' in config
    installer = manifest["linux_installer"]
    assert installer == "tools/update-synta-v110.sh"
    content = (ROOT / installer).read_bytes()
    assert manifest["files"][installer] == hashlib.sha256(content).hexdigest()


def test_managed_channel_v110_redirects_legacy_income_link_after_port_override():
    manifest = json.loads((ROOT / "tools/managed-release.json").read_text())
    config = (ROOT / "pyproject.toml").read_text()
    assert manifest["version"] == 110
    assert 'version = "0.1.3+v100.110"' in config
    assert manifest["linux_installer"] == "tools/update-synta-v110.sh"
    assert "rlm/v100/dashboard_redirect.py" in manifest["files"]
    installer = (ROOT / manifest["linux_installer"]).read_text()
    assert "range(78, 111)" in installer
    assert "synta-income-learning.service" in installer
    assert "v100-dashboard.service" in installer
    assert "--port 8766" in installer
    assert "zz-merged-dashboard.conf" in installer
    assert "--target http://192.168.0.68:8765/?v=110" in installer


def test_v107_through_v110_updaters_are_monotonic_successors():
    v107 = (ROOT / "tools/update-synta-v107.sh").read_text()
    v108 = (ROOT / "tools/update-synta-v108.sh").read_text()
    v109 = (ROOT / "tools/update-synta-v109.sh").read_text()
    v110 = (ROOT / "tools/update-synta-v110.sh").read_text()
    manifest = json.loads((ROOT / "tools/managed-release.json").read_text())
    assert "tools/update-synta-v106.sh" not in manifest["files"]
    assert "range(78, 108)" in v107
    assert "requires v78-v107" in v107
    assert "0.1.3+v100.107" in v107
    assert "v107-package-recovery" in v107
    assert "range(78, 109)" in v108
    assert "requires v78-v108" in v108
    assert "0.1.3+v100.108" in v108
    assert "v108-package-recovery" in v108
    assert "range(78, 110)" in v109
    assert "requires v78-v109" in v109
    assert "0.1.3+v100.109" in v109
    assert "v109-package-recovery" in v109
    assert "range(78, 111)" in v110
    assert "requires v78-v110" in v110
    assert "0.1.3+v100.110" in v110
    assert "v110-package-recovery" in v110


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
    git = shutil.which("git")
    if git:
        git_bash = Path(git).parents[1] / "bin/bash.exe"
        if git_bash.is_file():
            bash = str(git_bash)
    assert bash, "Bash is required to validate the Linux installer"
    parsed = subprocess.run([bash, "-n", str(script)], capture_output=True, text=True)
    assert parsed.returncode == 0, parsed.stderr


@pytest.mark.parametrize("release", [104, 105, 106, 107, 108, 109, 110])
def test_recovery_release_keeps_resource_policy_and_has_package_rollback(release):
    content = (ROOT / f"tools/update-synta-v{release}.sh").read_text()
    assert "active candidate training/evaluation retained".lower() in content.lower()
    assert "flock -n -x 9" in content
    assert '"$RECOVERY_HELPER" restore' in content
    assert content.index("'snapshot'") < content.index("print(stop(root)")
    assert "resource-policy.json" not in content
    assert "attach-synta-3060.sh" not in content
    assert "synta-owned-nodes.service" not in content
    assert f"0.1.3+v100.{release}" in content
    bash = shutil.which("bash")
    git = shutil.which("git")
    if git:
        git_bash = Path(git).parents[1] / "bin/bash.exe"
        if git_bash.is_file():
            bash = str(git_bash)
    assert bash
    parsed = subprocess.run(
        [bash, "-n", str(ROOT / f"tools/update-synta-v{release}.sh")],
        capture_output=True,
        text=True,
    )
    assert parsed.returncode == 0, parsed.stderr
