"""Operator-only v103 handoff is explicit, syntactically valid and self-recovering."""

import subprocess
from pathlib import Path


def test_operator_cutover_has_valid_bash_syntax_and_explicit_consent():
    path = Path(__file__).resolve().parents[1] / "tools/operator-cutover-v103.sh"
    subprocess.run(["bash", "-n", str(path)], check=True, timeout=10)
    script = path.read_text()
    assert "--accept-interrupt-current-cycle" in script
    assert "mission-maintenance.lock" in script
    assert "release.json" in script
    assert "NEEDS_RECOVERY=1" in script
    assert "trap recover_on_failure EXIT" in script
    assert "--resume" in script
    assert "systemctl --user stop v100-mission.service" in script
    assert "bash \"$INSTALLER\" \"$REV\"" in script
    assert "NEEDS_RECOVERY=0" in script
    assert "0.1.3+v100.103" in script
    assert "v100.102" in script
    assert "paper" not in script.lower() or "profit is not implied" in script.lower()
