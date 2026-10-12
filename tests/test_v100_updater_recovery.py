import os
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(params=[82, 83, 84, 85, 86, 87, 88])
def script(request):
    return Path(__file__).parents[1] / f"tools/update-synta-v{request.param}.sh"


def host(tmp_path):
    root = tmp_path / "ai-v100"
    (root / "research").mkdir(parents=True)
    (root / "venvs/v100-continual/bin").mkdir(parents=True)
    (root / "env.sh").write_text(f'export AI_V100_ROOT="{root}"\n')
    environment = {**os.environ, "HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
    python = root / "venvs/v100-continual/bin/python"
    return root, environment, python


def executable(path, code):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/bash\n" + code)
    path.chmod(0o755)


def test_missing_uv_cannot_stop_mission(tmp_path, script):
    root, environment, python = host(tmp_path)
    executable(python, f'touch "{root}/python-called"\n')
    result = subprocess.run(
        ["/bin/bash", str(script), "a" * 40], env=environment, capture_output=True, text=True
    )
    assert result.returncode == 127
    assert "before any service change" in result.stderr
    assert not (root / "python-called").exists()


def test_noninteractive_local_uv_is_resolved_and_failed_install_recovers(tmp_path, script):
    root, environment, python = host(tmp_path)
    uv = tmp_path / ".local/bin/uv"
    executable(uv, 'if [[ "$1" == "--version" ]]; then echo uv-test; exit 0; fi\nexit 9\n')
    executable(
        python,
        f'''input="$(/bin/cat)"
if [[ "$input" == *"Preserved recovery profile unavailable"* ]]; then
    touch "{root}/recovery-attempted"
    exit 0
fi
printf '%s' '{root}/research/profile.json' > "$SYNTA_RESTORE_MARKER"
exit 0
''',
    )
    result = subprocess.run(
        ["/bin/bash", str(script), "a" * 40], env=environment, capture_output=True, text=True
    )
    assert result.returncode == 9
    assert "uv-test" in result.stdout
    assert (root / "recovery-attempted").exists()
    assert not list((root / "research").glob(".update-v*-restore.*"))
