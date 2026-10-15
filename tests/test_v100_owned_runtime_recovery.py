from pathlib import Path

import pytest

from rlm.v100.owned_node import worker_failure


@pytest.mark.parametrize(
    "line,expected",
    [
        ("ModuleNotFoundError: No module named 'numpy'", "Missing NumPy"),
        ("Requested interactive experiment device is unavailable", "CUDA training runtime"),
        ("Compute trial made no validated improvement", "weights not promoted"),
    ],
)
def test_known_worker_failure_does_not_hide_actionable_category(tmp_path, line, expected):
    log = tmp_path / "worker.log"
    log.write_text(line)
    assert expected in worker_failure(log, 1)


def test_unknown_worker_output_never_discloses_log_or_secret(tmp_path):
    log = tmp_path / "worker.log"
    log.write_text("password=private-value\nUnexpected arbitrary worker text")
    value = worker_failure(log, 7)
    assert "exited 7" in value
    assert "private-value" not in value and "arbitrary worker" not in value
    assert "exited 7" in worker_failure(tmp_path / "missing", 7)


def test_cuda_installer_retains_runtime_and_drains_before_gpu_execution():
    source = (Path(__file__).parents[1] / "tools/bootstrap-synta-3060.sh").read_text()
    assert 'VENV="$BASE/venvs/torch-2.6.0-cu124-py$PY_TAG"' in source
    assert "'numpy==2.1.3'" in source
    assert "Checkpoint round-trip failed" in source
    assert source.index('"$SOURCE/owned_maintenance.py"') < source.index("<<'CUDA_PROBE'")
    assert source.index("<<'CUDA_PROBE'") < source.index(
        'ln -sfn "$SOURCE" "$BASE/current"', source.index("<<'CUDA_PROBE'")
    )
