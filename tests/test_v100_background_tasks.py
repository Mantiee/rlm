import json
from pathlib import Path

import pytest

from rlm.v100 import financial_snapshot


def test_saved_financial_report_has_separate_bounded_budget(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"padding": "x" * (3 * 2**20)}))
    with pytest.raises(ValueError, match="bounded byte budget"):
        financial_snapshot.read_object(path)
    assert len(financial_snapshot.read_object(path, 32 * 2**20)["padding"]) == 3 * 2**20


def test_background_task_launcher_keeps_windowless_exit_and_owned_scope():
    source = (Path(__file__).parents[1] / "tools/install-synta-background-tasks.ps1").read_text()
    assert "shell.Run(" in source and ", 0, True)" in source
    assert "WScript.Quit code" in source
    assert "Task is outside the owned PowerShell scope" in source
    assert "Synta-Managed-Update" in source
    assert "Stop-ScheduledTask" not in source
