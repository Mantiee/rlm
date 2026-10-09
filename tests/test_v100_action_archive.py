import json

import pytest

from rlm.v100.activity import ActivityLog
from rlm.v100.activity_browser import days, page


def test_archive_pages_preserve_every_recorded_action_and_context(tmp_path):
    journal = ActivityLog(tmp_path, "A", "researcher")
    folder = tmp_path / "research"
    folder.mkdir()
    (folder / "goal.json").write_text(json.dumps({"id": "active-goal"}))
    ids = [
        journal.write(
            "tools",
            "tool-start",
            {"tool": "calculate", "arguments": {"expression": str(i)}},
            task_id="task",
        )
        for i in range(5)
    ]
    day = days(tmp_path)[0]
    cursor, seen = 0, []
    while True:
        result = page(tmp_path, day, cursor, 2)
        seen += result["events"]
        cursor = result["next_cursor"]
        if not result["has_more"]:
            break
    assert [e["id"] for e in seen] == ids
    assert all(e["context"]["operator_goal_id_at_event"] == "active-goal" for e in seen)
    assert all(e["context"]["task_id"] == "task" for e in seen)


def test_archive_rejects_escape_symlink_invalid_cursor_and_incomplete_tail(tmp_path):
    folder = tmp_path / "research/logs/activity/2026-10-09"
    folder.mkdir(parents=True)
    record = json.dumps({"id": "complete"}).encode() + b"\n"
    path = folder / "timeline.jsonl"
    path.write_bytes(record + b'{"id":')
    result = page(tmp_path, "2026-10-09")
    assert result["next_cursor"] == len(record)
    assert result["events"] == [{"id": "complete"}]
    for day, cursor, limit in [
        ("../secret", 0, 1),
        ("2026-10-09", -1, 1),
        ("2026-10-09", 999999, 1),
        ("2026-10-09", 0, 101),
    ]:
        with pytest.raises(ValueError):
            page(tmp_path, day, cursor, limit)
    path.unlink()
    secret = tmp_path / "secret.json"
    secret.write_text('{"secret": "never"}')
    path.symlink_to(secret)
    assert days(tmp_path) == []
    with pytest.raises(ValueError):
        page(tmp_path, "2026-10-09")
