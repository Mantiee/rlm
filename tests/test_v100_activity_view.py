import json

from rlm.v100 import live_status, mission_chat


def test_pending_identical_chat_reuses_request_and_exposes_position(tmp_path):
    first = mission_chat.submit(tmp_path, "What did you learn?")
    assert mission_chat.submit(tmp_path, "What did you learn?") == first
    second = mission_chat.submit(tmp_path, "Show reports")
    assert mission_chat.inspect(tmp_path, second)["queue_position"] == 2
    with mission_chat.connect(tmp_path) as db:
        db.execute("UPDATE requests SET state='completed' WHERE id=?", (first,))
    assert mission_chat.submit(tmp_path, "What did you learn?") != first


def test_waiting_state_distinguishes_processing_from_queued(monkeypatch):
    monkeypatch.setattr(mission_chat.time, "time", lambda: 120)
    assert "position 3" in mission_chat.waiting_label({"id": "test", "queue_position": 3})
    assert "20s elapsed" in mission_chat.waiting_label(
        {"id": "test", "processing": {"started": 100, "time_budget_seconds": 180}}
    )


def test_chat_interrupt_does_not_cancel_request_or_print_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda prompt: "Do useful work")

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(mission_chat, "wait_reply", interrupt)
    mission_chat.chat(tmp_path)
    assert "mission continues" in capsys.readouterr().out
    with mission_chat.connect(tmp_path) as db:
        assert db.execute("SELECT state FROM requests").fetchone()[0] == "queued"


def test_agent_activity_preserves_evidence_without_private_thinking(tmp_path):
    folder = tmp_path / "research/logs/activity/2026-10-09"
    folder.mkdir(parents=True)
    rows = [
        {
            "time": "2026-10-09T10:00:00Z",
            "actor": "researcher",
            "branch": "A",
            "category": "decisions",
            "kind": "model-output",
            "payload": {
                "content": {
                    "hypothesis": "Compare source signals",
                    "rationale": "Use independently timestamped sources",
                    "thinking": "PRIVATE",
                    "reasoning_content": "PRIVATE",
                }
            },
        },
        {
            "time": "2026-10-09T10:00:01Z",
            "actor": "researcher",
            "branch": "A",
            "category": "tools",
            "kind": "tool-start",
            "payload": {"tool": "fetch_source", "arguments": {"url": "https://example.com/"}},
        },
    ]
    (folder / "timeline.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    events = live_status.recent_events(tmp_path)
    views = live_status.agent_views(events, [])
    assert views[0]["state"] == "tool-start"
    assert "independently timestamped" in views[0]["declaration"]
    assert "example.com" in views[0]["task"]
    assert "PRIVATE" not in json.dumps(views)
    assert views[0]["source"].endswith("timeline.jsonl")
