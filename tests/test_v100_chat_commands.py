import json
import threading
import time

import pytest
import requests

from rlm.v100 import chat_commands, mission_chat
from tests.test_v100_remote_access import CONFIG, gateway, request


def wait_result(root, identity):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        row = mission_chat.inspect(root, identity)
        if row["state"] != "queued":
            return row
        time.sleep(0.02)
    pytest.fail("Gateway command did not finish")


def test_authenticated_gateway_executes_command_without_trainer_worker(tmp_path, monkeypatch):
    suite = tmp_path / "research/income-challenge-v1/development.jsonl"
    suite.parent.mkdir(parents=True)
    suite.write_text(
        json.dumps(
            {
                "id": "fixed",
                "match": "exact",
                "expected": "4",
                "skill": "arithmetic",
                "messages": [{"role": "user", "content": "2+2"}],
            }
        )
        + "\n"
    )
    calls = []

    def respond(root, directory, row, accepted_cpu=None):
        calls.append(row["id"])
        # Exercise a real persisted control operation, not an arithmetic canary.
        return original(root, directory, row, accepted_cpu)

    original = mission_chat.respond
    monkeypatch.setattr(mission_chat, "respond", respond)
    monkeypatch.setattr(mission_chat, "submit", lambda *a: pytest.fail("Old worker used"))
    with gateway(tmp_path) as (server, base):
        _, _, body = request(
            base,
            "/api/chat",
            body=json.dumps({"message": "/goal Investigate zero-cost public compute"}).encode(),
            Origin=CONFIG["origin"],
            **{"X-Synta-CSRF": server.csrf, "Content-Type": "application/json"},
        )
        row = json.loads(body)
        assert row["route"] == "gateway-commands"
        result = wait_result(tmp_path, row["id"])
        assert result["state"] == "completed"
        assert result["response"]["applied"][0]
        assert calls == [row["id"]]
        with mission_chat.connect(tmp_path) as db:
            assert (
                db.execute("SELECT count(*) FROM requests WHERE state='queued'").fetchone()[0] == 0
            )
    from rlm.v100.planning import read

    assert read(tmp_path)["long"]["text"] == "Investigate zero-cost public compute"


def test_interrupted_commands_are_failed_and_never_replayed(tmp_path, monkeypatch):
    identity = chat_commands.submit(tmp_path, "Execute task")
    with mission_chat.connect(tmp_path) as db:
        db.execute("UPDATE requests SET state=? WHERE id=?", (chat_commands.RUNNING, identity))
    monkeypatch.setattr(
        chat_commands, "response", lambda *a: pytest.fail("Interrupted command replayed")
    )
    stop = threading.Event()
    worker = threading.Thread(target=chat_commands.work, args=(tmp_path, stop))
    worker.start()
    result = wait_result(tmp_path, identity)
    stop.set()
    worker.join(2)
    assert result["state"] == "failed" and "not replayed" in result["error"]


def test_transport_failure_after_action_is_not_retried(tmp_path, monkeypatch):
    calls = []

    def response(*args):
        calls.append(1)
        raise requests.ConnectionError("stream broke after tool")

    monkeypatch.setattr(chat_commands, "response", response)
    identity = chat_commands.submit(tmp_path, "Execute task")
    stop = threading.Event()
    worker = threading.Thread(target=chat_commands.work, args=(tmp_path, stop))
    worker.start()
    result = wait_result(tmp_path, identity)
    stop.set()
    worker.join(2)
    assert result["state"] == "failed" and calls == [1]


def test_only_pending_commands_can_be_cancelled(tmp_path):
    identity = chat_commands.submit(tmp_path, "Execute task")
    assert chat_commands.submit(tmp_path, "Execute task") == identity
    assert mission_chat.submit(tmp_path, "/cancel-chat " + identity) == identity
    assert mission_chat.inspect(tmp_path, identity)["state"] == "cancelled"
    second = chat_commands.submit(tmp_path, "Another task")
    with mission_chat.connect(tmp_path) as db:
        db.execute("UPDATE requests SET state=? WHERE id=?", (chat_commands.RUNNING, second))
    with pytest.raises(ValueError, match="Only pending"):
        mission_chat.submit(tmp_path, "/cancel-chat " + second)


def test_command_probe_requires_actual_tool_receipt(tmp_path, monkeypatch):
    from rlm.v100 import chat_probe, chat_readonly
    from rlm.v100.common import atomic_json

    atomic_json(tmp_path / "research/remote-access/config.json", CONFIG)
    monkeypatch.setattr(chat_readonly, "submit", lambda *a: None)
    monkeypatch.setattr(mission_chat, "direct_facts", lambda *a: None)
    monkeypatch.setattr(
        chat_commands,
        "response",
        lambda *a: {
            "answer": "Current goal",
            "actions": [],
            "applied": [],
            "tool_receipts": [{"tool": "get_plan", "result": {"long": {"text": "Existing goal"}}}],
        },
    )
    with gateway(tmp_path) as (server, _):
        result = chat_probe.run(
            tmp_path, "a" * 40, timeout=5, port=server.server_port, command=True
        )
    assert result["stage"] == "verified" and result["tool_receipts"][0]["tool"] == "get_plan"


def test_greeting_followed_by_income_question_is_read_only():
    assert mission_chat.conversation_question("hejka kiedy bedziesz zarabial?")
    assert not mission_chat.conversation_question("hejka kiedy zaczniesz? uruchom zadanie")


def test_gateway_command_owns_accepted_cpu_fallback_and_closes_it(tmp_path, monkeypatch):
    from rlm.v100 import chat_backend

    events = []
    manager = type(
        "CPU", (), {"maintain": lambda self: None, "close": lambda self: events.append("closed")}
    )()
    monkeypatch.setattr(chat_backend, "AcceptedCPUChat", lambda *a: manager)

    def respond(root, directory, row, accepted_cpu):
        assert accepted_cpu is manager
        return {"answer": "Accepted CPU reply", "actions": [], "applied": []}

    monkeypatch.setattr(mission_chat, "respond", respond)
    identity = chat_commands.submit(tmp_path, "Execute task")
    stop = threading.Event()
    worker = threading.Thread(target=chat_commands.work, args=(tmp_path, stop))
    worker.start()
    assert wait_result(tmp_path, identity)["state"] == "completed"
    stop.set()
    worker.join(2)
    assert events == ["closed"]
