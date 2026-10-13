import json

from rlm.v100 import chat_progress, decision_memory, managed_bootstrap, mission_chat


def test_recover_status_timeout_never_replays_command_or_overwrites_reply(tmp_path, monkeypatch):
    question = "Możesz już skutecznie testami zarabiać? trenujesz już przez 4 dni…"
    with mission_chat.connect(tmp_path) as db:
        for identity, message, response in [
            ("question", question, None),
            ("command", question + " Kup BTC.", None),
            ("receipt", question, "{}"),
        ]:
            db.execute(
                "INSERT INTO requests(id,created,message,state,response,error) VALUES(?, 'now', ?, 'failed', ?, 'Read timed out')",
                (identity, message, response),
            )
    monkeypatch.setattr(
        chat_progress,
        "respond",
        lambda root, message: {"answer": "Current receipts", "actions": [], "applied": []},
    )
    result = mission_chat.resolve_failed_status_questions(tmp_path)
    assert result == {"resolved_read_only_questions": ["question"], "actions_replayed": 0}
    with mission_chat.connect(tmp_path) as db:
        records = {row["id"]: dict(row) for row in db.execute("SELECT * FROM requests")}
    assert records["question"]["state"] == "completed"
    assert json.loads(records["question"]["response"])["previous_error"] == "Read timed out"
    assert records["command"]["state"] == "failed"
    assert records["receipt"]["response"] == "{}"


def test_earnings_progress_question_does_not_require_inference():
    assert chat_progress.requested(
        "Możesz już skutecznie testami zarabiać? trenujesz już przez 4 dni…"
    )
    assert chat_progress.requested("Can you earn money yet? Have you been training?")
    assert not chat_progress.requested("Możesz zarabiać? Zrób testy i uruchom trening.")
    assert not chat_progress.requested("Czy zarabiasz? Kup BTC.")
    assert not chat_progress.requested("Start training so you can earn money")


def test_public_memory_is_bounded_goal_scoped_and_not_reasoning(tmp_path, monkeypatch):
    goal = "a" * 64
    monkeypatch.setattr(decision_memory, "goal_id", lambda root: goal)
    for i in range(20):
        decision_memory.record(
            tmp_path,
            [{"role": "user", "content": f"task {i}"}],
            {"answer": "😀" * 2000, "reasoning": "private reasoning"},
            [{"tool": "read_source"}],
        )
    path = tmp_path / "research/decision-memory" / (goal + ".json")
    assert path.stat().st_size <= decision_memory.MAX_BYTES
    notes = decision_memory.read(tmp_path, goal)
    assert 1 <= len(notes) <= 8
    assert "private reasoning" not in path.read_text()
    message = {"role": "user", "content": "current authenticated request"}
    attached = decision_memory.attach(tmp_path, [message])
    assert attached[-1] == message
    assert "unverified" in attached[0]["content"]
    assert len(attached[0]["content"]) < 2000
    assert decision_memory.attach(tmp_path, attached) == attached
    monkeypatch.setattr(decision_memory, "goal_id", lambda root: "b" * 64)
    assert decision_memory.attach(tmp_path, [message]) == [message]


def test_optional_memory_does_not_consume_near_full_context(tmp_path, monkeypatch):
    monkeypatch.setattr(decision_memory, "goal_id", lambda root: "a" * 64)
    decision_memory.record(
        tmp_path, [{"role": "user", "content": "task"}], {"answer": "Unverified proposal"}, []
    )
    messages = [{"role": "user", "content": "x" * 4000}]
    assert decision_memory.attach(tmp_path, messages, 8192) == messages
    assert decision_memory.attach(tmp_path, messages, 0) == messages


def test_invalid_or_missing_goal_does_not_block_memory(tmp_path, monkeypatch):
    def invalid(root):
        raise ValueError("invalid goal")

    monkeypatch.setattr(decision_memory, "goal_id", invalid)
    messages = [{"role": "user", "content": "task"}]
    assert decision_memory.attach(tmp_path, messages) == messages
    decision_memory.record(tmp_path, messages, {"answer": "result"}, [])
    assert not (tmp_path / "research/decision-memory").exists()


def test_gateway_runtime_is_separate_from_trainer(tmp_path, monkeypatch):
    monkeypatch.setattr(managed_bootstrap.Path, "home", lambda: tmp_path / "home")
    calls = []
    monkeypatch.setattr(
        managed_bootstrap.subprocess, "run", lambda command, **kwargs: calls.append(command)
    )
    result = managed_bootstrap.remote_chat_update(
        tmp_path,
        {"stage": "ready", "revision": "c" * 40, "code_path": str(tmp_path / "runtime")},
        "c" * 40,
    )
    assert result["stage"] == "configured"
    assert any("synta-remote.service" in command for command in calls)
    assert not any("v100-mission.service" in command for command in calls)
    assert not any("pip" in command for command in calls)
    before = len(calls)
    managed_bootstrap.remote_chat_update(
        tmp_path,
        {"stage": "ready", "revision": "c" * 40, "code_path": str(tmp_path / "runtime")},
        "c" * 40,
    )
    assert len(calls) == before
    assert (
        json.loads((tmp_path / "research/remote-chat-deployment.json").read_text())["revision"]
        == "c" * 40
    )


def test_unverified_runtime_never_restarts_gateway(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        managed_bootstrap.subprocess, "run", lambda command, **kwargs: calls.append(command)
    )
    assert (
        managed_bootstrap.remote_chat_update(tmp_path, {"stage": "blocked"}, "c" * 40)["stage"]
        == "blocked"
    )
    assert calls == []
