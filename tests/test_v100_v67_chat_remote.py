import json
from types import SimpleNamespace

import pytest

from rlm.v100 import (
    agent,
    chat_progress,
    chat_recovery,
    mission,
    mission_chat,
    planning,
    remote_connect,
)
from rlm.v100.common import atomic_json


@pytest.mark.parametrize(
    "message",
    [
        "to co dokladnie robisz, szczegoly, teraz",
        "no ale co aktualnie robisz?",
        "what exactly are you doing?",
        "dalej tylko tracisz",
    ],
)
def test_current_questions_are_read_only(message):
    assert mission_chat.conversation_question(message)


def test_activity_question_skips_gpu_and_previous_chat(tmp_path, monkeypatch):
    observed = {"answer": "Recorded current work.", "actions": [], "applied": []}
    monkeypatch.setattr(chat_progress, "respond", lambda root, message: observed)
    assert (
        mission_chat.direct_facts(tmp_path, "to co dokładnie robisz, szczegóły, teraz") == observed
    )
    assert (
        mission_chat.respond(
            tmp_path, tmp_path, {"message": "to co dokładnie robisz, szczegóły, teraz"}
        )
        == observed
    )


def test_activity_reply_shows_assignments_plans_actions_and_record_age(tmp_path, monkeypatch):
    from rlm.v100 import drones, live_status, mission_evidence

    monkeypatch.setattr(
        mission_evidence,
        "collect",
        lambda root: {
            "mission_running": True,
            "phase": "research",
            "learning": {},
            "optimizer_updates_observed": None,
            "accepted_weight_updates_this_run": 0,
            "reports": [],
            "run": None,
        },
    )
    jobs = [
        {
            "id": "job",
            "kind": "researcher",
            "branch": "A",
            "state": "running",
            "updated": 1,
            "assignment": "Fetch a new primary source; complete assignment retained.",
        }
    ]
    monkeypatch.setattr(drones, "inspect", lambda root: jobs)
    monkeypatch.setattr(
        planning, "read", lambda root: {"short": {"id": "plan", "text": "Test future outcomes."}}
    )
    events = [
        {
            "time": "2026-10-10T10:00:00Z",
            "kind": "tool-result",
            "actor": "researcher",
            "tool": "read_public_page",
            "summary": "report: source-receipt",
        }
    ]
    monkeypatch.setattr(live_status, "recent_events", lambda *a: events)
    result = chat_progress.respond(tmp_path, "co dokładnie robisz?")
    assert result["observed_jobs"] == jobs and result["observed_actions"] == events
    assert "source-receipt" in result["answer"] and "Test future outcomes." in result["answer"]
    assert "wiek zapisu" in result["answer"] and result["applied"] == []


@pytest.mark.parametrize("later", [False, True])
def test_question_plan_recovery_preserves_later_plans_and_originals(tmp_path, monkeypatch, later):
    monkeypatch.setattr(mission, "status", lambda root: {"running": False})
    previous = planning.update(tmp_path, "short", "Collect source evidence.", "A")
    bad = planning.update(tmp_path, "short", "Analyze falling knives without a receipt.", "A")
    identity = mission_chat.submit(tmp_path, "to co dokladnie robisz, szczegoly, teraz")
    with mission_chat.connect(tmp_path) as db:
        db.execute(
            "UPDATE requests SET state='completed',response=? WHERE id=?",
            (
                json.dumps(
                    {
                        "answer": "previous incomplete narrative",
                        "actions": [{"kind": "plan_short"}],
                        "applied": [bad],
                    }
                ),
                identity,
            ),
        )
    if later:
        good = planning.update(
            tmp_path, "short", "Operator requested a new independent test.", "user"
        )
    result = chat_recovery.repair(tmp_path)
    assert planning.read(tmp_path)["short"]["id"] == (good["id"] if later else previous["id"])
    assert result["requeued_questions"][0]["restored_plans"] == ([] if later else ["short"])
    assert mission_chat.inspect(tmp_path, identity)["state"] == "queued"
    assert (tmp_path / "research/plans" / (bad["id"] + ".json")).exists()
    assert chat_recovery.repair(tmp_path)["requeued_questions"] == []


def test_partial_grammatically_closed_answer_recovers_once_without_old_answer(tmp_path):
    calls = []
    partial = "Measured facts. " * 30 + "in the observed period when"

    def request(endpoint, payload):
        calls.append(payload)
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": partial
                        if len(calls) == 1
                        else "Nie mam potwierdzenia bieżącego zadania."
                    },
                }
            ]
        }

    client = SimpleNamespace(
        model_name="v100", timeout=30, template_args=lambda: {}, request=request
    )
    result = mission_chat.conversational_response(tmp_path, client, "co robi helper?", {}, {}, {})
    assert len(calls) == 2 and "tools" not in calls[1]
    assert partial not in json.dumps(calls[1])
    assert (
        result["answer"] == "Nie mam potwierdzenia bieżącego zadania." and result["actions"] == []
    )


def test_unconfirmed_finish_cannot_execute_tool():
    client = SimpleNamespace(
        sampling_args={"max_tokens": 10},
        context_window=100,
        model_name="fake",
        template_args=lambda: {},
        count_text=lambda *a, **k: 1,
    )
    client.request = (
        lambda endpoint, data: {"prompt": "small"}
        if endpoint == "/apply-template"
        else {"choices": [{"finish_reason": None, "message": {"content": "execute something"}}]}
    )
    with pytest.raises(ValueError, match="unconfirmed finish"):
        agent.native_turn(client, [{"role": "user", "content": "test"}])


@pytest.mark.parametrize("partial", [False, True])
def test_standalone_work_ignores_history_and_output_recovery_cannot_apply_plans(
    tmp_path, monkeypatch, partial
):
    from rlm.v100 import competition, mission_evidence, research_tools
    from tests.test_v100_continual import profile

    accepted = profile(tmp_path)
    accepted["resources"] = {"interactive_lab": True}
    atomic_json(tmp_path / "serving-active.json", accepted)
    monkeypatch.setattr(mission_chat, "load_profile", lambda *a: accepted)
    monkeypatch.setattr(
        mission, "status", lambda root: {"running": True, "state": {}, "learning": {}}
    )
    monkeypatch.setattr(mission_evidence, "collect", lambda root: {})
    previous = mission_chat.submit(tmp_path, "Previous unrelated question marker")
    with mission_chat.connect(tmp_path) as db:
        db.execute(
            "UPDATE requests SET state='completed',response=? WHERE id=?",
            (json.dumps({"answer": "Old answer marker", "actions": []}), previous),
        )
    client = SimpleNamespace(
        model_name="v100",
        timeout=60,
        sampling_args={"max_tokens": 1024},
        activity_context={},
        template_args=lambda: {},
    )
    calls = []

    def request(endpoint, payload=None):
        calls.append(endpoint)
        if endpoint == "/props":
            return {"model_path": accepted["server"]["model"]}
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "Brak potwierdzenia wykonania; nie zmieniono planu."},
                }
            ]
        }

    client.request = request
    monkeypatch.setattr(competition, "helper_client", lambda *a: client)
    seen = []

    def research(client, messages, schema, root):
        seen.extend(messages)
        action = {
            "kind": "plan_short",
            "text": "Unrequested proposal.",
            "target": "master",
            "thinking": False,
            "max_tokens": 256,
            "batch_tokens": 128,
            "enabled": False,
        }
        return {
            "content": json.dumps(
                {
                    "answer": "A complete result."
                    if not partial
                    else "A factual sentence. " * 20 + "and then in the",
                    "actions": [action] if partial else [],
                }
            ),
            "research_trace": [],
        }

    monkeypatch.setattr(research_tools, "research_turn", research)
    result = mission_chat.respond(
        tmp_path, tmp_path, {"message": "Zbadaj niezależne publiczne źródło", "id": "current"}
    )
    assert "Old answer marker" not in json.dumps(
        seen
    ) and "Previous unrelated question marker" not in json.dumps(seen)
    assert result["actions"] == [] and result["applied"] == []
    assert not (tmp_path / "research/plans/current.json").exists()
    assert calls == (["/props", "/v1/chat/completions"] if partial else ["/props"])
    if partial:
        assert result["answer_recovered"]


@pytest.mark.parametrize("configured", [False, True])
def test_phone_signin_is_not_serve_readiness(tmp_path, monkeypatch, configured):
    host = "debian1.example.ts.net"
    status = {"BackendState": "Running", "Self": {"DNSName": host}}
    serve = (
        {"Web": {host + ":443": {"Handlers": {"/": {"Proxy": "http://127.0.0.1:8786"}}}}}
        if configured
        else {}
    )
    monkeypatch.setattr(
        remote_connect.subprocess,
        "check_output",
        lambda args, **kw: json.dumps(serve if "serve" in args else status),
    )
    monkeypatch.setattr(
        remote_connect.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="active\n")
    )
    atomic_json(
        tmp_path / "research/remote-access/config.json",
        {
            "owner_login": "owner",
            "origin": "https://" + host,
            "dashboard": "http://192.168.0.68:8765",
        },
    )

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(remote_connect, "urlopen", lambda *a, **kw: Response())
    result = remote_connect.status(tmp_path)
    assert result["ready"] == configured
    if not configured:
        assert "approve" in result["next_step"] and "Serve" in result["blockers"][0]
    assert "iPhone" in result["scope"]
