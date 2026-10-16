import json
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import requests

from rlm.v100 import autonomous_planning as planner
from rlm.v100 import planning, remote_helper, windows_resource_proxy
from rlm.v100.common import atomic_json


def test_guard_uses_live_monotonic_age_despite_host_clock_difference():
    guard = {
        "contract": "synta-adaptive-windows-v1",
        "ready": True,
        "updated": 0,
        "heartbeat_age_seconds": 0.4,
        "reason": "spare headroom",
    }
    assert remote_helper.check_guard(guard) == guard
    guard["heartbeat_age_seconds"] = 120
    with pytest.raises(requests.ConnectionError, match="stale resource heartbeat"):
        remote_helper.check_guard(guard)


def test_guard_error_reports_missing_contract_instead_of_spare_headroom():
    with pytest.raises(requests.ConnectionError, match="missing guard contract"):
        remote_helper.check_guard({"ready": True, "updated": 0, "reason": "spare headroom"})


def test_proxy_reports_fresh_age_using_its_own_clock(tmp_path, monkeypatch):
    pressure = windows_resource_proxy.Pressure(tmp_path)
    pressure.state = {"ready": True, "updated": 0}
    pressure.last_sample = 10
    monkeypatch.setattr(windows_resource_proxy.time, "monotonic", lambda: 11)
    assert pressure.fresh_state()["heartbeat_age_seconds"] == 1
    assert pressure.fresh_state()["ready"]


def test_goal_changed_during_generation_prevents_plan_update(tmp_path, monkeypatch):
    monkeypatch.setattr(planning, "load_goal", lambda root: {"id": "new", "text": "income"})
    with pytest.raises(ValueError, match="Goal changed"):
        planning.update(tmp_path, "short", "Test one outcome", "A", expected_goal_id="old")
    assert not (tmp_path / "research/plans/current.json").exists()


def test_planner_replans_short_and_mid_without_replacing_operator_goal(tmp_path, monkeypatch):
    from rlm.v100 import agent, competition, mission_chat

    goal = {"id": "goal", "text": "lawful income"}
    monkeypatch.setattr(planner, "context", lambda root: {"plans": {"long": goal}, "evidence": {}})
    monkeypatch.setattr(planning, "load_goal", lambda root: goal)
    monkeypatch.setattr(remote_helper, "selected_helper", lambda *a: tmp_path / "profile.json")
    monkeypatch.setattr(planner, "load_profile", lambda *a: {})
    client = SimpleNamespace(timeout=5, sampling_args={}, model_name="qualified")
    monkeypatch.setattr(competition, "helper_client", lambda *a: client)
    monkeypatch.setattr(mission_chat, "backend_preflight", lambda client: None)
    monkeypatch.setattr(
        agent,
        "native_turn",
        lambda *a, **k: {
            "content": json.dumps(
                {
                    "short": "Run a cost-controlled paper test; success means beating cash out of sample.",
                    "mid": "Compare verified service deliverables and paper results; reject repeated failures.",
                }
            )
        },
    )
    result = planner.run(tmp_path)
    assert result["stage"] == "replanned"
    assert planning.read(tmp_path)["long"] == goal
    assert planning.read(tmp_path)["short"]["author"] == "A"


def test_invalid_plan_output_cannot_write_any_plan(tmp_path, monkeypatch):
    from rlm.v100 import agent, competition, mission_chat

    monkeypatch.setattr(
        planner, "context", lambda root: {"plans": {"long": {"id": "goal"}}, "evidence": {}}
    )
    monkeypatch.setattr(remote_helper, "selected_helper", lambda *a: tmp_path / "profile.json")
    monkeypatch.setattr(planner, "load_profile", lambda *a: {})
    monkeypatch.setattr(
        competition,
        "helper_client",
        lambda *a: SimpleNamespace(sampling_args={}, model_name="test"),
    )
    monkeypatch.setattr(mission_chat, "backend_preflight", lambda client: None)
    monkeypatch.setattr(agent, "native_turn", lambda *a, **k: {"content": '{"short":"incomplete"}'})
    with pytest.raises(ValueError, match="incomplete"):
        planner.run(tmp_path)
    assert not (tmp_path / "research/plans/current.json").exists()


def test_planning_failure_keeps_future_retries_and_other_collectors_available(tmp_path):
    future = Future()
    future.set_exception(RuntimeError("helper unavailable"))
    key = str(tmp_path.resolve())
    planner.PENDING[key] = future
    atomic_json(tmp_path / "research/plans/autonomous-status.json", {"fingerprint": "saved"})
    planner.tick(tmp_path)
    status = json.loads((tmp_path / "research/plans/autonomous-status.json").read_text())
    assert status["stage"] == "planning-blocked" and status["fingerprint"] == "saved"
    assert key not in planner.PENDING


def test_only_read_only_guard_failures_are_reopened_without_command_replay(tmp_path):
    from rlm.v100 import mission_chat

    question = mission_chat.submit(
        tmp_path, "Możesz już skutecznie testami zarabiać? Trenujesz już trzy dni."
    )
    command = mission_chat.submit(tmp_path, "Zrób sobie free compute i zarejestruj konto.")
    with mission_chat.connect(tmp_path) as db:
        db.execute(
            "UPDATE requests SET state='failed',error='Windows adaptive guard unavailable: spare headroom'"
        )
    result = mission_chat.reopen_unavailable_questions(tmp_path)
    assert result["reopened_questions"] == [question]
    with mission_chat.connect(tmp_path) as db:
        assert (
            db.execute("SELECT state FROM requests WHERE id=?", (question,)).fetchone()[0]
            == "queued"
        )
        assert (
            db.execute("SELECT state FROM requests WHERE id=?", (command,)).fetchone()[0]
            == "failed"
        )
    before = json.loads((tmp_path / "research/chat-recovery" / (question + ".json")).read_text())
    assert before["state"] == "failed"


def test_income_work_can_use_pinned_windows_helper_when_v100_is_training(tmp_path, monkeypatch):
    from rlm.v100 import competition, income_work, mission_chat, remote_nodes

    path = tmp_path / "windows.json"
    client = SimpleNamespace(timeout=5, sampling_args={}, identity=lambda: None)
    monkeypatch.setattr(remote_nodes, "selected_node", lambda root: None)
    monkeypatch.setattr(remote_helper, "selected_helper", lambda *a: path)
    monkeypatch.setattr(mission_chat, "preferences", lambda root: {})
    monkeypatch.setattr(
        income_work,
        "load_profile",
        lambda *a: {
            "runtime": {"base_url": "http://helper.invalid:11435"},
            "resources": {"device": "remote"},
        },
    )
    monkeypatch.setattr(mission_chat, "backend_preflight", lambda client: None)
    monkeypatch.setattr(income_work, "income_profile_enabled", lambda root, profile: True)
    monkeypatch.setattr(competition, "helper_client", lambda *a: client)
    assert income_work.owned_advisor(tmp_path, "A", "V100 training") is client
    assert client.research_token_ceiling == 2048
    assert client.activity_actor == "income-owned-advisor"
