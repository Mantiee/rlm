import json
from types import SimpleNamespace

import psutil
import pytest

from rlm.v100 import (
    chat_resources,
    goal_compute,
    mission_chat,
    research_tools,
    resource_budget,
    resource_optimization,
)
from rlm.v100.common import atomic_json


@pytest.mark.parametrize(
    "message",
    [
        "a to z cpu na debianie czy windowsie? możesz zwiększyć użycie albo optymalniej użycie czegokolwiek żeby wszystko było szybciej. zrób to",
        "Miałeś zrobić optymalizację całą. zrób ją.",
        "Optimize CPU and GPU resource allocation.",
    ],
)
def test_optimization_commands_are_execution_requests(message):
    assert resource_optimization.requested(message)


@pytest.mark.parametrize(
    "message",
    [
        "Optimize trading strategy",
        "Zrób backtest na CPU",
        "Jak działa optymalizacja?",
        "Napraw dashboard CPU",
    ],
)
def test_other_work_is_not_resource_tuning(message):
    assert not resource_optimization.requested(message)


def test_followups_bind_prior_operator_request_not_assistant_backtest(tmp_path):
    first = mission_chat.submit(tmp_path, "Zrób optymalizację CPU i RAM")
    with mission_chat.connect(tmp_path) as db:
        db.execute(
            "UPDATE requests SET state='completed',response=? WHERE id=?",
            (json.dumps({"answer": "Backtest completed"}), first),
        )
    for message in ("Zrób to wszystko.", "error, rób dalej"):
        identity = mission_chat.submit(tmp_path, message)
        assert (
            resource_optimization.context(tmp_path, {"id": identity, "message": message})
            == "Zrób optymalizację CPU i RAM"
        )
    unrelated = mission_chat.submit(tmp_path, "Zrób wykres transakcji")
    last = mission_chat.submit(tmp_path, "zrób to wszystko")
    assert unrelated != last
    assert (
        resource_optimization.context(tmp_path, {"id": last, "message": "zrób to wszystko"}) is None
    )


def test_tuning_is_direct_with_real_policy_receipt_and_preserved_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(resource_budget, "available_cpus", lambda: 12)
    monkeypatch.setattr(psutil, "cpu_percent", lambda interval: 10)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=20 * 2**30))
    evidence = {
        "helper": {"state": "disabled by operator"},
        "external_compute": {"workers": [{"stale": False, "phase": "idle"}], "jobs": []},
    }
    monkeypatch.setattr(chat_resources, "status", lambda root: evidence)
    calls = []
    monkeypatch.setattr(
        goal_compute,
        "propose",
        lambda root, branch: calls.append(branch)
        or {"state": "blocked", "reason": "Need verified outcomes"},
    )
    preferences = tmp_path / "research/user-preferences.json"
    atomic_json(preferences, {"remote_helper_enabled": False})
    before = preferences.read_bytes()
    monkeypatch.setattr(
        mission_chat, "load_profile", lambda *a: pytest.fail("LLM/profile routing reached")
    )
    result = mission_chat.respond(tmp_path, tmp_path, {"message": "Zrób optymalizację CPU i RAM"})
    policy = json.loads((tmp_path / "research/state/resource-policy.json").read_text())
    assert policy["max_cpu_slots"] == 4 and policy["reserve_ram_gib"] == 4
    assert resource_budget.local_budget(tmp_path)["cpu_slots"] == 4
    assert calls == ["A"] and preferences.read_bytes() == before
    assert result["responder"]["model"] == "controller-resource-optimization"
    assert result["applied"][0]["state"] == "applied"
    assert result["actions"] == [] and result["resource_contract"]["evidence"] == evidence
    assert not (tmp_path / "research/plans/current.json").exists()
    assert len(list((tmp_path / "research/resource-optimization").glob("*/receipt.json"))) == 1


@pytest.mark.parametrize(
    "cpu,ram,cpus,expected",
    [
        (10, 20, 12, 4),
        (70, 20, 12, 1),
        (90, 20, 12, 0),
        (10, 6, 12, 0),
        (10, 10, 12, 2),
        (10, 20, 4, 1),
    ],
)
def test_adaptive_tuned_budget_respects_pressure_memory_and_threads(
    tmp_path, monkeypatch, cpu, ram, cpus, expected
):
    atomic_json(
        tmp_path / "research/state/resource-policy.json",
        {"max_cpu_slots": 4, "reserve_ram_gib": 4, "ram_per_slot_gib": 3},
    )
    monkeypatch.setattr(resource_budget, "available_cpus", lambda: cpus)
    monkeypatch.setattr(psutil, "cpu_percent", lambda interval: cpu)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=ram * 2**30))
    assert resource_budget.local_budget(tmp_path)["cpu_slots"] == expected


def test_cpu_quota_and_affinity_count_ancestor_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(resource_budget.os, "sched_getaffinity", lambda pid: set(range(16)))
    group = tmp_path / "group"
    (group / "child").mkdir(parents=True)
    (group / "cpu.max").write_text("350000 100000")
    (group / "child/cpu.max").write_text("max 100000")
    membership = tmp_path / "membership"
    membership.write_text("0::/child\n")
    assert resource_budget.available_cpus(group, membership) == 3


def test_missing_result_is_explicit_non_retryable_failure_and_real_archive_reads(tmp_path):
    tools = research_tools.ResearchTools(tmp_path, {}, "A")
    missing = tools.execute("read_tool_result", {"sha256": "a" * 64, "offset": 0})
    assert missing["status"] == "failed" and missing["code"] == "archive_not_found"
    assert not missing["retryable"] and "No such file" not in missing["error"]
    descriptor = json.loads(research_tools.bounded_tool_result(tmp_path, {"text": "abc" * 2000}))
    page = tools.execute(
        "read_tool_result", {"sha256": descriptor["full_result_sha256"], "offset": 0}
    )
    assert page["sha256"] == descriptor["full_result_sha256"] and page["next_offset"] == 512


@pytest.mark.parametrize(
    "reply",
    [
        {"answer": "", "actions": []},
        {"answer": "All tools failed", "execution_status": "failed", "actions": []},
    ],
)
def test_empty_or_failed_reply_is_not_completed(tmp_path, monkeypatch, reply):
    identity = mission_chat.submit(tmp_path, "request")
    calls = []
    monkeypatch.setattr(mission_chat, "respond", lambda *a: calls.append(1) or reply)

    class Stop:
        def wait(self, seconds):
            return bool(calls)

    mission_chat.service_loop(tmp_path, tmp_path, Stop())
    result = mission_chat.inspect(tmp_path, identity)
    assert result["state"] == "failed" and result["error"]


def test_parallel_python_tasks_get_distinct_cpus_and_leave_reserve(monkeypatch):
    monkeypatch.setattr(resource_budget.os, "sched_getaffinity", lambda pid: set(range(12)))
    monkeypatch.setattr(resource_budget, "available_cpus", lambda: 6)
    first = resource_budget.cpu_assignment([])
    second = resource_budget.cpu_assignment([first])
    assert first == [1, 2] and second == [3, 4]
    assert not set(first).intersection(second)
    assert resource_budget.cpu_assignment([first, second]) == [5]
    assert resource_budget.cpu_assignment([first, second, [5]]) == []


@pytest.mark.parametrize("affinity", [[], [0, 0], [0, 2], [True], [0, 1, 2]])
def test_invalid_affinity_never_launches_sandbox(tmp_path, monkeypatch, affinity):
    from rlm.v100 import research_sandbox

    monkeypatch.setattr(research_sandbox.os, "sched_getaffinity", lambda pid: {0, 1})
    monkeypatch.setattr(
        research_sandbox, "runtime", lambda root: pytest.fail("Sandbox runtime reached")
    )
    with pytest.raises(ValueError, match="affinity"):
        research_sandbox.run(tmp_path, "A", "print(1)", cpu_affinity=affinity)


def test_empty_direct_reply_cannot_be_saved(tmp_path):
    with pytest.raises(ValueError, match="Empty direct"):
        mission_chat.save_direct_reply(tmp_path, "question", {"answer": ""})
    assert mission_chat.recent_requests(tmp_path) == []


def test_queued_windows_trial_has_visible_action_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(resource_budget, "available_cpus", lambda: 12)
    monkeypatch.setattr(psutil, "cpu_percent", lambda interval: 10)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=20 * 2**30))
    monkeypatch.setattr(
        chat_resources,
        "status",
        lambda root: {
            "external_compute": {"workers": [{"stale": False, "phase": "idle"}], "jobs": []}
        },
    )
    monkeypatch.setattr(
        goal_compute,
        "propose",
        lambda *a: {"state": "queued", "id": "owned-job", "weights_changed": False},
    )
    result = resource_optimization.apply(tmp_path, "Optimize CPU", "Optimize CPU")
    assert result["applied"][1]["id"] == "owned-job"
    assert result["applied"][1]["state"] == "queued"
    assert not result["applied"][1]["weights_changed"]
    assert "owned-job" in result["answer"]
