import json
import time
from pathlib import Path
from types import SimpleNamespace

from rlm.v100 import income_work
from rlm.v100.common import atomic_json
from rlm.v100.income_work import (
    WINDOWS_3090_ORIGIN,
    backend_on_cooldown,
    income_profile_enabled,
    income_remote_origin_enabled,
)


def test_income_remote_origin_policy_disables_only_windows_by_default(tmp_path):
    windows = "http://192.168.0.61:11435"
    linux = "http://192.168.0.9:11435"
    assert WINDOWS_3090_ORIGIN == windows
    assert not income_remote_origin_enabled(tmp_path, windows)
    assert income_remote_origin_enabled(tmp_path, linux)
    assert income_profile_enabled(tmp_path, {"resources": {"device": "cuda"}, "runtime": {}})

    atomic_json(
        tmp_path / "research/user-preferences.json",
        {"remote_origin_enabled": {windows: True}},
    )
    assert income_remote_origin_enabled(tmp_path, windows)
    assert income_remote_origin_enabled(tmp_path, linux)

    atomic_json(
        tmp_path / "research/user-preferences.json",
        {
            "remote_helper_enabled": False,
            "remote_origin_enabled": {windows: True},
        },
    )
    assert not income_remote_origin_enabled(tmp_path, windows)
    assert not income_remote_origin_enabled(tmp_path, linux)


def test_income_worker_skips_disabled_planner_before_preflight_and_uses_local_v100(
    tmp_path, monkeypatch
):
    import rlm.v100.competition as competition
    import rlm.v100.mission_chat as mission_chat
    import rlm.v100.model_roles as model_roles
    import rlm.v100.serving as serving

    planner = tmp_path / "research/model-roles/planner.json"
    planner.parent.mkdir(parents=True)
    planner.write_text("{}")
    run = tmp_path / "research/mission/run-current"
    run.mkdir(parents=True)
    (tmp_path / "research/mission").mkdir(parents=True, exist_ok=True)
    atomic_json(tmp_path / "research/mission/active.json", {"run": str(run)})
    accepted_path = run / "serving-active.json"
    accepted_path.write_text("{}")

    windows_profile = {
        "resources": {"device": "remote"},
        "runtime": {"base_url": WINDOWS_3090_ORIGIN},
    }
    local_v100_profile = {
        "resources": {"device": "cuda"},
        "runtime": {"base_url": "http://127.0.0.1:8080"},
        "server": {"model": "accepted-v100.gguf"},
    }
    monkeypatch.setattr(model_roles, "selected_profile", lambda root, role: planner)
    monkeypatch.setattr(
        income_work,
        "load_profile",
        lambda path, root: windows_profile if Path(path) == planner else local_v100_profile,
    )
    monkeypatch.setattr(
        competition,
        "helper_client",
        lambda *args: SimpleNamespace(
            timeout=60,
            sampling_args={},
            enable_thinking=True,
            activity_actor="tester",
        ),
    )
    preflights = []
    monkeypatch.setattr(mission_chat, "backend_preflight", lambda client: preflights.append(client))
    identities = []
    monkeypatch.setattr(
        serving,
        "assert_served_expert",
        lambda client, profile, root: identities.append(profile),
    )

    client = income_work.accepted_master(tmp_path, "A")

    assert client.activity_actor == "income-research"
    assert preflights == []
    assert identities == [local_v100_profile]


def test_income_origin_policy_keeps_linux_candidate_enabled_and_allows_explicit_disable(
    tmp_path,
):
    linux = "http://192.168.0.9:11435"
    other = "http://192.168.0.10:11435"
    assert income_remote_origin_enabled(tmp_path, linux)
    atomic_json(
        tmp_path / "research/user-preferences.json",
        {"remote_origin_enabled": {linux: False}},
    )
    assert not income_remote_origin_enabled(tmp_path, linux)
    assert income_remote_origin_enabled(tmp_path, other)


def test_backend_cooldown_only_skips_exact_recent_failing_http_503(tmp_path):
    file = tmp_path / "research/income-work/status.json"
    file.parent.mkdir(parents=True)
    file.write_text(
        json.dumps(
            {
                "state": "model-selected work blocked",
                "error": (
                    "503 Server Error: Service Unavailable for url: http://127.0.0.1:11435/api/chat"
                ),
                "updated": time.time(),
                "retry_at": time.time() + 20,
            }
        )
    )
    assert backend_on_cooldown(tmp_path, "http://127.0.0.1:11435")
    assert not backend_on_cooldown(tmp_path, "http://127.0.0.1:11436")
    assert not backend_on_cooldown(tmp_path, "http://127.0.0.1:11437")


def test_income_cooldown_expires_and_never_blocks_healthy_work(tmp_path):
    file = tmp_path / "research/income-work/status.json"
    file.parent.mkdir(parents=True)
    error = "503 for url: http://127.0.0.1:11435/api/chat"
    for retry_at in (time.time() - 1, time.time() + 100):
        file.write_text(
            json.dumps(
                {"state": "model-selected work blocked", "error": error, "retry_at": retry_at}
            )
        )
        assert backend_on_cooldown(tmp_path, "http://127.0.0.1:11435") == (retry_at > time.time())
    file.write_text(
        json.dumps(
            {"state": "model-selected work recorded", "error": error, "updated": time.time()}
        )
    )
    assert not backend_on_cooldown(tmp_path, "http://127.0.0.1:11435")


def test_cooldown_survives_selection_receipt_and_scopes_exact_host(tmp_path, monkeypatch):
    failed = {
        "state": "deferred",
        "error": ("503 Server Error: Service Unavailable for url: http://127.0.0.1:11435/api/chat"),
        "updated": time.time(),
        "retry_after_seconds": 30,
        "retry_at": time.time() + 30,
    }
    path = tmp_path / "research/income-work/status.json"
    atomic_json(path, failed)
    monkeypatch.setattr(income_work, "load_goal", lambda _: {"id": "goal"})
    observed = []

    def unavailable(root, branch):
        observed.append(backend_on_cooldown(root, "http://127.0.0.1:11435"))
        observed.append(backend_on_cooldown(root, "http://127.0.0.1:11436"))
        raise income_work.MasterNotReady("No admitted alternative yet")

    monkeypatch.setattr(income_work, "accepted_master", unavailable)
    result = income_work.execute_goal_work(
        tmp_path,
        {"id": "work", "branch": "A"},
        {"id": "goal"},
        {},
    )
    assert result["state"] == "model-selected work blocked"
    assert observed == [True, False]
    assert backend_on_cooldown(tmp_path, "http://127.0.0.1:11435")
    assert (tmp_path / "research/income-work/backend-cooldown.json").is_file()
    assert json.loads(path.read_text())["error"] == "No admitted alternative yet"


def test_inactive_or_expired_failure_is_never_saved_as_cooldown(tmp_path):
    receipt = tmp_path / "research/income-work/backend-cooldown.json"
    income_work.record_backend_cooldown(
        tmp_path,
        {
            "state": "deferred",
            "error": "503 for url: http://127.0.0.1:11435/api/chat",
            "updated": time.time() - 950,
            "retry_after_seconds": 30,
        },
    )
    assert not receipt.exists()


def test_generic_503_without_typed_deferred_receipt_does_not_set_cooldown(tmp_path):
    receipt = tmp_path / "research/income-work/backend-cooldown.json"
    income_work.record_backend_cooldown(
        tmp_path,
        {
            "state": "model-selected work blocked",
            "error": "503 for url: http://127.0.0.1:11435/api/chat",
            "updated": time.time(),
        },
    )
    assert not receipt.exists()
    income_work.record_backend_cooldown(
        tmp_path,
        {
            "state": "model-selected work recorded",
            "error": "503 for url: http://127.0.0.1:11435/api/chat",
            "updated": time.time(),
            "retry_after_seconds": 30,
        },
    )
    assert not receipt.exists()


def test_old_unbounded_503_receipt_does_not_block_for_fifteen_minutes(tmp_path):
    receipt = tmp_path / "research/income-work/backend-cooldown.json"
    atomic_json(
        receipt,
        {
            "state": "model-selected work blocked",
            "error": "503 for url: http://127.0.0.1:11435/api/chat",
            "updated": time.time(),
        },
    )
    assert not backend_on_cooldown(tmp_path, "http://127.0.0.1:11435")
