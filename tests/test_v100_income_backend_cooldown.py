import json
import time

from rlm.v100 import income_work
from rlm.v100.common import atomic_json
from rlm.v100.income_work import backend_on_cooldown


def test_backend_cooldown_only_skips_exact_recent_failing_http_503(tmp_path):
    file = tmp_path / "research/income-work/status.json"
    file.parent.mkdir(parents=True)
    file.write_text(
        json.dumps(
            {
                "state": "model-selected work blocked",
                "error": (
                    "503 Server Error: Service Unavailable for url: "
                    "http://192.168.0.61:11435/api/chat"
                ),
                "updated": time.time(),
            }
        )
    )
    assert backend_on_cooldown(tmp_path, "http://192.168.0.61:11435")
    assert not backend_on_cooldown(tmp_path, "http://192.168.0.9:11435")
    assert not backend_on_cooldown(tmp_path, "http://127.0.0.1:11435")


def test_income_cooldown_expires_and_never_blocks_healthy_work(tmp_path):
    file = tmp_path / "research/income-work/status.json"
    file.parent.mkdir(parents=True)
    error = "503 for url: http://192.168.0.61:11435/api/chat"
    for old in (time.time() - 950, time.time() + 100):
        file.write_text(
            json.dumps({"state": "model-selected work blocked", "error": error, "updated": old})
        )
        assert not backend_on_cooldown(tmp_path, "http://192.168.0.61:11435")
    file.write_text(
        json.dumps(
            {"state": "model-selected work recorded", "error": error, "updated": time.time()}
        )
    )
    assert not backend_on_cooldown(tmp_path, "http://192.168.0.61:11435")

def test_cooldown_survives_selection_receipt_and_scopes_exact_host(tmp_path, monkeypatch):
    failed = {
        "state": "model-selected work blocked",
        "error": (
            "503 Server Error: Service Unavailable for url: "
            "http://192.168.0.61:11435/api/chat"
        ),
        "updated": time.time(),
    }
    path = tmp_path / "research/income-work/status.json"
    atomic_json(path, failed)
    monkeypatch.setattr(income_work, "load_goal", lambda _: {"id": "goal"})
    observed = []

    def unavailable(root, branch):
        observed.append(backend_on_cooldown(root, "http://192.168.0.61:11435"))
        observed.append(backend_on_cooldown(root, "http://192.168.0.9:11435"))
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
    assert backend_on_cooldown(tmp_path, "http://192.168.0.61:11435")
    assert (tmp_path / "research/income-work/backend-cooldown.json").is_file()
    assert json.loads(path.read_text())["error"] == "No admitted alternative yet"


def test_inactive_or_expired_failure_is_never_saved_as_cooldown(tmp_path):
    receipt = tmp_path / "research/income-work/backend-cooldown.json"
    income_work.record_backend_cooldown(
        tmp_path,
        {
            "state": "model-selected work blocked",
            "error": "503 for url: http://192.168.0.61:11435/api/chat",
            "updated": time.time() - 950,
        },
    )
    assert not receipt.exists()
    income_work.record_backend_cooldown(
        tmp_path,
        {
            "state": "model-selected work recorded",
            "error": "503 for url: http://192.168.0.61:11435/api/chat",
            "updated": time.time(),
        },
    )
    assert not receipt.exists()
