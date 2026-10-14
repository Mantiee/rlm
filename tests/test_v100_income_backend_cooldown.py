import json
import time

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
