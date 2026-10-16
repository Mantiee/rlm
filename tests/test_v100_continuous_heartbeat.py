import json
import threading

from rlm.v100 import continuous


def test_heartbeat_refreshes_liveness_during_blocked_stage(tmp_path, monkeypatch):
    path = tmp_path / "heartbeat.json"
    refreshed = threading.Event()
    original_atomic_json = continuous.atomic_json
    writes = 0
    receipts = []

    def record_write(destination, value):
        nonlocal writes
        original_atomic_json(destination, value)
        writes += 1
        receipts.append(dict(value))
        if writes >= 2:
            refreshed.set()

    monkeypatch.setattr(continuous, "atomic_json", record_write)
    entered_block = threading.Event()
    release_block = threading.Event()
    result = {}

    def blocked_stage():
        heartbeat = continuous._HeartbeatRefresher(path, interval=1)
        result["heartbeat"] = heartbeat
        with heartbeat:
            heartbeat.record(
                {
                    "stage": "candidate-training-and-independent-tests",
                    "cycle_in_progress": 7,
                    "reason": "unchanged reason",
                    "optimizer_updates_observed": 12,
                }
            )
            entered_block.set()
            release_block.wait()

    worker = threading.Thread(target=blocked_stage)
    worker.start()
    try:
        assert entered_block.wait(2)
        assert refreshed.wait(3)
        receipt = json.loads(path.read_text())
        assert receipt["heartbeat_kind"] == "liveness"
        assert receipt["scope"] == (
            "Liveness only; current loop stage, not optimizer progress or acceptance"
        )
        assert receipt["stage"] == "candidate-training-and-independent-tests"
        assert receipt["cycle_in_progress"] == 7
        assert receipt["reason"] == "unchanged reason"
        assert receipt["optimizer_updates_observed"] == 12
        assert receipts[1]["updated"] > receipts[0]["updated"]
    finally:
        release_block.set()
        worker.join(3)

    assert not worker.is_alive()
    assert result["heartbeat"]._thread is not None
    assert not result["heartbeat"]._thread.is_alive()


def test_heartbeat_context_joins_refresher_when_stage_raises(tmp_path):
    heartbeat = continuous._HeartbeatRefresher(tmp_path / "heartbeat.json", interval=1)

    try:
        with heartbeat:
            heartbeat.record({"stage": "waiting-for-verified-data", "cycle_in_progress": 3})
            raise RuntimeError("synthetic stage failure")
    except RuntimeError as error:
        assert str(error) == "synthetic stage failure"
    else:
        raise AssertionError("The synthetic stage error should propagate")

    assert heartbeat._thread is not None
    assert not heartbeat._thread.is_alive()
