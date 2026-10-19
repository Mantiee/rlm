import pytest

from rlm.v100.dashboard_snapshot import learning_activity_status


def test_missing_heartbeat_is_planned_and_counters_stay_unknown():
    value = learning_activity_status(None, True, None, now=1000)
    assert value["state"] == "planned"
    assert value["optimizer_evidence"]["state"] == "unknown"
    assert value["optimizer_evidence"]["optimizer_updates"] is None


def test_fresh_heartbeat_is_not_presented_as_verified_training():
    value = learning_activity_status(
        {"stage": "candidate-training-and-independent-tests", "updated": 990},
        True,
        None,
        now=1000,
    )
    assert value["state"] == "running"
    assert "does not prove optimizer progress" in value["detail"]
    assert value["optimizer_evidence"]["state"] == "unknown"


def test_stale_heartbeat_and_separate_verified_optimizer_receipt():
    value = learning_activity_status(
        {"stage": "candidate-training", "updated": 600},
        True,
        99,
        now=1000,
        optimizer_receipt_updated_at=990,
    )
    assert value["state"] == "stale"
    assert value["optimizer_evidence"]["state"] == "verified"
    assert value["optimizer_evidence"]["optimizer_updates"] == 99
    assert value["optimizer_evidence"]["activity_state"] == "active"


@pytest.mark.parametrize(
    ("receipt_updated_at", "expected_state"),
    [(990, "active"), (600, "stale"), (None, "unknown"), (1001, "unknown")],
)
def test_optimizer_receipt_freshness_never_erases_verified_cumulative_updates(
    receipt_updated_at, expected_state
):
    value = learning_activity_status(
        {"stage": "research", "updated": 995},
        True,
        99,
        now=1000,
        optimizer_receipt_updated_at=receipt_updated_at,
    )
    evidence = value["optimizer_evidence"]
    assert evidence["state"] == "verified"
    assert evidence["optimizer_updates"] == 99
    assert evidence["activity_state"] == expected_state
    if expected_state == "active":
        assert evidence["receipt_age_seconds"] == 10
    else:
        assert evidence["activity_state"] != "active"


def test_recent_optimizer_receipt_does_not_claim_active_when_mission_is_stopped():
    value = learning_activity_status(
        {"stage": "training", "updated": 995},
        False,
        99,
        now=1000,
        optimizer_receipt_updated_at=990,
    )
    assert value["optimizer_evidence"]["state"] == "verified"
    assert value["optimizer_evidence"]["activity_state"] == "stale"


def test_timestamp_without_valid_optimizer_counter_does_not_claim_active_training():
    value = learning_activity_status(
        {"stage": "training", "updated": 995},
        True,
        None,
        now=1000,
        optimizer_receipt_updated_at=990,
    )
    assert value["optimizer_evidence"]["state"] == "unknown"
    assert value["optimizer_evidence"]["activity_state"] == "unknown"


def test_stopped_mission_does_not_show_fresh_heartbeat_as_running():
    value = learning_activity_status({"stage": "research", "updated": 995}, False, 0, now=1000)
    assert value["state"] == "stale"
    assert value["optimizer_evidence"]["state"] == "verified"
    assert value["optimizer_evidence"]["optimizer_updates"] == 0
    assert value["optimizer_evidence"]["activity_state"] == "unknown"


def test_future_or_malformed_heartbeat_is_unknown():
    value = learning_activity_status({"stage": "training", "updated": 1001}, True, None, now=1000)
    assert value["state"] == "unknown"
