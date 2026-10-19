import copy
import json

import pytest

from rlm.v100 import distributed_compute, goal_compute, goal_learning, goals, training


def record(identity, at=100, group="series", answer="up", threshold=1, branch="A"):
    source = "https://example.test/" + group
    content = {
        "question": branch + " observer question",
        "horizon_seconds": 60,
        "threshold": threshold,
        "target_at_prediction": {"source": source, "acquired": at, "value": 100},
        "evidence_at_prediction": [
            {"source": source, "acquired": at, "value": 100, "excerpt": "observed"}
        ],
    }
    return {
        "id": identity,
        "group": group,
        "verification": {"goal_id": "goal"},
        "messages": [
            {"role": "user", "content": json.dumps(content)},
            {"role": "assistant", "content": answer},
        ],
    }


def test_AB_observers_collapse_to_earliest_archived_representative_without_mutation():
    a, b = record("archive-first", branch="A"), record("archive-second", branch="B")
    before = copy.deepcopy([a, b])
    assert goal_compute.deduplicated_records([a, b]) == [a]
    assert goal_compute.deduplicated_records([b, a]) == [b]
    assert [a, b] == before


def test_threshold_horizon_and_ordered_evidence_define_distinct_episodes():
    original = record("one")
    threshold = record("threshold", threshold=2)
    horizon = record("horizon")
    value = json.loads(horizon["messages"][0]["content"])
    value["horizon_seconds"] = 120
    horizon["messages"][0]["content"] = json.dumps(value)
    assert len(goal_compute.deduplicated_records([original, threshold, horizon])) == 3
    ordered = record("ordered")
    value = json.loads(ordered["messages"][0]["content"])
    value["evidence_at_prediction"].append(
        {"source": "https://other.test", "acquired": 90, "value": 1}
    )
    ordered["messages"][0]["content"] = json.dumps(value)
    reversed_order = copy.deepcopy(ordered)
    reversed_order["id"] = "reversed"
    value["evidence_at_prediction"].reverse()
    reversed_order["messages"][0]["content"] = json.dumps(value)
    assert len(goal_compute.deduplicated_records([ordered, reversed_order])) == 2


@pytest.mark.parametrize("answer", ["down", "flat", "up"])
def test_representative_does_not_depend_on_label(answer):
    a, b = record("first", answer=answer), record("second", answer=answer, branch="B")
    assert goal_compute.deduplicated_records([a, b])[0] is a


@pytest.mark.parametrize(
    "labels", [("up", "down", "up"), ("down", "up", "down"), ("flat", "up", "down")]
)
def test_conflicting_episode_excludes_all_receipts_regardless_order_or_label(labels):
    rows = [record(str(i), answer=label) for i, label in enumerate(labels)]
    before = copy.deepcopy(rows)
    statistics = {}
    assert goal_compute.deduplicated_records(rows, statistics) == []
    assert goal_compute.deduplicated_records(list(reversed(rows))) == []
    assert statistics == {
        "raw_records": 3,
        "unique_consistent_episodes": 0,
        "ambiguous_episodes": 1,
        "ambiguous_records": 3,
    }
    assert rows == before


def test_malformed_labels_and_provenance_still_fail_closed():
    with pytest.raises(ValueError, match="directional label"):
        goal_compute.deduplicated_records([record("bad", answer="maybe")])
    malformed = record("malformed")
    content = json.loads(malformed["messages"][0]["content"])
    content["target_at_prediction"]["source"] = None
    malformed["messages"][0]["content"] = json.dumps(content)
    with pytest.raises(ValueError, match="provenance"):
        goal_compute.deduplicated_records([malformed])


def test_numerically_equal_thresholds_are_same_episode():
    a, b = record("a", threshold=1), record("b", threshold=1.0)
    assert goal_compute.deduplicated_records([a, b]) == [a]


def configure(monkeypatch, train, validation):
    monkeypatch.setattr(goals, "load_goal", lambda root: {"id": "goal"})
    monkeypatch.setattr(goal_learning, "records", lambda root: train + validation)
    monkeypatch.setattr(training, "load_records", lambda *args: (train, validation))
    monkeypatch.setattr(distributed_compute, "mailbox_path", lambda root: root / "mailbox")


def test_counts_gate_after_split_dedup_blocks_inflated_population(tmp_path, monkeypatch):
    train = [record(f"t{i}", at=i // 2, group="train") for i in range(8)]
    validation = [record(f"v{i}", at=i, group="val" + str(i % 2)) for i in range(4)]
    configure(monkeypatch, train, validation)
    monkeypatch.setattr(
        distributed_compute,
        "queue_job",
        lambda *args, **kwargs: pytest.fail("Must not submit inflated dataset"),
    )
    result = goal_compute.propose(tmp_path)
    assert result["state"] == "blocked"
    assert result["train"] == 4
    assert result["raw_training_receipts"] == 8
    assert result["validation"] == 4


def test_submission_verification_and_legacy_semantics(tmp_path, monkeypatch):
    train = [record(f"t{i}", at=i // 2, group="train") for i in range(16)]
    validation = [record(f"v{i}", at=i // 2, group="val" + str((i // 2) % 2)) for i in range(8)]
    configure(monkeypatch, train, validation)
    captured = {}

    def queue(*args, **kwargs):
        captured.update(train=args[7], validation=args[8], provenance=kwargs["provenance"])
        return {"state": "queued"}

    monkeypatch.setattr(distributed_compute, "queue_job", queue)
    assert goal_compute.propose(tmp_path)["state"] == "queued"
    assert len(captured["train"]) == 8
    assert len(captured["validation"]) == 4
    goal_compute.verify_job(tmp_path, captured)
    archived = json.loads(json.dumps(captured))
    archived["provenance"]["selection"] = goal_compute.LEGACY_SELECTION
    archived["provenance"]["record_ids"] = [row["id"] for row in train + validation]
    archived["train"] = [goal_compute.compact(row) for row in train]
    archived["validation"] = [goal_compute.compact(row) for row in validation]
    goal_compute.verify_job(tmp_path, archived)
    assert len(archived["train"]) == 16  # Old jobs are not silently rewritten.


def test_other_consistent_episodes_train_after_all_ambiguous_receipts_excluded(
    tmp_path, monkeypatch
):
    train = [record(f"t{i}", at=i, group="train") for i in range(8)]
    train += [
        record("ambiguous-a", at=1000, group="train"),
        record("ambiguous-b", at=1000, group="train", answer="down"),
    ]
    validation = [record(f"v{i}", at=i, group="val" + str(i % 2)) for i in range(4)]
    configure(monkeypatch, train, validation)
    captured = {}

    def queue(*args, **kwargs):
        captured.update(train=args[7], validation=args[8], provenance=kwargs["provenance"])
        return {"state": "queued"}

    monkeypatch.setattr(distributed_compute, "queue_job", queue)
    assert goal_compute.propose(tmp_path)["state"] == "queued"
    assert len(captured["train"]) == 8
    assert captured["provenance"]["deduplication_counts"]["train"]["ambiguous_episodes"] == 1
    assert captured["provenance"]["deduplication_counts"]["train"]["ambiguous_records"] == 2
    goal_compute.verify_job(tmp_path, captured)
    captured["provenance"]["deduplication_counts"]["train"]["ambiguous_records"] = 0
    with pytest.raises(ValueError, match="exclusion counts changed"):
        goal_compute.verify_job(tmp_path, captured)


def test_count_gate_after_ambiguous_exclusion(tmp_path, monkeypatch):
    train = [record(f"t{i}", at=i, group="train") for i in range(6)]
    train += [
        record("ambiguous-a", at=1000, group="train"),
        record("ambiguous-b", at=1000, group="train", answer="down"),
    ]
    validation = [record(f"v{i}", at=i, group="val" + str(i % 2)) for i in range(4)]
    configure(monkeypatch, train, validation)
    monkeypatch.setattr(
        distributed_compute,
        "queue_job",
        lambda *args, **kwargs: pytest.fail("Must not weaken counts after exclusion"),
    )
    result = goal_compute.propose(tmp_path)
    assert result["state"] == "blocked"
    assert result["train"] == 6
    assert result["deduplication_counts"]["train"]["ambiguous_records"] == 2
