import copy
import json

import pytest

from rlm.v100 import distributed_compute, goal_compute, goal_learning, goals, training


def population():
    train = [
        {"id": f"t{i}", "group": "train-source", "verification": {"goal_id": "goal"}}
        for i in range(60)
    ]
    validation = [
        {"id": "v-old", "group": "validation-other", "verification": {"goal_id": "goal"}},
        *[
            {"id": f"v{i}", "group": "validation-recent", "verification": {"goal_id": "goal"}}
            for i in range(20)
        ],
    ]
    for index, row in enumerate(train + validation):
        source = "https://example.test/" + row["group"]
        content = {
            "horizon_seconds": 60,
            "threshold": 1,
            "target_at_prediction": {"source": source, "acquired": index, "value": 100},
            "evidence_at_prediction": [{"source": source, "acquired": index, "value": 100}],
        }
        row["messages"] = [
            {"role": "user", "content": json.dumps(content)},
            {"role": "assistant", "content": "up"},
        ]
    return train, validation


def test_latest_subset_retains_second_group_without_changing_roles_or_population():
    train, validation = population()
    before = copy.deepcopy(validation)
    selected = goal_compute.selected_validation(validation)
    assert len(selected) == 16
    assert selected == [validation[0], *validation[-15:]]
    assert validation == before
    assert all(any(row is original for original in validation) for row in selected)
    goal_compute.validate_selection(train[-48:], selected)


def test_already_diverse_latest_subset_unchanged():
    _, validation = population()
    validation[-5]["group"] = "validation-other"
    assert goal_compute.selected_validation(validation) == validation[-16:]
    assert goal_compute.selected_validation(validation[:4]) == validation[:4]


def test_selection_uses_latest_missing_group_not_label_quality():
    _, validation = population()
    validation.insert(2, {"id": "second-old", "group": "validation-other", "answer": "bad"})
    assert goal_compute.selected_validation(validation)[0]["id"] == "second-old"


@pytest.mark.parametrize("failure", ["too-few-train", "too-few-validation", "one-group", "overlap"])
def test_actual_subset_gate_stays_strict(failure):
    train, validation = population()
    selected = goal_compute.selected_validation(validation)
    if failure == "too-few-train":
        train = train[:7]
    elif failure == "too-few-validation":
        selected = selected[:3]
    elif failure == "one-group":
        selected = validation[-16:]
    else:
        train[0]["group"] = selected[0]["group"]
    with pytest.raises(ValueError):
        goal_compute.validate_selection(train, selected)


def test_propose_and_verify_use_identical_bounded_source_disjoint_subset(tmp_path, monkeypatch):
    train, validation = population()
    monkeypatch.setattr(goals, "load_goal", lambda root: {"id": "goal"})
    monkeypatch.setattr(goal_learning, "records", lambda root: train + validation)
    monkeypatch.setattr(training, "load_records", lambda *args: (train, validation))
    monkeypatch.setattr(distributed_compute, "mailbox_path", lambda root: root / "mailbox")
    monkeypatch.setattr(
        goal_compute, "compact", lambda row: {"group": row["group"], "answer": row["id"]}
    )
    captured = {}

    def queue(*args, **kwargs):
        captured.update(train=args[7], validation=args[8], provenance=kwargs["provenance"])
        return {"state": "queued"}

    monkeypatch.setattr(distributed_compute, "queue_job", queue)
    result = goal_compute.propose(tmp_path)
    assert result["state"] == "queued"
    assert len(captured["train"]) == 48
    assert len(captured["validation"]) == 16
    assert len({row["group"] for row in captured["validation"]}) == 2
    assert not {row["group"] for row in captured["train"]} & {
        row["group"] for row in captured["validation"]
    }
    goal_compute.verify_job(tmp_path, captured)
    captured["validation"][0]["answer"] = "tampered"
    with pytest.raises(ValueError, match="projection or labels changed"):
        goal_compute.verify_job(tmp_path, captured)
