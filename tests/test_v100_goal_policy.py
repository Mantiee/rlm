import json

import pytest

from rlm.v100 import goal_learning as lab
from rlm.v100 import goal_policy
from tests.test_v100_goal_learning import world as goal_world

world = goal_world


def test_cold_start_is_explicit_and_future_prices_do_not_change_past_features(world):
    root, clock, values, spec, _ = world
    cold = goal_policy.suggest(root, spec["target"], 60, 1)
    assert cold["trained_outcomes"] == 0
    assert cold["probabilities"] == pytest.approx([1 / 3] * 3)
    target = lab.observation(root, spec["target"])
    before = goal_policy.vector(root, target, 60, 1)
    clock[0] += 120
    values["https://example.org/metric"] = 100000
    lab.observe(root, "https://example.org/metric", "value")
    assert goal_policy.vector(root, target, 60, 1) == before


def test_failed_forecasts_train_real_weights_without_touching_gemma_or_paper(world):
    root, clock, values, spec, _ = world
    for index in range(12):
        target = lab.observe(root, "https://example.org/metric", "value")
        forecast = lab.predict(
            root, "A", {**spec, "target": target["id"], "question": f"Trial {index}"}
        )
        clock[0] += 60
        values["https://example.org/metric"] -= 3
        after = lab.observe(root, "https://example.org/metric", "value")
        assert not lab.settle(root, forecast["id"], after["id"])["correct"]
        clock[0] += 1
    state = goal_policy.update(root)
    assert state["trained_outcomes"] == 12 and state["weights_changed"]
    assert state["weights"][0][0] > 0
    assert state["prequential_samples"] == 4
    assert state["prequential_flat_baseline_brier"] == 2
    assert state["prequential_frequency_baseline_brier"] < state["uniform_baseline_brier"]
    assert goal_policy.update(root) == state
    fresh = lab.observe(root, "https://example.org/metric", "value")
    learned = goal_policy.suggest(root, fresh["id"], 60, 1)
    assert sum(learned["probabilities"]) == pytest.approx(1)
    assert learned["probabilities"] != pytest.approx([1 / 3] * 3)
    assert not (root / "research/paper/ledger.sqlite3").exists()
    assert not (root / "models").exists()


def test_overlapping_forecasts_do_not_use_unavailable_labels_for_scores(world):
    root, clock, values, spec, _ = world
    forecasts = [
        lab.predict(root, "A", {**spec, "horizon_seconds": 60 + index}) for index in range(12)
    ]
    clock[0] = 1120
    values["https://example.org/metric"] = 90
    after = lab.observe(root, "https://example.org/metric", "value")
    for forecast in forecasts:
        lab.settle(root, forecast["id"], after["id"])
    state = goal_policy.update(root)
    assert state["trained_outcomes"] == 12
    assert state["prequential_samples"] == 0 and state["prequential_brier"] is None


def test_paired_reference_and_shadow_are_one_label_and_tampering_is_rejected(world):
    root, clock, values, spec, _ = world
    a = lab.predict(root, "A", spec)
    b = lab.predict(root, "B", spec)
    clock[0] += 60
    values["https://example.org/metric"] = 90
    after = lab.observe(root, "https://example.org/metric", "value")
    for forecast in (a, b):
        lab.settle(root, forecast["id"], after["id"])
    assert goal_policy.update(root)["trained_outcomes"] == 1
    path = root / "research/goal-policy/status.json"
    altered = json.loads(path.read_text())
    altered["weights"][0][0] += 1
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="artifact changed"):
        goal_policy.update(root)
