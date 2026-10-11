"""Online CPU forecast learner over independently settled, audited goal labels.

This shadow policy is separate from Gemma adapters and never places orders.
Prequential scores use only outcomes available before each forecast commitment.
"""

import json
import math
from pathlib import Path

from rlm.v100.common import atomic_json
from rlm.v100.goal_learning import database, digest, observation, records
from rlm.v100.goals import load_goal

FEATURES = ("bias", "last_change", "mean_change", "change_deviation", "history_count", "horizon")


def vector(root: Path, target: dict, horizon: int, threshold: float) -> list[float]:
    if type(horizon) is not int or not 60 <= horizon <= 604800:
        raise ValueError("Use the goal forecast horizon budget")
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Use a positive finite absolute threshold")
    with database(root) as db:
        # Evidence is host-observed and strictly no later than the target snapshot.
        rows = db.execute(
            "SELECT id, data FROM observations WHERE time<=? ORDER BY time DESC, id DESC",
            (target["acquired"],),
        )
        history = []
        for row in rows:
            candidate = json.loads(row["data"])
            if (candidate["url"], candidate["field"]) == (target["url"], target["field"]):
                history.append(observation(root, row["id"]))
                if len(history) == 8:
                    break
    history.sort(key=lambda row: row["acquired"])
    changes = [
        (later["value"] - earlier["value"]) / threshold
        for earlier, later in zip(history, history[1:], strict=False)
        if later["acquired"] > earlier["acquired"]
    ]
    mean = sum(changes) / len(changes) if changes else 0.0
    deviation = math.sqrt(sum((x - mean) ** 2 for x in changes) / len(changes)) if changes else 0.0
    values = [
        1.0,
        changes[-1] if changes else 0.0,
        mean,
        deviation,
        len(changes) / 7,
        math.log1p(horizon / 300),
    ]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Nonfinite causal forecast feature")
    return [max(-10.0, min(10.0, value)) for value in values]


def probabilities(weights: list[list[float]], features: list[float]) -> list[float]:
    scores = [sum(w * x for w, x in zip(row, features, strict=True)) for row in weights]
    peak = max(scores)
    exponents = [math.exp(score - peak) for score in scores]
    total = sum(exponents)
    return [value / total for value in exponents]


def update_weights(weights: list[list[float]], features: list[float], actual: int) -> None:
    predicted = probabilities(weights, features)
    for label, row in enumerate(weights):
        for index, value in enumerate(features):
            row[index] = max(
                -5.0,
                min(
                    5.0,
                    row[index]
                    + 0.03 * ((label == actual) - predicted[label]) * value
                    - 0.0003 * row[index],
                ),
            )


def update(root: Path) -> dict:
    verified = records(root)[-512:]
    goal_id = (load_goal(root) or {}).get("id")
    identity = digest({"goal": goal_id, "records": verified, "features": FEATURES, "version": 1})
    folder = root / "research/goal-policy"
    path = folder / "status.json"
    if path.exists():
        prior = json.loads(path.read_text())
        checksum = prior.pop("state_sha256")
        if digest(prior) != checksum:
            raise ValueError("Shadow policy artifact changed")
        if prior.get("data_sha256") == identity:
            return prior
    samples = []
    with database(root) as db:
        for row in verified:
            forecast = json.loads(
                db.execute(
                    "SELECT data FROM forecasts WHERE id=?", (row["verification"]["forecast_id"],)
                ).fetchone()[0]
            )
            spec = forecast["specification"]
            outcome = json.loads(
                db.execute(
                    "SELECT outcome FROM forecasts WHERE id=?",
                    (row["verification"]["forecast_id"],),
                ).fetchone()[0]
            )
            target = observation(root, spec["target"])
            resolved = observation(root, outcome["observation"])
            samples.append(
                {
                    "commitment": digest(
                        [spec["target"], spec["horizon_seconds"], spec["threshold"]]
                    ),
                    "created": forecast["created"],
                    "resolved": resolved["acquired"],
                    "features": vector(root, target, spec["horizon_seconds"], spec["threshold"]),
                    "actual": ("down", "flat", "up").index(row["messages"][-1]["content"]),
                    "id": row["verification"]["forecast_id"],
                }
            )
    # Reference and shadow forecasts on the same target are one training label,
    # not two independent observations. Retain the first available resolution.
    unique = {}
    for sample in sorted(samples, key=lambda row: row["resolved"]):
        unique.setdefault(sample["commitment"], sample)
    samples = list(unique.values())
    # Predict before applying labels that arrive later. Overlapping horizons cannot
    # leak their future labels into another earlier forecast's score.
    events = sorted(
        [(row["created"], 0, row["id"], row) for row in samples]
        + [(row["resolved"], 1, row["id"], row) for row in samples],
        key=lambda event: event[:3],
    )
    weights = [[0.0] * len(FEATURES) for _ in range(3)]
    seen, scores, flat_scores, frequency_scores = 0, [], [], []
    counts = [0, 0, 0]
    for _, kind, _, row in events:
        if kind == 0:
            if seen >= 8:
                prediction = probabilities(weights, row["features"])
                scores.append(
                    sum((p - (i == row["actual"])) ** 2 for i, p in enumerate(prediction))
                )
                flat_scores.append(0.0 if row["actual"] == 1 else 2.0)
                frequency = [(count + 1) / (seen + 3) for count in counts]
                frequency_scores.append(
                    sum((p - (i == row["actual"])) ** 2 for i, p in enumerate(frequency))
                )
        else:
            update_weights(weights, row["features"], row["actual"])
            counts[row["actual"]] += 1
            seen += 1
    result = {
        "schema": "synta-goal-policy-v1",
        "goal_id": goal_id,
        "data_sha256": identity,
        "features": list(FEATURES),
        "weights": weights,
        "trained_outcomes": seen,
        "weights_changed": bool(seen),
        "prequential_samples": len(scores),
        "prequential_brier": sum(scores) / len(scores) if scores else None,
        "uniform_baseline_brier": 2 / 3,
        "prequential_flat_baseline_brier": sum(flat_scores) / len(flat_scores)
        if flat_scores
        else None,
        "prequential_frequency_baseline_brier": sum(frequency_scores) / len(frequency_scores)
        if frequency_scores
        else None,
        "state": "shadow learning" if seen else "waiting for verified outcomes",
        "scope": "Online supervised CPU shadow policy; losses and failed forecasts included. This is not Gemma weight training, a trading signal or verified income. Prequential replay is exploratory; new shadow forecasts are prospective tests.",
    }
    atomic_json(path, {**result, "state_sha256": digest(result)})
    return result


def suggest(root: Path, target: str, horizon_seconds: int, threshold: float) -> dict:
    state = update(root)
    value = observation(root, target)
    if value["value"] is None:
        raise ValueError("Learned forecast needs a numeric target")
    return {
        "target": target,
        "probabilities": probabilities(
            state["weights"], vector(root, value, horizon_seconds, threshold)
        ),
        "trained_outcomes": state["trained_outcomes"],
        "policy_data_sha256": state["data_sha256"],
        "prequential_brier": state["prequential_brier"],
        "scope": state["scope"],
    }
