"""Small reward-trained shadow decision head; audited ledger only, never live orders."""

import fcntl
import hashlib
import json
import math
import time
from decimal import Decimal
from pathlib import Path

from rlm.v100.common import atomic_json
from rlm.v100.paper import PaperBook, number, sha, timestamp

FEATURES = ["bias", "spread", "entry_cost", "exit_cost", "leverage", "crypto", "equities", "sports"]


def features(inputs: dict, decision: dict) -> list[float]:
    quote, fees, instrument = (inputs[key] for key in ("quote", "fee_profile", "instrument"))
    bid, ask = (float(number(quote[key], positive=True)) for key in ("bid", "ask"))
    if bid > ask:
        raise ValueError("Crossed policy quote")
    costs = [
        sum(
            float(number(fees.get(prefix + key, "0")))
            for key in ("commission_bps", "venue_fee_bps", "fx_bps")
        )
        / 10000
        for prefix in ("entry_", "exit_")
    ]
    values = [
        1.0,
        (ask - bid) / ask,
        *costs,
        float(number(decision.get("leverage", "1"), positive=True)) / 3,
        *[float(instrument["market"] == market) for market in ("crypto", "equities", "sports")],
    ]
    if any(not math.isfinite(x) or abs(x) > 10 for x in values):
        raise ValueError("Policy feature outside bounded numerical range")
    return values


def samples(root: Path, branch: str) -> list[dict]:
    if branch not in ("A", "B"):
        raise ValueError("Policy branch must be A/B")
    if not (root / "research/paper/ledger.sqlite3").exists():
        return []
    book = PaperBook(root)
    try:
        decisions, result = {}, []
        pending_orders = {
            position["order_id"]
            for portfolio in book.state()["branches"].values()
            for position in portfolio["positions"].values()
        }
        for event in book.events():
            if event["kind"] == "decision":
                proposal = event["payload"]["proposal"]
                decisions[proposal["id"]] = event
            if event["kind"] != "observation":
                continue
            for index, trade in enumerate(event["payload"].get("trades", [])):
                if (
                    trade["action"] not in ("close", "liquidation", "settle")
                    or trade.get("branch") != branch
                ):
                    continue
                origin = decisions.get(trade["order_id"])
                if not origin or origin["sequence"] >= event["sequence"]:
                    raise ValueError("Reward has no earlier decision")
                inputs = origin["payload"].get("decision_inputs")
                if not inputs or not inputs.get("quote"):
                    continue  # Legacy events have no causal features; cannot train this policy.
                proposal = origin["payload"]["proposal"]
                if proposal["action"] != "open":
                    continue
                if timestamp(inputs["quote"]["available_at"]) > timestamp(origin["time"]):
                    raise ValueError("Future quote in policy inputs")
                allocation = number(trade["allocation"], positive=True)
                pnl = number(trade["payout"]) - allocation
                if pnl != Decimal(trade["net_pnl"]):
                    raise ValueError("Reward disagrees with audited payout")
                result.append(
                    {
                        "id": sha([event["event_hash"], index]),
                        "decision_at": origin["time"],
                        "resolved_at": event["time"],
                        "features": features(inputs, proposal),
                        "reward": max(-1.0, min(1.0, float(pnl / allocation))),
                        "pnl": str(pnl),
                        "allocation": str(allocation),
                        "order_id": trade["order_id"],
                    }
                )
        # A hundred partial closes of one order are one sample, not a hundred
        # independent decisions. Aggregate their audited capital and realized P&L.
        grouped = {}
        for row in result:
            old = grouped.get(row["order_id"])
            if old is None:
                grouped[row["order_id"]] = row
                continue
            if row["features"] != old["features"]:
                raise ValueError("One order has inconsistent causal policy features")
            old["pnl"] = str(Decimal(old["pnl"]) + Decimal(row["pnl"]))
            old["allocation"] = str(Decimal(old["allocation"]) + Decimal(row["allocation"]))
            old["reward"] = max(
                -1.0, min(1.0, float(Decimal(old["pnl"]) / Decimal(old["allocation"])))
            )
            old["resolved_at"] = max(old["resolved_at"], row["resolved_at"])
            old["id"] = sha([old["id"], row["id"]])
        return sorted(
            (row for identity, row in grouped.items() if identity not in pending_orders),
            key=lambda row: (row["decision_at"], row["id"]),
        )
    finally:
        book.close()


def probability(weights: list[float], vector: list[float]) -> float:
    score = max(-30.0, min(30.0, sum(a * b for a, b in zip(weights, vector, strict=True))))
    return 1 / (1 + math.exp(-score))


def fit(rows: list[dict], initial: list[float], steps: int = 100) -> list[float]:
    if not rows or type(steps) is not int or not 1 <= steps <= 500:
        raise ValueError("Bounded reward training needs samples and 1..500 steps")
    weights = list(initial)
    for _ in range(steps):
        gradient = [0.0] * len(FEATURES)
        for row in rows:
            p = probability(weights, row["features"])
            # Reward of taking this observed trade, versus cash=0. This is a
            # conditional, observational policy objective, NOT unbiased off-policy RL.
            advantage = row["reward"] - 0.02 * math.log(p / (1 - p))
            for i, x in enumerate(row["features"]):
                gradient[i] += advantage * p * (1 - p) * x / len(rows)
        weights = [
            max(-5.0, min(5.0, w + 0.1 * g - 0.001 * (w - old)))
            for w, g, old in zip(weights, gradient, initial, strict=True)
        ]
    return weights


def value(weights: list[float], rows: list[dict]) -> float:
    return (
        sum(probability(weights, row["features"]) * row["reward"] for row in rows) / len(rows)
        if rows
        else 0.0
    )


def train_locked(root: Path, branch: str) -> dict:
    rows = samples(root, branch)
    folder = root / "research/reward-policies" / branch
    fingerprint = sha(rows)
    path = folder / (fingerprint + ".json")
    if path.exists():
        return json.loads(path.read_text())
    initial = [0.0] * len(FEATURES)
    if len(rows) < 40:
        return {
            "status": "waiting for 40 independently resolved paper outcomes",
            "branch": branch,
            "samples": len(rows),
            "weights_changed": False,
        }
    boundary = rows[len(rows) * 4 // 5]["decision_at"]
    # Purge every training outcome resolved after the first held-out decision.
    training = [row for row in rows if timestamp(row["resolved_at"]) < timestamp(boundary)]
    heldout = [row for row in rows if timestamp(row["decision_at"]) >= timestamp(boundary)]
    if len(training) < 20 or len(heldout) < 8:
        return {
            "status": "insufficient purged chronological split",
            "samples": len(rows),
            "weights_changed": False,
        }
    weights = fit(training, initial)
    result = {
        "schema": "v100-reward-policy-v1",
        "branch": branch,
        "features": FEATURES,
        "weights": weights,
        "data_sha256": fingerprint,
        "training_samples": len(training),
        "heldout_samples": len(heldout),
        "training_last_resolution": max(row["resolved_at"] for row in training),
        "heldout_first_decision": boundary,
        "heldout_observed_reward_objective": value(weights, heldout),
        "cash_objective": 0,
        "baseline_objective": value(initial, heldout),
        "created_at": time.time(),
        "status": "shadow only; requires independent prospective paper validation",
        "weights_changed": weights != initial,
        "scope": "Reward-trained CPU head. Includes losses and modeled costs. Selection bias remains: only executed actions have rewards. No automatic Gemma replacement or trading authority.",
    }
    atomic_json(path, result)
    atomic_json(
        folder / "shadow.json",
        {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
    )
    return result


def inspect(root: Path, branch: str) -> dict:
    pointer = root / "research/reward-policies" / branch / "shadow.json"
    if not pointer.exists():
        return {
            "branch": branch,
            "status": "no shadow policy",
            "audited_samples": len(samples(root, branch)),
        }
    pin = json.loads(pointer.read_text())
    path = Path(pin["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != pin["sha256"]:
        raise ValueError("Reward policy artifact changed")
    return json.loads(path.read_text())


def train(root: Path, branch: str) -> dict:
    if branch not in ("A", "B"):
        raise ValueError("Policy branch must be A/B")
    folder = root / "research/reward-policies" / branch
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "train.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        return train_locked(root, branch)
