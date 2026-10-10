"""Forward-only financial task commissioning, independent of model tool choice.

These are declared uniform-probability reference forecasts, never claimed model
insight, trading signals, fees verification, or proof of profitable patterns.
"""

import json
import time
from pathlib import Path

from rlm.v100.activity import ActivityLog
from rlm.v100.common import atomic_json
from rlm.v100.goals import load_goal

PRODUCTS = ("BTC", "ETH", "LTC", "BCH", "ETC", "LINK", "SOL", "AVAX")


def tick(root: Path) -> dict:
    from rlm.v100.goal_learning import database, observe, predict

    goal = load_goal(root)
    if not goal:
        return {"state": "blocked", "reason": "No operator goal"}
    if not any(
        term in goal["text"].lower()
        for term in ("income", "crypto", "equities", "stock", "dochód", "zarab")
    ):
        return {
            "state": "blocked",
            "goal_id": goal["id"],
            "reason": "Current goal has no financial reference adapter",
        }
    active_path = root / "research/mission/active.json"
    if not active_path.exists():
        return {"state": "blocked", "reason": "No mission configuration"}
    active = json.loads(active_path.read_text())
    run = Path(active["run"]).resolve()
    if not run.is_relative_to((root / "research/mission").resolve()):
        raise ValueError("Mission path escaped its root")
    profile = json.loads((run / "input-profile.json").read_text())
    if not profile.get("resources", {}).get("paper_research_enabled"):
        return {
            "state": "blocked",
            "reason": "Reference market observations require a financial mission; other goals need their own measurable adapter",
        }
    path = root / "research/goal-observer/status.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    index = previous.get("next_index", 0) if previous.get("goal_id") == goal["id"] else 0
    now = time.time()
    with database(root) as db:
        pending = [
            json.loads(row[0])
            for row in db.execute(
                "SELECT data FROM forecasts WHERE outcome IS NULL AND due>=?", (now - 300,)
            )
        ]
        if len(pending) >= 24:
            return {
                "state": "waiting",
                "goal_id": goal["id"],
                "reason": "Reserved room for model forecasts; waiting for outcomes",
            }
        recent = {
            row["specification"]["question"]
            for row in pending
            if row["plan"]["long"]["id"] == goal["id"]
        }
    results = []
    for offset in range(2):
        product = PRODUCTS[(index + offset) % len(PRODUCTS)]
        question = f"Controller reference: {product}-USD direction over 300 seconds"
        if question in recent:
            continue
        try:
            observation = observe(
                root, f"https://api.coinbase.com/v2/prices/{product}-USD/spot", "data.amount"
            )
            value = observation["value"]
            if value <= 0:
                raise ValueError("Reference price must be positive")
            receipt = predict(
                root,
                "A",
                {
                    "question": question,
                    "rationale": "Uniform controller reference to commission verified future labels. Not a learned pattern or income claim. No real order.",
                    "evidence": [observation["id"]],
                    "target": observation["id"],
                    "horizon_seconds": 300,
                    "threshold": max(abs(value) * 0.001, 0.000001),
                    "probabilities": [1 / 3, 1 / 3, 1 / 3],
                },
            )
            results.append(
                {
                    "product": product,
                    "id": receipt["id"],
                    "state": receipt["state"],
                    "due": receipt["due"],
                }
            )
        except Exception as error:
            results.append({"product": product, "state": "failed", "error": str(error)[:300]})
    state = {
        "state": "collecting"
        if any(r.get("id") for r in results)
        else "waiting or source unavailable",
        "goal_id": goal["id"],
        "updated": time.time(),
        "next_index": (index + 2) % len(PRODUCTS),
        "results": results,
        "scope": "Forward-only controller reference labels, not model skill or profit; model hypotheses remain separate",
    }
    atomic_json(path, state)
    ActivityLog(root, "controller", "goal-observer").write(
        "metrics", "goal-reference-observation", state
    )
    return state
