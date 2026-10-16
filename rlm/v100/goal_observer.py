"""Forward-only financial task commissioning, independent of model tool choice.

Uniform references and separately identified learned CPU shadow forecasts are
never claimed Gemma insight, trading signals, fees verification or profit.
"""

import json
import re
import time
from pathlib import Path

from rlm.v100.activity import ActivityLog
from rlm.v100.common import atomic_json
from rlm.v100.goals import load_goal

PRODUCTS = ("BTC", "ETH", "LTC", "BCH", "ETC", "LINK", "SOL", "AVAX")


def reference_products(root: Path) -> tuple[str, ...]:
    """Use a predeclared source universe, never URL variants or outcome-selected splits.

    Distinct product series are source groups, not proof of statistical independence.
    This setting does not change existing immutable training/validation assignments.
    """
    path = root / "research/goal-learning/universe.json"
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Goal reference universe escaped its root")
    if not path.exists():
        return tuple(product + "-USD" for product in PRODUCTS)
    with path.open("rb") as handle:
        raw = handle.read(4097)
    if len(raw) > 4096:
        raise ValueError("Goal reference universe exceeds 4 KiB")
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"products"}:
        raise ValueError("Goal reference universe requires only products")
    products = value["products"]
    if (
        not isinstance(products, list)
        or not 1 <= len(products) <= 64
        or any(
            not isinstance(product, str) or re.fullmatch(r"[A-Z0-9]{2,16}-USD", product) is None
            for product in products
        )
        or len(set(products)) != len(products)
    ):
        raise ValueError("Choose1..64 distinct Coinbase USD product IDs, not URLs")
    return tuple(products)


def tick(root: Path) -> dict:
    from rlm.v100.goal_learning import database, digest, observe, predict

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
    products = reference_products(root)
    universe_sha256 = digest(list(products))
    path = root / "research/goal-observer/status.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    same_universe = (
        previous.get("universe_sha256", digest([product + "-USD" for product in PRODUCTS]))
        == universe_sha256
    )
    index = (
        previous.get("next_index", 0)
        if previous.get("goal_id") == goal["id"] and same_universe
        else 0
    )
    now = time.time()
    with database(root) as db:
        pending = [
            json.loads(row[0])
            for row in db.execute(
                "SELECT data FROM forecasts WHERE outcome IS NULL AND due>=?", (now - 300,)
            )
        ]
        if len(pending) > 20:
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
    from rlm.v100.goal_policy import suggest

    results = []
    for offset in range(2):
        product_id = products[(index + offset) % len(products)]
        product = product_id.removesuffix("-USD")
        question = f"Controller reference: {product}-USD direction over 300 seconds"
        if question in recent:
            continue
        try:
            observation = observe(
                root, f"https://api.coinbase.com/v2/prices/{product_id}/spot", "data.amount"
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
            learned = suggest(root, observation["id"], 300, max(abs(value) * 0.001, 0.000001))
            shadow = predict(
                root,
                "B",
                {
                    "question": f"Controller learned shadow: {product}-USD direction over 300 seconds",
                    "rationale": "CPU online shadow policy trained on independently settled labels; prospective test, not Gemma skill or a trading instruction. Policy data SHA256: "
                    + learned["policy_data_sha256"],
                    "evidence": [observation["id"]],
                    "target": observation["id"],
                    "horizon_seconds": 300,
                    "threshold": max(abs(value) * 0.001, 0.000001),
                    "probabilities": learned["probabilities"],
                },
            )
            recent.add(question)
            results.append(
                {
                    "product": product,
                    "id": receipt["id"],
                    "state": receipt["state"],
                    "due": receipt["due"],
                    "shadow_id": shadow["id"],
                    "shadow_probabilities": learned["probabilities"],
                    "shadow_trained_outcomes": learned["trained_outcomes"],
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
        "next_index": (index + 2) % len(products),
        "universe": list(products),
        "universe_sha256": universe_sha256,
        "results": results,
        "scope": "Forward-only controller reference labels, not model skill or profit; model hypotheses remain separate",
    }
    atomic_json(path, state)
    ActivityLog(root, "controller", "goal-observer").write(
        "metrics", "goal-reference-observation", state
    )
    return state
