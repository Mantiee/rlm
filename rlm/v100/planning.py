"""Versioned user goal and model-owned short/mid-term plans, separate from test rules."""

import fcntl
import json
import time
import uuid
from pathlib import Path

from rlm.v100.common import atomic_json
from rlm.v100.goals import load_goal, set_goal


def read(root: Path) -> dict:
    path = root / "research/plans/current.json"
    result = json.loads(path.read_text()) if path.exists() else {"short": {}, "mid": {}}
    result["long"] = load_goal(root)
    return result


def update(root: Path, horizon: str, text: str, actor: str) -> dict:
    if horizon not in ("long", "mid", "short") or actor not in ("user", "A", "B"):
        raise ValueError("Invalid plan horizon or author")
    if horizon == "long" and actor != "user":
        raise ValueError(
            "Only an explicit user chat request or CLI command changes the long-term goal"
        )
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= 2000:
        raise ValueError("Plan text must contain 1-2000 characters")
    goal = load_goal(root) or {}
    if (
        actor != "user"
        and horizon != "long"
        and any(word in goal.get("text", "").lower() for word in ("income", "dochód", "zarab"))
    ):
        if any(word in text.lower() for word in ("dashboard", "html", "css", "layout")) and not any(
            word in text.lower()
            for word in ("forecast", "outcome", "income", "source", "prognoz", "dochód")
        ):
            raise ValueError(
                "UI-only plans cannot replace financial goal work; use a separate dashboard task"
            )
    directory = root / "research/plans"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "plan.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        current = read(root)
        previous = current.get(horizon)
        value = {
            "id": uuid.uuid4().hex,
            "horizon": horizon,
            "text": text.strip(),
            "author": actor,
            "time": time.time(),
            "previous": previous,
            "goal_id": (load_goal(root) or {}).get("id"),
        }
        if horizon == "long":
            value["goal"] = set_goal(
                root, text, root / "research/income-challenge-v1/development.jsonl"
            )
            # Old plans stay visible but explicitly need replanning against the new goal.
            for key in ("short", "mid"):
                current.setdefault(key, {})["needs_replanning"] = True
        else:
            current[horizon] = {k: v for k, v in value.items() if k != "previous"}
        current.pop("long", None)
        atomic_json(directory / (value["id"] + ".json"), value)
        atomic_json(directory / "current.json", current)
    return {
        "horizon": horizon,
        "id": value["id"],
        "text": text.strip(),
        "effective": "next planning/research turn; existing experiments retain their goal snapshot",
    }


def realign_income_plans(root: Path) -> dict:
    """Explicit upgrade repair: archive stale UI-only plans, retain operator goal."""
    current = read(root)
    goal = current.get("long") or {}
    if not any(word in goal.get("text", "").lower() for word in ("income", "dochód", "zarab")):
        return {"changed": [], "reason": "Non-income goal preserved"}
    changed = []
    for horizon in ("short", "mid"):
        text = current.get(horizon, {}).get("text", "").lower()
        if any(word in text for word in ("dashboard", "css", "html", "layout")):
            replacement = (
                "Collect host-timestamped numeric evidence; precommit diverse falsifiable goal forecasts; resolve actual future outcomes including failures. Inspect tool receipts and blockers. Dashboard maintenance is secondary."
                if horizon == "short"
                else "Evaluate source-disjoint goal outcomes and lawful zero-deposit opportunities. Dispatch verified CPU trials when enough labels exist; compare V100 candidates against goal, retention and independent gates. Report measured progress; no real orders or income guarantees."
            )
            changed.append(update(root, horizon, replacement, "A"))
    return {
        "changed": changed,
        "long_term_goal_changed": False,
        "previous_plans": "retained in versioned plan archive",
    }
