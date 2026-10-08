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
        raise ValueError("Only an explicit user chat request or CLI command changes the long-term goal")
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= 2000:
        raise ValueError("Plan text must contain 1-2000 characters")
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
