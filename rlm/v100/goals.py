"""User-owned objectives; neither learner owns its evaluation criterion."""

import hashlib
import json
from pathlib import Path

from rlm.v100.common import atomic_json
from rlm.v100.protection import file_hash


def set_goal(root: Path, text: str, suite: Path) -> dict:
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= 2000:
        raise ValueError("Goal must contain 1-2000 characters")
    # Validate the fixed evaluator inputs before making this objective active.
    rows = [json.loads(line) for line in suite.read_text().splitlines() if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Goal requires a nonempty suite with unique IDs")
    if any(
        row.get("match") not in ("exact", "contains")
        or not isinstance(row.get("expected"), str)
        or not row["expected"]
        or not row.get("skill")
        or not row.get("messages")
        or any(m["role"] == "assistant" for m in row["messages"])
        for row in rows
    ):
        raise ValueError("Goal requires independently evaluable prompt fixtures")
    goal = {"text": text.strip(), "suite_sha256": file_hash(suite)}
    goal["id"] = hashlib.sha256(json.dumps(goal, sort_keys=True).encode()).hexdigest()
    atomic_json(root / "research/goals" / (goal["id"] + ".json"), goal)
    atomic_json(root / "research/goal.json", goal)
    return goal


def load_goal(root: Path, suite: Path | None = None) -> dict | None:
    path = root / "research/goal.json"
    if not path.exists():
        return None
    goal = json.loads(path.read_text())
    if set(goal) != {"id", "text", "suite_sha256"}:
        raise ValueError("Invalid goal snapshot")
    value = {key: goal[key] for key in ("text", "suite_sha256")}
    if hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest() != goal["id"]:
        raise ValueError("Goal snapshot changed")
    if suite is not None and file_hash(suite) != goal["suite_sha256"]:
        raise ValueError("This suite differs from the user-owned goal")
    return goal
