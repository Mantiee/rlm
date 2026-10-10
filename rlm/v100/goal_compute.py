"""Automatic bounded owned-CPU pilots on verified active-goal outcomes.

Compact features are a separate baseline, not a replacement for the master's
full-evidence training. Labels and source splits are validated on the host.
"""

import fcntl
import json
import time
from pathlib import Path

from rlm.v100.activity import ActivityLog
from rlm.v100.common import atomic_json
from rlm.v100.goal_learning import digest
from rlm.v100.protection import file_hash


def compact(row: dict) -> dict:
    value = json.loads(row["messages"][-2]["content"])
    answer = row["messages"][-1]["content"]
    if answer not in ("down", "flat", "up"):
        raise ValueError("Goal CPU trial requires verified directional labels")
    # The same projection is used for all labels; never include the future outcome
    # or the model's probability as a feature. Original full evidence stays archived.
    for excerpt_limit in (12, 6, 0):
        prompt = (
            json.dumps(
                {
                    "h": value["horizon_seconds"],
                    "t": value["threshold"],
                    "v": value["target_at_prediction"]["value"],
                    "s": [
                        [item["value"], item["excerpt"][:excerpt_limit]]
                        for item in value["evidence_at_prediction"]
                    ],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\nAnswer: "
        )
        if len((prompt + answer).encode()) + 2 <= 256:
            return {"group": row["group"], "prompt": prompt, "answer": answer}
    raise ValueError("Goal features exceed the pinned CPU kernel context; full example retained")


def propose(root: Path, branch: str = "A") -> dict:
    from rlm.v100.distributed_compute import mailbox_path, queue_job
    from rlm.v100.goal_learning import records
    from rlm.v100.goals import load_goal
    from rlm.v100.training import load_records

    goal = load_goal(root)
    if not goal:
        return {"state": "blocked", "reason": "No operator goal"}
    rows = sorted(records(root), key=lambda row: row["id"])
    if len(rows) < 12:
        return {
            "state": "blocked",
            "reason": "Need at least 12 verified active-goal outcomes",
            "available": len(rows),
        }
    key = digest({"goal_id": goal["id"], "records": rows, "projection": "compact-v1"})
    folder = root / "research/goal-compute" / key
    folder.mkdir(parents=True, exist_ok=True)
    receipt_path = folder / (branch + ".json")
    if receipt_path.exists():
        return {**json.loads(receipt_path.read_text()), "reused": True}
    dataset = folder / "verified-full-evidence.jsonl"
    raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if dataset.exists() and dataset.read_text() != raw:
        raise ValueError("Frozen goal compute input changed")
    if not dataset.exists():
        dataset.write_text(raw)
    train, validation = load_records(dataset, root / "research/state/splits.sqlite3")
    if len(train) < 8 or len(validation) < 4 or len({r["group"] for r in validation}) < 2:
        return {
            "state": "blocked",
            "reason": "Need 8 training and 4 source-disjoint validation outcomes across 2 validation groups",
            "train": len(train),
            "validation": len(validation),
        }
    # Never shorten a full master example. This independent tiny baseline consumes
    # a documented compact projection and can be rejected by host evaluation.
    selected_train = train[-48:]
    selected_validation = validation[-16:]
    provenance = {
        "kind": "goal_observation",
        "goal_id": goal["id"],
        "dataset": str(dataset),
        "dataset_sha256": file_hash(dataset),
        "record_ids": [row["id"] for row in selected_train + selected_validation],
        "projection": "compact-v1: horizon, threshold, initial target and signal values/excerpts",
        "scope": "Small observational baseline; no causal, profit or master acceptance claim",
    }
    result = queue_job(
        root,
        mailbox_path(root),
        branch,
        "gru" if branch == "A" else "transformer",
        40,
        "Test compact source-disjoint patterns for the current operator goal",
        "goal_observation",
        [compact(r) for r in selected_train],
        [compact(r) for r in selected_validation],
        provenance=provenance,
    )
    result.update(goal_id=goal["id"], provenance=provenance)
    atomic_json(receipt_path, result)
    ActivityLog(root, branch, "goal-compute").write("steps", "goal-compute-queued", result)
    return result


def tick(root: Path) -> None:
    from rlm.v100.distributed_compute import inspect
    from rlm.v100.goals import load_goal

    goal_id = (load_goal(root) or {}).get("id")

    preferences = root / "research/user-preferences.json"
    if not preferences.exists() or not json.loads(preferences.read_text()).get(
        "automatic_goal_compute"
    ):
        return
    folder = root / "research/goal-compute"
    folder.mkdir(parents=True, exist_ok=True)
    status = folder / "status.json"
    with (folder / "dispatch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        previous = json.loads(status.read_text()) if status.exists() else {}
        if previous.get("goal_id") == goal_id and time.time() < previous.get("next_poll", 0):
            return
        atomic_json(
            status,
            {
                "state": "checking",
                "goal_id": goal_id,
                "updated": time.time(),
                "next_poll": time.time() + 300,
            },
        )
        try:
            resources = inspect(root)
            workers = [
                w
                for w in resources["workers"]
                if not w.get("stale", True) and w.get("phase") == "idle"
            ]
            if not workers:
                result = {"state": "blocked", "reason": "No fresh idle owned CPU worker"}
            elif any(j["state"] in ("queued", "running", "imported") for j in resources["jobs"]):
                result = {"state": "waiting", "reason": "Existing owned jobs retain priority"}
            else:
                result = {
                    "state": "dispatched",
                    "branches": [propose(root, branch) for branch in ("A", "B")],
                }
                if all(r.get("state") == "blocked" for r in result["branches"]):
                    result["state"] = "blocked"
                elif all(
                    r.get("reused") or r.get("state") == "blocked" for r in result["branches"]
                ):
                    result["state"] = "no-new-data"
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            result = {"state": "blocked", "reason": str(error)[:400]}
        atomic_json(
            status,
            {**result, "goal_id": goal_id, "updated": time.time(), "next_poll": time.time() + 300},
        )


def verify_job(root: Path, job: dict) -> None:
    """Recheck label provenance and the exact compact projection before host scoring."""
    from rlm.v100.training import load_records

    provenance = job["provenance"]
    dataset = Path(provenance["dataset"])
    if dataset.is_symlink() or not dataset.resolve().is_relative_to(
        (root / "research/goal-compute").resolve()
    ):
        raise ValueError("Goal compute dataset escaped its immutable archive")
    if file_hash(dataset) != provenance["dataset_sha256"]:
        raise ValueError("Goal compute dataset changed")
    train, validation = load_records(dataset, root / "research/state/splits.sqlite3")
    expected = train[-48:] + validation[-16:]
    if provenance["record_ids"] != [row["id"] for row in expected]:
        raise ValueError("Goal compute record selection changed")
    if any(row["verification"]["goal_id"] != provenance["goal_id"] for row in expected):
        raise ValueError("Goal compute contains a different goal's outcomes")
    if job["train"] != [compact(row) for row in train[-48:]] or job["validation"] != [
        compact(row) for row in validation[-16:]
    ]:
        raise ValueError("Goal compute projection or labels changed")
