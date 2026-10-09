"""Goal-linked development evidence for both foundation and scratch master trials."""

from pathlib import Path

from rlm.v100.common import atomic_json


def prepare(root: Path, parent: dict, pool: Path, folder: Path) -> Path | None:
    from rlm.v100.competition import helper_client, managed_server
    from rlm.v100.evaluation import evaluate_suite
    from rlm.v100.goal_learning import development_suite
    from rlm.v100.goals import load_goal

    ledger = Path(
        parent["training"].get("split_ledger", root / "research/state/training-splits.sqlite3")
    )
    suite = development_suite(root, pool, ledger, folder / "goal-development.jsonl")
    if suite is None:
        return None
    atomic_json(folder / "goal-snapshot.json", {"goal_id": load_goal(root)["id"]})
    profile = folder / "goal-parent-profile.json"
    atomic_json(profile, parent)
    with managed_server(profile, root, folder / "goal-parent-server.log"):
        evaluate_suite(helper_client(parent, root), parent, suite, folder / "goal-parent.json")
    return suite


def evaluate(root: Path, candidate: dict, suite: Path | None, folder: Path, parent: dict) -> bool:
    if suite is None:
        return True
    from rlm.v100.competition import helper_client
    from rlm.v100.evaluation import evaluate_suite

    evaluate_suite(helper_client(candidate, root), candidate, suite, folder / "goal-candidate.json")
    return verify(root, folder, parent, candidate)["passed"]


def verify(root: Path, folder: Path, parent: dict, candidate: dict) -> dict:
    import json

    from rlm.v100.goal_learning import verified_gate
    from rlm.v100.goals import load_goal

    snapshot = json.loads((folder / "goal-snapshot.json").read_text())
    if snapshot["goal_id"] != (load_goal(root) or {}).get("id"):
        raise ValueError("Operator goal changed during architecture trial; replan before promotion")
    return verified_gate(
        folder, Path(parent["server"]["model"]), Path(candidate["server"]["model"])
    )
