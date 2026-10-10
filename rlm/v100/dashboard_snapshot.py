"""Fast independent file snapshots when an aggregate report is unavailable.

Never reuse previous-run training counters or invent empty financial ledgers.
"""

import json
from pathlib import Path


def supplement(root: Path, mission: dict, report: dict, errors: list) -> dict:
    from rlm.v100.goals import load_goal
    from rlm.v100.planning import read

    result = dict(report)
    from rlm.v100.public_benchmarks import progress_for_run

    benchmark = progress_for_run(root, Path(mission["run"]) if mission.get("run") else None)
    if benchmark.get("report") or benchmark.get("historical") or "official_benchmark" in result:
        result["official_benchmark"] = benchmark
    for key, filename in (
        ("drones", "research/drones-status.json"),
        ("external_compute", "research/compute-status.json"),
        ("desktop", "research/desktop/status.json"),
        ("income_opportunities", "research/income-opportunities/status.json"),
        ("income_dispatch", "research/income-opportunities/dispatch.json"),
        ("income_work", "research/income-work/status.json"),
        ("research_quality", "research/research-quality/latest.json"),
        ("source_acquisition", "research/source-acquisition/status.json"),
    ):
        path = root / filename
        if path.exists():
            try:
                if path.stat().st_size > 2 * 2**20:
                    raise ValueError("Snapshot exceeds 2 MiB")
                value = json.loads(path.read_text())
                if not isinstance(value, dict):
                    raise ValueError("Snapshot must be an object")
                result[key] = value
            except (OSError, ValueError) as error:
                errors.append(f"{key} snapshot unavailable: {str(error)[:200]}")
    try:
        goal = load_goal(root)
        if goal:
            if result.get("income_opportunities", {}).get("goal_id") != goal["id"]:
                result.pop("income_opportunities", None)
            if result.get("income_dispatch", {}).get("goal_id") != goal["id"]:
                result.pop("income_dispatch", None)
            if result.get("source_acquisition", {}).get("goal_id") != goal["id"]:
                result.pop("source_acquisition", None)
            for name in ("income_work", "research_quality"):
                if result.get(name, {}).get("goal_id") != goal["id"]:
                    result.pop(name, None)
            result["plans"] = read(root)
            evidence = dict(result.get("mission_evidence", {}))
            evidence.setdefault("run", mission.get("run"))
            learning = dict(evidence.get("goal_learning", {}))
            if learning.get("goal_id") != goal["id"]:
                learning = {
                    "goal_id": goal["id"],
                    "scope": "Goal read from its verified file; forecast and training metrics unavailable",
                }
            for name, filename in (
                ("cached", "research/goal-learning/status.json"),
                ("observer", "research/goal-observer/status.json"),
                ("owned_cpu_dispatch", "research/goal-compute/status.json"),
            ):
                path = root / filename
                if path.exists() and path.stat().st_size <= 2 * 2**20:
                    snapshot = json.loads(path.read_text())
                    if name == "cached" and snapshot.get("goal_id") == goal["id"]:
                        learning.update(snapshot)
                    elif name != "cached" and snapshot.get("goal_id", goal["id"]) == goal["id"]:
                        learning[name] = snapshot
            evidence["goal_learning"] = learning
            result["mission_evidence"] = evidence
    except (OSError, ValueError) as error:
        errors.append("Goal snapshot unavailable: " + str(error)[:200])
    return result
