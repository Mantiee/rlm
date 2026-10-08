"""Read-only, bounded training evidence for the operator and research agents."""

import json
from pathlib import Path


def read_record(path: Path) -> dict:
    if path.stat().st_size > 2 * 2**20:
        raise ValueError("Evidence record exceeds 2 MiB")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Evidence record must be an object")
    return value


def collect(root: Path) -> dict:
    from rlm.v100.mission import status

    mission = status(root)
    run = Path(mission["run"]) if mission.get("run") else None
    result = {
        "schema": "v100-mission-evidence-v1",
        "mission_running": mission["running"],
        "phase": mission.get("state", {}).get("phase", mission.get("phase")),
        "run": str(run) if run else None,
        "evaluation": mission.get("evaluation"),
        "learning": mission.get("learning", {}),
        "accepted_weight_updates_this_run": None,
        "optimizer_updates_observed": None,
        "training_runs": [],
        "reports": [],
        "errors": [],
        "scope": "Current mission evidence only. Missing metrics mean unknown, not zero. Attempted/global steps are not successful optimizer updates. Recorded updates do not prove accepted weights or profit. GUI readiness is not a prerequisite for native inference, calculator tools or GPU training. Arithmetic training is an optional experiment, not a prerequisite or proof of an income edge.",
    }
    if run:
        state_path = run / "learning/state.json"
        cycles = read_record(state_path).get("cycles", []) if state_path.exists() else []
        if state_path.exists():
            result["accepted_weight_updates_this_run"] = sum(
                row.get("status", "").startswith("selected for next serving") for row in cycles
            )
            result["reports"].append(str(state_path))
        for path in sorted(run.glob("baseline-[0-9]*.json")):
            try:
                report = read_record(path)
                cases = report.get("cases", [])
                result["reports"].append(
                    {
                        "path": str(path),
                        "cases": len(cases),
                        "passed": sum(bool(row.get("passed")) for row in cases),
                    }
                )
            except (ValueError, OSError) as error:
                result["reports"].append(
                    {"path": str(path), "summary": "Unavailable within read budget"}
                )
                result["errors"].append({"path": str(path), "detail": str(error)[:300]})
        paths = sorted(
            (run / "learning").glob("update-*/**/training_health.json"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        result["training_runs_omitted"] = max(0, len(paths) - 24)
        for path in paths[:24]:
            try:
                health = read_record(path)
                counters = {
                    key: health.get(key)
                    for key in (
                        "attempted_steps",
                        "optimizer_updates",
                        "amp_skipped_steps",
                        "finite_gradient_steps",
                    )
                }
                if any(type(value) is not int or value < 0 for value in counters.values()):
                    raise ValueError("Missing or invalid optimizer counters")
                if (
                    counters["attempted_steps"]
                    != counters["optimizer_updates"] + counters["amp_skipped_steps"]
                ):
                    raise ValueError("Inconsistent optimizer counters")
                result["training_runs"].append({"path": str(path), **counters})
            except (ValueError, OSError) as error:
                result["errors"].append({"path": str(path), "detail": str(error)[:300]})
        if result["training_runs"]:
            result["optimizer_updates_observed"] = sum(
                row["optimizer_updates"] for row in result["training_runs"]
            )
        for row in cycles[-4:]:
            if row.get("detail"):
                result["errors"].append(
                    {key: row[key] for key in ("cycle", "status", "detail", "trial") if key in row}
                )
        if mission.get("state", {}).get("error"):
            result["errors"].append(mission["state"])
    paths = sorted(
        (root / "research/public-benchmarks/results").glob("*.error.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )[:2]
    for path in paths:
        try:
            error = read_record(path)
            result["errors"].append(
                {
                    "path": str(path),
                    "scope": "Historical benchmark error, may precede current run",
                    "key": str(error.get("key", ""))[:300],
                    "detail": str(error.get("error", ""))[:600],
                }
            )
        except (ValueError, OSError) as error:
            result["errors"].append({"path": str(path), "detail": str(error)[:300]})
    return result
