"""Bounded current-run facts for the dashboard, independent of aggregate audits."""

import json
import time
from pathlib import Path


def read_receipt(path: Path, maximum: int = 65536) -> dict:
    if path.is_symlink() or path.stat().st_size > maximum:
        raise ValueError("Linked or oversized status receipt")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Status receipt must be an object")
    return {**value, "source": str(path), "source_updated": path.stat().st_mtime}


def snapshot(root: Path, mission: dict, report: dict) -> dict:
    now = time.time()
    view = {
        "collected_at": now,
        "stage": mission.get("state", {}).get("phase", "unknown"),
        "running": mission.get("running"),
        "run": mission.get("run"),
        "evaluation": {},
        "learner": {},
        "updater": {},
        "warnings": [],
    }
    run = Path(mission["run"]).resolve() if mission.get("run") else None
    if run and run.is_relative_to((root / "research/mission").resolve()):
        paths = list(run.glob("*.progress.json"))
        paths.extend(run.glob("learning/update-*/generation-*/[AB]/*.progress.json"))
        paths.extend(run.glob("learning/update-*/generation-*/*.progress.json"))
        for path in sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[:64]:
            try:
                row = read_receipt(path)
                if row.get("schema") != "v100-evaluation-progress-v1":
                    continue
                age = max(0, now - row["source_updated"])
                view["evaluation"] = {**row, "age_seconds": age, "stale": age > 900}
                if row.get("state") == "running" and age <= 900:
                    view["stage"] = "candidate evaluation"
                break
            except (OSError, ValueError) as error:
                view["warnings"].append(str(error)[:180])
        profile_path = Path(
            mission.get("learning", {}).get("live_profile") or run / "input-profile.json"
        ).resolve()
        if profile_path.is_relative_to((root / "research").resolve()):
            try:
                profile = read_receipt(profile_path, 2 * 2**20)
                training = profile.get("training", {})
                view["learner"] = {
                    "base_model": training.get("base_model"),
                    "adapter": training.get("init_adapter"),
                    "serving_model": profile.get("server", {}).get("model"),
                    "source": str(profile_path),
                    "scope": "Configured accepted learner; candidate/advisory models have separate identities",
                }
            except (OSError, ValueError) as error:
                view["warnings"].append(str(error)[:180])
    try:
        view["updater"] = read_receipt(root / "research/managed-update/status.json")
    except (OSError, ValueError) as error:
        view["updater"] = {"stage": "unknown", "error": str(error)[:180]}
    work = report.get("income_work", {})
    view["income"] = {
        k: work.get(k)
        for k in (
            "state",
            "updated",
            "sources_collected",
            "proposals_registered",
            "error",
            "result_path",
        )
    }
    view["verified_income"] = "unknown: no verified payment receipt in this snapshot"
    activity = report.get("learning_activity", {})
    stamps = [view["evaluation"].get("source_updated", 0), activity.get("updated", 0)]
    stamps = [x for x in stamps if isinstance(x, (int, float))]
    view["last_activity_at"] = max(stamps, default=0) or None
    view["activity_age_seconds"] = max(0, now - max(stamps)) if stamps and max(stamps) else None
    view["completed_cycles"] = mission.get("learning", {}).get("completed_cycles")
    view["accepted_updates"] = report.get("accepted_weight_updates_this_run")
    return view
