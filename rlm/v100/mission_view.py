"""Bounded current-run facts for the dashboard, independent of aggregate audits."""

import json
import re
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

_PACKAGE_CACHE = {}


def training_package(root: Path) -> dict:
    """Probe the trainer interpreter without the renderer's PYTHONPATH/metadata."""
    python = root / "venvs/v100-continual/bin/python"
    cached = _PACKAGE_CACHE.get(str(python))
    if cached and time.monotonic() - cached[0] < 60:
        return dict(cached[1])
    value = {
        "version": None,
        "observed_at": time.time(),
        "scope": "Isolated trainer interpreter probe; not renderer metadata",
    }
    try:
        if not python.is_file():
            raise OSError("Trainer interpreter unavailable")
        result = subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                "from importlib.metadata import version;print(version('rlms'))",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        installed = result.stdout.strip()
        if not re.fullmatch(r"0\.1\.3\+v100\.\d+", installed):
            raise ValueError("Unknown trainer package identity")
        value["version"] = installed
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        value["error"] = str(error)[:180]
    _PACKAGE_CACHE[str(python)] = (time.monotonic(), value)
    return dict(value)


def read_receipt(path: Path, maximum: int = 65536) -> dict:
    if path.is_symlink() or path.stat().st_size > maximum:
        raise ValueError("Linked or oversized status receipt")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Status receipt must be an object")
    return {**value, "source": str(path), "source_updated": path.stat().st_mtime}


def snapshot(root: Path, mission: dict, report: dict) -> dict:
    now = time.time()
    package = training_package(root)
    view = {
        "collected_at": now,
        "stage": mission.get("state", {}).get("phase", "unknown"),
        "running": mission.get("running"),
        "run": mission.get("run"),
        "evaluation": {},
        "learner": {},
        "updater": {},
        "knowledge": {},
        "history": {},
        "warnings": [],
        "installed_package": package["version"],
        "training_package_probe": package,
        "renderer_package": version("rlms"),
    }
    run = Path(mission["run"]).resolve() if mission.get("run") else None
    if run and run.is_relative_to((root / "research/mission").resolve()):
        paths = list(run.glob("*.progress.json"))
        paths.extend(run.glob("learning/update-*/generation-*/[AB]/*.progress.json"))
        paths.extend(run.glob("learning/update-*/generation-*/[AB]/fresh-audit/*.progress.json"))
        paths.extend(run.glob("learning/update-*/generation-*/*.progress.json"))
        for path in sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[:64]:
            try:
                row = read_receipt(path)
                if row.get("schema") != "v100-evaluation-progress-v1":
                    continue
                age = max(0, now - row["source_updated"])
                view["evaluation"] = {
                    **row,
                    "age_seconds": age,
                    "stale": row.get("state") == "running"
                    and (age > 900 or mission.get("running") is not True),
                }
                if row.get("state") == "running" and mission.get("running") is not True:
                    view["warnings"].append(
                        "Mission stopped; the recorded evaluation does not confirm an active worker"
                    )
                if row.get("state") == "running" and age <= 900 and mission.get("running") is True:
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
    view["updater"]["installation_confirmed"] = (
        view["updater"].get("stage", "").startswith("package-ready")
        and view["updater"].get("version") == view["installed_package"]
    )
    for name, filename in (
        ("import", "import-status.json"),
        ("index", "index-progress.json"),
        ("goal_use", "goal-use.json"),
    ):
        try:
            receipt = read_receipt(root / "research/knowledge" / filename)
            view["knowledge"][name] = {
                **receipt,
                "age_seconds": max(0, now - receipt.get("updated", receipt["source_updated"])),
            }
        except (OSError, ValueError):
            view["knowledge"][name] = {"stage": "unknown"}
    paths = sorted(
        (root / "research/mission").glob("run-*/learning/state.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:32]
    completed = 0
    for path in paths:
        try:
            cycles = read_receipt(path, 2 * 2**20).get("cycles", [])
            completed += len(cycles)
        except (OSError, ValueError, TypeError):
            continue
    view["history"] = {
        "completed_cycles": completed,
        "runs_checked": len(paths),
        "scope": "Retained cycle receipts in the latest 32 runs; not optimizer steps or verified revenue",
    }
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
