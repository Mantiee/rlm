"""Evidence-based commissioning: unknown is distinct from success and failure."""

from pathlib import Path


def assess(root: Path, mission: dict, report: dict, layout: dict) -> dict:
    evidence = report.get("mission_evidence", {}) or {}
    external = report.get("external_compute", {}) or {}
    workers = external.get("workers", [])
    checks = []

    def add(name, state, detail, source):
        checks.append({"name": name, "state": state, "detail": detail, "source": source})

    add(
        "Mission",
        "passed" if mission.get("running") else "failed",
        mission.get("state", {}).get("phase", "unknown"),
        mission.get("log"),
    )
    add(
        "Dashboard layout",
        "passed" if layout.get("state") == "validated layout active" else "unknown",
        layout.get("state", "No publication receipt"),
        str(root / "research/dashboard/layout-status.json"),
    )
    add(
        "Owned worker heartbeat",
        "passed" if any(not w.get("stale", True) for w in workers) else "unknown",
        "Heartbeat proves availability, not execution",
        "external_compute.workers",
    )
    completed = [
        j
        for j in external.get("jobs", [])
        if j.get("state") in ("completed", "validated", "locally-validated", "rejected")
        or j.get("phase") in ("completed", "validated")
    ]
    add(
        "Owned worker execution",
        "observed" if completed else "unknown",
        f"{len(completed)} reported completed jobs; host validation remains separate",
        "external_compute.jobs",
    )
    validated = [j for j in external.get("jobs", []) if j.get("state") == "locally-validated"]
    add(
        "Owned worker host validation",
        "passed" if validated else "unknown",
        f"{len(validated)} host-validated trials; no master promotion implied",
        "external_compute.jobs.local_report",
    )
    updates = evidence.get("optimizer_updates_observed")
    add(
        "Production optimizer steps",
        "passed" if type(updates) is int and updates > 0 else "unknown",
        updates if updates is not None else "No verified counter",
        "mission_evidence.training_runs",
    )
    accepted = evidence.get("accepted_weight_updates_this_run")
    add(
        "Accepted production weights",
        "passed" if type(accepted) is int and accepted > 0 else "unknown",
        accepted if accepted is not None else "No verified acceptance receipt",
        "mission_evidence.reports",
    )
    goal = evidence.get("goal_learning", {})
    outcomes = goal.get("resolved_total", goal.get("recent_resolved"))
    add(
        "Goal outcomes",
        "observed" if type(outcomes) is int and outcomes > 0 else "unknown",
        outcomes if outcomes is not None else "Goal metrics unavailable",
        "research/goal-learning/ledger.sqlite3",
    )
    from rlm.v100.live_status import recent_events

    events = report["live_events"] if "live_events" in report else recent_events(root)
    add(
        "Streaming transport",
        "observed" if any(e.get("kind") == "inference-delta" for e in events) else "unknown",
        "Requires an observed delta on the installed backend",
        "research/logs/activity/*/timeline.jsonl",
    )
    return {
        "schema": "synta-readiness-v1",
        "checks": checks,
        "scope": "Per-check evidence only; no universal 100% completion claim. A rejected candidate can be correct behavior.",
        "unverified": [c["name"] for c in checks if c["state"] == "unknown"],
    }
