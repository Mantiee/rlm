"""Fast independent file snapshots when an aggregate report is unavailable.

Never reuse previous-run training counters or invent empty financial ledgers.
"""

import json
import math
import time
from pathlib import Path

# Only these status writers are known to put their own observation time in
# `updated`. File mtime is deliberately not used as a substitute.
STATUS_TIMESTAMP_FIELDS = {
    "drones": "updated",
    "income_opportunities": "updated",
    "income_dispatch": "updated",
    "income_work": "updated",
    "market_research": "updated",
    "backtest_audit": "updated",
    "owned_cpu_dispatch": "updated",
    "node3060": "updated",
    "paper_fee_source": "updated",
    "research_quality": "updated",
    "source_acquisition": "updated",
    "goal_learning": "updated",
    "observer": "updated",
}
STATUS_STALE_AFTER_SECONDS = 300


def learning_activity_status(
    activity: dict | None,
    mission_running: bool,
    optimizer_updates: int | None,
    now: float | None = None,
    optimizer_receipt_updated_at: float | None = None,
) -> dict:
    """Separate a fresh loop heartbeat from verified optimizer work."""
    now = time.time() if now is None else now
    activity = activity if isinstance(activity, dict) else {}
    stamp = activity.get("updated")
    age = None
    if type(stamp) in (int, float) and math.isfinite(stamp) and stamp > 0:
        age = now - float(stamp)
    if age is not None and age >= 0:
        if not mission_running or age > STATUS_STALE_AFTER_SECONDS:
            state = "stale"
            detail = "Heartbeat is old or mission is stopped; it does not show current work"
        else:
            state = "running"
            detail = "Fresh loop heartbeat only; it does not prove optimizer progress"
    elif not activity and mission_running:
        state = "planned"
        detail = "Mission is running, but no learning-stage heartbeat is recorded"
    else:
        state = "unknown"
        detail = "Heartbeat timestamp is missing, invalid, or in the future"
    optimizer_age = None
    if (
        type(optimizer_receipt_updated_at) in (int, float)
        and math.isfinite(optimizer_receipt_updated_at)
        and optimizer_receipt_updated_at > 0
    ):
        optimizer_age = now - float(optimizer_receipt_updated_at)
    if (
        type(optimizer_updates) is not int
        or optimizer_updates < 0
        or optimizer_age is None
        or optimizer_age < 0
    ):
        optimizer_activity = "unknown"
        optimizer_detail = (
            "No valid writer-timestamped optimizer receipt; active training is unconfirmed"
        )
    elif not mission_running or optimizer_age > STATUS_STALE_AFTER_SECONDS:
        optimizer_activity = "stale"
        optimizer_detail = (
            "Last optimizer receipt is old or mission is stopped; cumulative updates are historical"
        )
    else:
        optimizer_activity = "active"
        optimizer_detail = "Recent optimizer receipt; this confirms recent update activity only"
    if type(optimizer_updates) is int and optimizer_updates >= 0:
        training = {
            "state": "verified",
            "optimizer_updates": optimizer_updates,
            "scope": "Validated same-run training-health receipts; not proof of candidate acceptance",
        }
    else:
        training = {
            "state": "unknown",
            "optimizer_updates": None,
            "scope": "No valid same-run optimizer counter receipt",
        }
    training.update(
        activity_state=optimizer_activity,
        activity_detail=optimizer_detail,
        receipt_updated_at=(
            float(optimizer_receipt_updated_at) if optimizer_age is not None else None
        ),
        receipt_age_seconds=(
            round(max(0, optimizer_age), 3) if optimizer_age is not None else None
        ),
        stale_after_seconds=STATUS_STALE_AFTER_SECONDS,
    )
    return {
        "state": state,
        "detail": detail,
        "stage": activity.get("stage"),
        "observed_at": float(stamp) if age is not None else None,
        "age_seconds": round(max(0, age), 3) if age is not None else None,
        "stale_after_seconds": STATUS_STALE_AFTER_SECONDS,
        "optimizer_evidence": training,
    }


def status_freshness(name: str, value: dict | None, now: float | None = None) -> dict:
    """Describe status receipt age; never treat filesystem mtime as evidence."""
    now = time.time() if now is None else now
    field = STATUS_TIMESTAMP_FIELDS.get(name)
    base = {
        "source": name,
        "timestamp_field": field,
        "stale_after_seconds": STATUS_STALE_AFTER_SECONDS,
        "observed_at": None,
        "age_seconds": None,
    }
    if value is None:
        return {**base, "state": "unavailable", "detail": "Status file unavailable"}
    if field is None or field not in value:
        return {**base, "state": "unknown", "detail": "No trusted observation timestamp"}
    stamp = value[field]
    if type(stamp) not in (int, float) or not math.isfinite(stamp) or stamp <= 0:
        return {**base, "state": "unknown", "detail": "Observation timestamp invalid"}
    age = now - float(stamp)
    if age < 0:
        return {**base, "state": "unknown", "detail": "Observation timestamp is in the future"}
    return {
        **base,
        "state": "fresh" if age <= STATUS_STALE_AFTER_SECONDS else "stale",
        "observed_at": float(stamp),
        "age_seconds": round(age, 3),
        "detail": "Status receipt age only; underlying evidence has its own validity rules",
    }


def read_goal_status(
    path: Path, name: str, goal_id: str, now: float | None = None
) -> tuple[dict | None, dict]:
    """Read one bounded goal receipt and keep missing, invalid and mismatched distinct."""
    try:
        if not path.is_file():
            raise FileNotFoundError("Status file unavailable")
        if path.stat().st_size > 2 * 2**20:
            raise ValueError("Snapshot exceeds 2 MiB")
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("Snapshot must be an object")
    except (OSError, ValueError) as error:
        return None, {
            **status_freshness(name, None, now),
            "detail": str(error)[:200],
        }
    if value.get("goal_id") != goal_id:
        detail = (
            "Status receipt has no explicit goal ID"
            if "goal_id" not in value
            else "Status receipt belongs to a different goal"
        )
        return None, {
            **status_freshness(name, {}, now),
            "detail": detail,
        }
    return value, status_freshness(name, value, now)


def supplement(root: Path, mission: dict, report: dict, errors: list) -> dict:
    from rlm.v100.financial_snapshot import supplement as financial_supplement
    from rlm.v100.goals import load_goal
    from rlm.v100.planning import read

    result = financial_supplement(root, report, errors)
    from rlm.v100.paper_labs_snapshot import snapshot as paper_labs

    result["paper_labs"] = paper_labs(root)
    if mission.get("run"):
        path = Path(mission["run"]) / "learning/heartbeat.json"
        if path.exists() and path.stat().st_size <= 16384:
            try:
                result["learning_activity"] = json.loads(path.read_text())
            except (OSError, ValueError) as error:
                errors.append("Learning heartbeat unavailable: " + str(error)[:200])
        state_path = Path(mission["run"]) / "learning/state.json"
        if state_path.exists() and state_path.stat().st_size <= 2 * 2**20:
            try:
                state = json.loads(state_path.read_text())
                cycles = state.get("cycles", [])
                result["completed_learning_cycles"] = len(cycles)
                result["last_learning_cycle"] = cycles[-1] if cycles else None
                result["cpu_pilots"] = state.get("cpu_pilots", [])
                result["accepted_weight_updates_this_run"] = sum(
                    c.get("status", "").startswith("selected for next serving") for c in cycles
                )
            except (OSError, ValueError, TypeError) as error:
                errors.append("Learning state unavailable: " + str(error)[:200])
    from rlm.v100.mission_evidence import training_receipts

    evidence = dict(result.get("mission_evidence", {}))
    if evidence.get("run") != mission.get("run"):
        evidence = {}
    evidence["run"] = mission.get("run")
    training = (
        training_receipts(Path(mission["run"]))
        if mission.get("run")
        else {"optimizer_updates_observed": None, "training_runs": [], "errors": []}
    )
    errors.extend(
        "Training evidence unavailable: " + str(row)[:300] for row in training.pop("errors")
    )
    evidence.update(training)
    result["mission_evidence"] = evidence
    result["learning_activity_status"] = learning_activity_status(
        result.get("learning_activity"),
        bool(mission.get("running")),
        evidence.get("optimizer_updates_observed"),
        optimizer_receipt_updated_at=evidence.get("optimizer_receipt_updated_at"),
    )
    from rlm.v100.public_benchmarks import progress_for_run

    benchmark = progress_for_run(root, Path(mission["run"]) if mission.get("run") else None)
    if benchmark.get("report") or benchmark.get("historical") or "official_benchmark" in result:
        result["official_benchmark"] = benchmark
    status_freshness_rows = {}
    for key, filename in (
        ("drones", "research/drones-status.json"),
        ("external_compute", "research/compute-status.json"),
        ("desktop", "research/desktop/status.json"),
        ("income_opportunities", "research/income-opportunities/status.json"),
        ("income_dispatch", "research/income-opportunities/dispatch.json"),
        ("income_work", "research/income-work/status.json"),
        ("market_research", "research/market-research/status.json"),
        ("backtest_audit", "research/backtests/audit-status.json"),
        ("owned_cpu_dispatch", "research/goal-compute/status.json"),
        ("node3060", "research/owned-node/status.json"),
        ("paper_fee_source", "research/paper/fee-source-status.json"),
        ("paper_spot_rules", "research/paper/spot-rules-status.json"),
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
                status_freshness_rows[key] = status_freshness(key, value)
            except (OSError, ValueError) as error:
                errors.append(f"{key} snapshot unavailable: {str(error)[:200]}")
                status_freshness_rows[key] = {
                    **status_freshness(key, None),
                    "detail": str(error)[:200],
                }
        else:
            status_freshness_rows[key] = status_freshness(key, None)
    try:
        goal = load_goal(root)
        if goal:
            if result.get("income_opportunities", {}).get("goal_id") != goal["id"]:
                result.pop("income_opportunities", None)
            if result.get("income_dispatch", {}).get("goal_id") != goal["id"]:
                result.pop("income_dispatch", None)
            if result.get("source_acquisition", {}).get("goal_id") != goal["id"]:
                result.pop("source_acquisition", None)
            for name in (
                "income_work",
                "research_quality",
                "market_research",
                "owned_cpu_dispatch",
            ):
                if result.get(name, {}).get("goal_id") != goal["id"]:
                    result.pop(name, None)
            status_freshness_rows = {
                name: freshness
                for name, freshness in status_freshness_rows.items()
                if name in result
            }
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
                ("cpu_shadow_policy", "research/goal-policy/status.json"),
                ("observer", "research/goal-observer/status.json"),
                ("owned_cpu_dispatch", "research/goal-compute/status.json"),
            ):
                path = root / filename
                freshness_name = {
                    "cached": "goal_learning",
                    "cpu_shadow_policy": "cpu_shadow_policy",
                    "observer": "observer",
                    "owned_cpu_dispatch": "owned_cpu_dispatch",
                }[name]
                snapshot, freshness = read_goal_status(path, freshness_name, goal["id"])
                status_freshness_rows[freshness_name] = freshness
                if snapshot is not None and name == "cached":
                    # Cached goal-learning data requires an explicit goal identity.
                    if snapshot.get("goal_id") == goal["id"]:
                        learning.update(snapshot)
                    else:
                        status_freshness_rows[freshness_name] = {
                            **status_freshness(freshness_name, {}, time.time()),
                            "detail": "Status receipt has no matching explicit goal ID",
                        }
                elif snapshot is not None:
                    learning[name] = snapshot
            evidence["goal_learning"] = learning
            result["mission_evidence"] = evidence
    except (OSError, ValueError) as error:
        errors.append("Goal snapshot unavailable: " + str(error)[:200])
    from rlm.v100.mission_view import snapshot

    result["status_freshness"] = status_freshness_rows
    result["current_activity"] = snapshot(root, mission, result)
    return result
