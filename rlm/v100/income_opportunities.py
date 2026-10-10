"""Evidence-backed income hypotheses, never a fabricated revenue ledger.

Estimates are model proposals. Source retrieval verifies provenance, not the
operator's eligibility, demand, causality or eventual payment.
"""

import hashlib
import json
import math
import sqlite3
import time
from pathlib import Path

from rlm.v100.activity import ActivityLog
from rlm.v100.common import atomic_json
from rlm.v100.goal_learning import observation
from rlm.v100.goals import load_goal

TEXT_FIELDS = (
    "title",
    "domain",
    "mechanism",
    "eligibility",
    "blockers",
    "next_test",
    "failure_condition",
)
NUMBER_FIELDS = (
    "gross_pln_low",
    "gross_pln_high",
    "total_cost_pln",
    "labor_hours",
    "first_income_days",
    "upfront_spend_pln",
)


def register(root: Path, branch: str, specification: dict) -> dict:
    goal = load_goal(root)
    if goal is None or branch not in ("A", "B"):
        raise ValueError("Income hypothesis needs an active goal and research branch")
    required = {*TEXT_FIELDS, *NUMBER_FIELDS, "evidence"}
    if not isinstance(specification, dict) or set(specification) != required:
        raise ValueError("Income specification fields differ from the documented schema")
    for name in TEXT_FIELDS:
        value = specification[name]
        if not isinstance(value, str) or not 1 <= len(value.strip()) <= 1200:
            raise ValueError(name + " must contain 1-1200 characters")
    for name in NUMBER_FIELDS:
        value = specification[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1e9:
            raise ValueError(name + " must be a finite nonnegative estimate")
    if specification["labor_hours"] <= 0:
        raise ValueError("Labor must include research, preparation and fulfillment time")
    if specification["gross_pln_high"] < specification["gross_pln_low"]:
        raise ValueError("Gross income interval is reversed")
    from rlm.v100.income_policy import read as policy

    if specification["upfront_spend_pln"] != 0 and not policy(root)["capital_research"]:
        raise ValueError("Current mission admits only zero-upfront-spend experiments")
    identities = specification["evidence"]
    if not isinstance(identities, list) or not 1 <= len(identities) <= 8:
        raise ValueError("Provide 1-8 distinct host observation IDs")
    if any(not isinstance(value, str) or len(value) != 64 for value in identities):
        raise ValueError("Invalid host evidence identity")
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate evidence does not provide independent support")
    now = time.time()
    sources = [observation(root, identity) for identity in identities]
    if any(not 0 <= now - source["acquired"] <= 86400 for source in sources):
        raise ValueError("Income terms require host evidence refreshed within 24 hours")
    key = [
        goal["id"],
        specification["domain"].strip().casefold(),
        specification["title"].strip().casefold(),
    ]
    identity = hashlib.sha256(json.dumps(key).encode()).hexdigest()
    version = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    row = {
        "id": identity,
        "version": version,
        "goal_id": goal["id"],
        "branch": branch,
        "updated": now,
        "state": "evidence-backed hypothesis",
        "specification": specification,
        "estimated_net_pln_low": specification["gross_pln_low"] - specification["total_cost_pln"],
        "estimated_net_pln_high": specification["gross_pln_high"] - specification["total_cost_pln"],
        "actual_income_pln": None,
        "scope": "Unverified estimates before personal tax. Evidence provenance is verified; eligibility, demand, novelty and payment are not. No accounts, outreach, sale or order executed.",
    }
    directory = root / "research/income-opportunities"
    directory.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(directory / "ledger.sqlite3", timeout=5) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS candidates(id TEXT PRIMARY KEY, goal TEXT, data TEXT)"
        )
        db.execute("BEGIN IMMEDIATE")
        count = db.execute(
            "SELECT count(*) FROM candidates WHERE goal=?", (goal["id"],)
        ).fetchone()[0]
        exists = db.execute("SELECT 1 FROM candidates WHERE id=?", (identity,)).fetchone()
        if count >= 128 and not exists:
            raise ValueError(
                "Income hypothesis budget reached; retain and review existing candidates"
            )
        atomic_json(directory / "versions" / identity / (version + ".json"), row)
        db.execute(
            "INSERT OR REPLACE INTO candidates VALUES(?,?,?)",
            (identity, goal["id"], json.dumps(row)),
        )
    ActivityLog(root, branch, "income-opportunities").write(
        "decisions", "candidate-registered", row
    )
    status(root)
    return row


def status(root: Path) -> dict:
    goal = load_goal(root)
    goal_id = (goal or {}).get("id")
    directory = root / "research/income-opportunities"
    path = directory / "ledger.sqlite3"
    rows = []
    if path.exists():
        db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1)
        try:
            rows = [
                json.loads(row[0])
                for row in db.execute("SELECT data FROM candidates WHERE goal=?", (goal_id,))
            ]
        finally:
            db.close()
    # Present both speed and net-value estimates, without converting them to profit evidence.
    rows.sort(
        key=lambda row: (
            row["specification"]["first_income_days"],
            -row["estimated_net_pln_low"] / row["specification"]["labor_hours"],
        )
    )
    value = {
        "goal_id": goal_id,
        "updated": time.time(),
        "candidates": rows,
        "actual_income_pln": None,
        "scope": "Hypothesis comparison only. No verified revenue ledger connected; missing income is unknown, not zero.",
    }
    from rlm.v100.income_policy import read as policy

    value["research_policy"] = policy(root)
    atomic_json(directory / "status.json", value)
    return value


def commission(root: Path) -> dict:
    """One bounded collection/dossier assignment using the already serving master."""
    from rlm.v100.drones import cancel, schedule

    goal = load_goal(root)
    if not goal or not any(word in goal["text"].lower() for word in ("income", "dochód", "zarab")):
        return {"state": "not applicable"}
    active_path = root / "research/mission/active.json"
    if not active_path.exists():
        return {"state": "blocked", "reason": "No active mission configuration"}
    run = Path(json.loads(active_path.read_text())["run"]).resolve()
    if not run.is_relative_to((root / "research/mission").resolve()):
        raise ValueError("Mission profile outside mission directory")
    profile = json.loads((run / "input-profile.json").read_text())
    if not profile.get("resources", {}).get("paper_research_enabled"):
        return {"state": "blocked", "reason": "Income research disabled by operator"}
    from rlm.v100.market_research import commission as commission_market

    commission_market(root)
    path = root / "research/income-opportunities/dispatch.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    upgraded = previous.get("work_kind") != "income"
    if (previous.get("goal_id") != goal["id"] or upgraded) and previous.get("receipt", {}).get(
        "id"
    ):
        cancel(root, previous["receipt"]["id"])
    if (
        not upgraded
        and previous.get("goal_id") == goal["id"]
        and time.time() < previous.get("next_poll", 0)
    ):
        return previous
    try:
        receipt = schedule(root, "A", "income", goal["id"], 600)
        value = {
            "state": "scheduled",
            "work_kind": "income",
            "goal_id": goal["id"],
            "receipt": receipt,
            "next_poll": time.time() + 600,
            "updated": time.time(),
            "scope": "Job allocation only; inspect tool receipts and registered candidates for actual work",
        }
    except (OSError, ValueError, sqlite3.Error) as error:
        value = {
            "state": "blocked",
            "goal_id": goal["id"],
            "error": str(error)[:300],
            "next_poll": time.time() + 60,
            "updated": time.time(),
        }
    atomic_json(path, value)
    return value
