"""Reject obvious test mismatches and distinguish notes from executed work.

This conservative lexical check catches known category errors. It is not a
semantic proof that a test can establish causality, generalization or profit.
"""

import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path

from rlm.v100.common import atomic_json
from rlm.v100.goals import load_goal

MARKET = re.compile(r"\b(btc|eth|bitcoin|crypto|kraken|candles?|backtest|trading)\b", re.I)
NON_MARKET = re.compile(
    r"micro.?tasks?|affiliate|paid (?:studies|surveys)|user.?testing|software bounty|freelanc|mikrozada|ankiet",
    re.I,
)
EXECUTED = {"register_income_opportunity", "predict_goal_pattern"}


def assessment(root: Path, result: dict) -> dict:
    hypothesis, test = result.get("hypothesis", ""), result.get("suggested_test", "")
    reason = None
    if not hypothesis.strip() or not test.strip():
        reason = "Empty hypothesis or falsifiable test"
    elif NON_MARKET.search(hypothesis) and MARKET.search(test) and not NON_MARKET.search(test):
        reason = "Market price test cannot establish the proposed non-market income mechanism"
    trace = result.get("research_trace", [])
    evidence = []
    work = []
    for receipt in trace:
        value = receipt.get("result", {})
        if value.get("status") == "failed" or value.get("error"):
            continue
        identity = value.get("id")
        if receipt.get("tool") == "observe_goal_source" and identity:
            evidence.append(identity)
        if receipt.get("tool") in EXECUTED and identity:
            work.append({"tool": receipt["tool"], "id": identity})
    goal_id = (load_goal(root) or {}).get("id", "no-goal")
    normalized = [re.sub(r"\W+", " ", text.casefold()).strip() for text in (hypothesis, test)]
    identity = hashlib.sha256(json.dumps([goal_id, *normalized]).encode()).hexdigest()
    directory = root / "research/research-quality"
    directory.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(directory / "ledger.sqlite3", timeout=5) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS notes(id TEXT PRIMARY KEY, goal TEXT, count INTEGER)"
        )
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT count FROM notes WHERE id=?", (identity,)).fetchone()
        repeated = row is not None
        db.execute(
            "INSERT INTO notes VALUES(?,?,1) ON CONFLICT(id) DO UPDATE SET count=count+1",
            (identity, goal_id),
        )
        state = (
            "rejected test"
            if reason
            else "repeated advisory"
            if repeated and not work
            else "executed research receipt"
            if work
            else "advisory only - no executed test"
        )
        value = {
            "id": identity,
            "goal_id": goal_id,
            "updated": time.time(),
            "state": state,
            "reason": reason,
            "repeated": repeated,
            "evidence_ids": evidence,
            "work_receipts": work,
            "eligible_for_review": reason is None and (not repeated or bool(work)),
            "scope": "Lexical category and receipt audit only; no profit, causal or ML-label certification",
        }
        atomic_json(directory / "latest.json", value)
    return value


def repair_history(root: Path) -> dict:
    """Annotate old notes in place with a backup; retain their original text."""
    from rlm.v100.experiments import SharedLab

    lab = SharedLab(root / "research/state/competition.sqlite3")
    backup = root / "research/research-quality" / f"before-contract-{time.time_ns()}.sqlite3"
    backup.parent.mkdir(parents=True, exist_ok=True)
    changed = 0
    try:
        with sqlite3.connect(backup) as target:
            lab.db.backup(target)
        rows = lab.db.execute(
            "SELECT sequence,payload FROM events WHERE kind='worker-result'"
        ).fetchall()
        for sequence, encoded in rows:
            value = json.loads(encoded)
            if value.get("research_quality"):
                continue
            value["research_quality"] = assessment(root, value)
            value["status"] = value["research_quality"]["state"]
            with lab.db:
                lab.db.execute(
                    "UPDATE events SET payload=? WHERE sequence=?", (json.dumps(value), sequence)
                )
            changed += 1
    finally:
        lab.close()
    return {"annotated": changed, "backup": str(backup), "original_notes": "retained"}
