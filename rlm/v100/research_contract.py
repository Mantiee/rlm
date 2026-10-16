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
    r"micro.?tasks?|micro.?tasking|affiliate|paid (?:studies|surveys)|user.?testing|software bounty|freelanc|mikrozada|ankiet|prolific|data label(?:ing|ling)|outside markets|content or services",
    re.I,
)
PAPER = re.compile(r"paper[ -]trad|paper (?:profit|simulation|p.?&.?l)|trading simulation", re.I)


def python_process_receipt(root: Path, value: dict) -> dict | None:
    """Verify archived process provenance without certifying task correctness."""
    if (
        type(value.get("exit_code")) is not int
        or value["exit_code"] != 0
        or value.get("timed_out") is not False
    ):
        return None
    directory = value.get("directory")
    if not isinstance(directory, str):
        return None
    folder = Path(directory)
    archive = (root / "research/python-experiments").resolve()
    if folder.is_symlink() or not folder.resolve().is_relative_to(archive):
        return None
    receipt_path = folder / "result.json"
    if (
        not receipt_path.is_file()
        or receipt_path.is_symlink()
        or receipt_path.stat().st_size > 2 * 2**20
    ):
        return None
    try:
        archived = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if archived != value:
        return None
    artifacts = []
    total_bytes = 0
    work = folder / "work"
    if work.is_symlink():
        return None
    if work.is_dir():
        for path in work.rglob("*"):
            if len(artifacts) >= 20:
                break
            if (
                path.is_symlink()
                or not path.is_file()
                or not path.resolve().is_relative_to(work.resolve())
            ):
                continue
            size = path.stat().st_size
            if not 0 < size <= 32 * 2**20 - total_bytes:
                continue
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            total_bytes += size
            artifacts.append({"path": str(path), "bytes": size, "sha256": digest})
    output = value.get("output")
    if not artifacts and not (isinstance(output, str) and output.strip()):
        return None
    return {
        "tool": "run_research_python",
        "kind": "executed-unvalidated",
        "directory": str(folder),
        "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "exit_code": 0,
        "timed_out": False,
        "artifacts": artifacts,
        "scope": "Archived process exit and output provenance only; task correctness, usefulness and income unverified",
    }


def assessment(root: Path, result: dict, historical: bool = False) -> dict:
    hypothesis, test = result.get("hypothesis", ""), result.get("suggested_test", "")
    reason = None
    if not hypothesis.strip() or not test.strip():
        reason = "Empty hypothesis or falsifiable test"
    elif NON_MARKET.search(hypothesis) and MARKET.search(test) and not NON_MARKET.search(test):
        reason = "Market price test cannot establish the proposed non-market income mechanism"
    elif (
        NON_MARKET.search(hypothesis + " " + test)
        and PAPER.search(test)
        and re.search(r"compar|higher|faster|versus|\bvs\b|outperform|against", test, re.I)
    ):
        reason = "Simulated paper returns cannot be compared as actual earned labor income"
    trace = result.get("research_trace", [])
    evidence = []
    work = []
    proposals, forecasts, processes = [], [], []
    for receipt in trace:
        value = receipt.get("result", {})
        if (
            value.get("status") in {"failed", "blocked"}
            or value.get("state") == "blocked"
            or value.get("error")
        ):
            continue
        identity = value.get("id")
        if receipt.get("tool") == "observe_goal_source" and identity:
            evidence.append(identity)
        tool = receipt.get("tool")
        if tool == "register_income_opportunity" and identity:
            proposals.append(identity)
        elif tool == "predict_goal_pattern" and identity:
            forecasts.append(identity)
        elif tool == "backtest_prices" and value.get("report") and not value.get("cache_reused"):
            work.append({"tool": tool, "report": value["report"]})
        elif tool == "run_research_python":
            receipt = python_process_receipt(root, value)
            if receipt is not None:
                processes.append(receipt)
        elif tool == "sandbox_run" and value.get("exit_code") == 0:
            processes.append({"tool": tool, "scope": "Process exit only, not task validation"})
    goal_id = (load_goal(root) or {}).get("id", "no-goal")
    normalized = [re.sub(r"\W+", " ", text.casefold()).strip() for text in (hypothesis, test)]
    identity = hashlib.sha256(json.dumps([goal_id, *normalized]).encode()).hexdigest()
    test_identity = hashlib.sha256(json.dumps([goal_id, normalized[1]]).encode()).hexdigest()
    directory = root / "research/research-quality"
    directory.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(directory / "ledger.sqlite3", timeout=5) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS notes(id TEXT PRIMARY KEY, goal TEXT, count INTEGER)"
        )
        db.execute("CREATE TABLE IF NOT EXISTS advisory_tests(id TEXT PRIMARY KEY)")
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT count FROM notes WHERE id=?", (identity,)).fetchone()
        repeated_test = (
            db.execute("SELECT 1 FROM advisory_tests WHERE id=?", (test_identity,)).fetchone()
            is not None
        )
        repeated = row is not None or repeated_test
        db.execute("INSERT OR IGNORE INTO advisory_tests VALUES(?)", (test_identity,))
        db.execute(
            "INSERT INTO notes VALUES(?,?,1) ON CONFLICT(id) DO UPDATE SET count=count+1",
            (identity, goal_id),
        )
        state = (
            "rejected test"
            if reason
            else "repeated advisory"
            if repeated and not (work or forecasts or processes)
            else "executed research receipt"
            if work
            else "forecast committed - outcome pending"
            if forecasts
            else "process executed - task validation missing"
            if processes
            else "proposal registered - no executed test"
            if proposals
            else "advisory only - no executed test"
        )
        value = {
            "contract_version": 3,
            "historical_audit": historical,
            "id": identity,
            "goal_id": goal_id,
            "updated": time.time(),
            "state": state,
            "reason": reason,
            "repeated": repeated,
            "repeated_advisory_test": repeated_test,
            "evidence_ids": evidence,
            "work_receipts": work,
            "proposal_receipts": proposals,
            "forecast_commitments": forecasts,
            "process_receipts": processes,
            "eligible_for_review": reason is None
            and (not repeated or bool(work or forecasts or processes)),
            "scope": "Lexical category and receipt audit only; no profit, causal or ML-label certification",
        }
        if not historical:
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
            if value.get("research_quality", {}).get("contract_version") == 3:
                continue
            value["research_quality"] = assessment(root, value, historical=True)
            value["status"] = value["research_quality"]["state"]
            with lab.db:
                lab.db.execute(
                    "UPDATE events SET payload=? WHERE sequence=?", (json.dumps(value), sequence)
                )
            changed += 1
    finally:
        lab.close()
    return {"annotated": changed, "backup": str(backup), "original_notes": "retained"}
