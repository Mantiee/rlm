"""Transparent commissioning comparison, never proof of a profitable strategy."""

import hashlib
import json
import math
import re
import sqlite3
import time
from pathlib import Path

import requests

from rlm.v100.activity import ActivityLog
from rlm.v100.common import atomic_json
from rlm.v100.goals import load_goal
from rlm.v100.income_policy import normalized, read

PRICE_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=3600"


def requested(message: str) -> bool:
    value = normalized(message)
    return (
        "?" not in value
        and not re.search(r"\b(?:nie|dont|do not)\b", value)
        and bool(
            re.search(r"\b(?:przetestuj|test|zbieraj|collect|poszukaj)\b", value)
            and re.search(r"strateg|historycz|historical|backtest", value)
        )
    )


def triage(report: dict) -> dict:
    reasons = []
    result = report.get("development_test", {})
    net = result.get("net_return")
    if not result.get("fills"):
        reasons.append("No resolved simulated fills")
    for label, metric in (
        ("Cash comparison", net),
        ("Double-cost stress", report.get("double_cost_stress", {}).get("net_return")),
    ):
        if type(metric) not in (int, float) or not math.isfinite(metric) or metric <= 0:
            reasons.append(label + ": missing or nonpositive return")
    hold = report.get("buy_hold_baseline", {}).get("net_return")
    if (
        type(net) not in (int, float)
        or type(hold) not in (int, float)
        or not math.isfinite(hold)
        or net <= hold
    ):
        reasons.append("Does not beat the recorded buy-and-hold baseline")
    folds = report.get("walk_forward", [])
    if len(folds) < 3 or any(
        type(row.get("test", {}).get("net_return")) not in (int, float)
        or not math.isfinite(row["test"]["net_return"])
        or row["test"]["net_return"] <= 0
        for row in folds
    ):
        reasons.append("Missing or losing chronological development windows")
    return {
        "state": "rejected" if reasons else "hypothesis for independent future testing",
        "reasons": reasons,
        "profit_verified": False,
    }


def compare(root: Path) -> dict:
    from rlm.v100.backtesting import run
    from rlm.v100.research_tools import download_page

    goal_id = (load_goal(root) or {}).get("id")
    path = root / "research/market-research/status.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    if (
        previous.get("goal_id") == goal_id
        and previous.get("state") == "completed"
        and 0 <= time.time() - previous["updated"] < 3600
    ):
        for row in previous["comparisons"]:
            source = Path(row["report"]).resolve()
            if (
                not source.is_relative_to((root / "research/backtests").resolve())
                or hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]
            ):
                raise ValueError("Recorded historical report checksum differs; reuse refused")
        return {**previous, "cache_reused": True}
    log = ActivityLog(root, "A", "market-research")
    log.write("tools", "comparison-start", {"url": PRICE_URL, "goal_id": goal_id})
    value = {
        "goal_id": goal_id,
        "updated": time.time(),
        "state": "running",
        "comparisons": [],
        "actual_income_pln": None,
        "instrument": "BTC-USD",
        "scope": "Fixed commissioning test; historical development only, not a best-strategy or profit claim. No paper or real order. Fees/slippage are assumptions.",
    }
    atomic_json(path, value)
    try:
        snapshot = download_page(PRICE_URL)
        for rule in ("momentum", "mean_reversion"):
            report = run(
                root,
                "A",
                {
                    "price_url": PRICE_URL,
                    "event_url": "",
                    "rule": rule,
                    "fee_bps": 50,
                    "slippage_bps": 10,
                },
                source_snapshot=snapshot,
            )
            if (load_goal(root) or {}).get("id") != goal_id:
                raise ValueError(
                    "Goal changed during comparison; reports retained, not current evidence"
                )
            row = {
                key: report.get(key)
                for key in (
                    "report",
                    "parameters",
                    "sources",
                    "fetched_at",
                    "development_test",
                    "double_cost_stress",
                    "buy_hold_baseline",
                    "data_window",
                )
            }
            row["development_test"] = {
                k: v for k, v in row["development_test"].items() if k != "trace"
            }
            row["triage"] = triage(report)
            row["sha256"] = hashlib.sha256(Path(report["report"]).read_bytes()).hexdigest()
            value["comparisons"].append(row)
        value.update(state="completed", updated=time.time(), cache_reused=False)
        log.write("tools", "comparison-result", value)
    except (ValueError, OSError, KeyError, TypeError, requests.RequestException) as error:
        value.update(state="blocked", error=str(error), updated=time.time())
        log.write("errors", "comparison-blocked", value)
    atomic_json(path, value)
    from rlm.v100.backtest_audit import run as audit_backtests

    value["audit"] = audit_backtests(root, 4)
    atomic_json(path, value)
    return value


def commission(root: Path) -> dict:
    from rlm.v100.drones import cancel, schedule

    path = root / "research/market-research/dispatch.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    goal_id = (load_goal(root) or {}).get("id")
    if previous.get("receipt", {}).get("id") and (
        previous.get("goal_id") != goal_id or not read(root)["capital_research"]
    ):
        cancel(root, previous["receipt"]["id"])
    value = {"goal_id": goal_id, "updated": time.time(), "state": "not enabled"}
    if read(root)["capital_research"] and goal_id:
        try:
            value.update(
                state="commissioned", receipt=schedule(root, "A", "market-research", goal_id, 3600)
            )
        except (ValueError, OSError, sqlite3.Error) as error:
            value.update(state="blocked", error=str(error))
    atomic_json(path, value)
    return value


def execute(root: Path, goal_id: str) -> dict:
    if (load_goal(root) or {}).get("id") != goal_id or not read(root)["capital_research"]:
        return {"state": "blocked", "error": "Current goal/research policy does not admit this job"}
    active = root / "research/mission/active.json"
    if not active.exists():
        return {"state": "blocked", "error": "No active mission configuration"}
    run = Path(json.loads(active.read_text())["run"]).resolve()
    if not run.is_relative_to((root / "research/mission").resolve()):
        raise ValueError("Mission profile outside owned mission directory")
    profile = json.loads((run / "input-profile.json").read_text())
    if not profile.get("resources", {}).get("paper_research_enabled"):
        return {"state": "blocked", "error": "Income research disabled by operator"}
    return compare(root)
