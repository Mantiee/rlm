"""Separate sidecar provenance and read-only accounting in the main dashboard."""

import shutil
import sqlite3
import subprocess
from datetime import UTC, datetime

import pytest

from rlm.v100.dashboard_insights import SCRIPT
from rlm.v100.dashboard_layout import PAGE
from rlm.v100.paper_labs_snapshot import snapshot


def ledger(root, name, *, cash=9500, qty=0.001, bid=300000):
    path = root / f"research/experiments/{name}.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.executescript(
            "CREATE TABLE ticks(id INTEGER PRIMARY KEY, ts TEXT, bid REAL, ask REAL, a REAL, b REAL, valuation_basis TEXT);"
        )
        ts = datetime.fromtimestamp(1800000000, UTC).isoformat()
        db.execute(
            "INSERT INTO ticks VALUES(1,?,?,?,?,?,?)",
            (ts, bid, bid + 1, 9990, 9790, "legacy-liquidation-estimate-v1"),
        )
        db.execute(
            "INSERT INTO ticks VALUES(2,?,?,?,?,?,?)",
            (ts, bid, bid + 1, 10000, 9800, "gross-mark-to-bid-v2"),
        )
        if name == "live-paper-v3":
            db.executescript(
                "CREATE TABLE accounts(branch TEXT PRIMARY KEY,cash REAL,qty REAL); CREATE TABLE fills(id INTEGER PRIMARY KEY,ts TEXT,branch TEXT,side TEXT,qty REAL,price REAL,fee REAL);"
            )
            db.executemany(
                "INSERT INTO accounts VALUES(?,?,?)", [("a", 10000, 0), ("b", cash, qty)]
            )
        else:
            db.execute("CREATE TABLE portfolio(branch TEXT PRIMARY KEY,cash REAL,units REAL)")
            db.execute(
                "CREATE TABLE events(id INTEGER PRIMARY KEY,ts TEXT,kind TEXT,branch TEXT,details TEXT)"
            )
            db.execute(
                "INSERT INTO events VALUES(1,?,'fill','B',?)", (ts, '{"action":"BUY","qty":0.002}')
            )
            db.executemany(
                "INSERT INTO portfolio VALUES(?,?,?)", [("A", 10000, 0), ("B", cash, qty)]
            )
    return path


def test_separate_wallets_and_gross_basis_without_legacy_curve(tmp_path):
    path = ledger(tmp_path, "live-paper-v3")
    ledger(tmp_path, "live-paper", cash=9000, qty=0.002)
    before = path.read_bytes()
    v3, experiment = snapshot(tmp_path, 1800000000)
    assert v3["accounts"]["B"]["gross_equity"] == 9800
    assert experiment["accounts"]["B"]["gross_equity"] == 9600
    assert len(v3["gross_curve"]) == 1
    assert v3["gross_curve"][0]["b"] == 9800
    assert experiment["gross_curve"] == []
    assert experiment["recent_fills"][0]["details"]["action"] == "BUY"
    assert v3["ledger"] != experiment["ledger"]
    assert v3["promotion_eligible"] is False
    assert v3["full_execution_costs_verified"] is False
    assert path.read_bytes() == before


def test_missing_database_is_not_created_or_displayed_as_zero(tmp_path):
    results = snapshot(tmp_path)
    assert all(row["state"] == "unavailable" for row in results)
    assert all("accounts" not in row for row in results)
    assert not (tmp_path / "research").exists()


def test_stale_tick_is_explicit(tmp_path):
    ledger(tmp_path, "live-paper-v3")
    assert snapshot(tmp_path, 1800000181)[0]["quote_stale"] is True


def test_24h_activity_counts_distinguish_quotes_decisions_fills_and_trials(tmp_path):
    path = ledger(tmp_path, "live-paper-v3")
    now = 1800000000
    with sqlite3.connect(path) as db:
        db.executescript(
            "CREATE TABLE decisions(id INTEGER PRIMARY KEY,ts REAL,branch TEXT,action TEXT,reason TEXT);"
            "CREATE TABLE tests(id INTEGER PRIMARY KEY,ts TEXT,window INTEGER,threshold REAL,train_pct REAL,test_pct REAL,trades INTEGER);"
            "CREATE TABLE hypotheses(id INTEGER PRIMARY KEY,ts TEXT,correct INTEGER);"
        )
        ts = datetime.fromtimestamp(now - 3600, UTC).isoformat()
        db.execute("INSERT INTO decisions VALUES(1,?,'A','HOLD','test')", (now - 3600,))
        db.execute("INSERT INTO fills VALUES(1,?,'B','BUY',0.001,300000,1)", (ts,))
        db.execute("INSERT INTO tests VALUES(1,?,12,0.02,0.0,0.0,0)", (ts,))
        db.execute("INSERT INTO hypotheses VALUES(1,?,1)", (ts,))
    activity = snapshot(tmp_path, now)[0]["activity_24h"]
    assert activity["counts"] == {
        "quotes": 2,
        "decisions": 1,
        "fills": 1,
        "tests": 1,
        "scored_hypotheses": 1,
    }
    assert activity["lifetime_counts"]["fills"] == 1
    assert sum(row.get("decisions", 0) for row in activity["hourly"]) == 1
    assert "Aktywność zapisów — ostatnie 24 h" in PAGE
    assert "Wykonania paper" in PAGE


def test_failure_in_one_ledger_does_not_hide_other(tmp_path):
    ledger(tmp_path, "live-paper-v3", cash=-1)
    ledger(tmp_path, "live-paper")
    results = snapshot(tmp_path, 1800000000)
    assert results[0]["state"] == "unavailable"
    assert results[1]["state"] == "recorded"


@pytest.mark.parametrize("identity", ["paper-labs-panel", "trade-explorer", "paper"])
def test_wallet_and_canonical_sections_are_visible_under_trading(identity):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    program = (
        SCRIPT
        + "\nif(dashboardSectionGroup({id:"
        + repr(identity)
        + ",querySelector:()=>null})!=='trades')throw Error('Wallet hidden outside Trading');"
    )
    result = subprocess.run(
        [node, "-"], input=program, encoding="utf-8", capture_output=True, timeout=5
    )
    assert result.returncode == 0, result.stderr
