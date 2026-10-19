"""Read-only, separate provenance for the two legacy experimental paper ledgers."""

import json
import math
import sqlite3
import time
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path


def finite(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite legacy ledger value")
    return result


def activity_window_24h(db: sqlite3.Connection, now: float) -> dict:
    """Count recorded experiment activity without treating HOLDs as fills or profit."""
    since = now - 24 * 60 * 60
    since_iso = datetime.fromtimestamp(since, UTC).isoformat()
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    specs = {
        "quotes": ("ticks", "ts", "text"),
        "decisions": ("decisions", "ts", "unix"),
        "fills": ("fills", "ts", "text"),
        "tests": ("tests", "ts", "text"),
        "scored_hypotheses": ("hypotheses", "ts", "text"),
    }
    counts: dict[str, int | None] = {}
    lifetime: dict[str, int | None] = {}
    hourly: dict[str, dict[str, int | None]] = {}
    for label, (table, column, kind) in specs.items():
        if table not in tables:
            counts[label] = None
            lifetime[label] = None
            continue
        scored_filter = " AND correct IS NOT NULL" if label == "scored_hypotheses" else ""
        predicate = f"{column} >= ?" if kind == "unix" else f"julianday({column}) >= julianday(?)"
        cutoff: float | str = since if kind == "unix" else since_iso
        counts[label] = int(
            db.execute(
                f"SELECT count(1) FROM {table} WHERE {predicate}{scored_filter}", (cutoff,)
            ).fetchone()[0]
        )
        lifetime[label] = int(
            db.execute(f"SELECT count(1) FROM {table} WHERE 1=1{scored_filter}").fetchone()[0]
        )
        hour_expr = (
            f"strftime('%Y-%m-%dT%H', {column}, 'unixepoch')"
            if kind == "unix"
            else f"strftime('%Y-%m-%dT%H', {column})"
        )
        for row in db.execute(
            f"SELECT {hour_expr} AS hour,count(1) AS n FROM {table} "
            f"WHERE {predicate}{scored_filter} GROUP BY hour",
            (cutoff,),
        ):
            if row["hour"]:
                hourly.setdefault(row["hour"], {})[label] = int(row["n"])
    start = datetime.fromtimestamp(since, UTC).replace(minute=0, second=0, microsecond=0)
    hours = []
    for offset in range(25):
        hour = (start + timedelta(hours=offset)).strftime("%Y-%m-%dT%H")
        if hour > datetime.fromtimestamp(now, UTC).strftime("%Y-%m-%dT%H"):
            break
        row = {"hour": hour}
        for label, (table, _, _) in specs.items():
            if table in tables:
                row[label] = hourly.get(hour, {}).get(label, 0)
        hours.append(row)
    return {
        "window_hours": 24,
        "since": since_iso,
        "until": datetime.fromtimestamp(now, UTC).isoformat(),
        "counts": counts,
        "lifetime_counts": lifetime,
        "hourly": hours,
        "scope": "Recorded rows only. HOLDs are decisions, not fills; fills are not net income.",
    }


def read_lab(root: Path, name: str, now: float) -> dict:
    relative = f"research/experiments/{name}.sqlite3"
    path = root / relative
    result = {
        "id": name,
        "ledger": relative,
        "currency": "PLN",
        "mode": "isolated experimental paper",
        "promotion_eligible": False,
        "full_execution_costs_verified": False,
        "valuation": "Gross cash + units × last stored bid; reference FX, execution FX unverified",
        "scope": "Separate legacy ledger; no capital aggregation with canonical portfolios. No independent edge or trading-model training proof. Initial capital provenance is not stored in this database.",
    }
    if not path.is_file():
        return {**result, "state": "unavailable", "error": "Ledger missing"}
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Experimental ledger outside installation")
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1)) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        tick = db.execute("SELECT ts,bid,ask FROM ticks ORDER BY id DESC LIMIT 1").fetchone()
        accounts, fills, curve = {}, [], []
        if name == "live-paper-v3":
            rows = db.execute("SELECT branch,cash,qty FROM accounts LIMIT 3").fetchall()
            fill_rows = db.execute(
                "SELECT ts,branch,side,qty,price,fee FROM fills ORDER BY id DESC LIMIT 30"
            )
            fills = [dict(row) for row in fill_rows]
            columns = {row[1] for row in db.execute("PRAGMA table_info(ticks)")}
            if "valuation_basis" in columns:
                curve = [
                    dict(row)
                    for row in db.execute(
                        "SELECT ts,a,b FROM ticks WHERE valuation_basis='gross-mark-to-bid-v2' ORDER BY id DESC LIMIT 250"
                    )
                ][::-1]
            result["curve_valuation"] = (
                "gross-mark-to-bid-v2 only; earlier liquidation estimates excluded"
            )
            result["activity_24h"] = activity_window_24h(db, now)
        else:
            rows = db.execute("SELECT branch,cash,units AS qty FROM portfolio LIMIT 3").fetchall()
            for row in db.execute(
                "SELECT ts,branch,details FROM events WHERE kind='fill' ORDER BY id DESC LIMIT 30"
            ):
                if len(row["details"]) > 2**21:
                    raise ValueError("Experimental fill exceeds read budget")
                fills.append(
                    {
                        "ts": row["ts"],
                        "branch": row["branch"],
                        "details": json.loads(row["details"]),
                    }
                )
            # Legacy engine stores liquidation estimates, so do not join these to gross MTM.
            result["curve_valuation"] = (
                "Legacy liquidation curve omitted; no verified gross MTM history"
            )
        result["execution_audit"] = (
            "Recorded legacy fills; full event-chain and cash/position reconciliation pending"
        )
        if len(rows) != 2 or {row["branch"].upper() for row in rows} != {"A", "B"}:
            raise ValueError("Expected exactly two experimental accounts")
        bid = finite(tick["bid"]) if tick else None
        if bid is not None and (bid <= 0 or finite(tick["ask"]) < bid):
            raise ValueError("Invalid experimental bid/ask")
        for row in rows:
            cash, qty = finite(row["cash"]), finite(row["qty"])
            if cash < 0 or qty < 0:
                raise ValueError("Invalid experimental cash or quantity")
            accounts[row["branch"].upper()] = {
                "cash": cash,
                "units": qty,
                "gross_equity": cash + qty * bid if bid is not None else cash if qty == 0 else None,
            }
        age = None
        if tick:
            observed = datetime.fromisoformat(tick["ts"].replace("Z", "+00:00"))
            if observed.tzinfo is None:
                raise ValueError("Experimental tick timezone missing")
            age = now - observed.timestamp()
        result.update(
            state="recorded",
            accounts=accounts,
            recent_fills=fills,
            gross_curve=curve,
            quote=dict(tick) if tick else None,
            tick_age_seconds=age,
            quote_stale=age is None or not 0 <= age <= 180,
        )
        return result


def snapshot(root: Path, now: float | None = None) -> list[dict]:
    now = time.time() if now is None else now
    results = []
    for name in ("live-paper-v3", "live-paper"):
        try:
            results.append(read_lab(root, name, now))
        except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as error:
            results.append({"id": name, "state": "unavailable", "error": str(error)[:240]})
    return results
