"""Bounded, read-only chart data with execution receipts and explicit simulation scope."""

import json
import sqlite3
import time
from pathlib import Path

from rlm.v100.financial_snapshot import read_object


def paper_series(root: Path) -> list[dict]:
    from rlm.v100.paper import PaperBook, sha

    if not (root / "research/paper/ledger.sqlite3").exists():
        return []
    book = PaperBook(root, read_only=True)
    try:
        book.db.execute("BEGIN")
        state_row = book.db.execute("SELECT payload FROM state WHERE id=1").fetchone()
        if state_row is None or len(state_row["payload"]) > 2**21:
            raise ValueError("Missing or oversized paper state")
        state = json.loads(state_row["payload"])
        rows = book.db.execute("SELECT * FROM events ORDER BY sequence DESC LIMIT 600").fetchall()
        if not rows:
            raise ValueError("Paper chart has no immutable events")
        points, trades, previous, byte_count = [], [], None, 0
        for row in reversed(rows):
            byte_count += len(row["payload"])
            if byte_count > 4 * 2**20:
                raise ValueError("Paper chart tail exceeds 4 MiB")
            payload = json.loads(row["payload"])
            body = {key: row[key] for key in ("time", "kind", "branch", "previous_hash")}
            body["payload"] = payload
            if sha(body) != row["event_hash"] or (
                previous is not None and previous != row["previous_hash"]
            ):
                raise ValueError("Paper chart event checksum or tail chain differs")
            previous = row["event_hash"]
            for branch, equity in payload.get("equity", {}).items():
                points.append({"time": row["time"], "branch": branch, "equity": float(equity)})
            for index, trade in enumerate(payload.get("trades", [])):
                trades.append(
                    {
                        **trade,
                        "time": row["time"],
                        "sequence": row["sequence"],
                        "receipt": f"{row['sequence']}:{index}:{row['event_hash']}",
                    }
                )
        if rows and json.loads(rows[0]["payload"]).get("state_sha256") != sha(state):
            raise ValueError("Paper state checksum differs from chart tail")
        result = []
        for branch, portfolio in state["branches"].items():
            branch_points = [point for point in points if point["branch"] == branch]
            # Initial capital is the actual ledger value; never fabricate a fill.
            equity = float(book.equity(state, branch))
            if not branch_points or branch_points[-1]["equity"] != equity:
                branch_points.append({"time": rows[0]["time"], "branch": branch, "equity": equity})
            result.append(
                {
                    "id": "paper-" + branch,
                    "label": "Forward paper " + branch,
                    "currency": state["currency"],
                    "initial_capital": float(state["initial_capital"]),
                    "points": branch_points,
                    "executions": [t for t in trades if t.get("branch") == branch],
                    "net_pnl": equity - float(state["initial_capital"]),
                    "total_costs": float(portfolio["costs"]),
                    "scope": "Recorded forward paper ledger. Last 600 events, hash-checked tail; full audit is separate. Quotes may be stale. Not real money.",
                }
            )
        return result
    finally:
        book.close()


def historical_series(path: Path) -> dict:
    from rlm.v100.backtest_learning import verified
    from rlm.v100.backtesting import bars_from_source, simulate
    from rlm.v100.paper import timestamp

    verified(path)
    report = read_object(path)
    bars = bars_from_source(
        report["sources"][0]["url"],
        (path.parent / "source-0.json").read_text(),
        timestamp(report["fetched_at"]),
    )
    events = None
    if len(report["sources"]) > 1:
        events = [
            timestamp(row["available_at"]).timestamp()
            for row in json.loads((path.parent / "source-1.json").read_text())
        ]
    start = report["split_index"]
    args = report["parameters"]
    replay = simulate(
        bars,
        start,
        len(bars),
        args["rule"],
        report["selected_lookback_on_training_only"],
        args["fee_bps"],
        args["slippage_bps"],
        events,
        capture_executions=True,
    )
    return {
        "id": path.parent.name,
        "label": "Historical " + args["rule"] + " / " + path.parent.name,
        "currency": "normalized capital units",
        "initial_capital": 1,
        "points": [{"time": bars[start]["open_time"], "equity": 1.0, "price": bars[start]["open"]}]
        + [
            {
                "time": bars[start + i]["available_at"],
                "equity": point["marked_equity"],
                "price": bars[start + i]["close"],
            }
            for i, point in enumerate(replay["trace"])
        ],
        "executions": [
            {**row, "receipt": path.parent.name + ":" + str(i)}
            for i, row in enumerate(replay["executions"])
        ],
        "net_pnl": replay["net_return"],
        "total_costs": replay["total_costs"],
        "max_drawdown": replay["max_sampled_drawdown"],
        "scope": "Archived historical development sample, independently replayed. Assumed fees/slippage; not forward paper or actual income.",
        "report": str(path),
    }


def snapshot(root: Path) -> dict:
    series, errors = [], []
    try:
        series.extend(paper_series(root))
    except (ValueError, OSError, KeyError, TypeError, sqlite3.Error) as error:
        errors.append("Paper chart: " + str(error)[:200])
    paths = sorted(
        (root / "research/backtests").glob("run-*/report.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )[:4]
    for path in paths:
        try:
            series.append(historical_series(path))
        except (ValueError, OSError, KeyError, TypeError) as error:
            errors.append(path.parent.name + ": " + str(error)[:200])
    return {
        "updated": time.time(),
        "series": series,
        "errors": errors,
        "scope": "Recorded executions only. Historical and forward paper series are separate.",
    }
