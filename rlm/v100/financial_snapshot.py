"""Bounded financial display reads, independent of the full audit process."""

import json
import sqlite3
from pathlib import Path


def live_ledger(root: Path) -> dict:
    """Check one consistent state and its last event, without replaying all history."""
    from rlm.v100.paper import PaperBook, number, sha, timestamp

    book = PaperBook(root, read_only=True)
    try:
        book.db.execute("BEGIN")
        row = book.db.execute(
            "SELECT payload FROM state WHERE id=1 AND length(payload)<=2097152"
        ).fetchone()
        event = book.db.execute("SELECT * FROM events ORDER BY sequence DESC LIMIT 1").fetchone()
        if row is None or event is None or len(event["payload"]) > 2097152:
            raise ValueError("Ledger state missing or oversized")
        state, payload = json.loads(row["payload"]), json.loads(event["payload"])
        body = {key: event[key] for key in ("time", "kind", "branch", "previous_hash")}
        body["payload"] = payload
        if payload.get("state_sha256") != sha(state) or sha(body) != event["event_hash"]:
            raise ValueError("Ledger tail or state checksum differs")
        branches = {}
        for branch, portfolio in state["branches"].items():
            equity = book.equity(state, branch)
            branches[branch] = {
                "equity": str(equity),
                "cash": portfolio["cash"],
                "positions": portfolio["positions"],
                "net_pnl_before_personal_tax": str(equity - number(state["initial_capital"])),
                "modeled_costs": portfolio["costs"],
                "resolved_fills": None,
            }
        now = book.clock()
        active_fees = {i["fee_profile"] for i in state["instruments"].values()}
        blockers = [
            "expired fees: " + key
            for key in active_fees
            if timestamp(state["fee_profiles"][key]["valid_until"]) <= now
        ]
        for symbol in state["instruments"]:
            quote = state["quotes"].get(symbol)
            if not quote:
                blockers.append("missing quote: " + symbol)
            elif (now - timestamp(quote["available_at"])).total_seconds() > state["risk"][
                "quote_age_seconds"
            ]:
                blockers.append("stale quote: " + symbol)
        if not state["instruments"]:
            blockers.append("No registered instruments")
        return {
            "branches": branches,
            "currency": state["currency"],
            "executed_fills": None,
            "updated_at": event["time"],
            "collected_at": now.isoformat(),
            "ledger_sequence": event["sequence"],
            "blockers": blockers,
            "scope": "Current recorded state and last-event checksums checked; full historical audit and fill counts are separate. Equity uses recorded quotes, which may be stale.",
            "real_money_ready": False,
        }
    finally:
        book.close()


def read_object(path: Path) -> dict:
    if path.is_symlink() or path.stat().st_size > 2 * 2**20:
        raise ValueError("Financial snapshot is linked or exceeds 2 MiB")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Financial snapshot must be an object")
    return value


def supplement(root: Path, report: dict, errors: list) -> dict:
    result = dict(report)
    ledger = root / "research/paper/ledger.sqlite3"
    if ledger.exists():
        try:
            result["live_paper"] = live_ledger(root)
            result["paper_blockers"] = result["live_paper"]["blockers"]
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
            errors.append("Live paper state unavailable: " + str(error)[:200])
    if not result.get("recent_exploratory_backtests"):
        paths = sorted(
            (root / "research/backtests").glob("run-*/report.json"),
            key=lambda p: p.stat().st_mtime_ns,
            reverse=True,
        )[:4]
        rows = []
        for path in paths:
            try:
                value = read_object(path)
                if not isinstance(value.get("development_test"), dict):
                    raise ValueError("Backtest has no measured development result")
                row = {
                    k: value[k]
                    for k in (
                        "parameters",
                        "sources",
                        "scope",
                        "limitations",
                        "fetched_at",
                        "selected_lookback_on_training_only",
                        "buy_hold_baseline",
                        "double_cost_stress",
                        "data_window",
                    )
                    if k in value
                }
                for key in ("buy_hold_baseline", "double_cost_stress"):
                    if isinstance(row.get(key), dict):
                        row[key] = {k: v for k, v in row[key].items() if k != "trace"}
                row["development_test"] = {
                    k: v for k, v in value["development_test"].items() if k != "trace"
                }
                row["report"] = str(path)
                row["display_scope"] = (
                    "Stored historical result; not a new audit or forward paper return"
                )
                rows.append(row)
            except (OSError, ValueError) as error:
                errors.append(f"Backtest {path.name}: {str(error)[:200]}")
        result["recent_exploratory_backtests"] = rows
    index = root / "research/paper/latest-report.json"
    if not result.get("paper") and index.exists():
        try:
            saved = read_object(index)
            directory = Path(saved["directory"]).resolve()
            base = (root / "research/paper/reports").resolve()
            if directory.parent != base:
                raise ValueError("Paper report is outside the owned report folder")
            value = read_object(directory / "report.json")
            result["paper"] = {
                "currency": value["currency"],
                "branches": value["branches"],
                "executed_fills": len(value["trades"]),
                "updated_at": value["time"],
                "period": value["period"],
                "source": str(directory / "report.json"),
                "scope": "Saved paper report with its original time and period; not live or a fresh audit",
                "real_money_ready": False,
            }
            saved_blockers = [
                *["stale quote: " + s for s in value.get("stale_symbols", [])],
                *["expired fees: " + s for s in value.get("expired_fee_profiles", [])],
            ]
            if "live_paper" not in result:
                result["paper_blockers"] = saved_blockers
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append("Saved paper report unavailable: " + str(error)[:200])
    if not result.get("paper"):
        result["paper_display_status"] = "No stored paper report available; fills and P&L unknown"
    return result
