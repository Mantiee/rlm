"""Bounded financial display reads, independent of the full audit process."""

import json
import math
import sqlite3
import time
from pathlib import Path

LIVE_EXECUTION_POLICY = {
    "capital_limit_pln": 100,
    "daily_loss_stop_pln": 20,
    "daily_loss_stop_percent": 20,
    "max_total_loss_pln": 25,
    "max_total_loss_percent": 25,
    "leverage_enabled": False,
    "margin_enabled": False,
    "short_selling_enabled": False,
    "borrowing_enabled": False,
    "paid_services_enabled": False,
    "real_orders_enabled": False,
    "scope": "Configured operator ceiling. No broker order connector is enabled; this is not proof of account-side enforcement.",
}

BROKER_CONNECTION_READINESS = {
    "state": "not_connected",
    "account_reading": "not_configured",
    "order_submission": "not_implemented",
    "credential_storage": "OS secret store or injected environment only; never dashboard, source, ledger, or logs",
    "required_operator_inputs": [
        "broker/provider",
        "country and account type",
        "cash-only account with margin disabled",
    ],
    "first_connection_scope": "read-only account identity, currency, available cash, fee tier, and venue minimums",
    "activation_requirements": [
        "independent executable-cost verification",
        "account-side loss controls",
        "explicit separate operator enablement",
    ],
    "scope": "Readiness contract only; no credentials, account connection, or order endpoint is present.",
}

PAPER_RECEIPT_STALE_AFTER_SECONDS = 300


def _receipt_age_seconds(observed_at, now: float) -> float | None:
    from rlm.v100.paper import timestamp

    try:
        if type(observed_at) in (int, float) and math.isfinite(observed_at):
            stamp = float(observed_at)
        elif isinstance(observed_at, str):
            stamp = timestamp(observed_at).timestamp()
        else:
            return None
    except (ValueError, TypeError, OverflowError):
        return None
    age = now - stamp
    return age if math.isfinite(age) and age >= 0 else None


def model_decision_evidence(attempts: list[dict], now: float | None = None) -> dict:
    """Report only a completed model-decision receipt, never feed/hold refresh activity."""
    now = time.time() if now is None else now
    completed = [
        attempt
        for attempt in attempts
        if isinstance(attempt, dict)
        and attempt.get("status") == "decision-round-completed"
        and isinstance(attempt.get("model_version"), str)
        and attempt["model_version"]
        and type(attempt.get("decisions_recorded")) is int
        and attempt["decisions_recorded"] > 0
    ]
    latest = max(
        completed,
        key=lambda attempt: (
            attempt.get("sequence") if type(attempt.get("sequence")) is int else -1
        ),
        default=None,
    )
    if latest is None:
        return {
            "state": "unknown",
            "confirmed_at": None,
            "age_seconds": None,
            "model_version": None,
            "decisions_recorded": None,
            "scope": "No completed model-decision receipt; observations and HOLD freshness receipts are not decisions.",
        }
    age = _receipt_age_seconds(latest.get("time"), now)
    return {
        "state": (
            "unknown"
            if age is None
            else "fresh"
            if age <= PAPER_RECEIPT_STALE_AFTER_SECONDS
            else "stale"
        ),
        "confirmed_at": latest.get("time") if age is not None else None,
        "age_seconds": round(age, 3) if age is not None else None,
        "model_version": latest["model_version"],
        "decisions_recorded": latest["decisions_recorded"],
        "scope": "Completed model-decision receipt; HOLD is a decision, not a fill, profit, training update, or real order.",
    }


def paper_fill_evidence(report: dict | None, now: float | None = None) -> dict:
    """Age the last complete saved fill audit; a live quote or ledger tail is not a fill count."""
    now = time.time() if now is None else now
    count = report.get("executed_fills") if isinstance(report, dict) else None
    age = _receipt_age_seconds(report.get("updated_at"), now) if isinstance(report, dict) else None
    if type(count) is not int or count < 0 or age is None:
        state, count = "unknown", None
    else:
        state = "fresh" if age <= PAPER_RECEIPT_STALE_AFTER_SECONDS else "stale"
    last_fill_at = report.get("last_fill_at") if isinstance(report, dict) else None
    last_fill_age = _receipt_age_seconds(last_fill_at, now) if last_fill_at else None
    if count == 0 and state in ("fresh", "stale"):
        last_fill_state = "none-in-audited-period"
        last_fill_at = None
    elif last_fill_age is None:
        last_fill_state = "unknown"
        last_fill_at = None
    else:
        last_fill_state = "fresh" if last_fill_age <= PAPER_RECEIPT_STALE_AFTER_SECONDS else "stale"
    return {
        "state": state,
        "fill_count": count,
        "audit_at": report.get("updated_at")
        if age is not None and isinstance(report, dict)
        else None,
        "audit_age_seconds": round(age, 3) if age is not None else None,
        "last_fill_at": last_fill_at,
        "last_fill_state": last_fill_state,
        "last_fill_age_seconds": round(last_fill_age, 3) if last_fill_age is not None else None,
        "scope": "Full saved paper report only; stale reports are historical and do not prove the current fill count.",
    }


def paper_activity_stream(events: list[dict], pending_orders: int | None) -> dict:
    """Summarize a bounded, tail-linked paper ledger window without inferring actions."""
    entries = []

    def proposal_receipt(proposal: dict, payload: dict, source: str | None, status: str) -> dict:
        """Expose the model's own bounded proposal fields for an explanatory UI."""
        details = {
            key: proposal.get(key)
            for key in (
                "action",
                "symbol",
                "side",
                "budget",
                "leverage",
                "rationale",
                "model_action_blocked",
            )
            if proposal.get(key) is not None
        }
        if isinstance(details.get("rationale"), str):
            details["rationale"] = details["rationale"][:600]
        if isinstance(source, str) and source:
            details["source"] = source[:80]
        details["status"] = status[:80]
        inputs = payload.get("decision_inputs") or {}
        quote = inputs.get("quote") if isinstance(inputs, dict) else None
        if isinstance(quote, dict):
            details["quote"] = {
                key: quote[key]
                for key in ("bid", "ask", "observed_at", "available_at")
                if quote.get(key) is not None
            }
        return details

    counts = {
        "observations": 0,
        "model_decisions": 0,
        "automatic_freshness_holds": 0,
        "decision_round_attempts": 0,
        "paper_execution_attempts": 0,
        "confirmed_fills": 0,
    }
    fill_actions = {"buy", "short", "close", "liquidation", "bet", "settle"}
    attempt_actions = {"open", "close", "cancel"}
    for event in events:
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        kind = event.get("kind")
        sequence, at, branch = event.get("sequence"), event.get("time"), event.get("branch")
        if kind == "observation":
            observation = payload.get("observation") or {}
            if not isinstance(observation, dict):
                observation = {}
            counts["observations"] += 1
            entries.append(
                {
                    "category": "observation",
                    "time": at,
                    "sequence": sequence,
                    "branch": branch,
                    "label": f"{observation.get('kind', 'unknown')} observation",
                    "detail": observation.get("symbol")
                    or observation.get("category")
                    or "Recorded source input",
                }
            )
            for trade in payload.get("trades", []):
                if not isinstance(trade, dict) or trade.get("action") not in fill_actions:
                    continue
                counts["confirmed_fills"] += 1
                entries.append(
                    {
                        "category": "confirmed_fill",
                        "time": at,
                        "sequence": sequence,
                        "branch": trade.get("branch"),
                        "label": "Confirmed paper fill",
                        "detail": " ".join(
                            str(value)
                            for value in (
                                trade.get("action"),
                                trade.get("symbol"),
                                trade.get("quantity"),
                            )
                            if value is not None
                        ),
                    }
                )
        elif kind == "decision":
            proposal = payload.get("proposal") or {}
            if not isinstance(proposal, dict):
                continue
            action = proposal.get("action")
            source = proposal.get("proposal_source") or proposal.get("receipt_source")
            if source == "host-quote-freshness-guard":
                counts["automatic_freshness_holds"] += 1
                category, label = "automatic_hold", "Automatic quote-freshness HOLD"
            else:
                counts["model_decisions"] += source == "model-proposal"
                category, label = "decision", "Recorded paper decision"
            entries.append(
                {
                    "category": category,
                    "time": at,
                    "sequence": sequence,
                    "branch": branch,
                    "label": label,
                    "detail": " ".join(
                        str(value)
                        for value in (action, proposal.get("symbol"), source)
                        if value is not None
                    ),
                    "proposal": proposal_receipt(proposal, payload, source, "recorded"),
                }
            )
            if action in attempt_actions:
                counts["paper_execution_attempts"] += 1
                entries.append(
                    {
                        "category": "execution_attempt",
                        "time": at,
                        "sequence": sequence,
                        "branch": branch,
                        "label": "Paper order attempt recorded",
                        "detail": f"{action} {proposal.get('symbol', 'symbol unknown')} · fill not implied",
                    }
                )
        elif kind == "research":
            status = payload.get("status")
            rejected = payload.get("proposal") or {}
            if not isinstance(rejected, dict):
                rejected = {}
            if status == "proposal-rejected" and rejected.get("action") in attempt_actions:
                counts["paper_execution_attempts"] += 1
                entries.append(
                    {
                        "category": "execution_attempt",
                        "time": at,
                        "sequence": sequence,
                        "branch": branch,
                        "label": "Paper order attempt blocked",
                        "detail": " · ".join(
                            str(value)
                            for value in (
                                rejected.get("action"),
                                rejected.get("symbol"),
                                payload.get("error"),
                            )
                            if value
                        ),
                        "proposal": proposal_receipt(
                            rejected, payload, "model-proposal", "blocked"
                        ),
                    }
                )
            if isinstance(status, str) and (
                status.startswith("decision-round-")
                or status in ("decision-blocked", "proposal-rejected", "decision-unavailable")
            ):
                counts["decision_round_attempts"] += 1
                entries.append(
                    {
                        "category": "decision_attempt",
                        "time": at,
                        "sequence": sequence,
                        "branch": branch,
                        "label": "Model decision round receipt",
                        "detail": " · ".join(
                            str(value)
                            for value in (status, payload.get("reason"), payload.get("error"))
                            if value
                        ),
                    }
                )
    entries.sort(key=lambda entry: (str(entry.get("time") or ""), entry.get("sequence") or 0))
    return {
        "state": "verified-tail" if events else "unknown",
        "window_events": len(events),
        "window_scope": "Latest 100 ledger events; entries are checksum-linked to the verified current ledger tail, not lifetime totals.",
        "counts": {**counts, "pending_orders": pending_orders},
        "entries": entries[-200:],
    }


def live_ledger(root: Path) -> dict:
    """Check one consistent state and its last event, without replaying all history."""
    from rlm.v100.paper import PaperBook, execution_cost_blocker, number, sha, timestamp

    book = PaperBook(root, read_only=True)
    try:
        book.db.execute("BEGIN")
        row = book.db.execute(
            "SELECT payload FROM state WHERE id=1 AND length(payload)<=2097152"
        ).fetchone()
        event = book.db.execute("SELECT * FROM events ORDER BY sequence DESC LIMIT 1").fetchone()
        activity_rows = book.db.execute(
            "SELECT sequence,time,kind,branch,substr(payload,1,262145) AS payload,previous_hash,event_hash "
            "FROM events ORDER BY sequence DESC LIMIT 100"
        ).fetchall()
        decision_rows = book.db.execute(
            "SELECT sequence,time,payload FROM events WHERE kind='research' "
            "AND branch='controller' ORDER BY sequence DESC LIMIT 24"
        ).fetchall()
        if row is None or event is None or len(event["payload"]) > 2097152:
            raise ValueError("Ledger state missing or oversized")
        state, payload = json.loads(row["payload"]), json.loads(event["payload"])
        body = {key: event[key] for key in ("time", "kind", "branch", "previous_hash")}
        body["payload"] = payload
        if payload.get("state_sha256") != sha(state) or sha(body) != event["event_hash"]:
            raise ValueError("Ledger tail or state checksum differs")
        activity_stream = {"state": "unknown", "entries": [], "counts": None}
        try:
            parsed_events = []
            total_payload_bytes = 0
            previous_hash = None
            previous_sequence = None
            ordered_rows = list(reversed(activity_rows))
            for activity_row in ordered_rows:
                total_payload_bytes += len(activity_row["payload"])
                if len(activity_row["payload"]) > 256 * 2**10 or total_payload_bytes > 2 * 2**20:
                    raise ValueError("Recent paper event tail exceeds the display budget")
                event_payload = json.loads(activity_row["payload"])
                event_body = {
                    key: activity_row[key] for key in ("time", "kind", "branch", "previous_hash")
                }
                event_body["payload"] = event_payload
                if (
                    sha(event_body) != activity_row["event_hash"]
                    or (
                        previous_hash is not None and previous_hash != activity_row["previous_hash"]
                    )
                    or (
                        previous_sequence is not None
                        and activity_row["sequence"] != previous_sequence + 1
                    )
                ):
                    raise ValueError("Recent paper event tail checksum differs")
                previous_hash = activity_row["event_hash"]
                previous_sequence = activity_row["sequence"]
                parsed_events.append(
                    {
                        "sequence": activity_row["sequence"],
                        "time": activity_row["time"],
                        "kind": activity_row["kind"],
                        "branch": activity_row["branch"],
                        "payload": event_payload,
                    }
                )
            if not ordered_rows or ordered_rows[-1]["sequence"] != event["sequence"]:
                raise ValueError("Recent paper events do not end at the verified ledger tail")
            activity_stream = paper_activity_stream(parsed_events, len(state.get("orders", [])))
        except (ValueError, TypeError, KeyError) as error:
            activity_stream["detail"] = str(error)[:200]
        decision_attempts = []
        for row in decision_rows:
            receipt = json.loads(row["payload"])
            status = receipt.get("status", "")
            if status.startswith("decision-round-"):
                decision_attempts.append(
                    {
                        "sequence": row["sequence"],
                        "time": row["time"],
                        "status": status,
                        "cycle": receipt.get("cycle"),
                        "model_version": receipt.get("model_version"),
                        "decisions_recorded": receipt.get("decisions_recorded"),
                        "branch_actions": receipt.get("branch_actions", []),
                        "error": receipt.get("error"),
                        "detail": receipt.get("detail"),
                        "scope": receipt.get("scope"),
                    }
                )
                if len(decision_attempts) == 8:
                    break
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
            profile = state["fee_profiles"][state["instruments"][symbol]["fee_profile"]]
            cost_blocker = execution_cost_blocker(profile, quote or {})
            if cost_blocker:
                blockers.append(symbol + ": " + cost_blocker)
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
            "last_event": {
                "time": event["time"],
                "kind": event["kind"],
                "branch": event["branch"],
                "sequence": event["sequence"],
                "verified_tail": True,
            },
            "initial_capital": state["initial_capital"],
            "registered_instruments": len(state["instruments"]),
            "quotes_available": len(state["quotes"]),
            "decision_attempts": decision_attempts,
            "last_decision_attempt": decision_attempts[0] if decision_attempts else None,
            "activity_stream": activity_stream,
            "model_decision_evidence": model_decision_evidence(
                decision_attempts, now=now.timestamp()
            ),
            "blockers": blockers,
            "scope": "Current recorded state and last-event checksums checked; full historical audit and fill counts are separate. Equity uses recorded quotes, which may be stale.",
            "real_money_ready": False,
        }
    finally:
        book.close()


def read_object(path: Path, maximum: int = 2 * 2**20) -> dict:
    if path.is_symlink() or path.stat().st_size > maximum:
        raise ValueError("Financial snapshot is linked or exceeds its bounded byte budget")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Financial snapshot must be an object")
    return value


def supplement(root: Path, report: dict, errors: list) -> dict:
    result = dict(report)
    result["live_execution_policy"] = dict(LIVE_EXECUTION_POLICY)
    result["broker_connection_readiness"] = dict(BROKER_CONNECTION_READINESS)
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
                        "source_quality",
                        "metric_definitions",
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
                from rlm.v100.market_research import triage

                row["triage"] = triage(value)
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
            value = read_object(directory / "report.json", 32 * 2**20)
            trades = value.get("trades", [])
            valid_fill_times = []
            if isinstance(trades, list):
                for trade in trades:
                    if not isinstance(trade, dict) or not isinstance(trade.get("time"), str):
                        continue
                    fill_time = trade["time"]
                    fill_stamp = _receipt_age_seconds(fill_time, time.time())
                    if fill_stamp is not None:
                        valid_fill_times.append((time.time() - fill_stamp, fill_time))
            result["paper"] = {
                "currency": value["currency"],
                "branches": value["branches"],
                "executed_fills": len(value["trades"]),
                "updated_at": value["time"],
                "last_fill_at": min(valid_fill_times, default=(None, None))[1],
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
    result["paper_fill_evidence"] = paper_fill_evidence(result.get("paper"))
    return result
