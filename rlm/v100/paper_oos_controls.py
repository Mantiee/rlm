"""
OOS control ledger receipts for paper portfolio comparison.

Creates cost_assumptions_sha256 and immutable cash/buy-and-hold control
receipts from the same quote window as a candidate cohort.

All outputs are ASSUMED costs. No profit or edge claim.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from decimal import Decimal
from typing import Any


_SCHEMA = "synta-paper-oos-controls-v1"
_COST_SCHEMA = "synta-cost-assumptions-v1"


def cost_assumptions_sha256(fee: float, fx: float, slippage: float) -> str:
    """
    Return a deterministic SHA-256 digest binding the cost parameters.

    Parameters are normalised to 8 decimal places and serialised in a
    canonical JSON envelope so the hash is reproducible across runs.
    """
    for label, value in (("fee", fee), ("fx", fx), ("slippage", slippage)):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Cost assumption '{label}' must be a finite, non-negative number")
    if fee + fx >= 1 or slippage >= 1:
        raise ValueError("Cost assumptions exceed 100 % round-trip friction — invalid")
    payload = json.dumps(
        {
            "schema": _COST_SCHEMA,
            "fee_per_side": round(float(fee), 8),
            "fx_per_side": round(float(fx), 8),
            "slippage_per_side": round(float(slippage), 8),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _validate_quotes(quotes: list[dict]) -> list[dict]:
    """Light validation — same contract as paper_candidate_lab.validate_quotes."""
    if not isinstance(quotes, list) or len(quotes) < 4:
        raise ValueError("OOS controls require at least 4 quotes")
    required = {"bid", "ask", "bid_size", "ask_size", "timestamp"}
    for i, q in enumerate(quotes):
        if not isinstance(q, dict) or not required.issubset(q):
            raise ValueError(f"Quote {i} missing required keys")
        if not all(math.isfinite(float(q[k])) and float(q[k]) > 0 for k in required - {"timestamp"}):
            raise ValueError(f"Quote {i} has non-positive price/size")
    return quotes


def _epoch(ts: Any) -> float:
    if isinstance(ts, (int, float)) and math.isfinite(float(ts)):
        return float(ts)
    raise ValueError(f"Quote timestamp must be a finite number; got {ts!r}")


def cash_control_receipts(
    quotes: list[dict],
    *,
    portfolio_id: str,
    cohort_id: str,
    data_window_id: str,
    instrument_symbol: str,
    fee: float,
    fx: float,
    slippage: float,
    capital: float = 100.0,
) -> list[dict]:
    """
    Return append-only valuation receipts for a do-nothing cash control.

    The cash portfolio holds initial_capital_pln throughout, valued at each
    quote point. No fills are generated. All costs are ASSUMED.
    """
    rows = _validate_quotes(quotes)
    cost_sha = cost_assumptions_sha256(fee, fx, slippage)
    receipts: list[dict] = []
    for q in rows:
        at = _epoch(q["timestamp"])
        receipts.append({
            "schema": _SCHEMA,
            "portfolio_id": portfolio_id,
            "cohort_id": cohort_id,
            "data_window_id": data_window_id,
            "instrument_symbol": instrument_symbol,
            "cost_assumptions_sha256": cost_sha,
            "role": "cash",
            "kind": "valuation",
            "status": "valued",
            "occurred_at": at,
            "equity_pln": capital,
            "realized_pnl_pln": 0.0,
            "unrealized_pnl_pln": 0.0,
            "total_pnl_pln": 0.0,
            "mid_price": (float(q["bid"]) + float(q["ask"])) / 2,
            "cost_state": "ASSUMED",
            "note": "Cash control: initial capital held, zero fills, assumed costs",
        })
    return receipts


def buy_hold_control_receipts(
    quotes: list[dict],
    *,
    portfolio_id: str,
    cohort_id: str,
    data_window_id: str,
    instrument_symbol: str,
    fee: float,
    fx: float,
    slippage: float,
    capital: float = 100.0,
) -> list[dict]:
    """
    Return append-only receipts for a buy-and-hold control.

    One BUY fill is placed at the second quote (first next-quote opportunity),
    held to the end. Valued at each subsequent quote. All costs are ASSUMED.
    """
    rows = _validate_quotes(quotes)
    cost_sha = cost_assumptions_sha256(fee, fx, slippage)
    round_trip_fee = fee + fx

    # Entry at the second quote (index 1)
    entry = rows[1]
    entry_price = float(entry["ask"]) * (1.0 + slippage)
    quantity = min(
        capital / (entry_price * (1.0 + round_trip_fee)),
        float(entry["ask_size"]),
    )
    debit = quantity * entry_price * (1.0 + round_trip_fee)
    cash_after_entry = capital - debit

    receipts: list[dict] = []

    # BUY fill receipt at entry
    entry_at = _epoch(entry["timestamp"])
    receipts.append({
        "schema": _SCHEMA,
        "portfolio_id": portfolio_id,
        "cohort_id": cohort_id,
        "data_window_id": data_window_id,
        "instrument_symbol": instrument_symbol,
        "cost_assumptions_sha256": cost_sha,
        "role": "buy_hold",
        "kind": "fill",
        "action": "BUY",
        "status": "filled",
        "occurred_at": entry_at,
        "quantity": quantity,
        "price_pln": entry_price,
        "fee_pln": quantity * entry_price * round_trip_fee,
        "notional_pln": quantity * entry_price,
        "equity_pln": cash_after_entry + quantity * float(entry["bid"]) * (1.0 - slippage) * (1.0 - round_trip_fee),
        "realized_pnl_pln": 0.0,
        "unrealized_pnl_pln": cash_after_entry + quantity * float(entry["bid"]) * (1.0 - slippage) * (1.0 - round_trip_fee) - capital,
        "total_pnl_pln": cash_after_entry + quantity * float(entry["bid"]) * (1.0 - slippage) * (1.0 - round_trip_fee) - capital,
        "cost_state": "ASSUMED",
        "note": "Buy-and-hold control: single entry at first available quote",
    })

    # Valuation receipts at every subsequent quote
    for q in rows[1:]:
        at = _epoch(q["timestamp"])
        mark = float(q["bid"]) * (1.0 - slippage) * (1.0 - round_trip_fee)
        equity = cash_after_entry + quantity * mark
        unrealized = equity - capital
        receipts.append({
            "schema": _SCHEMA,
            "portfolio_id": portfolio_id,
            "cohort_id": cohort_id,
            "data_window_id": data_window_id,
            "instrument_symbol": instrument_symbol,
            "cost_assumptions_sha256": cost_sha,
            "role": "buy_hold",
            "kind": "valuation",
            "status": "valued",
            "occurred_at": at,
            "equity_pln": equity,
            "realized_pnl_pln": 0.0,
            "unrealized_pnl_pln": unrealized,
            "total_pnl_pln": unrealized,
            "mid_price": (float(q["bid"]) + float(q["ask"])) / 2,
            "cost_state": "ASSUMED",
            "note": "Buy-and-hold control: held position valuation",
        })

    return receipts


def oos_control_summary(
    quotes: list[dict],
    *,
    portfolio_id: str,
    cohort_id: str,
    data_window_id: str,
    instrument_symbol: str,
    fee: float,
    fx: float,
    slippage: float,
    capital: float = 100.0,
) -> dict:
    """
    Return a summary dict with cost_assumptions_sha256, receipt counts, and
    terminal equity for both cash and buy-and-hold controls.

    No profit claim. All costs ASSUMED. Used for cohort registration.
    """
    cost_sha = cost_assumptions_sha256(fee, fx, slippage)
    cash_rcpts = cash_control_receipts(
        quotes,
        portfolio_id=portfolio_id,
        cohort_id=cohort_id,
        data_window_id=data_window_id,
        instrument_symbol=instrument_symbol,
        fee=fee,
        fx=fx,
        slippage=slippage,
        capital=capital,
    )
    bh_rcpts = buy_hold_control_receipts(
        quotes,
        portfolio_id=portfolio_id,
        cohort_id=cohort_id,
        data_window_id=data_window_id,
        instrument_symbol=instrument_symbol,
        fee=fee,
        fx=fx,
        slippage=slippage,
        capital=capital,
    )
    bh_valuations = [r for r in bh_rcpts if r["kind"] == "valuation"]
    terminal_bh_equity = bh_valuations[-1]["equity_pln"] if bh_valuations else capital
    return {
        "schema": _SCHEMA,
        "cost_assumptions_sha256": cost_sha,
        "instrument_symbol": instrument_symbol,
        "cohort_id": cohort_id,
        "data_window_id": data_window_id,
        "quote_count": len(quotes),
        "cash_receipt_count": len(cash_rcpts),
        "buy_hold_receipt_count": len(bh_rcpts),
        "terminal_cash_equity_pln": capital,
        "terminal_buy_hold_equity_pln": round(terminal_bh_equity, 6),
        "cost_state": "ASSUMED",
        "note": (
            "Cash always equals initial capital; buy-and-hold terminal equity is from assumed "
            "round-trip costs. Neither is a profit claim or OOS evidence."
        ),
    }
