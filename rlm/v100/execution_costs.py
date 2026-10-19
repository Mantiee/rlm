"""Deterministic shadow depth estimates; never admission or guaranteed fills.

Inputs are normalized snapshots, not a venue API response. No networking, ledger
writes, order placement or promotion flags are performed here.
"""

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal, DecimalException, InvalidOperation, localcontext
from urllib.parse import urlparse

SCHEMA = "synta-shadow-depth-cost-v1"


def timestamp(value):
    if not isinstance(value, str) or len(value) > 80:
        raise ValueError("Invalid timestamp")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamps require an explicit timezone")
    return result.astimezone(UTC)


def amount(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
        raise ValueError("Amounts require decimal strings or integers")
    if isinstance(value, str) and len(value) > 100:
        raise ValueError("Decimal representation exceeds bound")
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("Invalid decimal amount") from error
    if (
        not result.is_finite()
        or result < 0
        or (positive and result == 0)
        or (result != 0 and not Decimal("1e-30") <= result <= Decimal("1e30"))
    ):
        raise ValueError("Amounts must be finite and nonnegative")
    return result


def evidence(value, decision_at, max_age_seconds):
    if not isinstance(value, dict):
        raise ValueError("Source evidence must be an object")
    if not isinstance(value["source_url"], str):
        raise ValueError("Invalid source URL")
    url = urlparse(value["source_url"])
    digest = value["source_sha256"]
    if (
        url.scheme != "https"
        or not url.hostname
        or url.username
        or url.password
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("Missing public source provenance")
    available = timestamp(value["available_at"])
    age = Decimal(str((decision_at - available).total_seconds()))
    if age < 0 or (max_age_seconds is not None and age > max_age_seconds):
        raise ValueError("Source is future or stale at decision time")
    return {
        "source_url": value["source_url"],
        "source_sha256": digest,
        "available_at": value["available_at"],
    }


def book_levels(book):
    sides = []
    for name in ("bids", "asks"):
        rows = book[name]
        if not isinstance(rows, list) or not 1 <= len(rows) <= 10000:
            raise ValueError("Book requires bounded nonempty sides")
        if any(not isinstance(row, (list, tuple)) or len(row) != 2 for row in rows):
            raise ValueError("Normalized levels require price and base quantity")
        levels = [(amount(row[0], positive=True), amount(row[1], positive=True)) for row in rows]
        if any(len(row) != 2 for row in rows):
            raise ValueError("Normalized levels require price and base quantity")
        if any(
            left[0] <= right[0] if name == "bids" else left[0] >= right[0]
            for left, right in zip(levels, levels[1:], strict=False)
        ):
            raise ValueError("Book levels must be strictly sorted without duplicates")
        sides.append(levels)
    if sides[0][0][0] >= sides[1][0][0]:
        raise ValueError("Locked or crossed book")
    return sides


def walk(levels, quantity):
    remaining, gross = quantity, Decimal(0)
    consumed = []
    for price, size in levels:
        taken = min(remaining, size)
        gross += price * taken
        consumed.append({"price": str(price), "quantity": str(taken)})
        remaining -= taken
        if remaining == 0:
            return gross, consumed
    raise ValueError("Insufficient depth; partial estimates cannot become fills")


def walk_quote_notional(levels, quote_notional):
    """Walk visible levels for an exact gross quote-currency notional."""
    remaining, gross, quantity = quote_notional, Decimal(0), Decimal(0)
    consumed = []
    for price, size in levels:
        available = price * size
        spend = min(remaining, available)
        taken = spend / price
        gross += spend
        quantity += taken
        consumed.append({"price": str(price), "quantity": str(taken)})
        remaining -= spend
        if remaining == 0:
            return gross, quantity, consumed
    raise ValueError("Insufficient depth; partial estimates cannot become fills")


def estimate(
    book,
    fee,
    *,
    side,
    quantity=None,
    quote_notional=None,
    decision_at,
    reporting_currency,
    fx=None,
    max_age_seconds=30,
):
    """Return an UNKNOWN or SHADOW_ESTIMATE receipt, always ineligible for execution.

    Fee is an evidenced quote-currency taker_bps/minimum schedule. Optional FX
    is an independently evidenced currency-pair depth book: base must equal
    trade quote currency and quote must equal reporting currency. A buy consumes
    FX asks to acquire trade currency; a sell liquidates proceeds at FX bids.
    Specify exactly one of base ``quantity`` or gross trade-currency
    ``quote_notional``; commission is calculated separately from that notional.
    FX fee is charged in reporting currency. Depth is an estimate of visible
    liquidity, excluding latency, queue dynamics and hidden liquidity.
    """
    receipt = {
        "schema": SCHEMA,
        "state": "UNKNOWN",
        "execution_eligible": False,
        "guaranteed_fill": False,
        # A source-backed estimate is still not evidence that this account can
        # execute at the quoted tier, nor that future execution costs are known.
        # Keep this explicit so downstream dashboards cannot infer verification
        # from SHADOW_ESTIMATE or from a published fee schedule alone.
        "full_execution_costs_verified": False,
        "unverified_cost_components": [
            "account_fee_tier_and_rebates",
            "latency_and_market_movement",
            "queue_position_and_hidden_or_future_liquidity",
        ],
        "scope": "Shadow visible-depth estimate only; no account fee-tier, latency, queue, or future-liquidity verification",
        "side": side if isinstance(side, str) else None,
        "decision_at": decision_at if isinstance(decision_at, str) else None,
    }
    try:
        with localcontext() as context:
            context.prec = 50
            result = calculate(
                book,
                fee,
                side,
                quantity,
                quote_notional,
                decision_at,
                reporting_currency,
                fx,
                max_age_seconds,
            )
        receipt.update(result, state="SHADOW_ESTIMATE")
    except (ValueError, KeyError, TypeError, IndexError, DecimalException) as error:
        receipt["reason"] = str(error)
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False)
    receipt["receipt_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    return receipt


def calculate(
    book, fee, side, quantity, quote_notional, decision_at, reporting_currency, fx, max_age
):
    if not isinstance(book, dict) or not isinstance(fee, dict):
        raise ValueError("Book and fee must be objects")
    if side not in ("buy", "sell"):
        raise ValueError("Unsupported side")
    now = timestamp(decision_at)
    if not isinstance(now, datetime):
        raise ValueError("Invalid decision time")
    age = amount(max_age, positive=True)
    if (quantity is None) == (quote_notional is None):
        raise ValueError("Specify exactly one of base quantity or quote notional")
    if quantity is not None:
        requested = amount(quantity, positive=True)
        request_kind = "base_quantity"
    else:
        requested = amount(quote_notional, positive=True)
        request_kind = "quote_notional"
    provenance = {"book": evidence(book, now, age), "fee": evidence(fee, now, None)}
    base, quote = book["base_currency"], book["quote_currency"]
    if not all(
        isinstance(value, str) and value.isascii() and value.isalnum() and 2 <= len(value) <= 12
        for value in (base, quote, reporting_currency)
    ):
        raise ValueError("Invalid currency identity")
    if base == quote or fee["currency"] != quote:
        raise ValueError("Inconsistent trade or fee currency")
    if timestamp(fee["valid_until"]) <= now:
        raise ValueError("Fee schedule expired")
    bids, asks = book_levels(book)
    levels = asks if side == "buy" else bids
    if request_kind == "base_quantity":
        quantity = requested
        gross, consumed = walk(levels, quantity)
    else:
        gross, quantity, consumed = walk_quote_notional(levels, requested)
    bps = amount(fee["taker_bps"])
    if bps >= 10000:
        raise ValueError("Fee rate must be below 100 percent")
    commission = max(gross * bps / 10000, amount(fee["minimum"]))
    trade_cash = gross + commission if side == "buy" else gross - commission
    if trade_cash <= 0:
        raise ValueError("Fees consume entire proceeds")
    reporting_cash, fx_fee, fx_consumed = trade_cash, Decimal(0), []
    if quote != reporting_currency:
        if not isinstance(fx, dict) or fx.get("kind") != "executable_depth":
            raise ValueError("Execution FX UNKNOWN: requires a sourced two-sided depth estimate")
        if fx["base_currency"] != quote or fx["quote_currency"] != reporting_currency:
            raise ValueError("FX direction or currency mismatch")
        provenance["fx"] = evidence(fx, now, age)
        provenance["fx_fee"] = evidence(fx["fee"], now, None)
        if (
            fx["fee"]["currency"] != reporting_currency
            or timestamp(fx["fee"]["valid_until"]) <= now
        ):
            raise ValueError("FX fee currency or validity mismatch")
        fx_bids, fx_asks = book_levels(fx)
        reporting_cash, fx_consumed = walk(fx_asks if side == "buy" else fx_bids, trade_cash)
        fx_bps = amount(fx["fee"]["taker_bps"])
        if fx_bps >= 10000:
            raise ValueError("FX fee rate must be below 100 percent")
        fx_fee = max(reporting_cash * fx_bps / 10000, amount(fx["fee"]["minimum"]))
        reporting_cash += fx_fee if side == "buy" else -fx_fee
        if reporting_cash <= 0:
            raise ValueError("FX fees consume entire proceeds")
    return {
        "quantity": str(quantity),
        "requested_size": {"kind": request_kind, "value": str(requested)},
        "base_currency": base,
        "quote_currency": quote,
        "reporting_currency": reporting_currency,
        "gross_quote": str(gross),
        "vwap_quote": str(gross / quantity),
        "book_impact_quote": str(abs(gross - quantity * levels[0][0])),
        "commission_quote": str(commission),
        "trade_cash_quote": str(trade_cash),
        "reporting_cash": str(reporting_cash),
        "fx_fee_reporting": str(fx_fee),
        "cash_direction": "debit" if side == "buy" else "credit",
        "consumed_levels": consumed,
        "fx_consumed_levels": fx_consumed,
        "provenance": provenance,
    }
