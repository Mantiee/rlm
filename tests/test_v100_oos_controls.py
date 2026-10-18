"""Tests for paper_oos_controls: cost hash, cash control, buy-and-hold control."""
import hashlib
import json
import math
import pytest
from rlm.v100.paper_oos_controls import (
    cost_assumptions_sha256,
    cash_control_receipts,
    buy_hold_control_receipts,
    oos_control_summary,
)


def _quotes(n=10, bid=50_000.0, ask=50_100.0, spread_bps=20):
    """Synthetic BTC-KRAKEN quote list."""
    return [
        {
            "timestamp": float(1_700_000_000 + i * 3600),
            "bid": bid,
            "ask": ask,
            "bid_size": 1.0,
            "ask_size": 1.0,
        }
        for i in range(n)
    ]


def _rising_quotes(n=10, start_bid=50_000.0):
    return [
        {
            "timestamp": float(1_700_000_000 + i * 3600),
            "bid": start_bid * (1 + i * 0.01),
            "ask": start_bid * (1 + i * 0.01) * 1.001,
            "bid_size": 1.0,
            "ask_size": 1.0,
        }
        for i in range(n)
    ]


_ID = dict(
    portfolio_id="test-portfolio-abc123",
    cohort_id="test-cohort-xyz789",
    data_window_id="a" * 40,
    instrument_symbol="BTC-KRAKEN",
)


# --- cost_assumptions_sha256 ---

def test_cost_sha256_deterministic():
    h1 = cost_assumptions_sha256(0.0025, 0.001, 0.001)
    h2 = cost_assumptions_sha256(0.0025, 0.001, 0.001)
    assert h1 == h2
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)


def test_cost_sha256_different_for_different_params():
    assert cost_assumptions_sha256(0.0025, 0.001, 0.001) != cost_assumptions_sha256(0.003, 0.001, 0.001)
    assert cost_assumptions_sha256(0.0025, 0.001, 0.001) != cost_assumptions_sha256(0.0025, 0.002, 0.001)


def test_cost_sha256_rejects_negative():
    with pytest.raises(ValueError, match="non-negative"):
        cost_assumptions_sha256(-0.001, 0.001, 0.001)


def test_cost_sha256_rejects_over_100pct():
    with pytest.raises(ValueError, match="exceed"):
        cost_assumptions_sha256(0.6, 0.6, 0.001)


# --- cash_control_receipts ---

def test_cash_control_length():
    q = _quotes(8)
    rcpts = cash_control_receipts(q, fee=0.0025, fx=0.001, slippage=0.001, **_ID)
    assert len(rcpts) == 8


def test_cash_control_equity_constant():
    q = _rising_quotes(10)
    rcpts = cash_control_receipts(q, fee=0.0025, fx=0.001, slippage=0.001, **_ID)
    for r in rcpts:
        assert r["equity_pln"] == 100.0
        assert r["realized_pnl_pln"] == 0.0
        assert r["unrealized_pnl_pln"] == 0.0
        assert r["kind"] == "valuation"
        assert r["status"] == "valued"
        assert r["role"] == "cash"


def test_cash_control_has_cost_sha():
    q = _quotes()
    rcpts = cash_control_receipts(q, fee=0.0025, fx=0.001, slippage=0.001, **_ID)
    sha = cost_assumptions_sha256(0.0025, 0.001, 0.001)
    assert all(r["cost_assumptions_sha256"] == sha for r in rcpts)


def test_cash_control_pnl_identity():
    q = _quotes()
    rcpts = cash_control_receipts(q, fee=0.0025, fx=0.001, slippage=0.001, **_ID)
    for r in rcpts:
        assert r["equity_pln"] - 100 == r["realized_pnl_pln"] + r["unrealized_pnl_pln"]


# --- buy_hold_control_receipts ---

def test_buy_hold_has_fill_and_valuations():
    q = _quotes(8)
    rcpts = buy_hold_control_receipts(q, fee=0.0025, fx=0.001, slippage=0.001, **_ID)
    fills = [r for r in rcpts if r["kind"] == "fill"]
    vals = [r for r in rcpts if r["kind"] == "valuation"]
    assert len(fills) == 1
    assert fills[0]["action"] == "BUY"
    assert fills[0]["status"] == "filled"
    assert len(vals) >= 1


def test_buy_hold_fill_role():
    q = _quotes()
    rcpts = buy_hold_control_receipts(q, fee=0.0025, fx=0.001, slippage=0.001, **_ID)
    assert all(r["role"] == "buy_hold" for r in rcpts)


def test_buy_hold_rising_market_positive_return():
    q = _rising_quotes(10)
    rcpts = buy_hold_control_receipts(q, fee=0.0025, fx=0.001, slippage=0.001, **_ID)
    vals = [r for r in rcpts if r["kind"] == "valuation"]
    # In a rising market the terminal equity should exceed initial capital
    assert vals[-1]["equity_pln"] > 100.0


def test_buy_hold_pnl_identity():
    q = _rising_quotes(12)
    rcpts = buy_hold_control_receipts(q, fee=0.0025, fx=0.001, slippage=0.001, **_ID)
    vals = [r for r in rcpts if r["kind"] == "valuation"]
    for r in vals:
        assert math.isclose(
            r["equity_pln"] - 100,
            r["realized_pnl_pln"] + r["unrealized_pnl_pln"],
            rel_tol=1e-9,
        ), f"PnL identity failed: {r}"


def test_buy_hold_cost_sha():
    q = _quotes()
    rcpts = buy_hold_control_receipts(q, fee=0.0025, fx=0.001, slippage=0.001, **_ID)
    sha = cost_assumptions_sha256(0.0025, 0.001, 0.001)
    assert all(r["cost_assumptions_sha256"] == sha for r in rcpts)


# --- oos_control_summary ---

def test_summary_has_required_fields():
    q = _quotes(10)
    s = oos_control_summary(q, fee=0.0025, fx=0.001, slippage=0.001, **_ID)
    assert "cost_assumptions_sha256" in s
    assert s["instrument_symbol"] == "BTC-KRAKEN"
    assert s["quote_count"] == 10
    assert s["cost_state"] == "ASSUMED"


def test_summary_terminal_cash_equals_capital():
    q = _quotes()
    s = oos_control_summary(q, fee=0.0025, fx=0.001, slippage=0.001, **_ID)
    assert s["terminal_cash_equity_pln"] == 100.0


def test_summary_requires_4_quotes():
    with pytest.raises(ValueError):
        oos_control_summary(_quotes(3), fee=0.0025, fx=0.001, slippage=0.001, **_ID)
