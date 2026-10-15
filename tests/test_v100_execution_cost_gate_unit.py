"""Deterministic fail-closed tests for execution-cost evidence admission."""

import copy

import pytest

from rlm.v100.execution_cost_gate import SCHEMA, execution_cost_blocker

NOW = "2026-10-15T04:00:00+00:00"
SYMBOL = "BTC-USD"


def source(**overrides):
    return {
        "source_url": "https://evidence.example.test/receipt",
        "source_sha256": "a" * 64,
        "available_at": "2026-10-15T03:59:00+00:00",
        "valid_until": "2026-10-15T04:30:00+00:00",
        "instrument": SYMBOL,
        "scope": "synthetic account and instrument test evidence",
        **overrides,
    }


def evidence():
    venue_values = {
        field: "0"
        for field in (
            "entry_commission_bps",
            "entry_minimum_commission",
            "entry_venue_fee_bps",
            "exit_commission_bps",
            "exit_minimum_commission",
            "exit_venue_fee_bps",
        )
    }
    return {
        "schema": SCHEMA,
        "state": "independently_verified",
        "instrument": SYMBOL,
        "portfolio_currency": "USD",
        "verifier": "synthetic-test-reviewer",
        "verified_at": "2026-10-15T03:59:00+00:00",
        "components": {
            "venue_fee": {
                **source(),
                "state": "independently_verified",
                "covered_profile_values": venue_values,
            },
            "account_fee_tier": {**source(), "state": "independently_verified"},
            "slippage": {
                **source(),
                "state": "independently_verified",
                "covered_profile_values": {"slippage_bps": "0"},
            },
            "fx": {
                "state": "not_applicable",
                "base_currency": "USD",
                "quote_currency": "USD",
                "covered_profile_values": {"entry_fx_bps": "0", "exit_fx_bps": "0"},
            },
        },
    }


def profile(**overrides):
    return {
        "currency": "USD",
        "execution_costs_verified": True,
        "execution_cost_evidence": evidence(),
        "entry_commission_bps": "0",
        "entry_minimum_commission": "0",
        "entry_venue_fee_bps": "0",
        "exit_commission_bps": "0",
        "exit_minimum_commission": "0",
        "exit_venue_fee_bps": "0",
        "slippage_bps": "0",
        "entry_fx_bps": "0",
        "exit_fx_bps": "0",
        **overrides,
    }


def quote(**overrides):
    return {
        "symbol": SYMBOL,
        "currency": "USD",
        "price_source_currency": "USD",
        "available_at": NOW,
        "fx": None,
        **overrides,
    }


def fx_quote(**overrides):
    return {
        **source(),
        "kind": "executable_depth",
        "base_currency": "EUR",
        "quote_currency": "USD",
        "bid": "1.08",
        "ask": "1.09",
        **overrides,
    }


def cross_currency_evidence():
    record = evidence()
    record["portfolio_currency"] = "USD"
    record["components"]["fx"] = {
        **source(),
        "state": "independently_verified",
        "base_currency": "EUR",
        "quote_currency": "USD",
        "covered_profile_values": {"entry_fx_bps": "20", "exit_fx_bps": "20"},
    }
    return record


def test_structured_but_self_asserted_evidence_stays_blocked_without_trust_root():
    candidate = profile()
    candidate["execution_cost_evidence"].update(
        attestation_key_id="synthetic-test-key",
        attestation_signature="fake-signature",
    )
    assert "no trusted provenance verifier" in execution_cost_blocker(candidate, quote())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.pop("execution_costs_verified"),
        lambda p: p.update(execution_costs_verified=None),
        lambda p: p.update(execution_costs_verified=False),
        lambda p: p.update(execution_cost_evidence={"state": "independently_verified"}),
        lambda p: p["execution_cost_evidence"].update(verifier=" "),
        lambda p: p["execution_cost_evidence"].update(instrument="OTHER"),
        lambda p: p["execution_cost_evidence"]["components"].pop("slippage"),
        lambda p: p["execution_cost_evidence"]["components"]["venue_fee"].update(
            source_sha256="not-a-hash"
        ),
        lambda p: p["execution_cost_evidence"]["components"]["account_fee_tier"].update(
            valid_until="2026-10-15T03:59:59+00:00"
        ),
        lambda p: p["execution_cost_evidence"]["components"]["slippage"].update(
            source_url="http://evidence.example.test/receipt"
        ),
        lambda p: p["execution_cost_evidence"]["components"]["venue_fee"][
            "covered_profile_values"
        ].update(entry_venue_fee_bps="1"),
        lambda p: p["execution_cost_evidence"]["components"]["fx"].pop("quote_currency"),
    ],
)
def test_missing_or_malformed_cost_provenance_blocks(mutate):
    candidate = profile()
    mutate(candidate)
    result = execution_cost_blocker(candidate, quote())
    assert result is not None
    assert "UNKNOWN" in result


def test_cross_currency_requires_current_executable_fx_not_reference_rate():
    candidate = profile(
        execution_cost_evidence=cross_currency_evidence(),
        entry_fx_bps="20",
        exit_fx_bps="20",
    )
    incoming = quote(
        currency="USD",
        price_source_currency="EUR",
        fx={
            "rate": "1.09",
            "source_url": "https://api.example.test/fx",
            "source_sha256": "b" * 64,
            "available_at": NOW,
            "note": "reference rate",
        },
    )
    assert "Execution FX UNKNOWN" in execution_cost_blocker(candidate, incoming)


@pytest.mark.parametrize(
    "change",
    [
        {"source_sha256": "bad"},
        {"instrument": "OTHER"},
        {"valid_until": "2026-10-15T03:59:59+00:00"},
        {"ask": "1.07"},
        {"base_currency": "USD"},
    ],
)
def test_malformed_or_mismatched_executable_fx_blocks(change):
    candidate = profile(
        execution_cost_evidence=cross_currency_evidence(),
        entry_fx_bps="20",
        exit_fx_bps="20",
    )
    incoming = quote(
        price_source_currency="EUR",
        fx=fx_quote(**change),
    )
    assert "Execution FX UNKNOWN" in execution_cost_blocker(candidate, incoming)


def test_valid_fx_shape_needs_both_profile_and_quote_evidence():
    candidate = profile(
        execution_cost_evidence=cross_currency_evidence(),
        entry_fx_bps="20",
        exit_fx_bps="20",
    )
    incoming = quote(price_source_currency="EUR", fx=fx_quote())
    assert "no trusted provenance verifier" in execution_cost_blocker(candidate, incoming)
    incoming["fx"] = None
    assert "Execution FX UNKNOWN" in execution_cost_blocker(candidate, incoming)


def test_inputs_are_not_mutated():
    candidate, incoming = profile(), quote()
    original = copy.deepcopy((candidate, incoming))
    execution_cost_blocker(candidate, incoming)
    assert (candidate, incoming) == original
