"""Fail-closed validation for paper execution-cost evidence.

This gate validates evidence shape and scope. It does not itself authenticate a
source or independently certify that a supplied assertion is true.
"""

import re
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from urllib.parse import urlparse

SCHEMA = "synta-execution-cost-evidence-v1"
COMPONENTS = ("venue_fee", "account_fee_tier", "slippage", "fx")
# Deliberately empty until a public key and signature-verification workflow are
# reviewed and shipped. Text fields and hashes supplied with a profile are not
# themselves trusted evidence.
TRUSTED_PROVENANCE_KEYS = {}
PROFILE_VALUES = {
    "venue_fee": (
        "entry_commission_bps",
        "entry_minimum_commission",
        "entry_venue_fee_bps",
        "exit_commission_bps",
        "exit_minimum_commission",
        "exit_venue_fee_bps",
    ),
    "slippage": ("slippage_bps",),
    "fx": ("entry_fx_bps", "exit_fx_bps"),
}


def _time(value):
    if not isinstance(value, str) or len(value) > 80:
        raise ValueError("invalid timestamp")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timestamp needs timezone")
    return result.astimezone(UTC)


def _source_record(value, *, decision_at, symbol):
    if not isinstance(value, dict):
        raise ValueError("missing source record")
    url = urlparse(value.get("source_url", ""))
    digest = value.get("source_sha256")
    if (
        url.scheme != "https"
        or not url.hostname
        or url.username
        or url.password
        or not isinstance(digest, str)
        or not re.fullmatch(r"[a-f0-9]{64}", digest)
        or value.get("instrument") != symbol
        or not isinstance(value.get("scope"), str)
        or not value["scope"].strip()
    ):
        raise ValueError("invalid source provenance or scope")
    available_at = _time(value.get("available_at"))
    valid_until = _time(value.get("valid_until"))
    if available_at > decision_at or valid_until <= decision_at:
        raise ValueError("source is future-dated or expired")


def _profile_values(value, profile, fields):
    recorded = value.get("covered_profile_values")
    if not isinstance(recorded, dict) or set(recorded) != set(fields):
        raise ValueError("cost evidence does not specify every covered fee value")
    for field in fields:
        if field not in profile:
            raise ValueError("cost profile is missing a covered fee value")
        try:
            supplied, current = Decimal(str(recorded[field])), Decimal(str(profile[field]))
        except (InvalidOperation, TypeError, ValueError):
            raise ValueError("covered fee value is malformed") from None
        if not supplied.is_finite() or not current.is_finite() or supplied != current:
            raise ValueError("covered fee value does not match current profile")


def _trusted_attestation(evidence, profile, quote):
    """No trust root is configured, so self-asserted receipts never authorize."""
    key_id = evidence.get("attestation_key_id") if isinstance(evidence, dict) else None
    signature = evidence.get("attestation_signature") if isinstance(evidence, dict) else None
    if not TRUSTED_PROVENANCE_KEYS or key_id not in TRUSTED_PROVENANCE_KEYS or not signature:
        return False
    # Signature verification over canonical evidence + profile + quote has not
    # been implemented; a configured key alone must never authorize a fill.
    del profile, quote
    return False


def _fx_applies(profile, quote, source_currency):
    try:
        assumed_fx = any(
            Decimal(str(profile.get(field, "0"))) != 0 for field in ("entry_fx_bps", "exit_fx_bps")
        )
    except (InvalidOperation, ValueError):
        return True
    return assumed_fx or source_currency != profile.get("currency") or quote.get("fx") is not None


def _validate_fx_quote(value, *, decision_at, symbol, source_currency, portfolio_currency):
    _source_record(value, decision_at=decision_at, symbol=symbol)
    if (
        value.get("kind") != "executable_depth"
        or value.get("base_currency") != source_currency
        or value.get("quote_currency") != portfolio_currency
    ):
        raise ValueError("FX evidence is not an executable quote in the required direction")
    try:
        bid, ask = Decimal(str(value["bid"])), Decimal(str(value["ask"]))
    except (KeyError, InvalidOperation, TypeError):
        raise ValueError("FX quote requires bid and ask") from None
    if not bid.is_finite() or not ask.is_finite() or bid <= 0 or ask < bid:
        raise ValueError("invalid executable FX spread")


def execution_cost_blocker(profile: dict, quote: dict) -> str | None:
    """Require complete, current, instrument-scoped evidence; otherwise UNKNOWN.

    A structurally valid receipt is a necessary gate, not cryptographic proof of
    the truth of its claims. The current Kraken bootstrap intentionally cannot
    meet this schema, so published fees and modeled costs remain blocked.
    """
    try:
        if not isinstance(profile, dict) or not isinstance(quote, dict):
            raise ValueError("profile or quote missing")
        if profile.get("execution_costs_verified") is not True:
            raise ValueError("full-cost verification is not explicitly attested")
        symbol = quote.get("symbol")
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("instrument identity missing")
        decision_at = _time(quote.get("available_at"))
        evidence = profile.get("execution_cost_evidence")
        if (
            not isinstance(evidence, dict)
            or evidence.get("schema") != SCHEMA
            or evidence.get("state") != "independently_verified"
            or evidence.get("instrument") != symbol
            or evidence.get("portfolio_currency") != profile.get("currency")
            or not isinstance(evidence.get("verifier"), str)
            or not evidence["verifier"].strip()
            or _time(evidence.get("verified_at")) > decision_at
        ):
            raise ValueError("structured independent evidence missing or out of scope")
        components = evidence.get("components")
        if not isinstance(components, dict) or any(name not in components for name in COMPONENTS):
            raise ValueError("required cost component evidence missing")
        for name in ("venue_fee", "account_fee_tier", "slippage"):
            component = components[name]
            if (
                not isinstance(component, dict)
                or component.get("state") != "independently_verified"
            ):
                raise ValueError(f"{name} is not independently verified")
            _source_record(component, decision_at=decision_at, symbol=symbol)
            if name in PROFILE_VALUES:
                _profile_values(component, profile, PROFILE_VALUES[name])

        source_currency = quote.get("price_source_currency", quote.get("currency"))
        portfolio_currency = profile.get("currency")
        fx = components["fx"]
        if _fx_applies(profile, quote, source_currency):
            if not isinstance(fx, dict) or fx.get("state") != "independently_verified":
                return "Execution FX UNKNOWN: executable conversion evidence is missing"
            try:
                _source_record(fx, decision_at=decision_at, symbol=symbol)
                _profile_values(fx, profile, PROFILE_VALUES["fx"])
                if (
                    fx.get("base_currency") != source_currency
                    or fx.get("quote_currency") != portfolio_currency
                ):
                    raise ValueError("FX component currencies do not match the portfolio")
                _validate_fx_quote(
                    quote.get("fx"),
                    decision_at=decision_at,
                    symbol=symbol,
                    source_currency=source_currency,
                    portfolio_currency=portfolio_currency,
                )
            except (KeyError, TypeError, ValueError, InvalidOperation, OverflowError):
                return "Execution FX UNKNOWN: missing or invalid executable conversion evidence"
        elif (
            not isinstance(fx, dict)
            or fx.get("state") != "not_applicable"
            or source_currency != portfolio_currency
            or fx.get("base_currency") != source_currency
            or fx.get("quote_currency") != portfolio_currency
        ):
            return "Execution FX UNKNOWN: currency match is not evidenced"
        if not _trusted_attestation(evidence, profile, quote):
            return "Execution costs UNKNOWN: no trusted provenance verifier is configured"
    except (KeyError, TypeError, ValueError, InvalidOperation, OverflowError):
        return "Execution costs UNKNOWN: missing or invalid independent cost evidence"
    return None
