import copy
from decimal import Decimal, localcontext

import pytest

from rlm.v100.execution_costs import estimate

NOW = "2026-10-15T04:00:00+00:00"


def source():
    return {
        "source_url": "https://example.org/snapshot",
        "source_sha256": "a" * 64,
        "available_at": NOW,
    }


def fixture():
    book = {
        **source(),
        "base_currency": "BTC",
        "quote_currency": "USD",
        "bids": [["99", "1"], ["98", "2"]],
        "asks": [["100", "1"], ["102", "2"]],
    }
    fee = {
        **source(),
        "currency": "USD",
        "taker_bps": "100",
        "minimum": "0",
        "valid_until": "2026-10-16T04:00:00+00:00",
    }
    return book, fee


def run(book=None, fee=None, **kwargs):
    default_book, default_fee = fixture()
    return estimate(
        book or default_book,
        fee or default_fee,
        side=kwargs.pop("side", "buy"),
        quantity=kwargs.pop("quantity", "2"),
        decision_at=NOW,
        reporting_currency=kwargs.pop("reporting_currency", "USD"),
        **kwargs,
    )


def test_two_level_buy_sell_and_no_double_charged_spread():
    buy, sell = run(), run(side="sell")
    assert buy["state"] == sell["state"] == "SHADOW_ESTIMATE"
    assert Decimal(buy["vwap_quote"]) == 101
    assert Decimal(buy["commission_quote"]) == Decimal("2.02")
    assert Decimal(buy["reporting_cash"]) == Decimal("204.02")
    assert Decimal(sell["reporting_cash"]) == Decimal("195.03")
    assert Decimal(buy["book_impact_quote"]) == 2
    assert not buy["execution_eligible"] and not buy["guaranteed_fill"]
    assert Decimal(sell["reporting_cash"]) - Decimal(buy["reporting_cash"]) == Decimal("-8.99")


def test_fx_uses_asks_for_buy_bids_for_sell_and_independent_fees():
    _, fee = fixture()
    fx = {
        **source(),
        "kind": "executable_depth",
        "base_currency": "USD",
        "quote_currency": "PLN",
        "bids": [["4", "500"]],
        "asks": [["4.1", "500"]],
        "fee": {**fee, "currency": "PLN", "taker_bps": "50"},
    }
    buy = run(reporting_currency="PLN", fx=fx)
    sell = run(side="sell", reporting_currency="PLN", fx=fx)
    assert Decimal(buy["reporting_cash"]) == Decimal("204.02") * Decimal("4.1") * Decimal("1.005")
    assert Decimal(sell["reporting_cash"]) == Decimal("195.03") * 4 * Decimal(".995")
    assert buy["provenance"]["fx"]["source_sha256"] == "a" * 64
    fx["asks"][0][1] = "1"
    assert run(reporting_currency="PLN", fx=fx)["state"] == "UNKNOWN"


@pytest.mark.parametrize(
    "change",
    [
        {"bids": [["100", "1"]]},
        {"asks": [["102", "1"], ["100", "2"]]},
        {"asks": [["100", "1"], ["100", "2"]]},
        {"asks": [["100", "0"]]},
        {"asks": [["NaN", "1"]]},
        {"asks": [["-1", "1"]]},
        {"asks": [[100.0, "1"]]},
        {"available_at": "2026-10-15T04:00:01+00:00"},
        {"available_at": "2026-10-15T03:59:29+00:00"},
        {"source_sha256": ""},
        {"source_url": "https://user:secret@example.org/feed"},
        {"base_currency": "USD"},
    ],
)
def test_invalid_books_are_unknown_without_execution(change):
    book, _ = fixture()
    book.update(change)
    receipt = run(book=book)
    assert receipt["state"] == "UNKNOWN"
    assert not receipt["execution_eligible"]
    assert "reporting_cash" not in receipt


@pytest.mark.parametrize(
    "change",
    [
        {"currency": "PLN"},
        {"taker_bps": "10000"},
        {"minimum": "Infinity"},
        {"valid_until": NOW},
        {"available_at": "2026-10-15T04:00:01+00:00"},
    ],
)
def test_unknown_fee_is_not_zero(change):
    _, fee = fixture()
    fee.update(change)
    assert run(fee=fee)["state"] == "UNKNOWN"


def test_insufficient_depth_and_missing_or_reference_fx():
    assert run(quantity="4")["state"] == "UNKNOWN"
    assert run(reporting_currency="PLN")["state"] == "UNKNOWN"
    assert run(reporting_currency="PLN", fx={"kind": "reference"})["state"] == "UNKNOWN"


def test_deterministic_replay_does_not_mutate_input_or_use_global_decimal_context():
    book, fee = fixture()
    original = copy.deepcopy((book, fee))
    first = run(book=book, fee=fee)
    with localcontext() as context:
        context.prec = 6
        assert run(book=book, fee=fee) == first
    assert (book, fee) == original
    book["available_at"] = "2026-10-15T04:00:01+00:00"
    assert run(book=book, fee=fee)["state"] == "UNKNOWN"
    assert first == run(book=original[0], fee=original[1])


def test_minimum_commission_and_fee_consuming_sell_proceeds():
    _, fee = fixture()
    fee["minimum"] = "5"
    assert Decimal(run(fee=fee)["commission_quote"]) == 5
    fee["minimum"] = "999"
    assert run(fee=fee, side="sell")["state"] == "UNKNOWN"


@pytest.mark.parametrize(
    "change", [{"available_at": None}, {"source_url": None}, {"asks": [["1e999999", "1"]]}]
)
def test_malformed_and_unbounded_snapshot_fails_closed(change):
    book, _ = fixture()
    book.update(change)
    assert run(book=book)["state"] == "UNKNOWN"


def test_fee_publication_can_be_old_but_must_be_known_and_unexpired():
    _, fee = fixture()
    fee["available_at"] = "2026-10-10T04:00:00+00:00"
    assert run(fee=fee)["state"] == "SHADOW_ESTIMATE"


def test_string_level_and_nonobject_inputs_are_rejected():
    book, fee = fixture()
    book["asks"] = ["12"]
    assert run(book=book)["state"] == "UNKNOWN"
    assert (
        estimate([], fee, side="buy", quantity="1", decision_at=NOW, reporting_currency="USD")[
            "state"
        ]
        == "UNKNOWN"
    )
