"""Fresh published fees must not turn assumed FX/slippage into certified fills."""

import pytest

from rlm.v100.financial_snapshot import live_ledger
from tests.test_v100_paper import decide, quote, setup


@pytest.mark.parametrize(
    "evidence",
    [
        {"execution_costs_verified": False},
        {"cost_evidence": "Slippage and FX are explicit paper assumptions"},
    ],
)
def test_fresh_fees_with_unverified_full_costs_block_orders_and_report_reason(tmp_path, evidence):
    book, clock, _ = setup(tmp_path, fee_changes=evidence)
    try:
        book.ingest(quote(clock))
        before = book.state()
        with pytest.raises(ValueError, match="Execution costs UNKNOWN"):
            decide(book)
        assert book.state() == before
        assert any("Execution costs UNKNOWN" in row for row in live_ledger(tmp_path)["blockers"])
        assert decide(book, action="hold")["kind"] == "decision"
    finally:
        book.close()


def test_reference_fx_does_not_replace_missing_independent_cost_receipt(tmp_path):
    book, clock, _ = setup(tmp_path)
    try:
        book.ingest(
            quote(clock, fx={"note": "NBP reference rate, not an executable broker FX quote"})
        )
        with pytest.raises(ValueError, match="Execution costs UNKNOWN"):
            decide(book)
    finally:
        book.close()


def test_unverified_costs_prevent_a_pending_order_from_being_created(tmp_path):
    book, clock, config = setup(tmp_path)
    try:
        book.ingest(quote(clock))
        before = book.state()
        with pytest.raises(ValueError, match="Execution costs UNKNOWN"):
            decide(book)
        state = book.state()
        assert state == before
        assert state["orders"] == []
        assert state["branches"]["A"]["cash"] == "10000"
        assert state["branches"]["A"]["positions"] == {}
        assert config["fee_profiles"][0].get("execution_cost_evidence") is None
    finally:
        book.close()
