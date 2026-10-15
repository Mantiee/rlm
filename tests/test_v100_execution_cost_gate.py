"""Fresh published fees must not turn assumed FX/slippage into certified fills."""

import copy

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


def test_reference_fx_cannot_authorize_a_fill_with_otherwise_known_fixture_costs(tmp_path):
    book, clock, _ = setup(tmp_path)
    try:
        book.ingest(
            quote(clock, fx={"note": "NBP reference rate, not an executable broker FX quote"})
        )
        with pytest.raises(ValueError, match="Execution FX UNKNOWN"):
            decide(book)
    finally:
        book.close()


def test_pending_order_is_cancelled_when_costs_become_unverified(tmp_path):
    book, clock, config = setup(tmp_path)
    try:
        book.ingest(quote(clock))
        decide(book)
        fee = copy.deepcopy(config["fee_profiles"][0])
        fee.update(id="fees-unverified-v2", execution_costs_verified=False)
        book.configure(
            {"fee_profiles": [fee], "fee_updates": [{"symbol": "TEST", "fee_profile": fee["id"]}]}
        )
        clock.advance()
        event = book.ingest(quote(clock))
        state = book.state()
        assert state["orders"] == []
        assert state["branches"]["A"]["cash"] == "10000"
        assert state["branches"]["A"]["positions"] == {}
        assert event["payload"]["trades"][0]["reason"] == "unverified-execution-costs"
        assert event["payload"]["trades"][0]["action"] == "cancel"
    finally:
        book.close()
