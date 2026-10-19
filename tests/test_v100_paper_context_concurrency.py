"""Concurrency tests isolate accounting with synthetic fees; never authorize live execution."""

import json
from datetime import timedelta
from types import SimpleNamespace

import pytest

from rlm.v100 import paper_agents
from rlm.v100.paper import PaperBook
from tests.test_v100_paper import allow_cost_math_only, configuration, decide, quote, setup


def proposal(action="open"):
    return {
        "action": action,
        "symbol": "TEST",
        "side": "long",
        "budget": "400",
        "leverage": "1",
        "rationale": "Synthetic hypothesis from captured input",
    }


def test_quote_arriving_during_model_preserves_actual_asof_inputs(tmp_path, monkeypatch):
    allow_cost_math_only(monkeypatch)
    book, clock, _ = setup(tmp_path, fee_changes={"slippage_bps": "10"})
    book.ingest(quote(clock))
    context = book.context("A")
    clock.advance()
    # Independent writer represents the feed thread while inference is running.
    writer = PaperBook(tmp_path, clock)
    writer.ingest(quote(clock, "100.01"))
    writer.close()
    event = book.decide("A", proposal(), context["sequence"])
    assert event["payload"]["proposal"]["action"] == "open"
    inputs = event["payload"]["decision_inputs"]
    assert inputs["quote"]["ask"] == "100"
    assert inputs["latest_quote_at_acceptance"]["ask"] == "100.01"
    assert inputs["asof_sequence"] == context["sequence"]
    assert book.state()["branches"]["A"]["positions"] == {}
    clock.advance()
    book.ingest(quote(clock, "100.01"))
    assert "TEST" in book.state()["branches"]["A"]["positions"]


def test_unrelated_branch_research_and_decision_do_not_invalidate(tmp_path):
    book, clock, _ = setup(tmp_path)
    book.ingest(quote(clock))
    context = book.context("A")
    book.note("B", {"observation": "Independent research"})
    decide(book, "hold", branch="B")
    event = book.decide("A", proposal("hold"), context["sequence"])
    assert event["payload"]["proposal"]["action"] == "hold"


def test_own_portfolio_change_rejects_captured_input(tmp_path, monkeypatch):
    allow_cost_math_only(monkeypatch)
    book, clock, _ = setup(tmp_path)
    book.ingest(quote(clock))
    context = book.context("A")
    decide(book)
    clock.advance()
    book.ingest(quote(clock))
    with pytest.raises(ValueError, match="portfolio, instrument or cost policy changed"):
        book.decide("A", proposal("hold"), context["sequence"])


def test_future_snapshot_timestamp_rejected(tmp_path):
    book, clock, _ = setup(tmp_path)
    book.ingest(quote(clock))
    context = book.context("A")
    clock.now -= timedelta(seconds=1)
    with pytest.raises(ValueError, match="future paper decision sequence"):
        book.decide("A", proposal("hold"), context["sequence"])


def test_new_quotes_cannot_extend_stale_model_inputs(tmp_path, monkeypatch):
    allow_cost_math_only(monkeypatch)
    book, clock, _ = setup(tmp_path)
    book.ingest(quote(clock))
    context = book.context("A")
    clock.advance(301)
    book.ingest(quote(clock))
    with pytest.raises(ValueError, match="input quote is stale"):
        book.decide("A", proposal(), context["sequence"])


def test_adverse_quote_move_rejects_slow_proposal(tmp_path, monkeypatch):
    allow_cost_math_only(monkeypatch)
    book, clock, _ = setup(tmp_path, fee_changes={"slippage_bps": "10"})
    book.ingest(quote(clock))
    context = book.context("A")
    clock.advance()
    book.ingest(quote(clock, "101"))
    with pytest.raises(ValueError, match="beyond modeled slippage"):
        book.decide("A", proposal(), context["sequence"])


def test_later_execution_cannot_escape_slow_proposal_price_bound(tmp_path, monkeypatch):
    allow_cost_math_only(monkeypatch)
    book, clock, _ = setup(tmp_path, fee_changes={"slippage_bps": "10"})
    book.ingest(quote(clock))
    context = book.context("A")
    clock.advance()
    book.ingest(quote(clock))
    book.decide("A", proposal(), context["sequence"])
    clock.advance()
    book.ingest(quote(clock, "101"))
    assert book.state()["branches"]["A"]["positions"] == {}
    assert book.state()["orders"] == []


def test_cost_gate_still_blocks_after_concurrent_tick(tmp_path):
    book, clock, _ = setup(tmp_path)
    book.ingest(quote(clock))
    context = book.context("A")
    clock.advance()
    book.ingest(quote(clock))
    with pytest.raises(ValueError, match="Execution costs"):
        book.decide("A", proposal(), context["sequence"])


def test_fee_version_change_requires_recomputation(tmp_path):
    book, clock, _ = setup(tmp_path)
    book.ingest(quote(clock))
    context = book.context("A")
    update = configuration(clock)["fee_profiles"][0]
    update["id"] = "test-fees-v2"
    book.configure(
        {
            "fee_profiles": [update],
            "fee_updates": [{"symbol": "TEST", "fee_profile": "test-fees-v2"}],
        }
    )
    with pytest.raises(ValueError, match="instrument or cost policy changed"):
        book.decide("A", proposal("hold"), context["sequence"])


def test_no_snapshot_cannot_accept_an_old_sequence(tmp_path):
    book, clock, _ = setup(tmp_path)
    book.ingest(quote(clock))
    sequence = book.sequence()
    clock.advance()
    book.ingest(quote(clock))
    with pytest.raises(ValueError, match="no host-captured input snapshot"):
        book.decide("A", proposal("hold"), sequence)


def test_gate_hold_preserves_original_model_intent_and_reason(tmp_path, monkeypatch):
    book, clock, _ = setup(tmp_path)
    book.ingest(quote(clock))
    original = {**proposal(), "accepted_research": []}
    monkeypatch.setattr(paper_agents, "helper_client", lambda *a: SimpleNamespace())
    monkeypatch.setattr(
        paper_agents, "native_turn", lambda *a, **kw: {"content": json.dumps(original)}
    )
    from rlm.v100 import mission_chat

    monkeypatch.setattr(mission_chat, "emit_alert", lambda *a: None)
    events = paper_agents.paper_decision_round(book, {"runtime": {"max_output_tokens": 512}})
    for event in events:
        recorded = event["payload"]["proposal"]
        assert recorded["host_action"] == recorded["action"] == "hold"
        assert recorded["model_proposal"] == original
        assert recorded["model_action_blocked"] == "open"
        assert "Execution costs UNKNOWN" in recorded["gate_reason"]
    assert book.state()["orders"] == []
