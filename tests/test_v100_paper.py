import copy
import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest

from rlm.v100 import paper_agents, paper_feeds, paper_tools
from rlm.v100.paper import FEE_FIELDS, PaperBook, fee_total
from rlm.v100.paper_reports import scheduled_reports, summarize, write_report


class Clock:
    def __init__(self):
        self.now = datetime(2026, 10, 7, 10, tzinfo=UTC)

    def __call__(self):
        return self.now

    def advance(self, seconds=1):
        self.now += timedelta(seconds=seconds)
        return self.now.isoformat()


def evidence(clock):
    return {
        "source_url": "https://example.org/independent-feed",
        "source_sha256": "a" * 64,
        "available_at": clock().isoformat(),
    }


def configuration(clock, product="spot", market="crypto", symbol="TEST", fee_changes=None):
    # Synthetic fees/rules for deterministic engine tests, never live platform rates.
    profile = {
        "id": "test-fees-v1",
        **evidence(clock),
        "currency": "PLN",
        "operator_verified": True,
        "valid_until": (clock() + timedelta(days=30)).isoformat(),
        "void_refunds_stake_tax": True,
        "void_refunds_entry_fees": False,
        "winnings_tax_basis": "whole-payout",
        "winnings_tax_base": "after-commission",
        **dict.fromkeys(FEE_FIELDS, "0"),
        **(fee_changes or {}),
    }
    instrument = {
        "symbol": symbol,
        "market": market,
        "product": product,
        "cluster": market,
        "feed_id": "independent:test",
        "quantity_step": "0.01",
        "fee_profile": profile["id"],
        "price_basis": "raw-unadjusted",
        "operator_verified": True,
        **evidence(clock),
    }
    if product == "isolated-linear":
        instrument.update(loss_cap="position", auto_add_margin=False, maintenance_fraction="0.005")
    if product == "bet":
        instrument["starts_at"] = (clock() + timedelta(hours=1)).isoformat()
    return {"fee_profiles": [profile], "instruments": [instrument]}


def setup(tmp_path, product="spot", market="crypto", fee_changes=None):
    clock = Clock()
    book = PaperBook(tmp_path, clock)
    book.initialize()
    config = configuration(clock, product, market, fee_changes=fee_changes)
    book.configure(config)
    return book, clock, config


def quote(clock, price="100", symbol="TEST", product="spot", previous=None, size="100", **extras):
    result = {
        "kind": "quote",
        "symbol": symbol,
        "feed_id": "independent:test",
        "bid": price,
        "ask": price,
        "bid_size": size,
        "ask_size": size,
        "currency": "PLN",
        **evidence(clock),
    }
    if product == "isolated-linear":
        result.update(
            mark=price,
            mark_low=price,
            mark_high=price,
            interval_start=previous or clock().isoformat(),
            funding_events=[],
        )
    return {**result, **extras}


def decide(book, action="open", branch="A", budget="400", leverage="1", symbol="TEST", side="long"):
    return book.decide(
        branch,
        {
            "action": action,
            "symbol": symbol,
            "side": side,
            "budget": budget,
            "leverage": leverage,
            "rationale": "Synthetic deterministic test hypothesis",
        },
        book.sequence(),
    )


def test_decision_is_sealed_before_next_quote_and_reopen_preserves_portfolios(tmp_path):
    book, clock, _ = setup(tmp_path)
    book.ingest(quote(clock))
    event = decide(book)
    assert book.state()["branches"]["A"]["positions"] == {}
    clock.advance()
    book.ingest(quote(clock, "105"))
    position = book.state()["branches"]["A"]["positions"]["TEST"]
    assert Decimal(position["entry"]) == 105
    assert book.events()[event["sequence"] - 1]["kind"] == "decision"
    expected = book.state()
    book.close()
    reopened = PaperBook(tmp_path, clock)
    assert reopened.state() == expected
    with pytest.raises(FileExistsError, match="reset"):
        reopened.initialize()
    reopened.close()


def test_unknown_expired_or_asymmetric_fees_are_not_ignored(tmp_path):
    clock = Clock()
    book = PaperBook(tmp_path, clock)
    book.initialize()
    config = configuration(clock)
    config["fee_profiles"][0]["entry_minimum_commission"] = None
    with pytest.raises(ValueError, match="unknown costs"):
        book.configure(config)
    assert book.state()["fee_profiles"] == {}
    config = configuration(
        clock,
        fee_changes={
            "entry_commission_bps": "10",
            "entry_minimum_commission": "2",
            "entry_venue_fee_bps": "5",
            "entry_fx_bps": "10",
            "exit_venue_fee_bps": "20",
        },
    )
    profile = config["fee_profiles"][0]
    assert fee_total(profile, Decimal(1000), closing=False) == Decimal("3.5")
    assert fee_total(profile, Decimal(1000)) == Decimal("2")
    book.configure(config)
    book.ingest(quote(clock))
    clock.advance(31 * 86400)
    with pytest.raises(ValueError, match="expired"):
        decide(book)
    book.close()


def test_net_profit_includes_both_sides_fees_spread_and_slippage(tmp_path):
    book, clock, _ = setup(
        tmp_path,
        fee_changes={
            "entry_minimum_commission": "2",
            "exit_minimum_commission": "3",
            "slippage_bps": "10",
        },
    )
    book.ingest(quote(clock, bid="99", ask="100"))
    decide(book)
    clock.advance()
    book.ingest(quote(clock, bid="99", ask="100"))
    position = copy.deepcopy(book.state()["branches"]["A"]["positions"]["TEST"])
    decide(book, "close")
    clock.advance()
    event = book.ingest(quote(clock, bid="110", ask="111"))
    closing = event["payload"]["trades"][0]
    quantity = Decimal(position["quantity"])
    expected = quantity * (Decimal("109.89") - Decimal("100.1")) - 5
    assert Decimal(closing["net_pnl"]) == expected
    assert Decimal(book.state()["branches"]["A"]["cash"]) == 10000 + expected
    assert Decimal(book.state()["branches"]["A"]["costs"]) == 5
    book.close()


def test_partial_fill_is_limited_by_liquidity_and_remaining_budget_is_reserved(tmp_path):
    book, clock, _ = setup(tmp_path)
    book.ingest(quote(clock))
    decide(book)
    clock.advance()
    book.ingest(quote(clock, size="1"))
    state = book.state()
    assert state["branches"]["A"]["positions"]["TEST"]["quantity"] == "1"
    assert Decimal(state["orders"][0]["remaining_budget"]) == 300
    clock.advance()
    book.ingest(quote(clock, size="1"))
    assert Decimal(book.state()["branches"]["A"]["positions"]["TEST"]["quantity"]) == 2
    decide(book, "cancel")
    assert book.state()["orders"] == []
    book.close()


@pytest.mark.parametrize(
    "alteration", ["future", "stale", "rewind", "wrong-feed", "wrong-currency"]
)
def test_bad_observations_do_not_mutate_ledger(tmp_path, alteration):
    book, clock, _ = setup(tmp_path)
    book.ingest(quote(clock))
    before = book.sequence()
    clock.advance()
    candidate = quote(clock)
    if alteration == "future":
        candidate["available_at"] = (clock() + timedelta(seconds=1)).isoformat()
    elif alteration == "stale":
        candidate["available_at"] = (clock() - timedelta(hours=1)).isoformat()
    elif alteration == "rewind":
        candidate["available_at"] = (clock() - timedelta(seconds=1)).isoformat()
    elif alteration == "wrong-feed":
        candidate["feed_id"] = "model-invented-price"
    else:
        candidate["currency"] = "USD"
    with pytest.raises(ValueError):
        book.ingest(candidate)
    assert book.sequence() == before
    book.close()


def test_stale_decision_context_and_editing_history_are_rejected(tmp_path):
    book, clock, _ = setup(tmp_path)
    book.ingest(quote(clock))
    old = book.sequence()
    decide(book, "hold")
    with pytest.raises(ValueError, match="context changed"):
        book.decide("B", {"action": "hold", "rationale": "Cannot peek ahead"}, old)
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        book.db.execute("UPDATE events SET kind='fake-win'")
    state = book.state()
    state["branches"]["A"]["cash"] = "9999999"
    book.db.execute("UPDATE state SET payload=?", (json.dumps(state),))
    book.db.commit()
    with pytest.raises(ValueError, match="outside its ledger"):
        book.state()
    book.close()


def test_diversification_counts_pending_correlated_exposure(tmp_path):
    book, clock, config = setup(tmp_path)
    for index in range(1, 5):
        instrument = {**config["instruments"][0], "symbol": f"TEST{index}"}
        book.configure({"instruments": [instrument]})
        book.ingest(quote(clock, symbol=instrument["symbol"]))
    for symbol in ("TEST1", "TEST2", "TEST3"):
        decide(book, symbol=symbol)
    with pytest.raises(ValueError, match="cluster_fraction"):
        decide(book, symbol="TEST4")
    with pytest.raises(ValueError, match="position_fraction"):
        decide(book, branch="B", symbol="TEST4", budget="501")
    book.close()


def test_isolated_liquidation_cannot_drain_other_cash_and_uses_interval_extreme(tmp_path):
    book, clock, _ = setup(tmp_path, product="isolated-linear")
    initial = quote(clock, product="isolated-linear")
    book.ingest(initial)
    decide(book, leverage="3")
    clock.advance()
    filling = quote(clock, product="isolated-linear", previous=initial["available_at"])
    book.ingest(filling)
    clock.advance()
    event = book.ingest(
        quote(clock, product="isolated-linear", previous=filling["available_at"], mark_low="40")
    )
    trade = event["payload"]["trades"][0]
    assert trade["action"] == "liquidation"
    assert Decimal(trade["net_pnl"]) == -400
    assert Decimal(book.state()["branches"]["A"]["cash"]) == 9600
    assert book.state()["branches"]["B"]["cash"] == "10000"
    book.close()


@pytest.mark.parametrize("changed", [{"loss_cap": "account"}, {"auto_add_margin": True}])
def test_unbounded_or_cross_margin_product_is_not_admitted(tmp_path, changed):
    clock = Clock()
    book = PaperBook(tmp_path, clock)
    book.initialize()
    config = configuration(clock, product="isolated-linear")
    config["instruments"][0].update(changed)
    with pytest.raises(ValueError, match="position-isolated"):
        book.configure(config)
    book.close()


def test_settled_funding_is_charged_once_at_its_actual_mark(tmp_path):
    book, clock, _ = setup(tmp_path, product="isolated-linear")
    first = quote(clock, product="isolated-linear")
    book.ingest(first)
    decide(book, leverage="3")
    clock.advance()
    second = quote(clock, product="isolated-linear", previous=first["available_at"])
    book.ingest(second)
    clock.advance()
    third = quote(
        clock,
        product="isolated-linear",
        previous=second["available_at"],
        funding_events=[{"at": clock().isoformat(), "rate_bps": "100", "mark": "100"}],
    )
    book.ingest(third)
    assert book.equity(book.state(), "A") == 9988
    decide(book, "close")
    clock.advance()
    event = book.ingest(quote(clock, product="isolated-linear", previous=third["available_at"]))
    assert Decimal(event["payload"]["trades"][0]["net_pnl"]) == -12
    assert Decimal(book.state()["branches"]["A"]["cash"]) == 9988
    book.close()


def test_sports_tax_fees_and_independent_outcome_settlement(tmp_path):
    book, clock, _ = setup(
        tmp_path,
        product="bet",
        market="sports",
        fee_changes={
            "stake_tax_bps": "1200",
            "winnings_commission_bps": "500",
            "winnings_tax_bps": "1000",
        },
    )
    book.ingest(quote(clock, "2"))
    decide(book, budget="100")
    clock.advance()
    book.ingest(quote(clock, "2"))
    assert book.state()["branches"]["A"]["positions"]["TEST"]["margin"] == "88.00"
    clock.advance(3600)
    snapshot = {
        "kind": "outcome",
        "symbol": "TEST",
        "feed_id": "independent:test",
        "result": "won",
        **evidence(clock),
    }
    event = book.ingest(snapshot)
    assert Decimal(event["payload"]["trades"][0]["payout"]) == Decimal("154.44")
    assert Decimal(book.state()["branches"]["A"]["cash"]) == Decimal("10054.44")
    with pytest.raises(ValueError, match="settlement"):
        book.ingest(snapshot)
    book.close()


def test_stock_split_preserves_value_and_dividend_uses_past_entitlement(tmp_path):
    book, clock, _ = setup(tmp_path, market="equities")
    book.ingest(quote(clock))
    decide(book)
    clock.advance()
    book.ingest(quote(clock))
    clock.advance()
    entitlement = clock().isoformat()
    book.ingest(
        {
            "kind": "corporate",
            "symbol": "TEST",
            "feed_id": "independent:test",
            "action": "split",
            "ratio": "4",
            "corporate_id": "split-1",
            "effective_at": entitlement,
            **evidence(clock),
        }
    )
    assert book.equity(book.state(), "A") == 10000
    assert Decimal(book.state()["branches"]["A"]["positions"]["TEST"]["quantity"]) == 16
    decide(book, "close")
    clock.advance()
    book.ingest(quote(clock, "25"))
    clock.advance()
    book.ingest(
        {
            "kind": "corporate",
            "symbol": "TEST",
            "feed_id": "independent:test",
            "action": "dividend",
            "corporate_id": "dividend-1",
            "effective_at": clock().isoformat(),
            "entitlement_at": entitlement,
            "amount_per_unit": "1",
            "withholding_bps": "1900",
            "currency": "PLN",
            **evidence(clock),
        }
    )
    assert Decimal(book.state()["branches"]["A"]["cash"]) == Decimal("10003.24")
    book.close()


def test_reports_contain_audited_trades_charts_and_no_real_money_promotion(tmp_path):
    book, clock, _ = setup(tmp_path)
    book.ingest(quote(clock))
    decide(book)
    clock.advance()
    book.ingest(quote(clock))
    decide(book, "close")
    clock.advance()
    book.ingest(quote(clock, "110"))
    directory = write_report(book)
    report = json.loads((directory / "report.json").read_text())
    assert report["real_money_ready"] is False
    assert Decimal(report["branches"]["A"]["net_pnl_before_personal_tax"]) == 40
    assert "A buy" in (directory / "report.html").read_text()
    assert "A close" in (directory / "report.html").read_text()
    assert "buy" in (directory / "trades.csv").read_text()
    assert report["audit_tail_sha256"] == book.events()[-1]["event_hash"]
    clock.advance(301)
    assert summarize(book)["stale_symbols"] == ["TEST"]
    book.close()


def test_local_daily_and_weekly_schedule_is_once_per_period_and_uses_warsaw(tmp_path):
    book, clock, _ = setup(tmp_path)
    clock.now = datetime(2026, 10, 11, 18, tzinfo=UTC)  # Sunday 20:00 Warsaw.
    first = scheduled_reports(book)
    assert len(first) == 2
    assert scheduled_reports(book) == []
    clock.advance(86400)
    assert len(scheduled_reports(book)) == 1
    book.close()


def test_coinbase_collector_uses_public_book_and_archived_nbp_conversion(tmp_path, monkeypatch):
    book, clock, config = setup(tmp_path)
    instrument = {**config["instruments"][0], "symbol": "BTC", "feed_id": "coinbase:BTC-USD"}
    book.configure({"instruments": [instrument]})
    urls = []

    def fetch(root, url, *args):
        urls.append(url)
        if "nbp.pl" in url:
            value = {"rates": [{"mid": "4", "effectiveDate": "2026-10-07"}]}
        else:
            value = {"bids": [["100", "2", 1]], "asks": [["101", "3", 1]], "sequence": 12}
        return value, hashlib.sha256(json.dumps(value).encode()).hexdigest(), clock().isoformat()

    monkeypatch.setattr(paper_feeds, "public_json", fetch)
    paper_feeds.poll_crypto(book)
    captured = book.state()["quotes"]["BTC"]
    assert captured["bid"] == "400" and captured["ask"] == "404"
    assert captured["fx"]["note"].startswith("NBP reference")
    assert all("/orders" not in url for url in urls)
    book.close()


def test_financial_rnd_parent_can_reject_every_worker_and_hold(tmp_path, monkeypatch):
    book, clock, _ = setup(tmp_path)
    book.ingest(quote(clock))
    profile = {"runtime": {"model_version": "test-gemma"}}
    monkeypatch.setattr(paper_agents, "helper_client", lambda *a: SimpleNamespace())
    seen = []

    def research(client, branch, job, observations, root):
        seen.append((branch, job["role"], observations[0]["paper_context"]["goal"]))
        return {
            "role": job["role"],
            "observation": "No verified edge",
            "hypothesis": "Maybe",
            "suggested_test": "Test future data",
            "exercises": [],
        }

    monkeypatch.setattr(paper_agents, "research_task", research)
    monkeypatch.setattr(
        paper_agents,
        "review_research",
        lambda *a: {"useful_indices": [], "conclusion": "Reject all"},
    )
    monkeypatch.setattr(
        paper_agents,
        "native_turn",
        lambda *a, **kw: {
            "content": json.dumps(
                {
                    "action": "hold",
                    "symbol": "",
                    "side": "long",
                    "budget": 0,
                    "leverage": 1,
                    "rationale": "Hold cash until a falsifiable edge exists",
                    "accepted_research": [],
                }
            )
        },
    )
    result = paper_agents.paper_round(book, profile, research_rounds=1)
    assert len(result) == 2 and len(seen) == 4
    assert {branch for branch, role, goal in seen} == {"A", "B"}
    assert all("No future data" in goal for branch, role, goal in seen)
    assert book.state()["orders"] == []
    book.close()


def test_tester_stress_is_read_only_and_uses_host_fees(tmp_path):
    book, clock, _ = setup(tmp_path, product="isolated-linear")
    book.ingest(quote(clock, product="isolated-linear"))
    before = book.sequence()
    # Tool's real clock is later than the fixture: keep the test clock explicit.
    factory = PaperBook
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(paper_tools, "PaperBook", lambda root: factory(root, clock))
    try:
        result = paper_tools.execute(
            tmp_path,
            "A",
            "paper_test_position",
            {"symbol": "TEST", "budget": 400, "leverage": 3, "side": "long", "adverse_bps": 6000},
        )
        assert result["hypothetical"] is True and result["liquidated"] is True
        assert Decimal(result["stress_net"]) == -400
        assert book.sequence() == before and book.state()["orders"] == []
        with pytest.raises(ValueError, match="Unknown"):
            paper_tools.execute(
                tmp_path,
                "A",
                "paper_test_position",
                {
                    "symbol": "FAKE",
                    "budget": 400,
                    "leverage": 3,
                    "side": "long",
                    "adverse_bps": 6000,
                },
            )
    finally:
        monkeypatch.undo()
        book.close()


def test_void_refunds_are_itemized_and_costs_reconcile(tmp_path):
    book, clock, _ = setup(
        tmp_path,
        product="bet",
        market="sports",
        fee_changes={"entry_minimum_commission": "2", "stake_tax_bps": "1200"},
    )
    book.ingest(quote(clock, "2"))
    decide(book, budget="100")
    clock.advance()
    book.ingest(quote(clock, "2"))
    clock.advance(3600)
    book.ingest(
        {
            "kind": "outcome",
            "symbol": "TEST",
            "feed_id": "independent:test",
            "result": "void",
            **evidence(clock),
        }
    )
    state = book.state()
    assert Decimal(state["branches"]["A"]["cash"]) == 9998
    assert Decimal(state["branches"]["A"]["costs"]) == 2
    book.close()


def test_winnings_below_declared_tax_threshold_are_not_flat_taxed(tmp_path):
    book, clock, _ = setup(
        tmp_path,
        product="bet",
        market="sports",
        fee_changes={"winnings_tax_bps": "1000", "winnings_tax_threshold": "250"},
    )
    book.ingest(quote(clock, "2"))
    decide(book, budget="100")
    clock.advance()
    book.ingest(quote(clock, "2"))
    clock.advance(3600)
    event = book.ingest(
        {
            "kind": "outcome",
            "symbol": "TEST",
            "feed_id": "independent:test",
            "result": "won",
            **evidence(clock),
        }
    )
    assert Decimal(event["payload"]["trades"][0]["payout"]) == 200
    book.close()


def test_reports_preserve_period_profit_and_sources_observed_after_publication(tmp_path):
    book, clock, _ = setup(tmp_path)
    publication = (clock() - timedelta(days=60)).isoformat()
    event = book.ingest(
        {
            "kind": "news",
            "category": "quarterly",
            "excerpt": "Quarterly report, fiscal end predates publication",
            **evidence(clock),
            "available_at": publication,
        }
    )
    assert event["payload"]["observation"]["observed_at"] == clock().isoformat()
    assert book.context("A")["sequence"] == event["sequence"]
    report = summarize(book, "daily")
    assert report["branches"]["A"]["period_marked_pnl"] == "0"
    book.close()


def test_loop_lease_rejects_duplicate_workers(tmp_path):
    with paper_agents.loop_lease(tmp_path):
        with pytest.raises(BlockingIOError):
            with paper_agents.loop_lease(tmp_path):
                pytest.fail("A duplicate loop could duplicate decisions and reports")


def test_cli_initializes_and_reports_without_server_profile(tmp_path, monkeypatch, capsys):
    from rlm.v100.cli import main

    monkeypatch.setattr("sys.argv", ["v100-continual", "--root", str(tmp_path), "paper-init"])
    main()
    assert "PAPER LAB READY" in capsys.readouterr().out
    assert not (tmp_path / "research/v100.toml").exists()
    monkeypatch.setattr("sys.argv", ["v100-continual", "--root", str(tmp_path), "paper-report"])
    main()
    assert "Paper report:" in capsys.readouterr().out
    assert (tmp_path / "research/paper/latest-report.json").exists()
