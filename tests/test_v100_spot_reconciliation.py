"""Conservative, audited repair of changed public paper quantity minima."""

import copy
import hashlib
import importlib
import json
from datetime import timedelta
from pathlib import Path

import pytest

from rlm.v100.paper import sha
from tests.test_v100_paper import setup


@pytest.fixture
def recovery(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    module = importlib.import_module("reconcile_synta_spot_rules")
    book, clock, _ = setup(tmp_path)
    with book.db:
        state = book.state()
        state["instruments"]["TEST"].update(
            feed_id="kraken:XBTUSD",
            quote_currency="USD",
            quantity_step="0.00000001",
            minimum_quantity="0.00005",
            minimum_notional_quote="0.5",
        )
        book.record(state, "test-fixture", {})
    publication = {
        "error": [],
        "result": {
            "XXBTZUSD": {
                "altname": "XBTUSD",
                "base": "XXBT",
                "quote": "ZUSD",
                "status": "online",
                "lot_decimals": 8,
                "ordermin": "0.00006",
                "costmin": "0.5",
            }
        },
    }
    body = json.dumps(publication).encode()
    digest = hashlib.sha256(body).hexdigest()
    snapshot = tmp_path / "research/paper/sources" / (digest + ".json")
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(body)
    evidence = {
        "source_url": "https://api.kraken.com/0/public/AssetPairs?pair=XBTUSD",
        "source_sha256": digest,
        "available_at": clock().isoformat(),
    }
    yield module, book, clock, publication, evidence, snapshot
    book.close()


def test_update_records_history_and_preserves_accounts_fees_risk(recovery):
    module, book, _, publication, evidence, _ = recovery
    before = copy.deepcopy(book.state())
    proposal = module.propose(before, publication)
    result = module.reconcile(book, publication, evidence, sha(proposal))
    after = book.state()
    assert result["state"] == "reconciled"
    assert after["instruments"]["TEST"]["minimum_quantity"] == "0.00006"
    for key in ("branches", "fee_profiles", "risk", "orders", "quotes"):
        assert after[key] == before[key]
    assert book.events()[-1]["kind"] == "spot-rule-reconciliation"
    sequence = book.sequence()
    assert module.reconcile(book, publication, evidence, sha(proposal))["state"] == "unchanged"
    assert book.sequence() == sequence


@pytest.mark.parametrize(
    "hazard", ["position", "order", "concurrent", "corrupt", "stale", "future"]
)
def test_refusal_is_atomic_and_retains_prior_history(recovery, hazard):
    module, book, clock, publication, evidence, snapshot = recovery
    expected = sha(module.propose(book.state(), publication))
    if hazard in ("position", "order"):
        with book.db:
            state = book.state()
            if hazard == "position":
                state["branches"]["A"]["positions"]["TEST"] = {"quantity": "1"}
            else:
                state["orders"].append({"symbol": "TEST"})
            book.record(state, "test-fixture", {})
    elif hazard == "concurrent":
        expected = "0" * 64
    elif hazard == "corrupt":
        snapshot.write_text("{}")
    elif hazard == "stale":
        clock.advance(301)
    elif hazard == "future":
        evidence["available_at"] = (clock() + timedelta(seconds=1)).isoformat()
    before, sequence = book.state(), book.sequence()
    with pytest.raises(ValueError):
        module.reconcile(book, publication, evidence, expected)
    assert book.state() == before
    assert book.sequence() == sequence


@pytest.mark.parametrize(
    "field,value",
    [
        ("ordermin", "0.00001"),
        ("costmin", "1"),
        ("lot_decimals", 7),
        ("base", "FOREIGN"),
        ("status", "cancel_only"),
    ],
)
def test_other_rule_changes_or_relaxation_require_manual_migration(recovery, field, value):
    module, book, _, publication, _, _ = recovery
    publication["result"]["XXBTZUSD"][field] = value
    with pytest.raises(ValueError):
        module.propose(book.state(), publication)


def test_unrelated_watchlist_instrument_is_preserved(recovery):
    module, book, _, publication, evidence, _ = recovery
    with book.db:
        state = book.state()
        state["instruments"]["OTHER"] = {
            **state["instruments"]["TEST"],
            "feed_id": "kraken:OTHERUSD",
        }
        book.record(state, "test-fixture", {})
    before = book.state()["instruments"]["OTHER"]
    proposal = module.propose(book.state(), publication)
    module.reconcile(book, publication, evidence, sha(proposal))
    assert book.state()["instruments"]["OTHER"] == before
