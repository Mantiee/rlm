import copy
import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from rlm.v100 import market_adapters


def odds_config():
    return {
        "id": "sports-one",
        "provider": "sports-odds",
        "free_only": True,
        "settings": {
            "sport": "basketball_nba",
            "region": "eu",
            "event_id": "abc",
            "bookmaker": "book",
            "outcome": "Home",
            "symbol": "HOME-BET",
            "feed_id": "odds:abc",
            "paper_stake_cap": "10",
            "settlement_rule": "research-only",
        },
    }


def test_free_adapter_never_accepts_paid_or_arbitrary_origin(tmp_path):
    good = {
        "id": "derivative",
        "provider": "bybit-linear",
        "free_only": True,
        "settings": {"contract": "BTCUSDT"},
    }
    market_adapters.register(tmp_path, good)
    for changed in (
        {**good, "free_only": False},
        {**good, "provider": "paid-history"},
        {**good, "url": "https://example.org"},
        {**good, "settings": {"contract": "../../private"}},
    ):
        with pytest.raises(ValueError):
            market_adapters.register(tmp_path, changed)


def test_stock_round_lots_and_source_timestamp_are_preserved():
    cfg = {
        "provider": "alpaca-iex",
        "settings": {
            "ticker": "EXAMPLE",
            "symbol": "EX",
            "feed_id": "iex:EX",
            "round_lot_shares": 100,
        },
    }
    data = {
        "quotes": {"EXAMPLE": {"t": "2026-10-07T10:00:00Z", "bp": 10, "ap": 11, "bs": 2, "as": 3}}
    }
    row = market_adapters.normalize(
        cfg,
        data,
        "a" * 64,
        "2026-10-07T10:00:01Z",
        "https://data.alpaca.markets/v2/stocks/quotes/latest",
        "PLN",
    )[0]
    assert row["bid_size"] == "200" and row["ask_size"] == "300"
    assert row["available_at"] == data["quotes"]["EXAMPLE"]["t"] and row["currency"] == "USD"
    with pytest.raises(ValueError, match="future"):
        market_adapters.normalize(
            cfg, data, "a" * 64, "2026-10-07T09:59:00Z", "https://example.org", "USD"
        )


def test_sports_uses_market_timestamp_not_observation_or_bookmaker_timestamp():
    cfg = odds_config()
    data = [
        {
            "id": "abc",
            "commence_time": "2026-10-07T18:00:00Z",
            "bookmakers": [
                {
                    "key": "book",
                    "last_update": "2026-10-07T10:00:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2026-10-07T09:59:30Z",
                            "outcomes": [{"name": "Home", "price": 1.9}],
                        }
                    ],
                }
            ],
        }
    ]
    row = market_adapters.normalize(
        cfg, data, "b" * 64, "2026-10-07T10:00:01Z", "https://example.org", "PLN"
    )[0]
    assert row["available_at"] == "2026-10-07T09:59:30Z" and row["kind"] == "news"
    assert "decimal_odds" in row["excerpt"]


@pytest.mark.parametrize(
    "home,away,outcome,rule,result",
    [
        (2, 1, "Home", "final-score-h2h-draw", "won"),
        (1, 2, "Home", "final-score-h2h-draw", "lost"),
        (1, 1, "Draw", "final-score-h2h-draw", "won"),
        (1, 1, "Home", "final-score-h2h-void-tie", "void"),
    ],
)
def test_sports_score_rules_are_explicit_and_causal(home, away, outcome, rule, result):
    cfg = odds_config()
    cfg["settings"].update(outcome=outcome, settlement_rule=rule)
    score = [
        {
            "id": "abc",
            "completed": True,
            "home_team": "Home",
            "away_team": "Away",
            "last_update": "2026-10-07T20:00:00Z",
            "scores": [{"name": "Home", "score": str(home)}, {"name": "Away", "score": str(away)}],
        }
    ]
    assert (
        market_adapters.settle(cfg, score, "c" * 64, "2026-10-07T20:00:01Z", "https://example.org")[
            0
        ]["result"]
        == result
    )
    with pytest.raises(ValueError, match="future"):
        market_adapters.settle(cfg, score, "c" * 64, "2026-10-07T19:00:00Z", "https://example.org")


def test_derivative_ticker_never_fabricates_interval_or_settled_funding():
    cfg = {"provider": "bybit-linear", "settings": {"contract": "ETHUSDT"}}
    data = {
        "retCode": 0,
        "result": {"list": [{"symbol": "ETHUSDT", "fundingRate": "0.001", "markPrice": "1200"}]},
    }
    row = market_adapters.normalize(
        cfg,
        data,
        "a" * 64,
        "2026-10-07T10:00:00Z",
        "https://api.bybit.com/v5/market/tickers",
        "PLN",
    )[0]
    assert row["kind"] == "news" and "not settled funding" in row["excerpt"]
    assert "mark_low" not in row and "funding_events" not in row


def test_monthly_sports_quota_blocks_network_and_never_prints_key(tmp_path, monkeypatch):
    monkeypatch.setenv("V100_ODDS_KEY", "private-secret-value")
    folder = tmp_path / "research/market-adapters"
    folder.mkdir(parents=True)
    identity = "sports-odds:" + datetime.now(UTC).strftime("%Y-%m")
    (folder / "quota.json").write_text(json.dumps({identity: 100}))
    with pytest.raises(ValueError, match="quota exhausted") as error:
        market_adapters.request(tmp_path, "sports-odds", odds_config()["settings"])
    assert "private-secret" not in str(error.value)


def test_bookmaker_rules_are_not_certified_by_model_registration(tmp_path):
    cfg = odds_config()
    cfg["settings"]["settlement_rule"] = "final-score-h2h-draw"
    with pytest.raises((ValueError, KeyError), match="Initialize|matching|state"):
        market_adapters.register(tmp_path, cfg)
    changed = copy.deepcopy(cfg)
    changed["settings"]["settlement_rule"] = "model-says-safe"
    with pytest.raises(ValueError, match="explicitly documented"):
        market_adapters.register(tmp_path, changed)


def test_completed_game_can_settle_after_odds_disappear(tmp_path, monkeypatch):
    cfg = odds_config()
    cfg["settings"]["settlement_rule"] = "final-score-h2h-draw"
    folder = tmp_path / "research/market-adapters"
    folder.mkdir(parents=True)
    (folder / "sports-one.json").write_text(json.dumps(cfg))
    score = [
        {
            "id": "abc",
            "completed": True,
            "home_team": "Home",
            "away_team": "Away",
            "last_update": "2026-01-02T20:00:00Z",
            "scores": [{"name": "Home", "score": "2"}, {"name": "Away", "score": "1"}],
        }
    ]
    calls = []

    def request(root, provider, settings, scores=False):
        calls.append(scores)
        assert scores, "Odds have disappeared for the completed event"
        return score, "c" * 64, "2026-01-02T20:00:01Z", "https://example.org/scores"

    monkeypatch.setattr(market_adapters, "request", request)
    book = SimpleNamespace(
        root=tmp_path,
        state=lambda: {
            "outcomes": {},
            "instruments": {"HOME-BET": {"starts_at": "2026-01-02T18:00:00Z"}},
        },
        ingest=lambda row: row,
    )
    result = market_adapters.poll(book)
    assert calls == [True] and result[0]["result"] == "won"


def test_malformed_mapping_is_deferred_without_stopping_other_sources(tmp_path, monkeypatch):
    folder = tmp_path / "research/market-adapters"
    folder.mkdir(parents=True)
    (folder / "broken.json").write_text("[]")
    events = []
    monkeypatch.setattr(
        market_adapters.ActivityLog, "write", lambda self, *args, **kw: events.append(args)
    )
    assert market_adapters.poll(SimpleNamespace(root=tmp_path)) == []
    assert events[0][1] == "free-feed-deferred"


def test_market_poll_rotates_four_sources_and_paces_repeated_requests(tmp_path, monkeypatch):
    folder = tmp_path / "research/market-adapters"
    folder.mkdir(parents=True)
    for index in range(6):
        (folder / f"source-{index}.json").write_text(
            json.dumps(
                {
                    "id": str(index),
                    "provider": "bybit-linear",
                    "settings": {"contract": "BTCUSDT"},
                }
            )
        )
    calls = []

    def unavailable(root, provider, settings):
        calls.append(provider)
        raise ValueError("temporarily unavailable")

    monkeypatch.setattr(market_adapters, "request", unavailable)
    monkeypatch.setattr(market_adapters.ActivityLog, "write", lambda *args, **kw: None)
    book = SimpleNamespace(root=tmp_path)
    market_adapters.poll(book)
    assert len(calls) == 4
    market_adapters.poll(book)
    assert len(calls) == 6
    market_adapters.poll(book)
    assert len(calls) == 6
