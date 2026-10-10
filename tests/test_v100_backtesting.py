import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rlm.v100 import backtesting, research_tools


def candles():
    return [[3600 * index, 90, 200, 100 + index, 101 + index, 1000] for index in range(100)]


def test_backtest_selects_parameters_only_before_chronological_split(tmp_path, monkeypatch):
    data = candles()
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=3600"
    monkeypatch.setattr(research_tools, "download_page", lambda target: (url, json.dumps(data)))
    args = {
        "price_url": url,
        "event_url": "",
        "rule": "momentum",
        "fee_bps": 40,
        "slippage_bps": 10,
    }
    first = backtesting.run(tmp_path, "A", args)
    for row in data[70:]:
        row[3] *= 10
        row[4] *= 10
        row[1] *= 10
        row[2] *= 10
    second = backtesting.run(tmp_path, "B", args)
    assert (
        first["selected_lookback_on_training_only"] == second["selected_lookback_on_training_only"]
    )
    assert first["split_index"] == 70
    assert first["double_cost_stress"]["net_return"] < first["development_test"]["net_return"]
    saved = json.loads(Path(first["report"]).read_text())
    assert all(
        row["signal_available_at"] <= row["fill_at"] for row in saved["development_test"]["trace"]
    )
    assert "not verified financial training labels" in first["scope"]


def test_future_prices_and_unpublished_events_cannot_change_earlier_signals():
    bars = backtesting.bars_from_source(
        "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=3600",
        json.dumps(candles()),
        datetime.now(UTC),
    )
    original = backtesting.simulate(bars, 21, 100, "momentum", 5, 40, 10)
    changed = copy.deepcopy(bars)
    changed[80]["close"] *= 100
    after = backtesting.simulate(changed, 21, 100, "momentum", 5, 40, 10)
    assert original["trace"][:59] == after["trace"][:59]
    no_events = backtesting.simulate(bars, 21, 100, "buy_hold", 5, 40, 10, [10**15])
    assert no_events["fills"] == 0 and no_events["net_return"] == 0


def test_duplicate_or_ambiguous_bars_are_rejected():
    data = candles()
    data[-1][0] = data[-2][0]
    with pytest.raises(ValueError, match="overlap"):
        backtesting.bars_from_source(
            "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=3600",
            json.dumps(data),
            datetime.now(UTC),
        )
