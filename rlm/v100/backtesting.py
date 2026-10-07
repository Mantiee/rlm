"""Bounded chronological spot-price experiments, never certified profit labels."""

import hashlib
import json
import math
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from rlm.v100.common import atomic_json
from rlm.v100.paper import timestamp


def bars_from_source(url: str, body: str, fetched_at: datetime) -> list[dict]:
    data = json.loads(body)
    origin = urlparse(url)
    if not isinstance(data, list) or not 60 <= len(data) <= 3000:
        raise ValueError("Backtest needs 60-3000 timestamped closed price bars")
    bars = []
    coinbase = origin.hostname == "api.exchange.coinbase.com" and origin.path.endswith("/candles")
    interval = int(parse_qs(origin.query).get("granularity", ["300"])[0]) if coinbase else None
    if coinbase and interval not in (60, 300, 900, 3600, 21600, 86400):
        raise ValueError("Unsupported Coinbase candle interval")
    for row in data:
        if coinbase:
            if not isinstance(row, list) or len(row) != 6:
                raise ValueError("Invalid public Coinbase OHLCV candle")
            opened, closed, open_price, close_price = (
                float(row[0]),
                float(row[0]) + interval,
                float(row[3]),
                float(row[4]),
            )
        else:
            opened, closed = (
                timestamp(row["open_time"]).timestamp(),
                timestamp(row["available_at"]).timestamp(),
            )
            open_price, close_price = float(row["open"]), float(row["close"])
        if (
            not all(math.isfinite(value) for value in (opened, closed, open_price, close_price))
            or open_price <= 0
            or close_price <= 0
            or closed <= opened
        ):
            raise ValueError("Invalid bar availability or price")
        if closed <= fetched_at.timestamp():
            bars.append(
                {
                    "open_time": opened,
                    "available_at": closed,
                    "open": open_price,
                    "close": close_price,
                }
            )
    bars.sort(key=lambda row: row["open_time"])
    if len(bars) < 60 or any(
        left["available_at"] > right["open_time"] or left["open_time"] >= right["open_time"]
        for left, right in zip(bars, bars[1:], strict=False)
    ):
        raise ValueError("Bars overlap, have ambiguous availability, or lack sufficient history")
    return bars


def simulate(
    bars: list[dict],
    start: int,
    stop: int,
    rule: str,
    lookback: int,
    fee_bps: float,
    slippage_bps: float,
    events: list[float] | None = None,
) -> dict:
    cash, units, peak, drawdown, fills = 1.0, 0.0, 1.0, 0.0, 0
    fee, slip = fee_bps / 10000, slippage_bps / 10000
    trace = []
    for index in range(start, stop):
        # Only a completed PREVIOUS bar can choose the next opening fill.
        previous = index - 1
        delta = bars[previous]["close"] - bars[previous - lookback]["close"]
        active = rule == "buy_hold" or (delta > 0 if rule == "momentum" else delta < 0)
        known_at = bars[previous]["available_at"]
        if events is not None:
            window = bars[previous - lookback]["available_at"]
            active = active and any(window <= at <= known_at for at in events)
        price = bars[index]["open"]
        if active and not units:
            units, cash = cash / (price * (1 + slip) * (1 + fee)), 0.0
            fills += 1
        elif not active and units:
            cash, units = units * price * (1 - slip) * (1 - fee), 0.0
            fills += 1
        equity = cash + units * bars[index]["close"] * (1 - slip) * (1 - fee)
        peak, drawdown = max(peak, equity), max(drawdown, 1 - equity / max(peak, equity))
        trace.append(
            {
                "signal_available_at": known_at,
                "fill_at": bars[index]["open_time"],
                "marked_equity": equity,
            }
        )
    if units:
        cash = units * bars[stop - 1]["close"] * (1 - slip) * (1 - fee)
        fills += 1  # Forced sample-end liquidation, explicitly reported.
    return {
        "net_return": cash - 1,
        "max_sampled_drawdown": drawdown,
        "fills": fills,
        "sample_end_liquidation": True,
        "trace": trace,
    }


def run(root: Path, branch: str, arguments: dict) -> dict:
    from rlm.v100.research_tools import download_page

    if set(arguments) != {"price_url", "event_url", "rule", "fee_bps", "slippage_bps"} or arguments[
        "rule"
    ] not in ("momentum", "mean_reversion", "buy_hold"):
        raise ValueError("Invalid backtest request")
    for field in ("fee_bps", "slippage_bps"):
        value = arguments[field]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1000:
            raise ValueError("Backtest costs must be explicit bounded assumptions")
    price_url, body = download_page(arguments["price_url"])
    now = datetime.now(UTC)
    bars = bars_from_source(price_url, body, now)
    sources = [
        {"url": price_url, "body": body, "sha256": hashlib.sha256(body.encode()).hexdigest()}
    ]
    events = None
    if arguments["event_url"]:
        event_url, event_body = download_page(arguments["event_url"])
        rows = json.loads(event_body)
        if not isinstance(rows, list) or not 1 <= len(rows) <= 512:
            raise ValueError("Event series needs 1-512 published_at/available_at records")
        events = []
        for row in rows:
            published, available = timestamp(row["published_at"]), timestamp(row["available_at"])
            if available < published or available > now:
                raise ValueError(
                    "Event availability cannot precede publication or follow retrieval"
                )
            events.append(available.timestamp())
        sources.append(
            {
                "url": event_url,
                "body": event_body,
                "sha256": hashlib.sha256(event_body.encode()).hexdigest(),
            }
        )
    from rlm.v100.protection import file_hash

    key = hashlib.sha256(
        json.dumps(
            {"parameters": arguments, "bars": bars, "events": events}, sort_keys=True, default=str
        ).encode()
    ).hexdigest()
    cache = root / "research/backtests/cache" / (key + ".json")
    if cache.exists():
        saved = json.loads(cache.read_text())
        path = Path(saved["report"]).resolve()
        if (
            not path.is_relative_to((root / "research/backtests").resolve())
            or file_hash(path) != saved["sha256"]
        ):
            raise ValueError("Cached backtest report changed")
        previous = json.loads(path.read_text())
        for section in ("training", "development_test", "cost_stress"):
            if isinstance(previous.get(section), dict):
                previous[section].pop("trace", None)
        return {
            **previous,
            "cache_reused": True,
            "requested_branch": branch,
            "cache_note": "Same data and parameters are not independent evidence",
        }
    folds = []
    bounds = [int(len(bars) * fraction) for fraction in (0.5, 0.65, 0.8)] + [len(bars)]
    for start, end in zip(bounds, bounds[1:], strict=False):
        past = {
            lookback: simulate(
                bars,
                21,
                start,
                arguments["rule"],
                lookback,
                arguments["fee_bps"],
                arguments["slippage_bps"],
                events,
            )
            for lookback in (5, 10, 20)
        }
        lookback = max(past, key=lambda length: past[length]["net_return"])
        fold = {"start": start, "end": end, "lookback_selected_only_on_past": lookback}
        for label, rule, multiplier in (
            ("test", arguments["rule"], 1),
            ("double_cost", arguments["rule"], 2),
            ("buy_hold", "buy_hold", 1),
        ):
            outcome = simulate(
                bars,
                start,
                end,
                rule,
                lookback,
                arguments["fee_bps"] * multiplier,
                arguments["slippage_bps"] * multiplier,
                events if rule != "buy_hold" else None,
            )
            fold[label] = {k: v for k, v in outcome.items() if k != "trace"}
        fold["cash_net_return"] = 0
        folds.append(fold)
    split = int(len(bars) * 0.7)
    candidates = [5, 10, 20]
    training = {
        lookback: simulate(
            bars,
            21,
            split,
            arguments["rule"],
            lookback,
            arguments["fee_bps"],
            arguments["slippage_bps"],
            events,
        )
        for lookback in candidates
    }
    chosen = max(candidates, key=lambda lookback: training[lookback]["net_return"])
    test = simulate(
        bars,
        split,
        len(bars),
        arguments["rule"],
        chosen,
        arguments["fee_bps"],
        arguments["slippage_bps"],
        events,
    )
    stress = simulate(
        bars,
        split,
        len(bars),
        arguments["rule"],
        chosen,
        arguments["fee_bps"] * 2,
        arguments["slippage_bps"] * 2,
        events,
    )
    baseline = simulate(
        bars, split, len(bars), "buy_hold", chosen, arguments["fee_bps"], arguments["slippage_bps"]
    )
    directory = root / "research/backtests" / ("run-" + uuid.uuid4().hex[:12])
    directory.mkdir(parents=True)
    for index, source in enumerate(sources):
        (directory / f"source-{index}.json").write_text(source["body"])
    result = {
        "branch": branch,
        "walk_forward": folds,
        "cache_reused": False,
        "fetched_at": now.isoformat(),
        "parameters": arguments,
        "sources": [
            {key: value for key, value in source.items() if key != "body"} for source in sources
        ],
        "bars": len(bars),
        "split_index": split,
        "selected_lookback_on_training_only": chosen,
        "development_test": test,
        "double_cost_stress": stress,
        "buy_hold_baseline": baseline,
        "scope": "Exploratory historical spot-price simulation, not verified financial training labels or proof of an income edge",
        "limitations": [
            "Cost assumptions are not certified broker fees",
            "No intrabar liquidity, tax, leverage, shorting or corporate-action validation",
            "Source availability timestamps are vendor claims; archive retrieval does not prove historical availability",
            "Repeated development tests can overfit; require future independent paper outcomes",
            "Sports odds and settlement need a dedicated simulator; do not pass odds as stock prices",
        ],
        "report": str(directory / "report.json"),
    }
    atomic_json(directory / "report.json", result)
    atomic_json(
        cache,
        {"report": str(directory / "report.json"), "sha256": file_hash(directory / "report.json")},
    )
    return {
        key: (
            {subkey: subvalue for subkey, subvalue in value.items() if subkey != "trace"}
            if key in ("development_test", "double_cost_stress", "buy_hold_baseline")
            else value
        )
        for key, value in result.items()
    }
