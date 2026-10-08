"""Read-only free data adapters; no broker orders or subscription endpoints.

Provider data is evidence, not account eligibility or certified transaction
costs. Stocks use IEX only, sports archive current odds, and derivative tickers
are research until a feed proves complete adverse marks and settled funding.
"""

import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import requests

from rlm.v100.activity import ActivityLog
from rlm.v100.common import atomic_json
from rlm.v100.paper import identifier, number, timestamp

DOCUMENTATION = {
    "alpaca-iex": "https://docs.alpaca.markets/us/reference/stocklatestquotes-1",
    "sports-odds": "https://the-odds-api.com/liveapi/guides/v4/",
    "bybit-linear": "https://bybit-exchange.github.io/docs/v5/market/tickers",
}


def register(root: Path, configuration: dict) -> dict:
    if (
        set(configuration) != {"id", "provider", "settings", "free_only"}
        or configuration["free_only"] is not True
    ):
        raise ValueError("Adapter requires an explicit free-only configuration")
    identity = identifier(configuration["id"])
    provider = configuration["provider"]
    if provider not in DOCUMENTATION or not isinstance(configuration["settings"], dict):
        raise ValueError("Unknown free market adapter")
    settings = configuration["settings"]
    if provider == "alpaca-iex":
        if set(settings) != {"ticker", "symbol", "feed_id", "round_lot_shares"} or not re.fullmatch(
            r"[A-Z][A-Z0-9.-]{0,15}", settings["ticker"]
        ):
            raise ValueError("IEX requires ticker, registered feed and documented round-lot size")
        if (
            type(settings["round_lot_shares"]) is not int
            or not 1 <= settings["round_lot_shares"] <= 10000
        ):
            raise ValueError("Unknown round-lot size cannot become assumed liquidity")
    elif provider == "sports-odds":
        required = {
            "sport",
            "region",
            "event_id",
            "bookmaker",
            "outcome",
            "symbol",
            "feed_id",
            "paper_stake_cap",
            "settlement_rule",
        }
        if set(settings) != required or settings["region"] not in ("eu", "uk", "us", "au"):
            raise ValueError("Sports adapter needs one exact event/bookmaker/outcome and region")
        if settings["settlement_rule"] not in (
            "research-only",
            "final-score-h2h-draw",
            "final-score-h2h-void-tie",
        ):
            raise ValueError("A bookmaker settlement rule must be explicitly documented")
        if not re.fullmatch(r"[a-z0-9_]{1,80}", settings["sport"]) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,128}", settings["event_id"]
        ):
            raise ValueError("Invalid sports/event identifier")
        number(settings["paper_stake_cap"], positive=True)
    else:
        if set(settings) != {"contract"} or not re.fullmatch(
            r"[A-Z0-9]{2,30}", settings["contract"]
        ):
            raise ValueError("Derivative research requires one public linear contract")
    if any(not isinstance(v, (str, int)) or isinstance(v, bool) for v in settings.values()):
        raise ValueError("Adapter settings contain unsupported values")
    if provider != "bybit-linear":
        from rlm.v100.paper import PaperBook

        book = PaperBook(root)
        try:
            instrument = book.state()["instruments"].get(settings["symbol"])
            if (provider == "alpaca-iex" or settings["settlement_rule"] != "research-only") and (
                not instrument
                or instrument["feed_id"] != settings["feed_id"]
                or instrument["market"] != ("equities" if provider == "alpaca-iex" else "sports")
                or instrument["product"] != ("spot" if provider == "alpaca-iex" else "bet")
            ):
                raise ValueError(
                    "Execution adapter needs independently registered matching market/feed rules"
                )
            if provider == "sports-odds" and settings["settlement_rule"] != "research-only":
                if any(
                    instrument.get(key) != settings[key]
                    for key in ("event_id", "bookmaker", "outcome", "settlement_rule")
                ):
                    raise ValueError(
                        "Sports mapping differs from operator-verified event/grading rules"
                    )
        finally:
            book.close()
    folder = root / "research/market-adapters"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (identity + ".json")
    if path.exists() and json.loads(path.read_text()) != configuration:
        raise ValueError("Market adapter mappings are immutable; use a new ID")
    if not path.exists() and len(list(folder.glob("*.json"))) >= 16:
        raise ValueError("At most sixteen market adapters")
    atomic_json(path, configuration)
    return {
        "id": identity,
        "provider": provider,
        "documentation": DOCUMENTATION[provider],
        "fees_changed": False,
        "credentials": "Operator supplies V100_ALPACA_KEY/V100_ALPACA_SECRET or V100_ODDS_KEY in the controller environment; not in model-visible configuration",
    }


def request(
    root: Path, provider: str, settings: dict, scores: bool = False
) -> tuple[object, str, str, str]:
    headers, params = {}, {}
    if provider == "alpaca-iex":
        url = "https://data.alpaca.markets/v2/stocks/quotes/latest"
        headers = {
            "APCA-API-KEY-ID": os.environ.get("V100_ALPACA_KEY", ""),
            "APCA-API-SECRET-KEY": os.environ.get("V100_ALPACA_SECRET", ""),
        }
        if not all(headers.values()):
            raise ValueError("Free IEX credentials unavailable")
        params = {"symbols": settings["ticker"], "feed": "iex"}
    elif provider == "sports-odds":
        key = os.environ.get("V100_ODDS_KEY", "")
        if not key:
            raise ValueError("Free sports data key unavailable")
        endpoint = "scores" if scores else "odds"
        url = "https://api.the-odds-api.com/v4/sports/" + settings["sport"] + "/" + endpoint
        params = (
            {"apiKey": key, "daysFrom": 1}
            if scores
            else {
                "apiKey": key,
                "regions": settings["region"],
                "markets": "h2h",
                "oddsFormat": "decimal",
                "dateFormat": "iso",
                "eventIds": settings["event_id"],
            }
        )
    elif provider == "bybit-linear":
        url = "https://api.bybit.com/v5/market/tickers"
        params = {"category": "linear", "symbol": settings["contract"]}
    else:
        raise ValueError("No arbitrary credential-bearing endpoint")
    now = datetime.now(UTC)
    quota_path = root / "research/market-adapters/quota.json"
    quota = json.loads(quota_path.read_text()) if quota_path.exists() else {}
    period = now.strftime("%Y-%m") if provider == "sports-odds" else now.strftime("%Y-%m-%d")
    identity = provider + ":" + period
    spent = quota.get(identity, 0)
    # One sports region/market costs one credit; scores with daysFrom costs two.
    cost = 2 if provider == "sports-odds" and scores else 1
    limit = 100 if provider == "sports-odds" else 300
    if spent + cost > limit:
        raise ValueError("Free adapter quota exhausted; no paid fallback")
    quota[identity] = spent + cost
    atomic_json(quota_path, quota)  # Reserve before network, including failures.
    try:
        with requests.Session() as session:
            session.trust_env = False
            with session.get(
                url,
                params=params,
                headers=headers,
                timeout=(5, 15),
                allow_redirects=False,
                stream=True,
            ) as response:
                if response.status_code != 200:
                    raise ValueError("Free provider unavailable: HTTP " + str(response.status_code))
                chunks, size = [], 0
                for chunk in response.iter_content(65536):
                    size += len(chunk)
                    if size > 2 * 2**20:
                        raise ValueError("Market snapshot exceeds 2 MiB")
                    chunks.append(chunk)
                if (
                    provider == "sports-odds"
                    and "x-requests-remaining" in response.headers
                    and int(response.headers["x-requests-remaining"]) <= 0
                ):
                    quota[identity] = limit
                    atomic_json(quota_path, quota)
        body = b"".join(chunks)
        value = json.loads(body)
    except requests.RequestException:
        # Requests exceptions can contain the credential-bearing URL. Never log it.
        raise ValueError("Free provider network unavailable") from None
    digest = hashlib.sha256(body).hexdigest()
    source = root / "research/paper/sources" / (digest + ".json")
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        with source.open("xb") as handle:
            handle.write(body)
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ValueError("Provider archive changed")
    return value, digest, datetime.now(UTC).isoformat(), url


def normalize(
    configuration: dict, data, digest: str, retrieved: str, url: str, currency: str
) -> list[dict]:
    provider, settings = configuration["provider"], configuration["settings"]
    source = {"source_url": url, "source_sha256": digest}
    if provider == "alpaca-iex":
        quote = data["quotes"][settings["ticker"]]
        available = quote["t"]
        if timestamp(available) > timestamp(retrieved):
            raise ValueError("IEX quote timestamp is in the future")
        lot = settings["round_lot_shares"]
        return [
            {
                **source,
                "kind": "quote",
                "symbol": settings["symbol"],
                "feed_id": settings["feed_id"],
                "available_at": available,
                "currency": "USD",
                "bid": quote["bp"],
                "ask": quote["ap"],
                "bid_size": str(number(quote["bs"]) * lot),
                "ask_size": str(number(quote["as"]) * lot),
                "venue_scope": "IEX single exchange; not consolidated NBBO",
                "price_basis": "raw-unadjusted; corporate actions require separate independent records",
            }
        ]
    if provider == "bybit-linear":
        if data.get("retCode") != 0:
            raise ValueError("Derivative source rejected the request")
        ticker = data["result"]["list"][0]
        return [
            {
                **source,
                "kind": "news",
                "category": "research",
                "available_at": retrieved,
                "excerpt": json.dumps(
                    {
                        "contract": settings["contract"],
                        "ticker": ticker,
                        "scope": "Public derivative research. fundingRate is predicted, not settled funding. Ticker does not prove adverse interval marks; no leveraged fills enabled.",
                    }
                )[:6000],
            }
        ]
    event = next(item for item in data if item["id"] == settings["event_id"])
    bookmaker = next(item for item in event["bookmakers"] if item["key"] == settings["bookmaker"])
    market = next(item for item in bookmaker["markets"] if item["key"] == "h2h")
    available = market["last_update"]
    if timestamp(available) > timestamp(retrieved):
        raise ValueError("Sports market timestamp is in the future")
    odds = next(item["price"] for item in market["outcomes"] if item["name"] == settings["outcome"])
    if number(odds) <= 1:
        raise ValueError("Decimal sports odds must exceed one")
    if settings["settlement_rule"] == "research-only":
        return [
            {
                **source,
                "kind": "news",
                "category": "research",
                "available_at": available,
                "excerpt": json.dumps(
                    {
                        "event": event["id"],
                        "start": event["commence_time"],
                        "outcome": settings["outcome"],
                        "decimal_odds": odds,
                        "scope": "Current archived odds, not historical free API data or certified bookmaker settlement",
                    }
                ),
            }
        ]
    return [
        {
            **source,
            "kind": "quote",
            "symbol": settings["symbol"],
            "feed_id": settings["feed_id"],
            "available_at": available,
            "currency": currency,
            "bid": odds,
            "ask": odds,
            "bid_size": settings["paper_stake_cap"],
            "ask_size": settings["paper_stake_cap"],
            "liquidity_scope": "Operator paper stake cap; provider supplies no executable bookmaker liquidity",
        }
    ]


def settle(
    configuration: dict, scores: list[dict], digest: str, retrieved: str, url: str
) -> list[dict]:
    settings = configuration["settings"]
    if settings["settlement_rule"] == "research-only":
        return []
    event = next((item for item in scores if item["id"] == settings["event_id"]), None)
    if not event or not event["completed"] or not event.get("scores"):
        return []
    values = {item["name"]: number(item["score"]) for item in event["scores"]}
    if len(values) != 2 or set(values) != {event["home_team"], event["away_team"]}:
        raise ValueError("Only documented final-score two-team markets can settle automatically")
    maximum = max(values.values())
    tied = len(set(values.values())) == 1
    selected = settings["outcome"]
    if selected not in values and selected != "Draw":
        raise ValueError("Unknown sports outcome")
    result = (
        "void"
        if tied and settings["settlement_rule"] == "final-score-h2h-void-tie"
        else "won"
        if (tied and selected == "Draw") or (not tied and values.get(selected) == maximum)
        else "lost"
    )
    available = event["last_update"]
    if timestamp(available) > timestamp(retrieved):
        raise ValueError("Sports settlement timestamp is in the future")
    return [
        {
            "kind": "outcome",
            "symbol": settings["symbol"],
            "feed_id": settings["feed_id"],
            "result": result,
            "available_at": available,
            "source_url": url,
            "source_sha256": digest,
            "settlement_scope": "Operator-selected final-score rule; score-provider corrections and bookmaker-specific grading remain separate risks",
        }
    ]


def poll(book, cancelled=None) -> list[dict]:
    folder = book.root / "research/market-adapters"
    result = []
    paths = sorted(p for p in folder.glob("*.json") if p.name != "quota.json")
    if not paths:
        return result
    cursor_path = book.root / "research/state/market-adapter-poll.json"
    cursor = (
        json.loads(cursor_path.read_text())
        if cursor_path.exists()
        else {"offset": 0, "attempts": {}}
    )
    offset = cursor["offset"] % len(paths)
    selected = (paths[offset:] + paths[:offset])[:4]
    cursor["offset"] = (offset + len(selected)) % len(paths)
    atomic_json(cursor_path, cursor)
    for path in selected:
        if cancelled and cancelled():
            break
        configuration = {"id": path.stem, "provider": "unknown"}
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError("Market adapter configuration must be an object")
            configuration = value
            settings = configuration["settings"]
            now = datetime.now(UTC).timestamp()
            interval = 1800 if configuration["provider"] == "sports-odds" else 300
            if now - cursor["attempts"].get(path.stem, 0) < interval:
                continue
            cursor["attempts"][path.stem] = now
            atomic_json(cursor_path, cursor)
            if (
                configuration["provider"] == "sports-odds"
                and settings["settlement_rule"] != "research-only"
            ):
                state = book.state()
                if settings["symbol"] in state["outcomes"]:
                    continue
                instrument = state["instruments"][settings["symbol"]]
                if datetime.now(UTC) >= timestamp(instrument["starts_at"]):
                    data, digest, retrieved, url = request(
                        book.root, "sports-odds", settings, scores=True
                    )
                    for row in settle(configuration, data, digest, retrieved, url):
                        result.append(book.ingest(row))
                    continue
            data, digest, retrieved, url = request(
                book.root, configuration["provider"], configuration["settings"]
            )
            rows = normalize(configuration, data, digest, retrieved, url, book.state()["currency"])
            if configuration["provider"] == "alpaca-iex" and book.state()["currency"] != "USD":
                if book.state()["currency"] != "PLN":
                    raise ValueError("IEX reference FX currently supports USD to PLN only")
                from rlm.v100.paper_feeds import public_json

                fx_url = "https://api.nbp.pl/api/exchangerates/rates/a/usd/?format=json"
                fx_data, fx_digest, observed = public_json(book.root, fx_url)
                record = fx_data["rates"][-1]
                rate = number(record["mid"], positive=True)
                for row in rows:
                    row.update(
                        bid=str(number(row["bid"], positive=True) * rate),
                        ask=str(number(row["ask"], positive=True) * rate),
                        currency="PLN",
                        price_source_currency="USD",
                        fx={
                            "rate": str(rate),
                            "effective_date": record["effectiveDate"],
                            "source_url": fx_url,
                            "source_sha256": fx_digest,
                            "available_at": observed,
                            "note": "NBP reference conversion, not executable FX; original stock quote age is retained",
                        },
                    )
            for row in rows:
                result.append(book.ingest(row))
        except (ValueError, KeyError, IndexError, StopIteration, OSError) as error:
            ActivityLog(book.root, "controller", "market-adapter").write(
                "errors",
                "free-feed-deferred",
                {
                    "id": configuration.get("id", path.stem),
                    "provider": configuration.get("provider", "unknown"),
                    "reason": str(error)[:240],
                },
            )
    return result
