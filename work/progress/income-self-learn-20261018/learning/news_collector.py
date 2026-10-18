"""
Synta NewsCollector — zbiera newsy z darmowych źródeł dla instrumentów w portfolio.

Źródła:
  - NewsAPI (darmowy klucz: https://newsapi.org/register)
  - RSS feeds (Reuters, Bloomberg public RSS)
  - Reddit (r/investing, r/stocks, r/CryptoCurrency) — Pushshift/JSON API
  - Alpha Vantage News (darmowy tier)

Wyniki: data/news_cache.json, data/news_latest.json
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
import hashlib
import urllib.request as urlreq
import urllib.parse as urlparse
import xml.etree.ElementTree as ET

logger = logging.getLogger("synta.news_collector")

DATA_DIR = Path(os.environ.get("SYNTA_DATA_DIR", "data"))
NEWS_CACHE = DATA_DIR / "news_cache.json"
NEWS_LATEST = DATA_DIR / "news_latest.json"
NEWS_API_KEY = os.environ.get("NEWSAPI_KEY", "")
AV_API_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "demo")


@dataclass
class NewsItem:
    item_id: str
    source: str
    title: str
    description: str
    url: str
    published_at: str
    symbols: list[str]
    sentiment_hint: str = "neutral"  # naive keyword-based


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _fetch(url: str, timeout: int = 12) -> bytes:
    req = urlreq.Request(url, headers={"User-Agent": "Synta/1.0"})
    with urlreq.urlopen(req, timeout=timeout) as r:
        return r.read()


def _naive_sentiment(text: str) -> str:
    text_l = text.lower()
    pos = sum(text_l.count(w) for w in ["surge", "gain", "rally", "bullish", "beat", "record", "growth"])
    neg = sum(text_l.count(w) for w in ["crash", "plunge", "bear", "miss", "decline", "loss", "recession"])
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


# ---------------------------------------------------------------------------
# Source: NewsAPI
# ---------------------------------------------------------------------------

def fetch_newsapi(symbols: list[str]) -> list[NewsItem]:
    if not NEWS_API_KEY:
        logger.debug("NEWSAPI_KEY not set, skipping NewsAPI")
        return []
    items: list[NewsItem] = []
    query = " OR ".join(symbols[:5])
    params = urlparse.urlencode({
        "q": query, "language": "en", "sortBy": "publishedAt",
        "pageSize": "20", "apiKey": NEWS_API_KEY,
    })
    url = f"https://newsapi.org/v2/everything?{params}"
    try:
        raw = _fetch(url)
        data = json.loads(raw)
        for art in data.get("articles", [])[:20]:
            title = art.get("title", "") or ""
            desc = art.get("description", "") or ""
            iid = _sha(title + art.get("url", ""))
            matched = [s for s in symbols if s.upper() in (title + desc).upper()]
            items.append(NewsItem(
                item_id=f"newsapi_{iid}",
                source="newsapi",
                title=title,
                description=desc[:300],
                url=art.get("url", ""),
                published_at=art.get("publishedAt", _now_iso()),
                symbols=matched or symbols[:1],
                sentiment_hint=_naive_sentiment(title + desc),
            ))
    except Exception as exc:
        logger.warning("NewsAPI fetch failed: %s", exc)
    return items


# ---------------------------------------------------------------------------
# Source: Alpha Vantage News Sentiment
# ---------------------------------------------------------------------------

def fetch_alpha_vantage_news(symbols: list[str]) -> list[NewsItem]:
    items: list[NewsItem] = []
    tickers = ",".join(symbols[:5])
    params = urlparse.urlencode({
        "function": "NEWS_SENTIMENT", "tickers": tickers,
        "limit": "20", "apikey": AV_API_KEY,
    })
    url = f"https://www.alphavantage.co/query?{params}"
    try:
        raw = _fetch(url)
        data = json.loads(raw)
        for art in data.get("feed", [])[:20]:
            title = art.get("title", "")
            desc = art.get("summary", "")[:300]
            iid = _sha(title)
            ticker_sentiments = art.get("ticker_sentiment", [])
            matched = [ts["ticker"] for ts in ticker_sentiments if ts.get("ticker") in symbols]
            # AV provides sentiment label
            overall = art.get("overall_sentiment_label", "Neutral").lower()
            if "positive" in overall or "bullish" in overall:
                sentiment = "positive"
            elif "negative" in overall or "bearish" in overall:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            items.append(NewsItem(
                item_id=f"av_{iid}",
                source="alpha_vantage",
                title=title,
                description=desc,
                url=art.get("url", ""),
                published_at=art.get("time_published", _now_iso()),
                symbols=matched or symbols[:1],
                sentiment_hint=sentiment,
            ))
    except Exception as exc:
        logger.warning("Alpha Vantage news fetch failed: %s", exc)
    return items


# ---------------------------------------------------------------------------
# Source: Reddit JSON API
# ---------------------------------------------------------------------------

def fetch_reddit_news(symbols: list[str]) -> list[NewsItem]:
    items: list[NewsItem] = []
    subreddits = ["investing", "stocks", "CryptoCurrency", "finance"]
    for sub in subreddits[:2]:
        url = f"https://www.reddit.com/r/{sub}/new.json?limit=25"
        try:
            raw = _fetch(url)
            data = json.loads(raw)
            posts = data.get("data", {}).get("children", [])
            for p in posts[:10]:
                d = p.get("data", {})
                title = d.get("title", "")
                text = d.get("selftext", "")[:300]
                matched = [s for s in symbols if s.upper() in (title + text).upper()]
                if not matched:
                    continue
                iid = _sha(title + d.get("id", ""))
                ts = datetime.fromtimestamp(d.get("created_utc", 0), tz=timezone.utc).isoformat()
                items.append(NewsItem(
                    item_id=f"reddit_{iid}",
                    source=f"reddit_{sub}",
                    title=title,
                    description=text,
                    url=f"https://reddit.com{d.get('permalink', '')}",
                    published_at=ts,
                    symbols=matched,
                    sentiment_hint=_naive_sentiment(title + text),
                ))
        except Exception as exc:
            logger.warning("Reddit %s fetch failed: %s", sub, exc)
    return items


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _load_cache() -> dict[str, dict]:
    if NEWS_CACHE.exists():
        try:
            return {item["item_id"]: item for item in json.loads(NEWS_CACHE.read_text())}
        except Exception:
            pass
    return {}


def collect(symbols: list[str]) -> list[NewsItem]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = _load_cache()
    all_items: list[NewsItem] = []
    all_items.extend(fetch_newsapi(symbols))
    all_items.extend(fetch_alpha_vantage_news(symbols))
    all_items.extend(fetch_reddit_news(symbols))

    new_items = [i for i in all_items if i.item_id not in cache]
    logger.info("Collected %d news items (%d new)", len(all_items), len(new_items))

    for item in new_items:
        cache[item.item_id] = asdict(item)

    # Keep last 500 items
    recent = sorted(cache.values(), key=lambda x: x.get("published_at", ""), reverse=True)[:500]
    NEWS_CACHE.write_text(json.dumps(recent, indent=2))
    NEWS_LATEST.write_text(json.dumps(new_items[:50], indent=2, default=asdict))
    return new_items


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    symbols = os.environ.get("SYNTA_SYMBOLS", "BTC,ETH,AAPL,MSFT,TSLA").split(",")
    while True:
        collect(symbols)
        time.sleep(900)  # 15 min
