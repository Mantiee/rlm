"""
Synta SocialCollector — Twitter/Reddit sentiment dla symboli z portfolio.

Źródła darmowe:
  - Reddit pushshift/JSON API (sentiment per symbol)
  - Twitter/X Basic free tier (bearer token) — optional, requires TWITTER_BEARER_TOKEN
  - StockTwits public API (no auth for reading streams)
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import hashlib
import urllib.request as urlreq
import urllib.parse as urlparse

logger = logging.getLogger("synta.social_collector")

DATA_DIR = Path(os.environ.get("SYNTA_DATA_DIR", "data"))
SOCIAL_CACHE = DATA_DIR / "social_cache.json"
SOCIAL_LATEST = DATA_DIR / "social_latest.json"
TWITTER_BEARER = os.environ.get("TWITTER_BEARER_TOKEN", "")


@dataclass
class SocialPost:
    post_id: str
    platform: str
    symbol: str
    text: str
    author: str
    posted_at: str
    sentiment: str  # positive / negative / neutral
    score: float  # -1.0 to 1.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _fetch(url: str, headers: Optional[dict] = None, timeout: int = 12) -> bytes:
    req = urlreq.Request(url, headers={"User-Agent": "Synta/1.0", **(headers or {})})
    with urlreq.urlopen(req, timeout=timeout) as r:
        return r.read()


POSITIVE_WORDS = {"bull", "moon", "pump", "buy", "gain", "surge", "ath", "hodl", "long", "breakout"}
NEGATIVE_WORDS = {"bear", "dump", "sell", "crash", "short", "rekt", "fear", "scam", "drop", "tank"}


def _score_text(text: str) -> tuple[str, float]:
    words = set(text.lower().split())
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    if pos == neg:
        return "neutral", 0.0
    score = (pos - neg) / max(pos + neg, 1)
    return ("positive" if score > 0 else "negative"), round(score, 3)


# ---------------------------------------------------------------------------
# Reddit
# ---------------------------------------------------------------------------

def fetch_reddit_sentiment(symbol: str) -> list[SocialPost]:
    posts: list[SocialPost] = []
    subs = ["wallstreetbets", "investing", "stocks", "CryptoCurrency"]
    for sub in subs[:2]:
        try:
            q = urlparse.urlencode({"q": symbol, "sort": "new", "limit": "25", "restrict_sr": "1"})
            url = f"https://www.reddit.com/r/{sub}/search.json?{q}"
            raw = _fetch(url)
            data = json.loads(raw)
            for p in data.get("data", {}).get("children", [])[:10]:
                d = p.get("data", {})
                text = d.get("title", "") + " " + d.get("selftext", "")[:200]
                sentiment, score = _score_text(text)
                pid = _sha(d.get("id", text))
                ts = datetime.fromtimestamp(d.get("created_utc", 0), tz=timezone.utc).isoformat()
                posts.append(SocialPost(
                    post_id=f"reddit_{pid}",
                    platform="reddit",
                    symbol=symbol,
                    text=text[:300],
                    author=d.get("author", "unknown"),
                    posted_at=ts,
                    sentiment=sentiment,
                    score=score,
                ))
        except Exception as exc:
            logger.warning("Reddit sentiment %s/%s: %s", sub, symbol, exc)
    return posts


# ---------------------------------------------------------------------------
# StockTwits
# ---------------------------------------------------------------------------

def fetch_stocktwits_sentiment(symbol: str) -> list[SocialPost]:
    posts: list[SocialPost] = []
    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
        raw = _fetch(url)
        data = json.loads(raw)
        for msg in data.get("messages", [])[:20]:
            text = msg.get("body", "")
            entities = msg.get("entities", {})
            sentiment_label = entities.get("sentiment", {})
            if isinstance(sentiment_label, dict):
                st_raw = sentiment_label.get("basic", "neutral").lower()
            else:
                st_raw = "neutral"
            if st_raw == "bullish":
                sentiment, score = "positive", 0.5
            elif st_raw == "bearish":
                sentiment, score = "negative", -0.5
            else:
                sentiment, score = _score_text(text)
            pid = _sha(str(msg.get("id", text)))
            ts = msg.get("created_at", _now_iso())
            posts.append(SocialPost(
                post_id=f"stocktwits_{pid}",
                platform="stocktwits",
                symbol=symbol,
                text=text[:300],
                author=msg.get("user", {}).get("username", "unknown"),
                posted_at=ts,
                sentiment=sentiment,
                score=score,
            ))
    except Exception as exc:
        logger.warning("StockTwits %s: %s", symbol, exc)
    return posts


# ---------------------------------------------------------------------------
# Twitter (optional, requires bearer token)
# ---------------------------------------------------------------------------

def fetch_twitter_sentiment(symbol: str) -> list[SocialPost]:
    if not TWITTER_BEARER:
        return []
    posts: list[SocialPost] = []
    try:
        q = urlparse.urlencode({"query": f"${symbol} -is:retweet lang:en", "max_results": "20",
                                "tweet.fields": "created_at,author_id"})
        url = f"https://api.twitter.com/2/tweets/search/recent?{q}"
        raw = _fetch(url, headers={"Authorization": f"Bearer {TWITTER_BEARER}"})
        data = json.loads(raw)
        for t in data.get("data", [])[:20]:
            text = t.get("text", "")
            sentiment, score = _score_text(text)
            pid = _sha(t.get("id", text))
            posts.append(SocialPost(
                post_id=f"twitter_{pid}",
                platform="twitter",
                symbol=symbol,
                text=text[:300],
                author=t.get("author_id", "unknown"),
                posted_at=t.get("created_at", _now_iso()),
                sentiment=sentiment,
                score=score,
            ))
    except Exception as exc:
        logger.warning("Twitter %s: %s", symbol, exc)
    return posts


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def compute_sentiment_summary(posts: list[SocialPost]) -> dict:
    if not posts:
        return {"total": 0, "positive": 0, "negative": 0, "neutral": 0, "avg_score": 0.0}
    from collections import Counter
    counts = Counter(p.sentiment for p in posts)
    avg_score = sum(p.score for p in posts) / len(posts)
    return {
        "total": len(posts),
        "positive": counts.get("positive", 0),
        "negative": counts.get("negative", 0),
        "neutral": counts.get("neutral", 0),
        "avg_score": round(avg_score, 4),
    }


def collect(symbols: list[str]) -> dict[str, dict]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict] = {}
    if SOCIAL_CACHE.exists():
        try:
            cache = json.loads(SOCIAL_CACHE.read_text())
        except Exception:
            pass

    results: dict[str, dict] = {}
    for sym in symbols:
        all_posts: list[SocialPost] = []
        all_posts.extend(fetch_reddit_sentiment(sym))
        all_posts.extend(fetch_stocktwits_sentiment(sym))
        all_posts.extend(fetch_twitter_sentiment(sym))
        new_posts = [p for p in all_posts if p.post_id not in cache]
        for p in new_posts:
            cache[p.post_id] = asdict(p)
        results[sym] = compute_sentiment_summary(all_posts)
        logger.info("Symbol %s: %d posts, summary=%s", sym, len(all_posts), results[sym])

    # keep last 2000
    recent = dict(list(cache.items())[-2000:])
    SOCIAL_CACHE.write_text(json.dumps(recent, indent=2))
    SOCIAL_LATEST.write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    symbols = os.environ.get("SYNTA_SYMBOLS", "BTC,ETH,AAPL,MSFT").split(",")
    while True:
        collect(symbols)
        time.sleep(900)
