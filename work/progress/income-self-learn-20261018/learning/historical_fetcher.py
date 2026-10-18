"""
Synta HistoricalFetcher — pobiera dane historyczne OHLCV.

Źródła (darmowe):
  - yfinance (pip install yfinance) — giełdy globalne, krypto
  - Alpha Vantage TIME_SERIES_DAILY (darmowy klucz, 25 req/dzień)
  - CoinGecko (krypto, bez klucza, 10-30 req/min)

Wyniki: data/ohlcv/{symbol}.json
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
import urllib.request as urlreq
import urllib.parse as urlparse

logger = logging.getLogger("synta.historical_fetcher")

DATA_DIR = Path(os.environ.get("SYNTA_DATA_DIR", "data"))
OHLCV_DIR = DATA_DIR / "ohlcv"
AV_API_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "demo")


@dataclass
class OHLCVBar:
    symbol: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fetch(url: str, timeout: int = 20) -> bytes:
    req = urlreq.Request(url, headers={"User-Agent": "Synta/1.0"})
    with urlreq.urlopen(req, timeout=timeout) as r:
        return r.read()


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Alpha Vantage
# ---------------------------------------------------------------------------

def fetch_av_daily(symbol: str, outputsize: str = "compact") -> list[OHLCVBar]:
    bars: list[OHLCVBar] = []
    try:
        params = urlparse.urlencode({
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": outputsize,
            "apikey": AV_API_KEY,
        })
        url = f"https://www.alphavantage.co/query?{params}"
        raw = _fetch(url)
        data = json.loads(raw)
        series = data.get("Time Series (Daily)", {})
        for date_str, vals in sorted(series.items(), reverse=True)[:100]:
            bars.append(OHLCVBar(
                symbol=symbol,
                timestamp=date_str + "T00:00:00Z",
                open=_safe_float(vals.get("1. open")),
                high=_safe_float(vals.get("2. high")),
                low=_safe_float(vals.get("3. low")),
                close=_safe_float(vals.get("4. close")),
                volume=_safe_float(vals.get("5. volume")),
                source="alpha_vantage",
            ))
    except Exception as exc:
        logger.warning("AV daily %s: %s", symbol, exc)
    return bars


# ---------------------------------------------------------------------------
# CoinGecko (crypto)
# ---------------------------------------------------------------------------

COINGECKO_ID_MAP = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
    "BNB": "binancecoin", "XRP": "ripple", "ADA": "cardano",
    "DOGE": "dogecoin", "DOT": "polkadot", "MATIC": "matic-network",
}


def fetch_coingecko_ohlcv(symbol: str, days: int = 90) -> list[OHLCVBar]:
    cg_id = COINGECKO_ID_MAP.get(symbol.upper(), "")
    if not cg_id:
        return []
    bars: list[OHLCVBar] = []
    try:
        params = urlparse.urlencode({"vs_currency": "usd", "days": str(days), "interval": "daily"})
        url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/ohlc?{params}"
        raw = _fetch(url)
        data = json.loads(raw)
        for item in data:
            ts_ms, o, h, l, c = item
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()
            bars.append(OHLCVBar(
                symbol=symbol,
                timestamp=ts,
                open=_safe_float(o),
                high=_safe_float(h),
                low=_safe_float(l),
                close=_safe_float(c),
                volume=0.0,
                source="coingecko",
            ))
    except Exception as exc:
        logger.warning("CoinGecko %s: %s", symbol, exc)
    return bars


# ---------------------------------------------------------------------------
# yfinance (optional — if installed)
# ---------------------------------------------------------------------------

def fetch_yfinance(symbol: str, period: str = "3mo", interval: str = "1d") -> list[OHLCVBar]:
    bars: list[OHLCVBar] = []
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        for idx, row in hist.iterrows():
            ts = idx.isoformat() if hasattr(idx, "isoformat") else str(idx)
            bars.append(OHLCVBar(
                symbol=symbol,
                timestamp=ts,
                open=_safe_float(row.get("Open")),
                high=_safe_float(row.get("High")),
                low=_safe_float(row.get("Low")),
                close=_safe_float(row.get("Close")),
                volume=_safe_float(row.get("Volume")),
                source="yfinance",
            ))
    except ImportError:
        logger.debug("yfinance not installed")
    except Exception as exc:
        logger.warning("yfinance %s: %s", symbol, exc)
    return bars


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _is_crypto(symbol: str) -> bool:
    return symbol.upper() in COINGECKO_ID_MAP


def fetch_symbol(symbol: str) -> list[OHLCVBar]:
    # Try yfinance first (best coverage)
    bars = fetch_yfinance(symbol)
    if bars:
        return bars
    # Crypto fallback
    if _is_crypto(symbol):
        return fetch_coingecko_ohlcv(symbol)
    # Stock fallback
    return fetch_av_daily(symbol)


def collect(symbols: list[str]) -> dict[str, int]:
    OHLCV_DIR.mkdir(parents=True, exist_ok=True)
    result: dict[str, int] = {}
    for sym in symbols:
        bars = fetch_symbol(sym)
        if bars:
            out_file = OHLCV_DIR / f"{sym.upper()}.json"
            existing: list[dict] = []
            if out_file.exists():
                try:
                    existing = json.loads(out_file.read_text())
                except Exception:
                    pass
            existing_ts: set[str] = {b["timestamp"] for b in existing}
            new_bars = [b for b in bars if asdict(b)["timestamp"] not in existing_ts]
            combined = existing + [asdict(b) for b in new_bars]
            combined.sort(key=lambda x: x["timestamp"])
            out_file.write_text(json.dumps(combined[-1000:], indent=2))
            result[sym] = len(new_bars)
            logger.info("OHLCV %s: %d new bars (%d total)", sym, len(new_bars), len(combined))
        else:
            logger.warning("No data for %s", sym)
            result[sym] = 0
        time.sleep(0.5)  # rate limit
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    symbols = os.environ.get("SYNTA_SYMBOLS", "BTC,ETH,AAPL,MSFT,TSLA").split(",")
    collect(symbols)
