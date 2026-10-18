"""
Synta QuarterlyParser — pobiera i parsuje quarterly earnings reports.

Źródła:
  - SEC EDGAR XBRL/JSON API (darmowe, bez klucza)
  - Yahoo Finance earnings calendar (scraping, fallback)
  - Alpha Vantage EARNINGS endpoint (darmowy tier)
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import urllib.request as urlreq
import urllib.parse as urlparse

logger = logging.getLogger("synta.quarterly_parser")

DATA_DIR = Path(os.environ.get("SYNTA_DATA_DIR", "data"))
EARNINGS_FILE = DATA_DIR / "earnings.json"
AV_API_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "demo")


@dataclass
class EarningsReport:
    symbol: str
    fiscal_quarter: str
    report_date: str
    eps_actual: Optional[float]
    eps_estimate: Optional[float]
    revenue_actual: Optional[float]
    revenue_estimate: Optional[float]
    surprise_pct: Optional[float]
    source: str
    fetched_at: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fetch(url: str, timeout: int = 15) -> bytes:
    req = urlreq.Request(url, headers={"User-Agent": "Synta/1.0 (research; contact: synta-bot)"})
    with urlreq.urlopen(req, timeout=timeout) as r:
        return r.read()


def _safe_float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Alpha Vantage EARNINGS
# ---------------------------------------------------------------------------

def fetch_av_earnings(symbol: str) -> list[EarningsReport]:
    reports: list[EarningsReport] = []
    try:
        params = urlparse.urlencode({"function": "EARNINGS", "symbol": symbol, "apikey": AV_API_KEY})
        url = f"https://www.alphavantage.co/query?{params}"
        raw = _fetch(url)
        data = json.loads(raw)
        for q in data.get("quarterlyEarnings", [])[:8]:
            eps_a = _safe_float(q.get("reportedEPS"))
            eps_e = _safe_float(q.get("estimatedEPS"))
            surprise = _safe_float(q.get("surprisePercentage"))
            reports.append(EarningsReport(
                symbol=symbol,
                fiscal_quarter=q.get("fiscalDateEnding", ""),
                report_date=q.get("reportedDate", ""),
                eps_actual=eps_a,
                eps_estimate=eps_e,
                revenue_actual=None,
                revenue_estimate=None,
                surprise_pct=surprise,
                source="alpha_vantage",
                fetched_at=_now_iso(),
            ))
    except Exception as exc:
        logger.warning("AV earnings %s: %s", symbol, exc)
    return reports


# ---------------------------------------------------------------------------
# SEC EDGAR — company facts (EPS / Net Income)
# ---------------------------------------------------------------------------

_CIK_CACHE: dict[str, str] = {}


def _get_cik(symbol: str) -> Optional[str]:
    if symbol in _CIK_CACHE:
        return _CIK_CACHE[symbol]
    try:
        url = f"https://efts.sec.gov/LATEST/search-index?q=%22{symbol}%22&dateRange=custom&startdt=2020-01-01&forms=10-Q&hits.hits._source=period_of_report,display_date_filed,entity_name,file_num"
        raw = _fetch(url)
        data = json.loads(raw)
        hits = data.get("hits", {}).get("hits", [])
        if hits:
            entity = hits[0].get("_source", {}).get("entity_name", "")
            # Try company tickers endpoint
            ticker_url = "https://www.sec.gov/files/company_tickers.json"
            raw2 = _fetch(ticker_url)
            tickers_data = json.loads(raw2)
            for entry in tickers_data.values():
                if entry.get("ticker", "").upper() == symbol.upper():
                    cik = str(entry.get("cik_str", "")).zfill(10)
                    _CIK_CACHE[symbol] = cik
                    return cik
    except Exception as exc:
        logger.debug("CIK lookup %s: %s", symbol, exc)
    return None


def fetch_sec_earnings(symbol: str) -> list[EarningsReport]:
    reports: list[EarningsReport] = []
    cik = _get_cik(symbol)
    if not cik:
        return reports
    try:
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        raw = _fetch(url)
        data = json.loads(raw)
        facts = data.get("facts", {})
        us_gaap = facts.get("us-gaap", {})
        # Try EPS basic
        eps_data = us_gaap.get("EarningsPerShareBasic", {}).get("units", {}).get("USD/shares", [])
        eps_by_period: dict[str, float] = {}
        for entry in eps_data:
            if entry.get("form") in ("10-Q", "10-K") and "end" in entry:
                eps_by_period[entry["end"]] = float(entry.get("val", 0))
        for period, eps in list(eps_by_period.items())[-8:]:
            reports.append(EarningsReport(
                symbol=symbol,
                fiscal_quarter=period,
                report_date=period,
                eps_actual=eps,
                eps_estimate=None,
                revenue_actual=None,
                revenue_estimate=None,
                surprise_pct=None,
                source="sec_edgar",
                fetched_at=_now_iso(),
            ))
    except Exception as exc:
        logger.warning("SEC EDGAR %s: %s", symbol, exc)
    return reports


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect(symbols: list[str]) -> dict[str, list[EarningsReport]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if EARNINGS_FILE.exists():
        try:
            existing = json.loads(EARNINGS_FILE.read_text())
        except Exception:
            pass

    results: dict[str, list[EarningsReport]] = {}
    for sym in symbols:
        reports: list[EarningsReport] = []
        reports.extend(fetch_av_earnings(sym))
        if not reports:
            reports.extend(fetch_sec_earnings(sym))
        results[sym] = reports
        logger.info("Earnings %s: %d reports", sym, len(reports))

    all_new = [asdict(r) for rlist in results.values() for r in rlist]
    combined = existing + all_new
    # dedup by symbol+fiscal_quarter+source
    seen: set[str] = set()
    deduped: list[dict] = []
    for item in combined:
        key = f"{item['symbol']}_{item['fiscal_quarter']}_{item['source']}"
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    EARNINGS_FILE.write_text(json.dumps(deduped[-500:], indent=2))
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    symbols = os.environ.get("SYNTA_SYMBOLS", "AAPL,MSFT,TSLA").split(",")
    collect(symbols)
