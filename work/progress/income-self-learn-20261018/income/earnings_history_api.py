"""
Synta EarningsHistoryAPI — endpoint /api/earnings_history dla dashboardu.

Czyta income_results.json (gig income) i paper P&L z api snapshot,
zwraca skumulowane serie danych dla wykresu zarobków.

Format odpowiedzi:
{
  "timestamps": ["2026-10-01T00:00:00Z", ...],
  "sources": {
    "paper_pnl":  [0.0, 1.2, -0.5, ...],
    "gig_income": [0.0, 0.0, 5.0, ...]
  }
}

Montuje się jako handler w istniejącym dashboardzie Synta lub osobny
aiohttp/http.server.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger("synta.earnings_api")

DATA_DIR = Path(os.environ.get("SYNTA_DATA_DIR", "data"))
INCOME_RESULTS = Path(os.environ.get("SYNTA_INCOME_RESULTS", "income_results.json"))
PAPER_SNAPSHOT = DATA_DIR / "paper_snapshot.json"


def _days_ago(n: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=n)


def _parse_dt(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return datetime.now(timezone.utc)


def build_earnings_history(days: int = 30) -> dict[str, Any]:
    """Build cumulative earnings series for the last `days` days."""
    cutoff = _days_ago(days)

    # Build daily buckets
    buckets: dict[str, dict[str, float]] = {}
    for d in range(days):
        day = (_days_ago(days - 1 - d)).strftime("%Y-%m-%d")
        buckets[day] = {"paper_pnl": 0.0, "gig_income": 0.0}

    # --- Paper P&L from snapshot ---
    try:
        if PAPER_SNAPSHOT.exists():
            snap = json.loads(PAPER_SNAPSHOT.read_text())
            # Expected: {"history": [{"timestamp": "...", "equity": 100.5}, ...]}
            history = snap.get("history", [])
            initial_equity = float(snap.get("initial_equity", 100.0))
            prev = initial_equity
            for entry in history:
                ts = _parse_dt(entry.get("timestamp", ""))
                if ts < cutoff:
                    prev = float(entry.get("equity", prev))
                    continue
                day = ts.strftime("%Y-%m-%d")
                if day in buckets:
                    equity = float(entry.get("equity", prev))
                    buckets[day]["paper_pnl"] = equity - initial_equity
                prev = float(entry.get("equity", prev))
    except Exception as exc:
        logger.warning("Paper snapshot read: %s", exc)

    # --- Gig income from income_results.json ---
    try:
        if INCOME_RESULTS.exists():
            results = json.loads(INCOME_RESULTS.read_text())
            cumulative_gig = 0.0
            for rec in results:
                ts = _parse_dt(rec.get("fetched_at", ""))
                if ts < cutoff:
                    continue
                day = ts.strftime("%Y-%m-%d")
                # Only count confirmed (non-uncertain) income
                amount = 0.0
                if not rec.get("activation_uncertain", True) and rec.get("confirmed", False):
                    amount = float(rec.get("amount_usd", 0) or 0)
                if day in buckets:
                    buckets[day]["gig_income"] += amount
    except Exception as exc:
        logger.warning("Income results read: %s", exc)

    # Build cumulative series
    sorted_days = sorted(buckets.keys())
    cum_paper = 0.0
    cum_gig = 0.0
    timestamps, paper_series, gig_series = [], [], []
    for day in sorted_days:
        cum_paper += buckets[day]["paper_pnl"]
        cum_gig += buckets[day]["gig_income"]
        timestamps.append(day + "T00:00:00Z")
        paper_series.append(round(cum_paper, 4))
        gig_series.append(round(cum_gig, 4))

    return {
        "timestamps": timestamps,
        "sources": {
            "paper_pnl": paper_series,
            "gig_income": gig_series,
        },
        "days": days,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Minimal standalone HTTP server for testing ──────────────────────────────

def _make_handler():
    from http.server import BaseHTTPRequestHandler
    import urllib.parse as urlparse

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            logger.debug(fmt, *args)

        def do_GET(self):
            parsed = urlparse.urlparse(self.path)
            if parsed.path != "/api/earnings_history":
                self.send_response(404)
                self.end_headers()
                return
            params = dict(urlparse.parse_qsl(parsed.query))
            days = int(params.get("days", "30"))
            data = build_earnings_history(days)
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


if __name__ == "__main__":
    import sys
    from http.server import HTTPServer
    logging.basicConfig(level=logging.INFO)
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8766
    server = HTTPServer(("0.0.0.0", port), _make_handler())
    logger.info("EarningsHistoryAPI listening on port %d", port)
    server.serve_forever()
