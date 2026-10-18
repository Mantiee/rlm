"""
Synta ModelUpdater — używa zebranych danych do aktualizacji sygnałów strategii.

Wejście:
  - data/news_latest.json (sentiment per symbol)
  - data/social_latest.json (sentiment aggregate per symbol)
  - data/earnings.json (EPS surprise, revenue)
  - data/ohlcv/{symbol}.json (OHLCV bars)

Wyjście:
  - data/signals.json — zaktualizowane sygnały per symbol
  - data/model_log.json — historia aktualizacji

Strategia:
  Combined score = w_news * news_score + w_social * social_score
                  + w_earnings * eps_surprise + w_momentum * price_momentum
  Threshold > 0.3 → BUY_SIGNAL, < -0.3 → SELL_SIGNAL, else HOLD

Nie hardcode'uje wyborów portfela; każda zmiana jest logowana z atrybutem.
Żaden sygnał nie jest "zyskiem" — to propozycja do badania.
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

logger = logging.getLogger("synta.model_updater")

DATA_DIR = Path(os.environ.get("SYNTA_DATA_DIR", "data"))
SIGNALS_FILE = DATA_DIR / "signals.json"
MODEL_LOG_FILE = DATA_DIR / "model_log.json"

W_NEWS = float(os.environ.get("SYNTA_W_NEWS", "0.25"))
W_SOCIAL = float(os.environ.get("SYNTA_W_SOCIAL", "0.25"))
W_EARNINGS = float(os.environ.get("SYNTA_W_EARNINGS", "0.30"))
W_MOMENTUM = float(os.environ.get("SYNTA_W_MOMENTUM", "0.20"))

BUY_THRESHOLD = float(os.environ.get("SYNTA_BUY_THRESHOLD", "0.30"))
SELL_THRESHOLD = float(os.environ.get("SYNTA_SELL_THRESHOLD", "-0.30"))


@dataclass
class Signal:
    symbol: str
    signal: str  # BUY_SIGNAL | SELL_SIGNAL | HOLD
    combined_score: float
    news_score: float
    social_score: float
    earnings_score: float
    momentum_score: float
    basis: str
    updated_at: str
    confidence: str  # low | medium | high


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path, default) -> object:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception as exc:
            logger.warning("Load %s: %s", path, exc)
    return default


def _news_score(symbol: str, news_data: list[dict]) -> float:
    relevant = [n for n in news_data if symbol in n.get("symbols", [])]
    if not relevant:
        return 0.0
    scores = []
    for n in relevant[-20:]:
        hint = n.get("sentiment_hint", "neutral")
        if hint == "positive":
            scores.append(0.5)
        elif hint == "negative":
            scores.append(-0.5)
        else:
            scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0


def _social_score(symbol: str, social_data: dict) -> float:
    summary = social_data.get(symbol, {})
    return float(summary.get("avg_score", 0.0))


def _earnings_score(symbol: str, earnings_data: list[dict]) -> float:
    relevant = [e for e in earnings_data if e.get("symbol") == symbol and e.get("surprise_pct") is not None]
    if not relevant:
        return 0.0
    latest = sorted(relevant, key=lambda x: x.get("report_date", ""), reverse=True)[:2]
    surprises = [float(e["surprise_pct"]) for e in latest]
    avg = sum(surprises) / len(surprises)
    # Normalize: ±20% surprise → ±1.0
    return max(-1.0, min(1.0, avg / 20.0))


def _momentum_score(symbol: str, ohlcv_dir: Path) -> float:
    ohlcv_file = ohlcv_dir / f"{symbol.upper()}.json"
    if not ohlcv_file.exists():
        return 0.0
    try:
        bars = json.loads(ohlcv_file.read_text())
        if len(bars) < 10:
            return 0.0
        closes = [b.get("close", 0) for b in bars[-20:] if b.get("close")]
        if len(closes) < 5:
            return 0.0
        # 5-day vs 20-day momentum
        short = sum(closes[-5:]) / 5
        long_ = sum(closes) / len(closes)
        if long_ == 0:
            return 0.0
        return max(-1.0, min(1.0, (short - long_) / long_ * 10))
    except Exception as exc:
        logger.warning("Momentum %s: %s", symbol, exc)
        return 0.0


def _confidence(n_sources: int, data_age_h: float) -> str:
    if n_sources >= 3 and data_age_h < 2:
        return "high"
    if n_sources >= 2 and data_age_h < 24:
        return "medium"
    return "low"


def update_signals(symbols: list[str]) -> list[Signal]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ohlcv_dir = DATA_DIR / "ohlcv"

    news_data: list[dict] = _load_json(DATA_DIR / "news_latest.json", [])
    social_data: dict = _load_json(DATA_DIR / "social_latest.json", {})
    earnings_data: list[dict] = _load_json(DATA_DIR / "earnings.json", [])

    signals: list[Signal] = []
    for sym in symbols:
        ns = _news_score(sym, news_data)
        ss = _social_score(sym, social_data)
        es = _earnings_score(sym, earnings_data)
        ms = _momentum_score(sym, ohlcv_dir)

        combined = W_NEWS * ns + W_SOCIAL * ss + W_EARNINGS * es + W_MOMENTUM * ms

        if combined > BUY_THRESHOLD:
            signal = "BUY_SIGNAL"
        elif combined < SELL_THRESHOLD:
            signal = "SELL_SIGNAL"
        else:
            signal = "HOLD"

        n_sources = sum([
            bool(any(sym in n.get("symbols", []) for n in news_data)),
            sym in social_data,
            bool(any(e.get("symbol") == sym for e in earnings_data)),
            (ohlcv_dir / f"{sym.upper()}.json").exists(),
        ])
        basis = (
            f"news={ns:.3f} social={ss:.3f} earnings={es:.3f} momentum={ms:.3f} "
            f"combined={combined:.3f} sources={n_sources}"
        )
        sig = Signal(
            symbol=sym,
            signal=signal,
            combined_score=round(combined, 4),
            news_score=round(ns, 4),
            social_score=round(ss, 4),
            earnings_score=round(es, 4),
            momentum_score=round(ms, 4),
            basis=basis,
            updated_at=_now_iso(),
            confidence=_confidence(n_sources, data_age_h=0),
        )
        signals.append(sig)
        logger.info("Signal %s: %s (score=%.3f, confidence=%s)", sym, signal, combined, sig.confidence)

    SIGNALS_FILE.write_text(json.dumps([asdict(s) for s in signals], indent=2))

    log: list[dict] = _load_json(MODEL_LOG_FILE, [])
    log.append({
        "timestamp": _now_iso(),
        "symbols": symbols,
        "signals": {s.symbol: {"signal": s.signal, "score": s.combined_score} for s in signals},
    })
    MODEL_LOG_FILE.write_text(json.dumps(log[-200:], indent=2))

    return signals


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    symbols = os.environ.get("SYNTA_SYMBOLS", "BTC,ETH,AAPL,MSFT,TSLA").split(",")
    while True:
        update_signals(symbols)
        time.sleep(1800)  # 30 min
