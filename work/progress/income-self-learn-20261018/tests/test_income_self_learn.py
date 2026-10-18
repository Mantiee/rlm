"""Basic smoke tests for income + self-learning modules."""
import json
import sys
import os
import tempfile
from pathlib import Path

# Make modules importable
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["SYNTA_DATA_DIR"] = tempfile.mkdtemp()
os.environ["SYNTA_INCOME_RESULTS"] = str(Path(os.environ["SYNTA_DATA_DIR"]) / "income_results.json")
os.environ["SYNTA_GIG_STATE"] = str(Path(os.environ["SYNTA_DATA_DIR"]) / "gig_state.json")


def test_gig_worker_imports():
    from income.gig_worker import GigTask, IncomeRecord, execute_task, _is_executable
    t = GigTask(
        task_id="test_1", source="test", title="Python automation script",
        description="write python script", category="freelance",
        url="https://example.com", budget_usd=50.0, fetched_at="2026-10-18T00:00:00Z",
    )
    assert _is_executable(t)
    t2 = execute_task(t)
    assert t2.status == "skipped"


def test_gig_worker_not_executable():
    from income.gig_worker import GigTask, _is_executable
    t = GigTask(
        task_id="test_2", source="test", title="Design a logo for bakery",
        description="creative design work", category="design",
        url="https://example.com", budget_usd=100.0, fetched_at="2026-10-18T00:00:00Z",
    )
    assert not _is_executable(t)


def test_gig_worker_poll_once_runs():
    """poll_once should not raise even if all fetches fail."""
    from income.gig_worker import poll_once
    tasks = poll_once()
    assert isinstance(tasks, list)


def test_news_collector_imports():
    from learning.news_collector import collect, _naive_sentiment
    assert _naive_sentiment("stock surge rally bullish") == "positive"
    assert _naive_sentiment("crash dump bear rekt") == "negative"
    assert _naive_sentiment("stock moved today") == "neutral"


def test_social_collector_score():
    from learning.social_collector import _score_text
    sentiment, score = _score_text("moon bull buy gain rally breakout")
    assert sentiment == "positive"
    assert score > 0
    sentiment2, score2 = _score_text("crash dump bear short tank rekt")
    assert sentiment2 == "negative"
    assert score2 < 0


def test_quarterly_parser_safe_float():
    from learning.quarterly_parser import _safe_float
    assert _safe_float("1.23") == 1.23
    assert _safe_float(None) is None
    assert _safe_float("bad") is None


def test_historical_fetcher_crypto_detection():
    from learning.historical_fetcher import _is_crypto
    assert _is_crypto("BTC")
    assert _is_crypto("ETH")
    assert not _is_crypto("AAPL")


def test_model_updater_signal_hold():
    from learning.model_updater import update_signals
    # With empty data dirs, should produce HOLD signals without crashing
    signals = update_signals(["BTC", "AAPL"])
    assert len(signals) == 2
    for s in signals:
        assert s.signal in ("BUY_SIGNAL", "SELL_SIGNAL", "HOLD")
        assert -1.0 <= s.combined_score <= 1.0


def test_earnings_api_build_history():
    from income.earnings_history_api import build_earnings_history
    result = build_earnings_history(days=7)
    assert "timestamps" in result
    assert "sources" in result
    assert len(result["timestamps"]) == 7
    assert len(result["sources"]["paper_pnl"]) == 7
    assert len(result["sources"]["gig_income"]) == 7


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
