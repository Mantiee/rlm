"""Research-only multi-horizon experiment curriculum with explicit quality gates.

These are experiment specifications, not trained models, backtest results,
trade instructions or permission to place real-money orders.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Experiment:
    name: str
    hypothesis: str
    horizons: tuple[str, ...]
    controls: tuple[str, ...]
    evaluation: str
    failure_condition: str
    required_data: tuple[str, ...]


EXPERIMENTS = (
    Experiment(
        "Volatility regime switch",
        "Momentum may outperform cash in persistent low-noise trends, while avoiding volatile whipsaws.",
        ("1h", "4h", "1d", "1w"),
        ("cash", "buy-and-hold", "always-on momentum"),
        "Purged walk-forward folds; rolling volatility regimes; cost-adjusted return and drawdown.",
        "No improvement over cash or strongest control after doubled costs.",
        ("closed OHLCV bars", "venue fee schedule", "spread and slippage estimates"),
    ),
    Experiment(
        "Cross-asset lead-lag",
        "BTC moves may precede ETH moves on some horizons, after realistic data latency.",
        ("15m", "1h", "4h", "1d"),
        ("cash", "ETH momentum", "randomized lag"),
        "Lagged timestamp-aligned signals with embargo, latency, and placebo shifts.",
        "Signal vanishes after shifting timestamps or transaction costs.",
        ("timestamped BTC and ETH closed bars", "availability timestamps", "cost evidence"),
    ),
    Experiment(
        "Liquidity and reversal",
        "Extreme moves followed by volume normalization may revert over multi-hour windows.",
        ("1h", "4h", "1d"),
        ("cash", "buy-and-hold", "random entry baseline"),
        "Chronological event study and forward paper replay with spread stress.",
        "Significant drawdown or no net benefit beyond random entries.",
        ("closed OHLCV bars", "volume", "spread samples", "cost evidence"),
    ),
    Experiment(
        "Event response with causal timestamps",
        "Publicly timestamped announcements might generate delayed price reactions.",
        ("1h", "1d", "1w"),
        ("cash", "matched non-event windows", "randomized event times"),
        "Published-at versus available-at checks, deduplication and embargo.",
        "Leaky event availability, insufficient independent events or negative net return.",
        ("archived primary-source events", "retrieval receipts", "closed OHLCV bars"),
    ),
    Experiment(
        "Adaptive exposure under uncertainty",
        "Small or zero exposure during low-confidence regimes may beat always-invested signals.",
        ("4h", "1d", "1w"),
        ("cash", "fixed 5% position", "buy-and-hold"),
        "Walk-forward calibration of forecast confidence with drawdown and fee budgets.",
        "No calibration advantage or excess turnover erases returns.",
        ("time-stamped forecasts", "closed OHLCV bars", "fee and slippage assumptions"),
    ),
)


def curriculum() -> list[dict]:
    """Return isolated hypotheses; never claim positive returns from specifications."""
    return [asdict(experiment) for experiment in EXPERIMENTS]


def promotion_gate(evidence: dict) -> dict:
    """Fail closed when a proposed strategy has not passed independent evidence checks."""
    required = (
        "primary_data_valid",
        "chronological_out_of_sample",
        "no_future_leakage",
        "costs_verified",
        "double_costs_positive",
        "beats_cash_after_costs",
        "independent_forward_paper",
        "risk_limits_passed",
    )
    missing = [key for key in required if evidence.get(key) is not True]
    return {
        "paper_research_qualified": not missing,
        "missing_evidence": missing,
        "real_money_authorized": False,
        "scope": "Research readiness only. Human consent and exchange verification separate.",
    }
