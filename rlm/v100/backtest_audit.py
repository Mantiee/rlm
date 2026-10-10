"""Bounded reproducibility audit; never confuse a losing strategy with bad data."""

import hashlib
import json
import time
from pathlib import Path

from rlm.v100.common import atomic_json


def inspect_report(path: Path) -> dict:
    from rlm.v100.backtest_learning import verified

    result = {
        "report": str(path),
        "state": "quarantined",
        "postmortem_training_admitted": False,
        "forecast_training_admitted": False,
        "scope": "Archived source checksums, chronology, OHLCV ranges and simulator replay. This cannot certify vendor accuracy, causality or a future edge.",
    }
    try:
        record = verified(path)
        stored = json.loads(path.read_text())
        fee = stored["parameters"]["fee_bps"] / 10000
        slip = stored["parameters"]["slippage_bps"] / 10000
        pairs = stored["development_test"]["fills"] / 2
        flat_return = (((1 - fee) * (1 - slip)) / ((1 + fee) * (1 + slip))) ** pairs - 1
        result.update(
            state="reproduced",
            report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            postmortem_training_admitted=True,
            verified_sample=json.loads(record["messages"][-1]["content"]),
            flat_price_cost_only_return=flat_return,
            cost_explanation="Illustration for the reported number of full-capital buy/sell pairs at constant prices. Actual return also depends on price changes. Fee and slippage apply to each leg; these remain assumptions, not certified fees.",
        )
    except (ValueError, OSError, KeyError, TypeError, IndexError, ZeroDivisionError) as error:
        result["reason"] = str(error)[:400]
    return result


def run(root: Path, limit: int = 20) -> dict:
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError("Audit 1-20 recent stored reports")
    paths = sorted(
        (root / "research/backtests").glob("run-*/report.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )[:limit]
    result = {
        "updated": time.time(),
        "reports": [inspect_report(path) for path in paths],
        "metric_correction": "max_sampled_drawdown is peak-to-trough equity loss, not standard deviation. Double-cost net return is a different metric. Loss alone is not evidence of a data glitch.",
        "originals": "retained unchanged",
        "scope": "Recent reports only; every historical training record is checked again at admission. Unverified reports are excluded, including profitable ones.",
    }
    atomic_json(root / "research/backtests/audit-status.json", result)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().root), indent=2))
