"""Recomputed historical postmortems, never labels for future price predictions."""

import hashlib
import json
from pathlib import Path

from rlm.v100.backtesting import bars_from_source, simulate, source_quality
from rlm.v100.paper import sha, timestamp


def verified(path: Path) -> dict:
    if path.is_symlink() or path.stat().st_size > 2 * 2**20:
        raise ValueError("Unsafe or oversized historical report")
    report = json.loads(path.read_text())
    bodies = []
    for index, source in enumerate(report["sources"]):
        archive = path.parent / f"source-{index}.json"
        if archive.is_symlink() or archive.stat().st_size > 2 * 2**20:
            raise ValueError("Unsafe or oversized archived source")
        body = archive.read_text()
        if hashlib.sha256(body.encode()).hexdigest() != source["sha256"]:
            raise ValueError("Backtest archived source changed")
        bodies.append(body)
    bars = bars_from_source(report["sources"][0]["url"], bodies[0], timestamp(report["fetched_at"]))
    quality = source_quality(bars)
    if quality["gaps"] or quality["large_opening_jumps"]:
        raise ValueError("Archived bars require source review; excluded from postmortem training")
    events = None
    if len(bodies) > 1:
        events = []
        for item in json.loads(bodies[1]):
            published, available = timestamp(item["published_at"]), timestamp(item["available_at"])
            if not published <= available <= timestamp(report["fetched_at"]):
                raise ValueError("Invalid archived event timing")
            events.append(available.timestamp())
    arguments = report["parameters"]
    split = int(len(bars) * 0.7)
    choices = {
        length: simulate(
            bars,
            21,
            split,
            arguments["rule"],
            length,
            arguments["fee_bps"],
            arguments["slippage_bps"],
            events,
        )
        for length in (5, 10, 20)
    }
    chosen = max(choices, key=lambda length: choices[length]["net_return"])
    test = simulate(
        bars,
        split,
        len(bars),
        arguments["rule"],
        chosen,
        arguments["fee_bps"],
        arguments["slippage_bps"],
        events,
    )
    stress = simulate(
        bars,
        split,
        len(bars),
        arguments["rule"],
        chosen,
        arguments["fee_bps"] * 2,
        arguments["slippage_bps"] * 2,
        events,
    )
    baseline = simulate(
        bars, split, len(bars), "buy_hold", chosen, arguments["fee_bps"], arguments["slippage_bps"]
    )
    if (
        report["bars"] != len(bars)
        or report["split_index"] != split
        or report["selected_lookback_on_training_only"] != chosen
        or report["development_test"] != test
        or report.get("double_cost_stress") != stress
        or report.get("buy_hold_baseline") != baseline
    ):
        raise ValueError("Historical report differs from independently rerun simulator")
    outcome = test["net_return"]
    for value in (test, stress, baseline):
        peak, drawdown = 1.0, 0.0
        for point in value["trace"]:
            equity = point["marked_equity"]
            if point["signal_available_at"] > point["fill_at"]:
                raise ValueError("Future information in historical trace")
            peak = max(peak, equity)
            drawdown = max(drawdown, 1 - equity / peak)
        if (
            abs(value["net_return"] - (value["trace"][-1]["marked_equity"] - 1)) > 1e-10
            or abs(value["max_sampled_drawdown"] - drawdown) > 1e-10
        ):
            raise ValueError("Historical equity trace does not reproduce reported metrics")
    label = {
        "sample_result": "profit" if outcome > 0 else "loss" if outcome < 0 else "flat",
        "beats_cash_in_sample": outcome > 0,
        "beats_buy_hold_in_sample": outcome > baseline["net_return"],
        "positive_at_double_assumed_cost": stress["net_return"] > 0,
        "repeatable_income_established": False,
        "drawdown_is_standard_deviation": False,
        "negative_return_proves_data_corruption": False,
        "next_evidence": "independent forward paper observations with documented costs",
    }

    def compact(value):
        return {key: value[key] for key in ("net_return", "max_sampled_drawdown", "fills")}

    inputs = {
        "parameters": arguments,
        "historical_test": compact(test),
        "double_cost_test": compact(stress),
        "buy_hold": compact(baseline),
    }
    identity = "historical-review-" + sha(inputs)
    return {
        "group": identity,
        "document_ids": ["backtest-market-" + sha(arguments["price_url"].split("?")[0])],
        "messages": [
            {
                "role": "user",
                "content": "Retrospective exploratory simulation review, NOT a forecast. Use these already-observed results to assess the sample, cost sensitivity and what remains unproven. Return JSON.\n"
                + json.dumps(inputs, sort_keys=True),
            },
            {"role": "assistant", "content": json.dumps(label, sort_keys=True)},
        ],
        "verification": {"kind": "historical_postmortem", "accepted": True, "report": str(path)},
    }


def records(root: Path) -> list[dict]:
    result = []
    for path in sorted((root / "research/backtests").glob("run-*/report.json")):
        try:
            row = verified(path)
        except (ValueError, KeyError, OSError, TypeError, IndexError, ZeroDivisionError):
            continue  # Older reports without complete artifacts are not training evidence.
        result.append(row)
    return result
