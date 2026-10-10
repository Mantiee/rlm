"""Host-generated income and weight-update reports; no model claims become profit."""

import fcntl
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from rlm.v100.common import atomic_json
from rlm.v100.experiments import SharedLab
from rlm.v100.paper import PaperBook
from rlm.v100.paper_reports import summarize
from rlm.v100.researchers import compact_result


def snapshot(root: Path) -> dict:
    """Fast read-only facts for chat, without rerunning a financial ledger audit."""
    from rlm.v100.dashboard_snapshot import supplement
    from rlm.v100.mission import status

    mission = status(root)
    value, errors = {}, []
    path = root / "research/mission/latest-report.json"
    if path.exists():
        if path.stat().st_size > 2 * 2**20:
            errors.append("Aggregate snapshot exceeds 2 MiB")
        else:
            try:
                saved = json.loads(path.read_text())
                if saved.get("mission_evidence", {}).get("run") == mission.get("run"):
                    value = saved
            except (OSError, ValueError, AttributeError) as error:
                errors.append("Aggregate unavailable: " + str(error)[:200])
    value = supplement(root, mission, value, errors)
    value["phase"] = mission.get("state", {}).get("phase", mission.get("phase"))
    value["running"] = mission.get("running")
    value["collection_errors"] = errors
    value["facts_scope"] = (
        "Stored same-run audit with its original timestamp; live phase and file receipts read independently"
    )
    return value


def report(root: Path) -> dict:
    from rlm.v100.mission import status

    mission = status(root)
    run = Path(mission["run"]) if mission.get("run") else None
    learning_path = run / "learning/state.json" if run else None
    learning = (
        json.loads(learning_path.read_text()) if learning_path and learning_path.exists() else {}
    )
    cycles = learning.get("cycles", [])
    from rlm.v100.paper_outcomes import records as outcome_records

    outcomes = outcome_records(root)
    backtest_paths = sorted(
        (root / "research/backtests").glob("run-*/report.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )[:4]
    backtests = []
    for path in backtest_paths:
        previous = json.loads(path.read_text())
        backtests.append(
            {
                key: previous[key]
                for key in (
                    "report",
                    "parameters",
                    "sources",
                    "scope",
                    "limitations",
                    "selected_lookback_on_training_only",
                )
            }
        )
        backtests[-1]["development_test"] = {
            key: item for key, item in previous["development_test"].items() if key != "trace"
        }
        from rlm.v100.market_research import triage

        for key in ("fetched_at", "buy_hold_baseline", "double_cost_stress", "data_window"):
            backtests[-1][key] = previous.get(key)
        backtests[-1]["triage"] = triage(previous)
    metrics = {}
    if run:
        paths = sorted(
            (run / "learning").glob("update-*/**/metrics.jsonl"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )[:2]
        for path in paths:
            with path.open("rb") as handle:
                offset = max(0, path.stat().st_size - 16384)
                handle.seek(offset)
                if offset:
                    handle.readline()
                rows = [json.loads(line) for line in handle if line.endswith(b"\n")]
            metrics[str(path)] = rows[-1] if rows else None
    book = PaperBook(root, read_only=(root / "research/paper/ledger.sqlite3").exists())
    initialized = True
    try:
        state = book.state()
        paper = summarize(book)
    except ValueError as error:
        if str(error) != "Initialize the paper lab first":
            raise
        initialized = False
        state = {"instruments": {}, "fee_profiles": {}, "quotes": {}}
        paper = {
            "currency": "unknown",
            "branches": {},
            "trades": [],
            "stale_symbols": [],
            "expired_fee_profiles": [],
            "personal_income_tax": "unknown",
        }
    finally:
        book.close()
    shared_path = root / "research/state/competition.sqlite3"
    shared = SharedLab(shared_path, read_only=shared_path.exists())
    try:
        ideas = [
            {"branch": event["branch"], **compact_result(event["payload"])}
            for event in shared.recent(30)
            if event["kind"] == "worker-result"
            and event["payload"].get("mission_run") == (str(run) if run else None)
            and not event["payload"].get("research_quality", {}).get("historical_audit")
            and event["payload"].get("research_quality", {}).get("eligible_for_review", True)
        ][-6:]
    finally:
        shared.close()
    missing = [
        label
        for label, field in (
            ("registered instruments", "instruments"),
            ("documented fees", "fee_profiles"),
            ("fresh quote feeds", "quotes"),
        )
        if not state[field]
    ]
    missing.extend("stale quote: " + symbol for symbol in paper["stale_symbols"])
    missing.extend("expired fees: " + identity for identity in paper["expired_fee_profiles"])
    value = {
        "updated_at": datetime.now(UTC).isoformat(),
        "running": mission["running"],
        "phase": mission.get("state", {}).get("phase", mission.get("phase")),
        "evaluation": mission.get("evaluation"),
        "context_window": mission.get("state", {}).get("context_window"),
        "completed_learning_cycles": len(cycles),
        "available_audited_paper_outcomes": len(outcomes),
        "accepted_weight_updates_this_run": sum(
            row.get("status", "").startswith("selected for next serving") for row in cycles
        ),
        "last_learning_cycle": cycles[-1] if cycles else None,
        "training_metrics": metrics,
        "fixed_suite_baseline": {
            "passed": mission.get("state", {}).get("baseline_passed"),
            "total": mission.get("state", {}).get("baseline_total"),
        },
        "learning_scope": "Verified formal exercises and audited realized paper postmortems, including losses; hypotheses are not proven strategies",
        "paper_blockers": missing,
        "paper": {
            "currency": paper["currency"],
            "branches": paper["branches"],
            "executed_fills": len(paper["trades"]) if initialized else None,
            "real_money_ready": False,
            "tax": paper["personal_income_tax"],
        },
        "research_hypotheses_not_verified_income": ideas,
        "recent_exploratory_backtests": backtests,
        "evidence_required": "A timestamped independent test and net result after costs before claiming an income edge",
    }
    from rlm.v100.planning import read as read_plans

    value["plans"] = read_plans(root)
    from rlm.v100.mission_evidence import collect

    value["mission_evidence"] = collect(root)
    for name, path in (
        ("drones", root / "research/drones-status.json"),
        ("external_compute", root / "research/compute-status.json"),
        ("desktop", root / "research/desktop/status.json"),
        ("supervisor", root / "research/supervisor/status.json"),
        ("income_opportunities", root / "research/income-opportunities/status.json"),
        ("income_dispatch", root / "research/income-opportunities/dispatch.json"),
        ("income_work", root / "research/income-work/status.json"),
        ("market_research", root / "research/market-research/status.json"),
        ("research_quality", root / "research/research-quality/latest.json"),
        ("source_acquisition", root / "research/source-acquisition/status.json"),
        ("owned_cpu_dispatch", root / "research/goal-compute/status.json"),
        ("paper_fee_source", root / "research/paper/fee-source-status.json"),
    ):
        value[name] = json.loads(path.read_text()) if path.exists() else {"state": "not started"}
    from rlm.v100.public_benchmarks import progress_for_run

    value["official_benchmark"] = progress_for_run(root, run)
    baseline_reports = value["mission_evidence"].get("reports", [])
    baseline = next(
        (row for row in baseline_reports if isinstance(row, dict) and row.get("cases")), None
    )
    if baseline and value["fixed_suite_baseline"]["total"] is None:
        value["fixed_suite_baseline"] = {"passed": baseline["passed"], "total": baseline["cases"]}
    from rlm.v100.goals import load_goal

    current_goal = (load_goal(root) or {}).get("id")
    for name in (
        "income_opportunities",
        "income_dispatch",
        "income_work",
        "research_quality",
        "source_acquisition",
        "owned_cpu_dispatch",
    ):
        if value[name].get("goal_id") != current_goal:
            value[name] = {"state": "not started for current goal", "goal_id": current_goal}
    live_path = learning.get("live_profile")
    active_profile = (
        json.loads(Path(live_path).read_text()) if live_path and Path(live_path).exists() else {}
    )
    value["accepted_branch_versions"] = active_profile.get("resources", {}).get(
        "branch_lineages", {}
    )
    value["architecture_history"] = active_profile.get("resources", {}).get(
        "architecture_history", []
    )
    value["active_foundation_expert"] = active_profile.get("resources", {}).get(
        "active_foundation_expert"
    )
    lines = [
        f"Phase: {value['phase']} | Running: {value['running']}",
        f"Learning cycles: {len(cycles)} | Accepted weight updates in this run: {value['accepted_weight_updates_this_run']}",
        "Successful optimizer updates observed in this run: "
        + str(value["mission_evidence"]["optimizer_updates_observed"])
        + " (None means no verified counters; not proof of zero)",
        f"Audited paper outcomes available for training: {len(outcomes)} (including losses; availability is not proof that an adapter used them)",
        f"Fixed baseline: {value['fixed_suite_baseline']['passed']}/{value['fixed_suite_baseline']['total']}",
        "Accepted independent branches: "
        + (", ".join(value["accepted_branch_versions"]) or "none yet; seeded from the same base"),
        "Official benchmark: " + json.dumps(value["official_benchmark"]),
        "Resident workers: " + json.dumps(value["drones"]),
        "External compute: " + json.dumps(value["external_compute"]),
        "Owned CPU training dispatch: " + json.dumps(value["owned_cpu_dispatch"]),
        "Income work: " + json.dumps(value["income_work"]),
        "Research quality: " + json.dumps(value["research_quality"]),
        "Paper fee source: " + json.dumps(value["paper_fee_source"]),
        "Active architecture: " + str(value["active_foundation_expert"] or "original"),
        "Private desktop: " + json.dumps(value["desktop"]),
    ]
    if cycles:
        lines.append("Last learning cycle: " + json.dumps(cycles[-1]))
    for branch, portfolio in paper["branches"].items():
        lines.append(
            f"{branch}: net paper P&L {portfolio['net_pnl_before_personal_tax']} {paper['currency']} before personal tax; modeled costs {portfolio['modeled_costs']}"
        )
    lines.append(
        "Paper blockers: " + (", ".join(missing) or "Check freshness and instrument eligibility")
    )
    for path, row in metrics.items():
        lines.append(f"Training {path}: {json.dumps(row)}")
    for idea in ideas:
        lines.extend(
            [
                f"\n{idea['branch']} / {idea.get('model', 'unknown')} [{idea.get('status', 'unverified hypothesis')}]",
                "Idea: " + idea.get("hypothesis", ""),
                "Next test: " + idea.get("suggested_test", ""),
            ]
        )
    destination = root / "research/mission/latest-report.txt"
    for backtest in backtests:
        lines.append("Backtest [exploratory]: " + json.dumps(backtest))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (destination.parent / "report.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        atomic_json(root / "research/mission/latest-report.json", value)
        temporary = destination.with_suffix(".txt.tmp")
        temporary.write_text("\n".join(lines) + "\n")
        temporary.replace(destination)
    return value


def watch(root: Path) -> None:
    """Follow native activity with UTC-date rollover and readable Polish local times."""
    offsets = {}
    try:
        while True:
            path = (
                root
                / "research/logs/activity"
                / datetime.now(UTC).strftime("%F")
                / "timeline.jsonl"
            )
            if path.exists():
                with path.open("rb") as handle:
                    offset = offsets.get(str(path))
                    if offset is None:
                        offset = max(0, path.stat().st_size - 65536)
                        handle.seek(offset)
                        if offset:
                            handle.readline()
                    else:
                        handle.seek(offset)
                    while True:
                        position = handle.tell()
                        line = handle.readline()
                        if not line or not line.endswith(b"\n"):
                            handle.seek(position)
                            break
                        event = json.loads(line)
                        clock = (
                            datetime.fromisoformat(event["time"])
                            .astimezone(ZoneInfo("Europe/Warsaw"))
                            .strftime("%H:%M:%S")
                        )
                        print(
                            f"\n[{clock}] {event['branch']}/{event['actor']} | {event['kind']}",
                            flush=True,
                        )
                        print(
                            json.dumps(event["payload"], ensure_ascii=False, indent=2), flush=True
                        )
                    offsets[str(path)] = handle.tell()
            time.sleep(1)
    except KeyboardInterrupt:
        return


def main() -> None:
    """Dedicated reporting entry point; no inference CLI initialization."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("command", choices=["mission-report"])
    args = parser.parse_args()
    report(args.root)


if __name__ == "__main__":
    main()
