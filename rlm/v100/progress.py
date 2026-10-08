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
    book = PaperBook(root)
    try:
        state = book.state()
        paper = summarize(book)
    finally:
        book.close()
    shared = SharedLab(root / "research/state/competition.sqlite3")
    try:
        ideas = [
            {"branch": event["branch"], **compact_result(event["payload"])}
            for event in shared.recent(30)
            if event["kind"] == "worker-result"
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
            "executed_fills": len(paper["trades"]),
            "real_money_ready": False,
            "tax": paper["personal_income_tax"],
        },
        "research_hypotheses_not_verified_income": ideas,
        "recent_exploratory_backtests": backtests,
        "evidence_required": "A timestamped independent test and net result after costs before claiming an income edge",
    }
    from rlm.v100.planning import read as read_plans

    value["plans"] = read_plans(root)
    for name, path in (
        ("drones", root / "research/drones-status.json"),
        ("desktop", root / "research/desktop/status.json"),
        ("supervisor", root / "research/supervisor/status.json"),
        ("official_benchmark", root / "research/public-benchmarks/progress.json"),
    ):
        value[name] = json.loads(path.read_text()) if path.exists() else {"state": "not started"}
    live_path = learning.get("live_profile")
    active_profile = (
        json.loads(Path(live_path).read_text()) if live_path and Path(live_path).exists() else {}
    )
    value["accepted_branch_versions"] = active_profile.get("resources", {}).get(
        "branch_lineages", {}
    )
    lines = [
        f"Phase: {value['phase']} | Running: {value['running']}",
        f"Learning cycles: {len(cycles)} | Accepted weight updates in this run: {value['accepted_weight_updates_this_run']}",
        f"Audited paper outcomes available for training: {len(outcomes)} (including losses; availability is not proof that an adapter used them)",
        f"Fixed baseline: {value['fixed_suite_baseline']['passed']}/{value['fixed_suite_baseline']['total']}",
        "Accepted independent branches: "
        + (", ".join(value["accepted_branch_versions"]) or "none yet; seeded from the same base"),
        "Official benchmark: " + json.dumps(value["official_benchmark"]),
        "Resident workers: " + json.dumps(value["drones"]),
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
                f"\n{idea['branch']} / {idea.get('model', 'unknown')} [unverified hypothesis]",
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
