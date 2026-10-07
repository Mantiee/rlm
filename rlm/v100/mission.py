"""Background income research and guarded learning, with adaptive context capacity."""

import copy
import fcntl
import json
import os
import signal
import subprocess
import uuid
from pathlib import Path

from rlm.v100.common import atomic_json, load_profile
from rlm.v100.competition import command, helper_client, managed_server, require_idle_gpu
from rlm.v100.evaluation import evaluate_suite
from rlm.v100.goals import load_goal, set_goal
from rlm.v100.inference import generation_conditions
from rlm.v100.protection import compare_reports, execution_hash, file_hash
from rlm.v100.serving import process_identity

OBJECTIVE = (
    "Find the fastest and largest lawful, repeatable net income with no deposits or paid APIs. "
    "Compare sports research, crypto and equities in forward-only paper simulations, plus "
    "other zero-deposit income opportunities. Include documented fees, funding, financing, "
    "slippage, FX and tax limitations. Minimize losses and correlated exposure. Research "
    "primary public sources, quarterly filings and relevant social evidence without future "
    "data. Compete as A/B, share useful evidence, use CPU researchers, and independently "
    "test self-upgrades. Train only independently verified examples; retain originals, "
    "prior model versions and checkpoints. No real orders, paid services, sales, account "
    "creation or spending are executed by this controller. Do not claim guaranteed profit "
    "or universal zero forgetting."
)
SEEDS = (
    "https://help.coinbase.com/en/exchange/trading-and-funding/exchange-fees",
    "https://api.exchange.coinbase.com/products/BTC-USD/book?level=1",
    "https://api.exchange.coinbase.com/products/ETH-USD/book?level=1",
)


def status(root: Path) -> dict:
    path = root / "research/mission/active.json"
    if not path.exists():
        return {"running": False, "phase": "not started"}
    result = json.loads(path.read_text())
    try:
        running = process_identity(result["pid"]) == result["process_start"]
    except (OSError, KeyError):
        running = False
    phase = Path(result["run"]) / "status.json"
    learning = Path(result["run"]) / "learning/state.json"
    progress = json.loads(learning.read_text()) if learning.exists() else {}
    state = json.loads(phase.read_text()) if phase.exists() else {"phase": "starting"}
    evaluation = Path(result["run"]) / f"baseline-{state.get('context_window')}.progress.json"
    return {
        **result,
        "running": running,
        "state": state,
        "evaluation": json.loads(evaluation.read_text()) if evaluation.exists() else None,
        "learning": {
            "completed_cycles": len(progress.get("cycles", [])),
            "last_cycle": progress.get("cycles", [None])[-1] if progress.get("cycles") else None,
            "live_profile": progress.get("live_profile"),
        },
    }


def start(root: Path, profile_path: Path) -> dict:
    root = root.resolve()
    directory = root / "research/mission"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "start.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if status(root)["running"]:
            raise FileExistsError("Mission is already running; use mission-status")
        folder = root / "research/income-challenge-v1"
        if not (folder / "manifest.json").exists():
            raise ValueError("Prepare the disjoint challenge curriculum before starting a mission")
        profile = load_profile(profile_path, root)
        from rlm.v100.remote_helper import selected_helper

        load_profile(selected_helper(root), root)
        if not (root / "research/paper/ledger.sqlite3").exists():
            raise ValueError("Missing initialized paper ledger; run paper-init")
        require_idle_gpu()
        run = directory / ("run-" + uuid.uuid4().hex[:12])
        run.mkdir()
        atomic_json(run / "input-profile.json", profile)
        log = run / "controller.log"
        with log.open("a") as handle:
            process = subprocess.Popen(
                command(root, run / "input-profile.json", "mission-loop", "--run", str(run)),
                stdout=handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONNOUSERSITE": "1"},
            )
        record = {
            "pid": process.pid,
            "process_start": process_identity(process.pid),
            "run": str(run),
            "log": str(log),
        }
        atomic_json(directory / "active.json", record)
        return {
            **record,
            "running": True,
            "note": "Started in background; inspect mission-status and controller.log",
        }


def stop(root: Path) -> dict:
    record = status(root)
    if not record["running"]:
        return record
    pid = record["pid"]
    arguments = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    if (
        b"mission-loop" not in arguments
        or str(record["run"]).encode() not in arguments
        or os.getpgid(pid) != pid
    ):
        raise ValueError("Mission process identity differs; no process was stopped")
    os.killpg(pid, signal.SIGTERM)  # This session contains only this mission and its children.
    return {"stop_requested": True, "pid": pid, "run": record["run"]}


def note(run: Path, phase: str, **values) -> None:
    atomic_json(run / "status.json", {"phase": phase, **values})
    print(json.dumps({"phase": phase, **values}, ensure_ascii=False), flush=True)


def gpu_free_gib() -> float:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", "0"],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip()) / 1024


def setup_profile(profile: dict, context: int) -> dict:
    chosen = copy.deepcopy(profile)
    chosen["runtime"].update(context_window=context, tool_protocol="json")
    chosen["server"].update(context_per_slot=context, draft_model="", jinja=True)
    chosen["memory"]["database"] = "research/state/mission-memory.sqlite3"
    chosen["training"]["max_steps"] = min(50, chosen["training"]["max_steps"])
    return chosen


def setup_helper(profile: dict) -> dict:
    chosen = copy.deepcopy(profile)
    from rlm.v100.remote_helper import remote_profile

    if remote_profile(chosen):
        return chosen
    chosen["runtime"].update(
        tool_protocol="json",
        max_output_tokens=768,
        max_timeout=max(900, chosen["runtime"]["max_timeout"]),
    )
    chosen["resources"]["compact_research_tools"] = True
    return chosen


def previous_baseline(root: Path, profile: dict, suite: Path) -> tuple[dict, Path] | None:
    """Reuse only complete matching evaluations, never partially finished cases."""
    contexts = root / "research/mission"
    paths = list(contexts.glob(f"run-*/baseline-{profile['runtime']['context_window']}.json"))
    if not paths:
        return None
    suite_hash = file_hash(suite)
    model_hash = file_hash(Path(profile["server"]["model"]))
    execution = execution_hash(profile)
    case_ids = {json.loads(line)["id"] for line in suite.read_text().splitlines() if line.strip()}
    for path in sorted(paths, key=lambda item: item.stat().st_mtime_ns, reverse=True)[:16]:
        report = json.loads(path.read_text())
        if (
            report.get("schema") != "v100-quality-v1"
            or report.get("suite_sha256") != suite_hash
            or report.get("model_sha256") != model_hash
            or report.get("execution_sha256") != execution
            or report.get("generation") != generation_conditions(profile)
            or report.get("memory_mode") != "fixed prompt fixtures; no live retrieval"
            or {row["id"] for row in report.get("cases", [])} != case_ids
        ):
            continue
        compare_reports(report, report)
        return report, path
    return None


def run(root: Path, profile: dict, directory: Path) -> None:
    from rlm.v100.continuous import learn_loop
    from rlm.v100.mission_memory import archive, compress
    from rlm.v100.paper import PaperBook
    from rlm.v100.paper_agents import financial_helper_profile
    from rlm.v100.paper_learning import PaperLearning
    from rlm.v100.research_tools import ResearchTools
    from rlm.v100.serving import assert_served_expert

    folder = root / "research/income-challenge-v1"
    suite, pool = folder / "development.jsonl", folder / "pool.jsonl"
    manifest = json.loads((folder / "manifest.json").read_text())
    for path in (suite, pool):
        if file_hash(path) != manifest["files_sha256"][path.name]:
            raise ValueError("Mission curriculum or evaluation suite changed")
    goal = load_goal(root)
    goal = set_goal(root, goal["text"] if goal else OBJECTIVE, suite)
    settings = {
        "schema": "v100-paper-learning-v1",
        "objective": goal["text"],
        "crypto": False,
        "ciks": [],
        "sec_contact": "",
        "observer_interval": 60,
        "research_rounds": 1,
        "other_income_rnd": True,
    }
    paper_settings = directory / "income-settings.json"
    atomic_json(paper_settings, settings)
    from rlm.v100.remote_helper import selected_helper

    helper_original = setup_helper(load_profile(selected_helper(root), root))
    helper_path = directory / "research-helper.json"
    atomic_json(helper_path, helper_original)
    helper_path = financial_helper_profile(helper_path, root)
    tool_reader = ResearchTools(root, {}, "A")
    note(
        directory,
        "collecting-income-sources",
        goal=goal["text"],
        paper_trades="blocked until verified fees and feeds exist",
    )
    for url in SEEDS:
        try:
            tool_reader.execute("read_public_page", {"url": url})
        except Exception as error:
            archive(
                root,
                "source-error:" + url,
                "Research source unavailable: " + url + "\n" + str(error)[:300],
            )
            print("Source unavailable:", url, type(error).__name__, str(error)[:300], flush=True)
    selected, baseline, selected_path = None, None, None
    for context in (32768, 16384, 8192):
        chosen = setup_profile(profile, context)
        path = directory / f"profile-{context}.json"
        atomic_json(path, chosen)
        require_idle_gpu()
        note(directory, "loading-context", context_window=context)
        try:
            with managed_server(path, root, directory / f"server-{context}.log") as serving:
                free = gpu_free_gib()
                if free < 4:
                    raise RuntimeError("Context leaves less than 4 GiB inference headroom")
                client = helper_client(serving, root)
                assert_served_expert(client, serving, root)
                selected, selected_path = serving, path
                note(
                    directory,
                    "income-research",
                    context_window=context,
                    free_vram_gib=round(free, 2),
                )
                with managed_server(helper_path, root, directory / "research-helper.log") as helper:
                    with PaperLearning(root, paper_settings) as income:
                        try:
                            income.research(serving, helper)
                            compress(root, serving)
                        except (ValueError, RuntimeError, OSError) as error:
                            print("Income research error; see activity logs:", error, flush=True)
                note(directory, "baseline-before-weight-updates", context_window=context)
                cached = previous_baseline(root, serving, suite)
                if cached:
                    baseline, original_report = cached
                    atomic_json(directory / f"baseline-{context}.json", baseline)
                    atomic_json(
                        directory / "baseline-reused.json",
                        {"source": str(original_report), "sha256": file_hash(original_report)},
                    )
                    print("Reused complete matching baseline:", original_report, flush=True)
                else:
                    baseline = evaluate_suite(
                        client, serving, suite, directory / f"baseline-{context}.json"
                    )
            break
        except (RuntimeError, TimeoutError) as error:
            selected = None
            note(directory, "context-retry", context_window=context, detail=str(error)[:300])
    if selected is None or baseline is None:
        raise RuntimeError("No tested context profile fits; inspect per-context server logs")
    note(
        directory,
        "research-and-learning-loop",
        context_window=selected["runtime"]["context_window"],
        baseline_passed=sum(row["passed"] for row in baseline["cases"]),
        baseline_total=len(baseline["cases"]),
        live_profile=str(selected_path),
        updates="verified candidate LoRAs with finite regression gates",
    )
    try:
        learn_loop(
            selected,
            root,
            pool,
            directory / "learning",
            suite,
            [baseline],
            helper_path,
            cycles=0,
            interval=300,
            train_timeout=7200,
            paper_config=paper_settings,
            initial_update=True,
        )
    finally:
        book = PaperBook(root)
        try:
            from rlm.v100.paper_reports import write_report

            print("Final income paper report:", write_report(book), flush=True)
        finally:
            book.close()


def worker(root: Path, profile: dict, directory: Path) -> None:
    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        run(root, profile, directory)
    except KeyboardInterrupt:
        note(directory, "stopped", checkpoints="retained")
    except Exception as error:
        note(directory, "failed", error=type(error).__name__, detail=str(error)[:600])
        raise
