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
from rlm.v100.competition import (
    command,
    helper_client,
    managed_server,
    require_idle_gpu,
    waiting_researcher,
)
from rlm.v100.evaluation import evaluate_suite
from rlm.v100.goals import load_goal, set_goal
from rlm.v100.inference import generation_conditions
from rlm.v100.protection import compare_reports, execution_hash, file_hash
from rlm.v100.serving import process_identity

OBJECTIVE = (
    "Improve the system's verified capabilities, efficiency and reliability through measured "
    "self-upgrades. Follow the operator's long-term goal and current short/mid-term plans. "
    "Find new independently verifiable learning data; preserve previous versions and skills. "
    "Do not optimize merely for published benchmark scores. No paid APIs or real orders."
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
    startup = Path(result["run"]) / f"server-{state.get('context_window')}.startup.json"
    return {
        **result,
        "running": running,
        "state": state,
        "server_startup": json.loads(startup.read_text()) if startup.exists() else None,
        "evaluation": json.loads(evaluation.read_text()) if evaluation.exists() else None,
        "learning": {
            "completed_cycles": len(progress.get("cycles", [])),
            "last_cycle": progress.get("cycles", [None])[-1] if progress.get("cycles") else None,
            "live_profile": progress.get("live_profile"),
        },
    }


def start(
    root: Path, profile_path: Path, max_context: int = 32768, flash_attention: str | None = None
) -> dict:
    if max_context not in (32768, 65536, 131072):
        raise ValueError("Mission context ceiling must be 32768, 65536 or 131072")
    if flash_attention not in (None, "auto", "on", "off"):
        raise ValueError("Flash Attention must be auto, on or off")
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
        profile.setdefault("resources", {})["mission_max_context"] = max_context
        if flash_attention is not None:
            profile["server"]["flash_attention"] = flash_attention
        from rlm.v100.remote_helper import selected_helper

        load_profile(selected_helper(root), root)
        if (
            profile.get("resources", {}).get("paper_research_enabled", False)
            and not (root / "research/paper/ledger.sqlite3").exists()
        ):
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
    if (root / "research/supervisor").exists():
        atomic_json(
            root / "research/supervisor/pause.json", {"reason": "operator requested mission-stop"}
        )
    record = status(root)
    if not record["running"]:
        return record
    pid = record["pid"]
    try:
        arguments = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if (
            b"mission-loop" not in arguments
            or str(record["run"]).encode() not in arguments
            or os.getpgid(pid) != pid
        ):
            # A process may exit after status() but before reading its command line.
            if not status(root)["running"]:
                return status(root)
            raise ValueError("Mission process identity differs; no process was stopped")
        if process_identity(pid) != record["process_start"]:
            raise ValueError("Mission process identity differs; no process was stopped")
        os.killpg(pid, signal.SIGTERM)  # Only the verified mission session.
    except (FileNotFoundError, ProcessLookupError):
        return status(root)
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
    chosen["server"].update(context_per_slot=context, jinja=True)
    from rlm.v100.mtp_gate import valid

    if not valid(chosen):
        chosen["server"]["draft_model"] = ""
        chosen.setdefault("resources", {}).pop("mtp_validation", None)
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
    from rlm.v100.mission_memory import archive, compress, repair
    from rlm.v100.paper import PaperBook
    from rlm.v100.paper_agents import financial_helper_profile
    from rlm.v100.paper_learning import PaperLearning
    from rlm.v100.research_tools import ResearchTools
    from rlm.v100.serving import assert_served_expert

    folder = root / "research/income-challenge-v1"
    print("Memory retrieval repair:", json.dumps(repair(root)), flush=True)
    suite, pool = folder / "development.jsonl", folder / "pool.jsonl"
    manifest = json.loads((folder / "manifest.json").read_text())
    for path in (suite, pool):
        if file_hash(path) != manifest["files_sha256"][path.name]:
            raise ValueError("Mission curriculum or evaluation suite changed")
    goal = load_goal(root)
    goal = set_goal(root, goal["text"] if goal else OBJECTIVE, suite)
    financial = profile.get("resources", {}).get("paper_research_enabled", False)
    book = PaperBook(root) if financial else None
    try:
        configured = book.state() if book else {"instruments": {}}
        crypto_ready = any(
            item["market"] == "crypto"
            and item["product"] == "spot"
            and item["feed_id"].startswith(("coinbase:", "kraken:"))
            for item in configured["instruments"].values()
        )
    finally:
        if book:
            book.close()
    settings = {
        "schema": "v100-paper-learning-v1",
        "objective": goal["text"],
        "crypto": crypto_ready,
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
    if financial:
        helper_path = financial_helper_profile(helper_path, root)
    tool_reader = ResearchTools(root, {}, "A")
    note(
        directory,
        "collecting-income-sources" if financial else "goal-research",
        goal=goal["text"],
        paper_trades="blocked until verified fees and feeds exist",
    )
    for url in SEEDS if financial else ():
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
    max_context = profile.get("resources", {}).get("mission_max_context", 32768)
    if max_context not in (32768, 65536, 131072):
        raise ValueError("Invalid snapshotted mission context ceiling")
    for context in (131072, 65536, 32768, 16384, 8192):
        if context > max_context:
            continue
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
                    "income-research" if financial else "goal-research",
                    context_window=context,
                    free_vram_gib=round(free, 2),
                    flash_attention_requested=serving["server"]["flash_attention"],
                    kv_cache_type=serving["server"]["cache_type"],
                )
                with waiting_researcher(
                    helper_path, root, directory / "research-helper.log", fallback=serving
                ) as helper:
                    try:
                        if financial:
                            with PaperLearning(root, paper_settings) as income:
                                income.research(serving, helper)
                        else:
                            from rlm.v100.competition import research_task

                            research_task(
                                helper_client(helper, root, "A"),
                                "A",
                                {
                                    "role": "researcher",
                                    "brief": "Investigate the operator goal and propose a new independently testable self-upgrade.",
                                },
                                [],
                                root,
                            )
                        compress(root, serving)
                    except (ValueError, RuntimeError, OSError) as error:
                        print("Goal research error; see activity logs:", error, flush=True)
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
                if serving.get("resources", {}).get("public_benchmarks"):
                    from rlm.v100.public_benchmarks import evaluate as public_evaluate
                    from rlm.v100.public_benchmarks import reusable as reusable_public

                    public_path = directory / "public-baseline.json"
                    note(directory, "official-public-baseline", context_window=context)
                    try:
                        cached_public = reusable_public(root, serving)
                        if cached_public:
                            atomic_json(public_path, json.loads(cached_public.read_text()))
                            atomic_json(
                                directory / "public-baseline-reused.json",
                                {"source": str(cached_public), "sha256": file_hash(cached_public)},
                            )
                            print("Reused pinned official baseline:", cached_public, flush=True)
                        else:
                            from rlm.v100.public_benchmarks import resume_baseline

                            resumed = resume_baseline(root, serving, public_path)
                            if resumed:
                                print(
                                    "Resumed exact matching official baseline:", resumed, flush=True
                                )
                            public_evaluate(root, serving, public_path)
                        serving["resources"]["public_baseline"] = str(public_path)
                        serving["resources"]["public_baseline_sha256"] = file_hash(public_path)
                    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
                        serving["resources"].pop("public_baseline", None)
                        atomic_json(
                            directory / "public-baseline-deferred.json",
                            {
                                "error": str(error)[:500],
                                "training": "paused until official baseline completes; research continues",
                            },
                        )
                    atomic_json(path, serving)
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
            paper_config=paper_settings if financial else None,
            initial_update=True,
        )
    finally:
        if financial:
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
        from contextlib import ExitStack

        from rlm.v100 import desktop, drones
        from rlm.v100.mission_chat import alongside

        with ExitStack() as services:
            services.enter_context(alongside(root, directory))
            if profile.get("resources", {}).get("resident_drones"):
                services.enter_context(drones.alongside(root))
                services.enter_context(desktop.alongside(root))
            run(root, profile, directory)
    except KeyboardInterrupt:
        note(directory, "stopped", checkpoints="retained")
    except Exception as error:
        note(directory, "failed", error=type(error).__name__, detail=str(error)[:600])
        raise
