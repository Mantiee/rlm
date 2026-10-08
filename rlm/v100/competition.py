"""Sequential GPU branches with CPU or separate remote-GPU research."""

import copy
import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from pathlib import Path
from urllib.parse import urlparse

import requests

from rlm.clients.llamacpp import LlamaCppClient
from rlm.v100.activity import ActivityLog
from rlm.v100.common import atomic_json, load_profile
from rlm.v100.experiments import SharedLab, judge_duel, load_duel, plan_duel
from rlm.v100.researchers import available_ram_gib, research_task, review_research


def command(root: Path, profile: Path, *arguments: str) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "rlm.v100.cli",
        "--root",
        str(root),
        "--profile",
        str(profile),
        *arguments,
    ]


def require_idle_gpu() -> None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    if len(result.stdout.strip().splitlines()) != 1 or int(result.stdout.strip()) < 28 * 1024:
        raise ValueError(
            "Need one GPU with 28 GiB free. Stop your inference servers first; this runner never kills them."
        )


@contextmanager
def managed_server(profile_path: Path, root: Path, log_path: Path):
    profile = load_profile(profile_path, root)
    from rlm.v100.remote_helper import remote_profile

    if remote_profile(profile):
        client = helper_client(profile, root)
        from rlm.v100.remote_helper import OllamaResearchClient

        assert isinstance(client, OllamaResearchClient)
        client.timeout = min(15, client.timeout)
        client.identity()
        measured = client.loaded()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as log:
            log.write(
                json.dumps({"remote_helper": profile["runtime"]["base_url"], "loaded": measured})
                + "\n"
            )
        # The Windows process is owned by its launcher, never by this context.
        yield profile
        return
    origin = urlparse(profile["runtime"]["base_url"])
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", origin.port))
    cpu = profile.get("resources", {}).get("device") == "cpu"
    minimum = profile.get("resources", {}).get("min_available_ram_gib", 6)
    if cpu and available_ram_gib() < minimum:
        raise ValueError("Insufficient available host RAM for a CPU researcher")
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": str(profile["server"]["threads"]),
        "MKL_NUM_THREADS": str(profile["server"]["threads"]),
        "PYTHONNOUSERSITE": "1",
    }
    if cpu:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        process = subprocess.Popen(
            command(root, profile_path, "serve"),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        try:
            if cpu:
                os.setpriority(os.PRIO_PROCESS, process.pid, 10)
            deadline = time.monotonic() + profile["runtime"]["max_timeout"]
            with requests.Session() as session:
                session.trust_env = False
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"Managed server exited; inspect {log_path}")
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Managed server startup timed out; inspect {log_path}")
                    try:
                        response = session.get(
                            profile["runtime"]["base_url"] + "/health", timeout=2
                        )
                    except (requests.ConnectionError, requests.Timeout):
                        time.sleep(0.25)
                        continue
                    if response.status_code == 200:
                        break
                    time.sleep(0.25)
            yield profile
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


@contextmanager
def waiting_researcher(
    profile_path: Path, root: Path, log_path: Path, fallback: dict | None = None
):
    """Retry only remote entry transport failures, never repeat work after yield."""
    from rlm.v100.remote_helper import remote_profile

    profile = load_profile(profile_path, root)
    if not remote_profile(profile):
        with managed_server(profile_path, root, log_path) as actual:
            yield actual
        return
    active_path = root / "research/mission/active.json"
    status_path, original_state = None, None
    if active_path.exists():
        active = json.loads(active_path.read_text())
        run = Path(active["run"]).resolve()
        if active.get("pid") == os.getpid() and run.is_relative_to(
            (root / "research/mission").resolve()
        ):
            status_path = run / "status.json"
            original_state = json.loads(status_path.read_text()) if status_path.exists() else {}
    attempt = 0
    while True:
        with ExitStack() as scope:
            try:
                actual = scope.enter_context(managed_server(profile_path, root, log_path))
            except (requests.ConnectionError, requests.Timeout, ValueError) as error:
                if isinstance(error, ValueError) and fallback is None:
                    raise
                attempt += 1
                event = {
                    "phase": "waiting-for-remote-helper",
                    "endpoint": profile["runtime"]["base_url"],
                    "attempt": attempt,
                    "retry_seconds": 30,
                    "error": type(error).__name__,
                    "detail": str(error)[:400],
                    "checkpoints": "retained; no candidate accepted",
                }
                if status_path is not None:
                    atomic_json(status_path, {**original_state, **event})
                ActivityLog(root, "controller", "tester").write(
                    "errors", "remote-helper-unavailable", event
                )
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a") as log:
                    log.write(json.dumps(event) + "\n")
                print(json.dumps(event), flush=True)
                if fallback is not None:
                    ActivityLog(root, "controller", "tester").write(
                        "steps",
                        "research-on-current-master",
                        {
                            "remote_unavailable": True,
                            "scope": "No second model loaded; retry external helper next cycle",
                        },
                    )
                    if status_path is not None:
                        atomic_json(status_path, original_state)
                    # Caller already owns and verified this serving master.
                    yield fallback
                    return
            else:
                if attempt:
                    if status_path is not None:
                        atomic_json(status_path, original_state)
                    event = {"endpoint": profile["runtime"]["base_url"], "attempts": attempt}
                    ActivityLog(root, "controller", "tester").write(
                        "steps", "remote-helper-reconnected", event
                    )
                    print("Remote helper reconnected:", json.dumps(event), flush=True)
                yield actual
                return
        time.sleep(30)


def validate_concurrent_researcher(profile: dict) -> None:
    from rlm.v100.remote_helper import remote_profile, validate_remote

    if remote_profile(profile):
        validate_remote(profile)
        return
    if (
        profile.get("resources", {}).get("device") != "cpu"
        or profile["server"].get("gpu_layers") != 0
        or profile["server"]["draft_model"]
        or profile["server"]["slots"] != 1
        or not 1 <= profile["server"]["threads"] <= 4
    ):
        raise ValueError(
            "Concurrent researcher must use a pinned remote helper or CPU only, one slot and at most four threads"
        )


def helper_client(
    profile: dict, root: Path | None = None, branch: str = "controller"
) -> LlamaCppClient:
    settings = profile["runtime"]
    from rlm.v100.inference import sampling_settings, thinking_enabled
    from rlm.v100.remote_helper import OllamaResearchClient, remote_profile

    remote = remote_profile(profile)
    client_type = OllamaResearchClient if remote else LlamaCppClient
    extras = (
        {
            "model_digest": profile["resources"]["model_digest"],
            "metadata_sha256": profile["resources"]["metadata_sha256"],
            "metadata_hash_scheme": profile["resources"].get(
                "metadata_hash_scheme", "ollama-full-v1"
            ),
            "max_vram_gib": profile["resources"]["max_vram_gib"],
            "helper_batch_tokens": profile["resources"].get("helper_batch_tokens", 64),
            "helper_duty_percent": profile["resources"].get("helper_duty_percent", 65),
        }
        if remote
        else {}
    )
    client = client_type(
        **extras,
        model_name=settings["model_name"],
        base_url=settings["base_url"],
        context_window=settings["context_window"],
        timeout=settings["max_timeout"],
        sampling_args=sampling_settings(profile),
        enable_thinking=thinking_enabled(profile),
        activity_root=str(root) if root else None,
        activity_branch=branch,
        activity_actor="tester"
        if profile.get("resources", {}).get("device") in ("cpu", "remote")
        else "model",
        activity_context={
            "device": profile.get("resources", {}).get("device", "local GPU"),
            "endpoint": settings["base_url"],
            "model_version": settings["model_version"],
            "target": profile["server"]["model"],
            "draft_model": profile["server"]["draft_model"],
            "draft_tokens": profile["server"]["draft_tokens"],
        },
    )
    client.research_config = profile.get("research", {})
    client.research_device = profile.get("resources", {}).get("device", "cuda")
    client.tool_protocol = settings.get("tool_protocol", "native")
    if (
        profile.get("resources", {}).get("device") == "cpu"
        and profile.get("resources", {}).get("compact_research_tools") is True
    ):
        from rlm.v100.research_tools import COMPACT_CPU_TOOLS

        client.research_tool_names = set(COMPACT_CPU_TOOLS)
    return client


def complete_metrics(path: Path) -> list[dict]:
    if not path.exists():
        return []
    # Ignore only the unterminated, still-being-written last line, not invalid completed rows.
    return [json.loads(line) for line in path.read_text().split("\n")[:-1] if line][-4:]


def collect_worker(future, branch: str, root: Path) -> dict:
    try:
        return future.result()
    except Exception as error:
        # Advisory work cannot invalidate the learner's complete checkpoints.
        # Record failures explicitly, allowing the parent to discard them.
        result = {"status": "failed", "error": type(error).__name__, "detail": str(error)[:400]}
        shared = SharedLab(root / "research/state/competition.sqlite3")
        try:
            shared.append(branch, "worker-result", result)
        finally:
            shared.close()
        return result


def train_branch(
    root: Path,
    output: Path,
    branch: str,
    item: dict,
    researcher: dict | None,
    train_timeout: int = 7200,
    code_candidate: Path | None = None,
) -> list[dict]:
    if not 1 <= train_timeout <= 86400:
        raise ValueError("Training timeout must be 1-86400 seconds")
    profile = json.loads(Path(item["profile"]).read_text())
    results, submitted, last_observed = [], 0, None
    jobs = item["decision"]["research_jobs"] if researcher else []
    metrics = Path(profile["training"]["output"]) / "metrics.jsonl"
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "PYTHONNOUSERSITE": "1",
    }
    training_command = command(root, Path(item["profile"]), "train", item["dataset"])
    if code_candidate:
        from rlm.v100.code_lab import code_training_command, sandbox_environment

        training_command = code_training_command(
            code_candidate, profile, Path(item["dataset"]), root, output / branch / "code-worker"
        )
        environment = sandbox_environment(gpu=True)
    with (
        (output / branch / "training.log").open("a") as log,
        ThreadPoolExecutor(max_workers=1) as workers,
    ):
        process = subprocess.Popen(
            training_command, stdout=log, stderr=subprocess.STDOUT, env=environment
        )
        future = None
        deadline = time.monotonic() + train_timeout
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Branch {branch} exceeded its training budget")
                if future is not None and future.done():
                    results.append(collect_worker(future, branch, root))
                    future = None
                observed = complete_metrics(metrics)
                if observed and json.dumps(observed[-1], sort_keys=True) != last_observed:
                    last_observed = json.dumps(observed[-1], sort_keys=True)
                    print(
                        json.dumps(
                            {
                                "branch": branch,
                                **observed[-1],
                                "available_ram_gib": round(available_ram_gib(), 2),
                            }
                        ),
                        flush=True,
                    )
                if jobs and submitted < len(jobs) and future is None and observed:
                    assert researcher is not None
                    from rlm.v100.remote_helper import remote_profile

                    if remote_profile(researcher) or available_ram_gib() >= researcher.get(
                        "resources", {}
                    ).get("min_available_ram_gib", 6):
                        job = jobs[submitted]
                        shared = SharedLab(root / "research/state/competition.sqlite3")
                        try:
                            shared.append(
                                branch,
                                "worker-task",
                                {"job": job, "observed_step": observed[-1].get("step")},
                            )
                        finally:
                            shared.close()
                        submitted += 1
                        future = workers.submit(
                            research_task,
                            helper_client(researcher, root, branch),
                            branch,
                            job,
                            observed,
                            root,
                        )
                time.sleep(0.5)
            if process.returncode:
                raise RuntimeError(
                    f"Branch {branch} training failed; inspect {output / branch / 'training.log'}"
                )
            if future is not None:
                results.append(collect_worker(future, branch, root))
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    atomic_json(
        output / branch / "workers.json",
        {"results": results, "submitted": submitted, "unsubmitted": len(jobs) - submitted},
    )
    return results


def run_duel(
    output: Path,
    root: Path,
    suite: Path,
    baseline: dict | list[dict],
    researcher_path: Path | None = None,
    train_timeout: int = 7200,
    code_candidates: dict[str, Path] | None = None,
) -> dict:
    from rlm.v100.protection import compare_reports, file_hash

    output, root, suite = output.resolve(), root.resolve(), suite.resolve()
    bundle = load_duel(output)
    if bundle.get("goal") and bundle["goal"]["suite_sha256"] != file_hash(suite):
        raise ValueError("Duel suite differs from its user-owned objective")
    if Path(bundle["root"]).resolve() != root.resolve():
        raise ValueError("Duel belongs to a different lab root")
    if (output / "judgment.json").exists():
        raise FileExistsError("Duel already completed")
    baselines = baseline if isinstance(baseline, list) else [baseline]
    if not baselines or any(parent["suite_sha256"] != file_hash(suite) for parent in baselines):
        raise ValueError("Baseline reports must describe this fixed development suite")
    for parent in baselines:
        compare_reports(parent, parent)
        for item in bundle["branches"].values():
            from rlm.v100.inference import generation_conditions

            expected = generation_conditions(json.loads(Path(item["profile"]).read_text()))
            if (
                parent["generation"] != expected
                or parent["memory_mode"] != "fixed prompt fixtures; no live retrieval"
            ):
                raise ValueError("Baseline generation or memory conditions differ")
    code_candidates = code_candidates or {}
    if not set(code_candidates) <= {"A", "B"}:
        raise ValueError("Unknown code-candidate branch")
    if researcher_path:
        helper = load_profile(researcher_path, root)
        validate_concurrent_researcher(helper)
    for path in code_candidates.values():
        from rlm.v100.code_lab import verify_code

        verify_code(path)
    atomic_json(
        output / "execution-plan.json",
        {
            "code_candidates": {
                branch: str(path.resolve()) for branch, path in code_candidates.items()
            },
            "train_timeout": train_timeout,
            "researcher_profile": str(researcher_path) if researcher_path else None,
        },
    )
    from rlm.v100.architectures import resource_lease

    # CPU or external GPU helper stays available while each local branch learns. A/B are not loaded
    # together. No arbitrary host server is stopped and no hidden audit is shared.
    with resource_lease(root, "cuda"):
        if researcher_path:
            with ExitStack() as resources:
                try:
                    researcher = resources.enter_context(
                        managed_server(researcher_path, root, output / "researcher.log")
                    )
                except (requests.RequestException, OSError, ValueError) as error:
                    from rlm.v100.remote_helper import remote_profile

                    if not remote_profile(load_profile(researcher_path, root)):
                        raise
                    atomic_json(
                        output / "helper-deferred.json",
                        {
                            "detail": str(error)[:400],
                            "scope": "Local A/B learning continues; no unverified remote output used",
                        },
                    )
                    researcher = None
                reports = run_branches(
                    output, root, suite, bundle, researcher, train_timeout, code_candidates
                )
        else:
            reports = run_branches(
                output, root, suite, bundle, None, train_timeout, code_candidates
            )
    return judge_duel(output, baseline, reports)


def run_branches(
    output: Path,
    root: Path,
    suite: Path,
    bundle: dict,
    researcher: dict | None,
    train_timeout: int,
    code_candidates: dict[str, Path],
) -> dict:
    from rlm.v100.efficiency import observed_training
    from rlm.v100.evaluation import evaluate_suite
    from rlm.v100.protection import file_hash
    from rlm.v100.serving import assert_served_expert

    reports = {}
    for branch in ("A", "B"):
        print(f"Branch {branch}: training, with separate helper research if configured", flush=True)
        require_idle_gpu()
        item = bundle["branches"][branch]
        profile = json.loads(Path(item["profile"]).read_text())
        started = time.monotonic()
        results = train_branch(
            root, output, branch, item, researcher, train_timeout, code_candidates.get(branch)
        )
        training_seconds = time.monotonic() - started
        print(f"Branch {branch}: exporting and independent development evaluation", flush=True)
        started = time.monotonic()
        subprocess.run(command(root, Path(item["profile"]), "export-model"), check=True)
        export_seconds = time.monotonic() - started
        serving = copy.deepcopy(profile)
        serving["server"].update(
            model=str(Path(profile["training"]["output"]) / "export-Q6_K.gguf"), draft_model=""
        )
        serving["runtime"]["model_version"] = f"{output.name}-{branch}"
        serving_path = output / branch / "serving.json"
        atomic_json(serving_path, serving)
        started = time.monotonic()
        with managed_server(serving_path, root, output / branch / "server.log"):
            client = helper_client(serving, root, branch)
            assert_served_expert(client, serving, root)
            report_path = output / branch / "development-quality.json"
            reports[branch] = evaluate_suite(client, serving, suite, report_path)
            if serving.get("resources", {}).get("public_benchmarks"):
                from rlm.v100.public_benchmarks import evaluate as public_evaluate

                public_evaluate(root, serving, output / branch / "public-quality.json")
            if results:
                review = review_research(client, branch, results, root)
                atomic_json(output / branch / "research-review.json", review)
        evaluation_seconds = time.monotonic() - started
        performance = {
            "schema": "v100-performance-v1",
            "model_sha256": reports[branch]["model_sha256"],
            "quality_report_sha256": file_hash(report_path),
            "training_wall_seconds": training_seconds,
            "export_wall_seconds": export_seconds,
            "evaluation_wall_seconds": evaluation_seconds,
            "total_wall_seconds": training_seconds + export_seconds + evaluation_seconds,
            "learner_reported": observed_training(
                Path(profile["training"]["output"]) / "metrics.jsonl"
            ),
            "scope": "Host wall time includes loading, validation, saves, pending helpers and export; single trial, not a global optimum",
        }
        atomic_json(output / branch / "performance.json", performance)
        shared = SharedLab(Path(bundle["shared_memory"]))
        try:
            shared.append(
                branch,
                "training",
                {
                    "parameters": item["decision"]["parameters"],
                    "measured_cost": {
                        key: performance[key]
                        for key in ("training_wall_seconds", "total_wall_seconds")
                    },
                    "learner_reported": performance["learner_reported"],
                    "best_eval_loss": json.loads(
                        (Path(profile["training"]["output"]) / "best.json").read_text()
                    )["eval_loss"],
                    "candidate": profile["training"]["output"],
                },
            )
        finally:
            shared.close()
    return reports


def evolve(
    profile: dict,
    root: Path,
    pool: Path,
    output: Path,
    suite: Path,
    baselines: list[dict],
    researcher_path: Path | None = None,
    generations: int = 2,
    train_timeout: int = 7200,
) -> dict:
    """Bounded development evolution; preserve every eligible predecessor's tests."""
    from rlm.v100.protection import assert_candidate_output, compare_reports, file_hash

    if type(generations) is not int or not 1 <= generations <= 4:
        raise ValueError("Evolution is limited to 1-4 generations")
    root, pool, output, suite = root.resolve(), pool.resolve(), output.resolve(), suite.resolve()
    initial = profile["training"].get("init_adapter")
    assert_candidate_output(
        output, Path(profile["training"]["base_model"]), Path(initial) if initial else None, root
    )
    if pool.is_relative_to(output) or suite.is_relative_to(output):
        raise ValueError("Evolution output overlaps its fixed inputs")
    if output.exists():
        raise FileExistsError("Evolution requires a new directory")
    if not baselines or any(parent["suite_sha256"] != file_hash(suite) for parent in baselines):
        raise ValueError("Evolution requires reports for its fixed development suite")
    for parent in baselines:
        compare_reports(parent, parent)
    output.mkdir(parents=True)
    current, replay, rounds = copy.deepcopy(profile), None, []
    gates = list(baselines)
    for generation in range(1, generations + 1):
        from rlm.v100.insights import extend_pool

        next_pool = output / f"pool-{generation:02d}.jsonl"
        if extend_pool(pool, root, next_pool):
            pool = next_pool
        require_idle_gpu()
        directory = output / f"generation-{generation:02d}"
        planning_profile = output / f"planner-{generation:02d}.json"
        planner = copy.deepcopy(current)
        planner["runtime"]["max_output_tokens"] = max(1536, current["runtime"]["max_output_tokens"])
        atomic_json(planning_profile, planner)
        # Parent is served only while choosing the next experiments, then unloaded
        # before GPU training. CPU research is concurrent with the training itself.
        with managed_server(planning_profile, root, output / f"planner-{generation:02d}.log"):
            from rlm.v100.lineages import read

            parents, _ = read(current)
            plan_duel(
                helper_client(planner, root),
                current,
                pool,
                directory,
                root,
                replay=replay,
                recent=True,
                **({"branch_parents": parents} if parents else {}),
            )
        from rlm.v100.self_code import admitted

        candidates = admitted(root)
        verdict = run_duel(
            directory,
            root,
            suite,
            gates,
            researcher_path,
            train_timeout,
            **({"code_candidates": candidates} if candidates else {}),
        )
        rounds.append({"directory": str(directory), "judgment": verdict})
        result = {
            "schema": "v100-evolution-v1",
            "rounds": rounds,
            "scope": "Bounded development selection; independent final audit and expert promotion remain separate",
        }
        atomic_json(output / "evolution.json", result)
        if verdict["winner"] is None:
            break
        # Equal task scores do not justify inventing a quality ranking. A is the
        # declared deterministic continuation choice, while both are preserved.
        winner = verdict.get("continuation_branch") or (
            "A" if verdict["winner"] == "tie" else verdict["winner"]
        )
        bundle = load_duel(directory)
        selected = json.loads(Path(bundle["branches"][winner]["profile"]).read_text())
        adapter = Path(selected["training"]["output"]) / "candidate"
        current["training"].update(init_adapter=str(adapter), teacher_adapter=str(adapter))
        current["server"].update(model=str(adapter.parent / "export-Q6_K.gguf"), draft_model="")
        current["runtime"]["model_version"] = f"{directory.name}-{winner}"
        from rlm.v100.lineages import record

        current.setdefault("resources", {})["branch_lineages"] = record(current, directory, verdict)
        replay = directory / "verified-pool.jsonl"
        for branch in ("A", "B"):
            if verdict["branches"][branch]["eligible"]:
                gates.append(
                    json.loads((directory / branch / "development-quality.json").read_text())
                )
    return result
