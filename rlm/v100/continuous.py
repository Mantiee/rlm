"""Repeated R&D, verified-data updates and guarded activation of new adapters."""

import copy
import json
import shutil
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path

from rlm.v100.common import atomic_json
from rlm.v100.competition import evolve, helper_client, managed_server, require_idle_gpu
from rlm.v100.experiments import SharedLab, load_duel
from rlm.v100.insights import extend_pool
from rlm.v100.protection import assert_candidate_output, file_hash
from rlm.v100.researchers import research_task, review_research


def learn_loop(
    profile: dict,
    root: Path,
    pool: Path,
    output: Path,
    suite: Path,
    baselines: list[dict],
    researcher_path: Path,
    cycles: int = 4,
    interval: int = 600,
    train_timeout: int = 7200,
    paper_config: Path | None = None,
    initial_update: bool = False,
) -> dict:
    """Zero cycles means run until interrupted; never runs on mere self-agreement."""
    from rlm.v100.architectures import prepare_inputs
    from rlm.v100.goals import load_goal
    from rlm.v100.protection import compare_reports
    from rlm.v100.training import load_records

    if (
        type(cycles) is not int
        or not 0 <= cycles <= 1000
        or type(interval) is not int
        or not 30 <= interval <= 86400
    ):
        raise ValueError("Use 0-1000 cycles and a 30-86400 second interval")
    root, pool, output, suite = root.resolve(), pool.resolve(), output.resolve(), suite.resolve()
    adapter = profile["training"].get("init_adapter")
    assert_candidate_output(
        output, Path(profile["training"]["base_model"]), Path(adapter) if adapter else None, root
    )
    if (
        pool.is_relative_to(output)
        or suite.is_relative_to(output)
        or researcher_path.resolve().is_relative_to(output)
        or (paper_config is not None and paper_config.resolve().is_relative_to(output))
    ):
        raise ValueError("Learning-loop output overlaps its fixed inputs")
    if output.exists():
        raise FileExistsError(
            "Learning loop requires a new directory; partial trials remain available for manual resume"
        )
    if not baselines or any(parent["suite_sha256"] != file_hash(suite) for parent in baselines):
        raise ValueError("Learning loop requires reports for its fixed development suite")
    for parent in baselines:
        compare_reports(parent, parent)
    load_records(pool, Path(profile["training"]["split_ledger"]))
    goal = load_goal(root, suite)
    paper = None
    if paper_config is not None:
        from rlm.v100.paper_agents import financial_helper_profile
        from rlm.v100.paper_learning import PaperLearning

        paper = PaperLearning(root, paper_config)
        researcher_path = financial_helper_profile(researcher_path, root)
    require_idle_gpu()
    output.mkdir(parents=True)
    current, gates = copy.deepcopy(profile), list(baselines)
    current_pool = output / "initial-pool.jsonl"
    current_pool.write_bytes(pool.read_bytes())
    live = output / "live.json"
    history = []
    state = {
        "schema": "v100-continuous-v1",
        "cycles": history,
        "live_profile": str(live),
        "goal": goal,
        "scope": "Automatic verified-data training and finite development-gated activation, not a universal no-forgetting guarantee",
    }
    if paper is not None:
        state["paper_settings_sha256"] = paper.settings_sha
    cycle = 0
    with ExitStack() as scopes:
        if paper is not None:
            scopes.enter_context(paper)
        while cycles == 0 or cycle < cycles:
            if load_goal(root, suite) != goal:
                raise ValueError("User objective changed; start a separate learning experiment")
            cycle += 1
            prepare_inputs(root, current_pool, suite)
            atomic_json(live, current)
            if paper is not None:
                paper.phase("serving-rnd", cycle, current)
            # Serve the current version during R&D/waiting. Only our own inference
            # process is stopped for the training phase; never another user's server.
            with managed_server(researcher_path, root, output / "researcher.log") as helper:
                with managed_server(live, root, output / "live-server.log"):
                    shared = SharedLab(root / "research/state/competition.sqlite3")
                    try:
                        observations = [
                            {
                                "branch": row["branch"],
                                "kind": row["kind"],
                                "public_excerpt": json.dumps(row["payload"], ensure_ascii=False)[
                                    :600
                                ],
                            }
                            for row in shared.recent(4)
                        ]
                    finally:
                        shared.close()
                    for branch in ("A", "B"):
                        try:
                            result = research_task(
                                helper_client(helper, root, branch),
                                branch,
                                {
                                    "role": "researcher",
                                    "brief": "Use public peer findings and available research tools to propose a NEW useful independently checkable learning challenge. Do not repeat old exercises.",
                                },
                                observations,
                                root,
                            )
                            review_research(
                                helper_client(current, root, branch), branch, [result], root
                            )
                        except (ValueError, RuntimeError, OSError) as error:
                            atomic_json(
                                output / f"research-error-{cycle:04d}-{branch}.json",
                                {"error": type(error).__name__, "detail": str(error)[:400]},
                            )
                    if paper is not None:
                        try:
                            paper.research(current, helper)
                        except (ValueError, RuntimeError, OSError) as error:
                            atomic_json(
                                output / f"income-error-{cycle:04d}.json",
                                {"detail": str(error)[:400]},
                            )
                            print(
                                "Income research failed; previous version retained:",
                                error,
                                flush=True,
                            )
                    if current["runtime"].get("tool_protocol") == "json":
                        from rlm.v100.mission_memory import compress

                        try:
                            print("Memory summaries created:", compress(root, current), flush=True)
                        except (ValueError, RuntimeError, OSError) as error:
                            print(
                                "Memory compression deferred; originals retained:",
                                error,
                                flush=True,
                            )
                    expanded = output / f"pool-{cycle:04d}.jsonl"
                    changed = extend_pool(current_pool, root, expanded)
                    if initial_update and cycle == 1 and not changed:
                        expanded.write_bytes(current_pool.read_bytes())
                        changed = True
                    if not changed:
                        history.append(
                            {"cycle": cycle, "status": "no new verified and admitted examples"}
                        )
                        atomic_json(output / "state.json", state)
                        if cycles == 0 or cycle < cycles:
                            deadline = time.monotonic() + interval
                            while time.monotonic() < deadline:
                                if paper is not None:
                                    paper.check()
                                time.sleep(max(0, min(1, deadline - time.monotonic())))
                        continue
            # Avoid training indefinitely on a stale dataset. Its immutable prior
            # snapshots and complete optimizer checkpoints are never overwritten.
            if len(expanded.read_text().splitlines()) > 10000:
                reason = "verified replay exceeds 10000 records"
            elif shutil.disk_usage(output).free < 160 * 2**30:
                reason = "less than 160 GiB free disk for independent A/B exports"
            else:
                reason = None
            if reason:
                history.append(
                    {
                        "cycle": cycle,
                        "status": "upgrade deferred; research continues",
                        "reason": reason,
                    }
                )
                atomic_json(output / "state.json", state)
                print("Upgrade deferred; originals retained:", reason, flush=True)
                deadline = time.monotonic() + interval
                while time.monotonic() < deadline:
                    if paper is not None:
                        paper.check()
                    time.sleep(max(0, min(1, deadline - time.monotonic())))
                continue
            trial = output / f"update-{cycle:04d}"
            if paper is not None:
                paper.phase("training-started", cycle, current)
            try:
                result = evolve(
                    current,
                    root,
                    expanded,
                    trial,
                    suite,
                    gates,
                    researcher_path,
                    generations=1,
                    train_timeout=train_timeout,
                )
            except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
                history.append(
                    {
                        "cycle": cycle,
                        "status": "upgrade failed; previous version retained",
                        "detail": str(error)[:400],
                        "trial": str(trial),
                    }
                )
                atomic_json(output / "state.json", state)
                print("Upgrade failed; serving previous version next cycle:", error, flush=True)
                deadline = time.monotonic() + interval
                while time.monotonic() < deadline:
                    if paper is not None:
                        paper.check()
                    time.sleep(max(0, min(1, deadline - time.monotonic())))
                continue
            generation = result["rounds"][-1]
            verdict = generation["judgment"]
            directory = Path(generation["directory"])
            if verdict["winner"] is not None:
                branch = verdict.get("continuation_branch") or (
                    "A" if verdict["winner"] == "tie" else verdict["winner"]
                )
                bundle = load_duel(directory)
                chosen = json.loads(Path(bundle["branches"][branch]["profile"]).read_text())
                adapter_path = Path(chosen["training"]["output"]) / "candidate"
                current = json.loads((directory / branch / "serving.json").read_text())
                current["training"].update(
                    init_adapter=str(adapter_path), teacher_adapter=str(adapter_path)
                )
                current_pool = directory / "verified-pool.jsonl"
                for ancestor in ("A", "B"):
                    if verdict["branches"][ancestor]["eligible"]:
                        gates.append(
                            json.loads(
                                (directory / ancestor / "development-quality.json").read_text()
                            )
                        )
                status = "selected for next serving phase after finite quality gates"
            else:
                # Parent remains live next cycle. A later trial may revisit the
                # verified pool using different model-selected settings.
                status = "rejected; previous version retained"
            history.append(
                {"cycle": cycle, "status": status, "trial": str(trial), "judgment": verdict}
            )
            atomic_json(live, current)
            atomic_json(output / "state.json", state)
            if paper is not None:
                paper.phase(status, cycle, current)
    return state
