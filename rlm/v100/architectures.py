"""Isolated full-weight architecture pilots with host-owned development scoring.

These are experimental submodels, not automatic replacements for llama.cpp.
V100 has no MIG: GPU trials are exclusive and allocation limits are advisory
outside PyTorch. CPU workers additionally have a hard virtual-memory limit.
"""

import ast
import contextlib
import copy
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from rlm.v100.code_lab import sandbox_command, sandbox_environment
from rlm.v100.common import atomic_json
from rlm.v100.goals import load_goal
from rlm.v100.protection import file_hash

DEFAULT_BUDGET = {
    "device": "cpu",
    "ram_gib": 8,
    "vram_gib": 0,
    "threads": 2,
    "steps": 40,
    "timeout": 120,
    "max_parameters": 2000000,
    "context_window": 256,
    "max_new_tokens": 32,
    "learning_rate": 0.001,
    "seed": 42,
    "vocab_size": 257,
    "artifact_mib": 512,
}


def prepare_inputs(root: Path, pool: Path, suite: Path) -> dict:
    """Snapshot a user-supplied development workload for autonomous CPU pilots."""
    from rlm.v100.training import load_records

    load_goal(root, suite)
    if pool.stat().st_size > 8 * 2**20 or suite.stat().st_size > 2 * 2**20:
        raise ValueError("Submodel pilot inputs exceed their text budget")
    load_records(pool, root / "research/state/architecture-splits.sqlite3")
    key = file_hash(pool) + "-" + file_hash(suite)
    directory = root / "research/architecture-inputs" / key
    directory.mkdir(parents=True, exist_ok=True)
    for name, source in (("pool.jsonl", pool), ("suite.jsonl", suite)):
        target = directory / name
        if not target.exists():
            shutil.copyfile(source, target)
        if file_hash(target) != file_hash(source):
            raise ValueError("Submodel workload snapshot changed")
    value = {
        "directory": str(directory.resolve()),
        "pool_sha256": file_hash(pool),
        "suite_sha256": file_hash(suite),
    }
    atomic_json(root / "research/architecture-inputs.json", value)
    return value


def prepared_inputs(root: Path) -> tuple[Path, Path]:
    settings = json.loads((root / "research/architecture-inputs.json").read_text())
    directory = Path(settings["directory"]).resolve()
    if not directory.is_relative_to((root / "research/architecture-inputs").resolve()):
        raise ValueError("Submodel workload escaped its lab directory")
    paths = []
    for name in ("pool", "suite"):
        path = directory / (name + ".jsonl")
        if path.is_symlink() or file_hash(path) != settings[name + "_sha256"]:
            raise ValueError("Submodel workload snapshot changed")
        paths.append(path)
    return tuple(paths)


def validate_budget(budget: dict) -> None:
    if set(budget) != set(DEFAULT_BUDGET) or budget["device"] not in ("cpu", "cuda"):
        raise ValueError("Invalid architecture budget")
    for key, minimum, maximum in (
        ("ram_gib", 4, 24),
        ("vram_gib", 0, 30),
        ("threads", 1, 8),
        ("steps", 1, 10000),
        ("timeout", 1, 7200),
        ("max_parameters", 1, 1000000000),
        ("context_window", 32, 262144),
        ("max_new_tokens", 1, 16384),
        ("seed", 0, 2**31 - 1),
        ("artifact_mib", 16, 65536),
    ):
        if type(budget[key]) is not int or not minimum <= budget[key] <= maximum:
            raise ValueError("Architecture resource or training budget exceeded")
    if budget["vocab_size"] != 257 or type(budget["vocab_size"]) is not int:
        raise ValueError("Architecture pilots use the fixed byte-token interface")
    rate = budget["learning_rate"]
    if type(rate) not in (int, float) or not 0 < rate <= 0.01:
        raise ValueError("Invalid architecture learning rate")
    if (budget["device"] == "cpu") != (budget["vram_gib"] == 0):
        raise ValueError("CUDA budget must be positive; CPU pilots cannot reserve VRAM")
    if budget["max_new_tokens"] >= budget["context_window"]:
        raise ValueError("Architecture output must fit its byte context")


def candidate_path(root: Path, candidate_id: str) -> Path:
    if not isinstance(candidate_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", candidate_id):
        raise ValueError("Invalid architecture candidate identifier")
    parent = (root / "research/architecture-candidates").resolve()
    target = parent / candidate_id
    if target.is_symlink() or not target.resolve().is_relative_to(parent):
        raise ValueError("Architecture candidate escaped its directory")
    return target


def create_candidate(
    root: Path, branch: str, candidate_id: str, code: str, hypothesis: str, joint: bool = False
) -> dict:
    if branch not in ("A", "B", "shared"):
        raise ValueError("Architecture owner must be A, B or shared")
    if type(joint) is not bool:
        raise ValueError("Joint proposal must be a boolean")
    if not isinstance(code, str) or not 1 <= len(code) <= 12000:
        raise ValueError("Architecture source exceeds its context budget")
    if not isinstance(hypothesis, str) or not 1 <= len(hypothesis) <= 1200:
        raise ValueError("Architecture needs a bounded hypothesis")
    tree = ast.parse(code)
    if not any(isinstance(n, ast.FunctionDef) and n.name == "build" for n in tree.body):
        raise ValueError("Architecture must implement build(config)")
    output = candidate_path(root, candidate_id)
    output.mkdir(parents=True, exist_ok=False)
    (output / "source").mkdir()
    (output / "source/model.py").write_text(code)
    (output / "checks").mkdir()
    runner = Path(__file__).with_name("architecture_worker.py")
    shutil.copyfile(runner, output / "checks/runner.py")
    report = {
        "schema": "v100-architecture-v1",
        "owner": "shared" if joint else branch,
        "hypothesis": hypothesis,
        "goal": load_goal(root),
        "code_sha256": file_hash(output / "source/model.py"),
        "runner_sha256": file_hash(output / "checks/runner.py"),
        "status": "untested isolated architecture, not a serving replacement",
    }
    atomic_json(output / "architecture.json", report)
    if joint:
        support_candidate(root, branch, candidate_id)
    from rlm.v100.experiments import SharedLab

    shared = SharedLab(root / "research/state/competition.sqlite3")
    try:
        shared.append(
            branch,
            "architecture-proposal",
            {
                "candidate_id": candidate_id,
                "owner": report["owner"],
                "hypothesis": hypothesis,
                "joint": joint,
            },
        )
    finally:
        shared.close()
    return {"candidate_id": candidate_id, **report}


def agreement_key(manifest: dict, budget: dict) -> str:
    value = {"code_sha256": manifest["code_sha256"], "goal": manifest["goal"], "budget": budget}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def support_candidate(
    root: Path, branch: str, candidate_id: str, budget: dict | None = None
) -> dict:
    if branch not in ("A", "B"):
        raise ValueError("Only A and B can agree to a shared trial")
    budget = copy.deepcopy(budget or DEFAULT_BUDGET)
    validate_budget(budget)
    output = candidate_path(root, candidate_id)
    manifest = verify_candidate(output)
    if manifest["owner"] != "shared":
        raise ValueError("This candidate did not request a joint experiment")
    key = agreement_key(manifest, budget)
    atomic_json(
        output / ("support-" + branch + ".json"), {"branch": branch, "agreement_sha256": key}
    )
    both = all(
        (output / ("support-" + peer + ".json")).exists()
        and json.loads((output / ("support-" + peer + ".json")).read_text())
        == {"branch": peer, "agreement_sha256": key}
        for peer in ("A", "B")
    )
    return {"candidate_id": candidate_id, "branch": branch, "both_agree": both}


def verify_candidate(output: Path) -> dict:
    report = json.loads((output / "architecture.json").read_text())
    if report["schema"] != "v100-architecture-v1":
        raise ValueError("Invalid architecture manifest")
    for directory, filename, key in (
        ("source", "model.py", "code_sha256"),
        ("checks", "runner.py", "runner_sha256"),
    ):
        paths = list((output / directory).rglob("*"))
        if any(p.is_symlink() for p in paths) or {p.name for p in paths} != {filename}:
            raise ValueError("Architecture snapshot contains unexpected files")
        if file_hash(output / directory / filename) != report[key]:
            raise ValueError("Architecture source or launcher changed")
    # An obsolete or edited launcher cannot certify the current protocol.
    if report["runner_sha256"] != file_hash(Path(__file__).with_name("architecture_worker.py")):
        raise ValueError("Architecture launcher differs from installed trusted code")
    return report


@contextlib.contextmanager
def resource_lease(root: Path, device: str):
    path = root / "research/state" / ("architecture-" + device + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another architecture trial owns this device budget") from error
        try:
            if device == "cuda":
                from rlm.v100.competition import require_idle_gpu

                require_idle_gpu()
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def phase(
    output: Path,
    run: Path,
    inputs: Path,
    budget: dict,
    mode: str,
    extra_mounts: list[tuple[Path, bool]] | None = None,
) -> float:
    import psutil

    task = [
        sys.executable,
        "-I",
        "/checks/runner.py",
        mode,
        str(inputs / "config.json"),
        str(inputs / "data.json"),
        str(run / "weights"),
    ]
    args = sandbox_command(
        output,
        task,
        [(inputs, False), (run / "weights", mode == "train"), *(extra_mounts or [])],
        gpu=budget["device"] == "cuda",
    )
    limiter = shutil.which("prlimit")
    if limiter is None:
        raise FileNotFoundError("Architecture experiments require prlimit and bubblewrap")
    limits = [
        limiter,
        "--fsize=" + str((budget["artifact_mib"] if mode == "train" else 2) * 2**20),
        "--nofile=256",
        "--nproc=512",
        "--cpu=" + str(budget["timeout"]),
    ]
    if budget["device"] == "cpu":
        limits += ["--as=" + str(budget["ram_gib"] * 2**30)]
    args = ["/usr/bin/nice", "-n", "10", *limits, "--", *args]
    environment = sandbox_environment(budget["device"] == "cuda")
    environment.update(
        OMP_NUM_THREADS=str(budget["threads"]), MKL_NUM_THREADS=str(budget["threads"])
    )
    start = time.monotonic()
    with (run / (mode + ".log")).open("w") as log:
        process = subprocess.Popen(
            args, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            while process.poll() is None:
                if time.monotonic() - start > budget["timeout"]:
                    raise TimeoutError("Architecture phase exceeded its wall-clock budget")
                try:
                    tree = [
                        psutil.Process(process.pid),
                        *psutil.Process(process.pid).children(recursive=True),
                    ]
                    rss = sum(p.memory_info().rss for p in tree if p.is_running())
                except psutil.NoSuchProcess:
                    rss = 0
                if rss > budget["ram_gib"] * 2**30:
                    raise RuntimeError("Architecture trial exceeded its sampled RAM budget")
                if budget["device"] == "cuda":
                    used = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if int(used.stdout.splitlines()[0]) > min(budget["vram_gib"] + 2, 30) * 1024:
                        raise RuntimeError("Architecture trial exceeded its sampled GPU budget")
                paths = list((run / "weights").rglob("*"))
                if (
                    any(p.is_symlink() for p in paths)
                    or sum(p.stat().st_size for p in paths if p.is_file())
                    > budget["artifact_mib"] * 2**20
                ):
                    raise RuntimeError("Architecture trial exceeded its artifact budget")
                time.sleep(0.2)
            if process.returncode:
                raise RuntimeError(
                    "Isolated architecture phase failed; inspect " + str(run / (mode + ".log"))
                )
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    verify_candidate(output)
    return time.monotonic() - start


def score_predictions(rows: list[dict], predictions: list[dict]) -> list[dict]:
    expected_ids = {row["id"] for row in rows}
    if (
        len(expected_ids) != len(rows)
        or {p.get("id") for p in predictions} != expected_ids
        or len(predictions) != len(rows)
    ):
        raise ValueError("Architecture must answer each fixed evaluation case exactly once")
    answers = {
        p["id"]: p["answer"]
        for p in predictions
        if set(p) == {"id", "answer"} and isinstance(p["answer"], str) and len(p["answer"]) <= 4096
    }
    if answers.keys() != expected_ids:
        raise ValueError("Invalid architecture predictions")
    return [
        {
            "id": row["id"],
            "answer": answers[row["id"]],
            "passed": answers[row["id"]].strip() == row["expected"].strip()
            if row["match"] == "exact"
            else row["expected"] in answers[row["id"]],
        }
        for row in rows
    ]


def run_candidate(
    root: Path,
    candidate_id: str,
    pool: Path,
    suite: Path,
    budget: dict | None = None,
    init_weights: Path | None = None,
) -> dict:
    from rlm.v100.training import load_records

    budget = copy.deepcopy(budget or DEFAULT_BUDGET)
    validate_budget(budget)
    output = candidate_path(root, candidate_id)
    manifest = verify_candidate(output)
    if manifest["owner"] == "shared":
        key = agreement_key(manifest, budget)
        if not all(
            (output / ("support-" + peer + ".json")).exists()
            and json.loads((output / ("support-" + peer + ".json")).read_text())
            == {"branch": peer, "agreement_sha256": key}
            for peer in ("A", "B")
        ):
            raise ValueError(
                "Shared architecture trial requires both A and B to agree to this code and budget"
            )
    goal = load_goal(root, suite)
    if manifest["goal"] != goal:
        raise ValueError("Architecture belongs to a different user goal")
    rows = [json.loads(line) for line in suite.read_text().splitlines() if line.strip()]
    if (
        not rows
        or len(rows) > 256
        or any(
            row.get("match") not in ("exact", "contains")
            or not isinstance(row.get("expected"), str)
            or not row["expected"]
            or not row.get("messages")
            or any(m["role"] == "assistant" for m in row["messages"])
            for row in rows
        )
    ):
        raise ValueError("Architecture trials require 1-256 fixed independently evaluated cases")
    if pool.stat().st_size > 8 * 2**20:
        raise ValueError("Architecture corpus exceeds 8 MiB pilot budget")
    train, validation = load_records(pool, root / "research/state/architecture-splits.sqlite3")
    run = output / "trial"
    with resource_lease(root, budget["device"]):
        run.mkdir(exist_ok=False)
        (run / "weights").mkdir()
        atomic_json(run / "budget.json", budget)
        # Neither fixed answers nor host controller files are mounted in the worker.
        with tempfile.TemporaryDirectory(prefix="v100-architecture-") as temp:
            inputs = Path(temp)
            configuration = dict(budget)
            atomic_json(inputs / "validation.json", validation[:32])
            configuration["validation_file"] = str(inputs / "validation.json")
            mounts = []
            if init_weights is not None:
                if init_weights.is_symlink() or not init_weights.is_file():
                    raise ValueError("Continuation weights must be an ordinary frozen file")
                configuration["init_weights"] = str(init_weights.resolve())
                configuration["init_weights_sha256"] = file_hash(init_weights)
                mounts = [(init_weights.resolve(), False)]
            atomic_json(inputs / "config.json", configuration)
            atomic_json(inputs / "data.json", train)
            train_time = (
                phase(output, run, inputs, budget, "train", mounts)
                if mounts
                else phase(output, run, inputs, budget, "train")
            )
            weights = run / "weights/weights.safetensors"
            if (
                weights.is_symlink()
                or not weights.is_file()
                or weights.stat().st_size > budget["artifact_mib"] * 2**20
            ):
                raise ValueError("Architecture did not produce a bounded ordinary weights file")
            weights_sha = file_hash(weights)
            atomic_json(
                inputs / "data.json",
                [{"id": row["id"], "messages": row["messages"]} for row in rows],
            )
            eval_time = phase(output, run, inputs, budget, "predict")
            if file_hash(weights) != weights_sha:
                raise ValueError("Architecture changed weights while being evaluated")
        if (run / "predict.log").stat().st_size > 2 * 2**20:
            raise ValueError("Architecture prediction output exceeds its protocol budget")
        predictions = [
            json.loads(line)
            for line in (run / "predict.log").read_text().splitlines()
            if line.startswith('{"id":')
        ]
        cases = score_predictions(rows, predictions)
        report = {
            "schema": "v100-architecture-quality-v1",
            "candidate_id": candidate_id,
            "owner": manifest["owner"],
            "goal": goal,
            "suite_sha256": file_hash(suite),
            "code_sha256": manifest["code_sha256"],
            "weights_sha256": weights_sha,
            "training_seconds": train_time,
            "evaluation_seconds": eval_time,
            "cases": cases,
            "passed_cases": sum(row["passed"] for row in cases),
            "budget": budget,
            "status": "scored full-weight submodel pilot; main serving model was not replaced",
            "scope": "Finite development suite; sampled resource monitoring is not hardware partitioning or proof against malicious code",
        }
        atomic_json(run / "quality.json", report)
        from rlm.v100.experiments import SharedLab

        shared = SharedLab(root / "research/state/competition.sqlite3")
        try:
            shared.append(
                manifest["owner"],
                "architecture-result",
                {
                    key: report[key]
                    for key in ("candidate_id", "passed_cases", "training_seconds", "status")
                },
            )
        finally:
            shared.close()
        return report
