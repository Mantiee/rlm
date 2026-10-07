"""Self-code candidates run in a namespace, never in the host controller."""

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

from rlm.v100.agent import native_turn
from rlm.v100.common import atomic_json
from rlm.v100.protection import file_hash

# These files contain the externally supplied system prompts. Candidate code may
# change its algorithms; the host prompt/evaluator/controller stay outside it.
PROMPT_FILES = {
    "rlm/v100/agent.py",
    "rlm/v100/experiments.py",
    "rlm/v100/researchers.py",
    "rlm/v100/cli.py",
    "rlm/v100/code_lab.py",
    "rlm/utils/prompts.py",
    "rlm/v100/research_tools.py",
    "rlm/v100/goals.py",
    "rlm/v100/architectures.py",
    "rlm/v100/architecture_worker.py",
    "rlm/v100/free_router.py",
    "rlm/v100/free_services.py",
    "rlm/v100/paper.py",
    "rlm/v100/paper_agents.py",
    "rlm/v100/paper_feeds.py",
    "rlm/v100/paper_reports.py",
    "rlm/v100/paper_cli.py",
    "rlm/v100/paper_tools.py",
}


def source_files(source: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(source), "ls-files", "-s"], check=True, capture_output=True, text=True
    )
    files = []
    for line in result.stdout.splitlines():
        info, name = line.split("\t", 1)
        if info.split()[0] not in ("100644", "100755"):
            raise ValueError("Experimental source snapshots do not support links or submodules")
        files.append(name)
    return files


def snapshot_source(source: Path, destination: Path) -> dict:
    destination.mkdir()
    hashes = {}
    for name in source_files(source):
        path = (source / name).resolve()
        if not path.is_relative_to(source.resolve()):
            raise ValueError("Source file escaped its repository")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[name] = file_hash(target)
    return hashes


def propose_code(client, source: Path, filename: str, output: Path) -> dict:
    if filename in PROMPT_FILES or not filename.startswith("rlm/") or not filename.endswith(".py"):
        raise ValueError("Choose algorithm code; system-prompt files are read-only")
    if filename not in source_files(source):
        raise ValueError("Self-code access is restricted to tracked project source files")
    path = (source / filename).resolve()
    if not path.is_relative_to(source.resolve()) or path.stat().st_size > 16000:
        raise ValueError("Source view exceeds the working context budget")
    original = path.read_text()
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["find", "replace", "hypothesis"],
        "properties": {
            "find": {"type": "string", "maxLength": 6000},
            "replace": {"type": "string", "maxLength": 6000},
            "hypothesis": {"type": "string", "maxLength": 1200},
        },
    }
    response = native_turn(
        client,
        [
            {
                "role": "system",
                "content": "Propose one falsifiable algorithm improvement to your experimental source. Return an exact unique old-code substring and replacement. Preserve compatibility. System prompts, host controller and independent quality tests are outside your writable workspace. Do not claim tests passed before execution.",
            },
            {"role": "user", "content": json.dumps({"file": filename, "source": original})},
        ],
        response_format={"type": "json_object", "schema": schema},
    )
    change = json.loads(response["content"])
    if set(change) != {"find", "replace", "hypothesis"} or not all(
        isinstance(value, str) for value in change.values()
    ):
        raise ValueError("Invalid self-code proposal")
    if (
        not change["find"]
        or original.count(change["find"]) != 1
        or change["replace"] == change["find"]
    ):
        raise ValueError("Code edit must match exactly once and change its target")
    if max(len(change["find"]), len(change["replace"])) > 6000 or len(change["hypothesis"]) > 1200:
        raise ValueError("Code edit exceeds its budget")
    modified = original.replace(change["find"], change["replace"], 1)
    ast.parse(modified)
    output = output.resolve()
    if source.resolve().is_relative_to(output) or output.is_relative_to(source.resolve()):
        raise ValueError("Self-code output must be separate from the source repository")
    if output.exists():
        raise FileExistsError("Code candidate already exists")
    output.mkdir(parents=True)
    hashes = snapshot_source(source, output / "source")
    trusted = snapshot_source(source, output / "checks")
    destination = output / "source" / filename
    destination.write_text(modified)
    hashes[filename] = file_hash(destination)
    for name in PROMPT_FILES:
        if name in hashes:
            (output / "source" / name).chmod(0o444)
    report = {
        "schema": "v100-code-v1",
        "source_files": hashes,
        "trusted_files": trusted,
        "change": {"file": filename, **change},
        "status": "untested candidate",
    }
    atomic_json(output / "code.json", report)
    return report


def verify_code(output: Path) -> dict:
    report = json.loads((output / "code.json").read_text())
    if report["schema"] != "v100-code-v1":
        raise ValueError("Invalid code candidate")
    for directory, key in (("source", "source_files"), ("checks", "trusted_files")):
        tree = output / directory
        actual = {str(path.relative_to(tree)) for path in tree.rglob("*") if path.is_file()}
        if actual != set(report[key]) or any(path.is_symlink() for path in tree.rglob("*")):
            raise ValueError("Unexpected files or links in code snapshot")
        for name, expected in report[key].items():
            path = (output / directory / name).resolve()
            if (
                not path.is_relative_to((output / directory).resolve())
                or file_hash(path) != expected
            ):
                raise ValueError("Experimental code or trusted checks changed")
    return report


def sandbox_command(
    output: Path, task: list[str], mounts: list[tuple[Path, bool]] | None = None, gpu: bool = False
) -> list[str]:
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise FileNotFoundError(
            "bubblewrap is required; untrusted code never falls back to host execution"
        )
    args = [bwrap, "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL"]
    prefixes = [Path("/usr"), Path("/lib"), Path("/lib64"), Path(sys.prefix), Path(sys.base_prefix)]
    for path in dict.fromkeys(prefixes):
        if path.exists():
            args += ["--ro-bind", str(path), str(path)]
    args += [
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--ro-bind",
        str(output / "source"),
        "/work",
        "--ro-bind",
        str(output / "checks"),
        "/checks",
        "--chdir",
        "/work",
    ]
    for path, writable in mounts or []:
        args += ["--bind" if writable else "--ro-bind", str(path), str(path)]
    if gpu:
        for path in sorted(Path("/dev").glob("nvidia*")):
            if path.is_char_device():
                args += ["--dev-bind", str(path), str(path)]
    return [*args, *task]


def sandbox_environment(gpu: bool = False) -> dict:
    return {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "XDG_CACHE_HOME": "/tmp/cache",
        "CUDA_CACHE_PATH": "/tmp/cuda-cache",
        "CUDA_VISIBLE_DEVICES": "0" if gpu else "",
    }


def check_code(output: Path, timeout: int = 600, cpu_threads: int = 8) -> dict:
    report = verify_code(output)
    if not 1 <= timeout <= 3600:
        raise ValueError("Invalid code-check timeout")
    if type(cpu_threads) is not int or not 1 <= cpu_threads <= 8:
        raise ValueError("Code checks require 1-8 CPU threads")
    runner = "import runpy,sys;sys.path.insert(0,'/work');sys.argv=['pytest','/checks/tests','-q','--basetemp=/tmp/checks','-p','no:cacheprovider'];runpy.run_module('pytest',run_name='__main__')"
    args = sandbox_command(output, [sys.executable, "-I", "-c", runner])
    if cpu_threads <= 4:
        args = ["/usr/bin/nice", "-n", "10", *args]
    environment = sandbox_environment()
    environment.update(OMP_NUM_THREADS=str(cpu_threads), MKL_NUM_THREADS=str(cpu_threads))
    with (output / "checks.log").open("w") as log:
        result = subprocess.run(
            args, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=timeout
        )
    # The mutable worker cannot write this result or the immutable checks tree.
    verify_code(output)
    verdict = {
        "passed": result.returncode == 0,
        "returncode": result.returncode,
        "source_files": report["source_files"],
        "scope": "Visible unit tests only; independent task quality is still required",
    }
    atomic_json(output / "check-result.json", verdict)
    if result.returncode:
        raise RuntimeError(f"Isolated code checks failed; inspect {output / 'checks.log'}")
    return verdict


def code_training_command(
    output: Path, profile: dict, dataset: Path, root: Path, destination: Path
) -> list[str]:
    report = verify_code(output)
    verdict = json.loads((output / "check-result.json").read_text())
    if not verdict["passed"] or verdict["source_files"] != report["source_files"]:
        raise ValueError("Only an unchanged code candidate with passed isolated checks may train")
    destination.mkdir()
    private_profile = json.loads(json.dumps(profile))
    # Candidate code can experiment with its private split copy. It cannot write
    # the global split ledger, protected experts, source pool or host prompt.
    ledger = destination / "splits.sqlite3"
    import sqlite3

    with sqlite3.connect(
        Path(profile["training"]["split_ledger"]).resolve().as_uri() + "?mode=ro", uri=True
    ) as source:
        with sqlite3.connect(ledger) as target:
            source.backup(target)
    private_profile["training"]["split_ledger"] = str(ledger)
    training_output = Path(profile["training"]["output"]).resolve()
    if training_output.exists() and any(training_output.iterdir()):
        raise FileExistsError("Code training requires a new, empty candidate output")
    training_output.mkdir(parents=True, exist_ok=True)
    config = destination / "worker-profile.json"
    atomic_json(config, private_profile)
    mounts = [
        (destination, True),
        (training_output, True),
        (dataset, False),
        (Path(profile["training"]["base_model"]), False),
    ]
    for key in ("init_adapter", "teacher_adapter"):
        if profile["training"].get(key):
            mounts.append((Path(profile["training"][key]), False))
    runner = "import json,sys;from pathlib import Path;sys.path.insert(0,'/work');from rlm.v100.training import train_model;train_model(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]),False,Path(sys.argv[3]))"
    return sandbox_command(
        output,
        [sys.executable, "-I", "-c", runner, str(config), str(dataset), str(root)],
        mounts,
        gpu=True,
    )
