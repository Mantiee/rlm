"""Host coordinator for untrusted results from operator-owned compute workers."""

import json
import math
import secrets
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from rlm.v100.common import atomic_json
from rlm.v100.compute_kernel import validate
from rlm.v100.compute_worker import read_json
from rlm.v100.protection import file_hash


def configure(root: Path, mailbox: Path) -> dict:
    from rlm.v100.mission import status

    if status(root)["running"]:
        raise ValueError("Configure the operator-owned shared folder while the mission is stopped")
    mailbox = mailbox.expanduser().resolve()
    if mailbox == root or root.is_relative_to(mailbox):
        raise ValueError("Use a dedicated compute mailbox, not the host home or mission root")
    for name in ("jobs", "claims", "results", "workers", "closed"):
        (mailbox / name).mkdir(parents=True, exist_ok=True)
    value = {
        "mailbox": str(mailbox),
        "schema": "v100-compute-config-v1",
        "scope": "Authenticated operator-owned filesystem; no free managed Colab worker farm",
    }
    atomic_json(root / "research/compute-config.json", value)
    return value


def mailbox_path(root: Path) -> Path:
    path = root / "research/compute-config.json"
    if not path.exists():
        raise ValueError(
            "Operator must first configure an authenticated dedicated shared compute folder"
        )
    return Path(json.loads(path.read_text())["mailbox"])


def fresh_records(domain: str, count: int = 64) -> list[dict]:
    from rlm.v100.insights import verified_record

    if (
        domain not in ("arithmetic", "linear_equation", "decimal_calculation")
        or not 32 <= count <= 128
    ):
        raise ValueError("Compute examples require a trusted proof domain and 32-128 records")
    rng, rows = secrets.SystemRandom(), {}
    while len(rows) < count:
        a, b, c = (rng.randint(2, 999) for _ in range(3))
        expression = (
            f"({a}-{b})*{c}"
            if domain == "arithmetic"
            else f"{a}*x+{b}={c}"
            if domain == "linear_equation"
            else f"{a}.{b:03d}*{c}/10000"
        )
        row = verified_record({"kind": domain, "expression": expression})
        rows[row["group"]] = row
    return list(rows.values())


def fresh_examples(domain: str, count: int = 64) -> tuple[list, list]:
    ordered = [
        {
            "group": r["group"],
            "prompt": r["messages"][-2]["content"] + "\nAnswer: ",
            "answer": r["messages"][-1]["content"],
        }
        for r in fresh_records(domain, count)
    ]
    return ordered[16:], ordered[:16]


def propose(
    root: Path,
    branch: str,
    architecture: str,
    steps: int,
    purpose: str,
    domain: str,
    width: int = 64,
    layers: int = 1,
    learning_rate: float = 0.001,
) -> dict:
    mailbox = mailbox_path(root)
    if branch not in ("A", "B") or not isinstance(purpose, str) or not 1 <= len(purpose) <= 600:
        raise ValueError("Choose A/B and a bounded compute experiment purpose")
    records = list((root / "research/compute-jobs").glob("*/state.json"))
    if (
        len(records) >= 128
        or sum(read_json(p)["state"] in ("queued", "running", "imported") for p in records) >= 16
    ):
        raise ValueError("Compute queue is full; finish or cancel existing experiments")
    train, validation = fresh_examples(domain)
    identity = uuid.uuid4().hex[:24]
    folder = root / "research/compute-jobs" / identity
    folder.mkdir(parents=True)
    kernel = Path(__file__).with_name("compute_kernel.py")
    job = {
        "schema": "v100-compute-job-v1",
        "id": identity,
        "branch": branch,
        "architecture": architecture,
        "steps": steps,
        "purpose": purpose,
        "domain": domain,
        "kernel_sha256": file_hash(kernel),
        "worker_sha256": file_hash(Path(__file__).with_name("compute_worker.py")),
        "width": width,
        "layers": layers,
        "context": 256,
        "device": "cpu",
        "threads": 2,
        "seconds": 120,
        "learning_rate": learning_rate,
        "train": train,
        "validation": validation,
    }
    validate(job)
    atomic_json(folder / "job.json", job)
    shutil.copyfile(kernel, folder / "kernel.py")
    (folder / "kernel.py").chmod(0o444)
    atomic_json(
        folder / "state.json",
        {
            "state": "queued",
            "attempts": 0,
            "branch": branch,
            "job_sha256": file_hash(folder / "job.json"),
        },
    )
    destination = mailbox / "jobs" / (identity + ".json")
    temporary = destination.with_suffix(".download")
    shutil.copyfile(folder / "job.json", temporary)
    temporary.replace(destination)
    return {
        "id": identity,
        "state": "queued",
        "weights_changed": False,
        "scope": "Tiny all-weight CPU model on fresh independent examples; not master weights or a proven goal improvement",
    }


def inspect(root: Path) -> dict:
    config = root / "research/compute-config.json"
    mailbox = mailbox_path(root) if config.exists() else None
    workers = (
        [
            read_json(p, 16384) | {"name": p.stem, "stale": time.time() - p.stat().st_mtime > 300}
            for p in sorted((mailbox / "workers").glob("*.json"))[:32]
        ]
        if mailbox
        else []
    )
    return {
        "configured": bool(mailbox),
        "workers": workers,
        "jobs": [
            {"id": p.parent.name, **read_json(p)}
            for p in sorted((root / "research/compute-jobs").glob("*/state.json"))[:128]
        ],
        "scope": "Owned compute only; remote claims are untrusted until local validation",
    }


def cancel(root: Path, identity: str) -> dict:
    if len(identity) != 24 or any(c not in "0123456789abcdef" for c in identity):
        raise ValueError("Invalid compute job ID")
    path = root / "research/compute-jobs" / identity / "state.json"
    state = read_json(path)
    state.update(state="cancelled")
    atomic_json(path, state)
    atomic_json(mailbox_path(root) / "closed" / (identity + ".json"), {"state": "cancelled"})
    return state


def tick(root: Path) -> None:
    mailbox = mailbox_path(root) if (root / "research/compute-config.json").exists() else None
    for path in sorted((root / "research/compute-jobs").glob("*/state.json"))[:128]:
        state, identity = read_json(path), path.parent.name
        if state["state"] == "imported":
            from rlm.v100.drones import schedule

            if not state.get("validation_job"):
                try:
                    scheduled = schedule(root, state["branch"], "compute-audit", identity, 0)
                    state["validation_job"] = scheduled["id"]
                    atomic_json(path, state)
                except ValueError:
                    # A full bounded queue defers validation, never a completed import.
                    pass
            continue
        if state["state"] not in ("queued", "running") or mailbox is None:
            continue
        lease_path = mailbox / "claims" / identity / "lease.json"
        if not lease_path.exists():
            continue
        try:
            lease = read_json(lease_path, 16384)
            if (
                not isinstance(lease.get("nonce"), str)
                or len(lease["nonce"]) != 32
                or any(c not in "0123456789abcdef" for c in lease["nonce"])
            ):
                raise ValueError("Invalid compute lease")
            destination = mailbox / "results" / (identity + "-" + lease["nonce"])
            receipt = (
                read_json(destination / "receipt.json", 16384)
                if (destination / "receipt.json").exists()
                else None
            )
            if receipt is None:
                if time.time() - lease["heartbeat"] < 300:
                    state.update(state="running", worker=lease["worker"])
                    atomic_json(path, state)
                    continue
                raise ValueError("Compute worker lease expired")
            if receipt.get("nonce") != lease["nonce"] or receipt.get("job_id") != identity:
                raise ValueError("Stale or unrelated compute result")
            if receipt.get("state") != "complete":
                raise ValueError(receipt.get("detail", "Remote compute failed"))
            if receipt.get("job_sha256") != state["job_sha256"]:
                raise ValueError("Compute input changed on remote worker")
            result = path.parent / ("received-" + lease["nonce"])
            result.mkdir(exist_ok=False)
            for name in ("report.json", "weights.safetensors"):
                source = destination / name
                if (
                    source.is_symlink()
                    or source.stat().st_size > 16 * 2**20
                    or file_hash(source) != receipt["files"].get(name)
                ):
                    raise ValueError("Compute artifact changed or exceeds its budget")
                shutil.copyfile(source, result / name)
                if file_hash(result / name) != receipt["files"][name]:
                    raise ValueError("Compute artifact changed during copy")
            report = read_json(result / "report.json")
            for key in ("initial_loss", "heldout_loss", "seconds"):
                value = report.get(key)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError("Nonfinite remote compute metrics")
            atomic_json(result / "receipt.json", receipt)
            state.update(
                state="imported",
                result=str(result),
                weights_promoted=False,
                detail="Remote metrics untrusted; queued local safe-tensor validation",
            )
            atomic_json(path, state)
            atomic_json(mailbox / "closed" / (identity + ".json"), {"state": "imported"})
        except (ValueError, KeyError, TypeError, OSError) as error:
            attempts = state["attempts"] + 1
            state.update(
                state="failed" if attempts >= 3 else "queued",
                attempts=attempts,
                detail=str(error)[:400],
            )
            atomic_json(path, state)
            if attempts >= 3:
                atomic_json(mailbox / "closed" / (identity + ".json"), {"state": "failed"})
            claim = mailbox / "claims" / identity
            if claim.exists():
                claim.rename(mailbox / "claims" / ("retired-" + identity + "-" + uuid.uuid4().hex))
    atomic_json(root / "research/compute-status.json", inspect(root))


def validate_locally(root: Path, identity: str) -> dict:
    import psutil

    if len(identity) != 24 or any(c not in "0123456789abcdef" for c in identity):
        raise ValueError("Invalid compute job ID")
    folder = root / "research/compute-jobs" / identity
    state = read_json(folder / "state.json")
    if (
        state["state"] not in ("imported", "validation-deferred")
        or file_hash(folder / "job.json") != state["job_sha256"]
    ):
        raise ValueError("Local compute validation requires frozen original input")
    job = read_json(folder / "job.json")
    if file_hash(folder / "kernel.py") != job["kernel_sha256"]:
        raise ValueError("Local trusted compute kernel changed")
    received = Path(state["result"])
    if not received.resolve().is_relative_to(folder.resolve()):
        raise ValueError("Compute result escapes its host trial")
    receipt = read_json(received / "receipt.json")
    weights = received / "weights.safetensors"
    if (
        weights.is_symlink()
        or weights.stat().st_size > 16 * 2**20
        or file_hash(weights) != receipt["files"]["weights.safetensors"]
    ):
        raise ValueError("Received compute weights changed")
    if psutil.virtual_memory().available < 6 * 2**30:
        raise ValueError("Local compute validation needs six GiB available host RAM")
    output = folder / ("local-validation-" + uuid.uuid4().hex[:12])
    with (folder / "local-validation.log").open("a") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(folder / "kernel.py"),
                str(folder / "job.json"),
                str(output),
                "--weights",
                str(weights),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        started = time.monotonic()
        try:
            while process.poll() is None:
                if time.monotonic() - started > 60:
                    raise RuntimeError("Local compute validation exceeded sixty seconds")
                try:
                    rss = psutil.Process(process.pid).memory_info().rss
                except psutil.NoSuchProcess:
                    process.wait()
                    break
                if psutil.virtual_memory().available < 4 * 2**30:
                    raise RuntimeError("Local compute validation stopped to release host RAM")
                if rss > 4 * 2**30:
                    raise RuntimeError("Local compute validation exceeded four GiB RAM")
                time.sleep(0.2)
            if process.returncode:
                raise RuntimeError(
                    "Local compute tensor validation failed; inspect local-validation.log"
                )
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
    report = read_json(output / "report.json")
    passed = report["heldout_loss"] < report["initial_loss"]
    state.update(
        state="locally-validated" if passed else "rejected",
        local_report=str(output / "report.json"),
        local_report_sha256=file_hash(output / "report.json"),
        weights_promoted=False,
        detail="Independent host loss evaluation of safe tensors; not a master promotion or proof of the user goal",
    )
    atomic_json(folder / "state.json", state)
    return state
