"""Interactive Colab proposals; never remote workers or quota bypass."""

import hashlib
import json
import math
import shutil
import uuid
import zipfile
from pathlib import Path

from rlm.v100.common import atomic_json


def propose(root: Path, branch: str, steps: int, purpose: str) -> dict:
    if (
        branch not in ("A", "B")
        or type(steps) is not int
        or not 10 <= steps <= 100
        or not isinstance(purpose, str)
        or not 1 <= len(purpose) <= 600
    ):
        raise ValueError("Colab pilot requires 10-100 steps and a short purpose")
    path = root / "research/colab-proposals" / (uuid.uuid4().hex[:12] + ".json")
    value = {
        "branch": branch,
        "steps": steps,
        "purpose": purpose,
        "max_seconds": 120,
        "status": "awaiting interactive user session",
        "notebook": "https://colab.research.google.com/github/Mantiee/rlm/blob/v100-rtx-reconnect/tools/colab-income-pilot.ipynb",
        "scope": "Optional interactive separate-model pilot; safe tensor import and independent local validation. No automatic account/session, distributed worker, cookie rotation, paid plan or live-model promotion.",
    }
    atomic_json(path, value)
    from rlm.v100.distributed_compute import fresh_records
    from rlm.v100.protection import file_hash

    # Each notebook gets new examples; it cannot train on benchmark/audit answers.
    records = fresh_records("decimal_calculation", 80)
    source = json.loads((root / "research/self-code-source.json").read_text())
    pool = "".join(json.dumps(row) + "\n" for row in records).encode()
    value.update(
        job_id=path.stem,
        source_revision=source["revision"],
        pool_sha256=hashlib.sha256(pool).hexdigest(),
    )
    kernel = Path(__file__).with_name("compute_kernel.py")
    compute_id = path.stem + value["pool_sha256"][:12]
    inputs = [
        {
            "group": row["group"],
            "prompt": row["messages"][-2]["content"] + "\nAnswer: ",
            "answer": row["messages"][-1]["content"],
        }
        for row in records
    ]
    value["experiment"] = {
        "schema": "v100-compute-job-v1",
        "id": compute_id,
        "branch": branch,
        "architecture": "gru",
        "steps": steps,
        "purpose": purpose,
        "kernel_sha256": file_hash(kernel),
        "width": 64,
        "layers": 1,
        "context": 256,
        "device": "cpu",
        "threads": 2,
        "seconds": 120,
        "learning_rate": 0.001,
        "train": inputs[16:],
        "validation": inputs[:16],
    }
    assets = path.with_suffix("")
    assets.mkdir()
    shutil.copyfile(kernel, assets / "kernel.py")
    (assets / "kernel.py").chmod(0o444)
    package = path.with_suffix(".zip")
    with zipfile.ZipFile(package, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("job.json", json.dumps(value))
        archive.writestr("pool.jsonl", pool)
    atomic_json(path, value)
    return {
        **value,
        "proposal": str(path),
        "upload_package": str(package),
        "package_sha256": file_hash(package),
    }


def import_result(root: Path, path: Path) -> dict:
    if path.stat().st_size > 64 * 2**20:
        raise ValueError("Colab result archive exceeds 64 MiB")
    with zipfile.ZipFile(path) as archive:
        allowed = {"manifest.json", "metrics.json"}
        names = set(archive.namelist())
        if (
            not allowed <= names
            or len(archive.infolist()) > 32
            or sum(i.file_size for i in archive.infolist()) > 64 * 2**20
            or any(info.file_size > 32 * 2**20 for info in archive.infolist())
            or archive.getinfo("manifest.json").file_size > 65536
            or archive.getinfo("metrics.json").file_size > 2**20
            or len(names) != len(archive.infolist())
        ):
            raise ValueError("Invalid or oversized Colab archive")
        manifest = json.loads(archive.read("manifest.json"))
        metrics = json.loads(archive.read("metrics.json"))
        identity = manifest.get("job_id", "")
        if len(identity) != 12 or any(c not in "0123456789abcdef" for c in identity):
            raise ValueError("Colab result has no known job ID")
        proposal = json.loads(
            (root / "research/colab-proposals" / (identity + ".json")).read_text()
        )
        for key in ("source_revision", "pool_sha256"):
            if manifest.get(key) != proposal[key]:
                raise ValueError("Colab result does not match the pinned job")
        if not isinstance(metrics, list) or not 1 <= len(metrics) <= proposal["steps"]:
            raise ValueError("Colab metrics exceed the planned trial")
        for index, row in enumerate(metrics, 1):
            if row.get("step") != index or any(
                type(row.get(key)) not in (int, float)
                or not math.isfinite(row[key])
                or row[key] < 0
                for key in ("train_loss", "heldout_loss", "seconds")
            ):
                raise ValueError("Invalid Colab metrics")
        destination = root / "research/colab-results" / (identity + "-" + uuid.uuid4().hex[:8])
        destination.mkdir(parents=True)
        # Do not extract or deserialize arbitrary pickle checkpoints or execute code.
        for name in allowed:
            (destination / name).write_bytes(archive.read(name))
    compute_id = None
    if "weights.safetensors" in names:
        from rlm.v100.compute_kernel import validate
        from rlm.v100.protection import file_hash

        experiment = proposal["experiment"]
        validate(experiment)
        kernel = root / "research/colab-proposals" / identity / "kernel.py"
        if file_hash(kernel) != experiment["kernel_sha256"]:
            raise ValueError("Pinned notebook kernel changed")
        with zipfile.ZipFile(path) as archive:
            raw = archive.read("weights.safetensors")
        if len(raw) > 16 * 2**20 or hashlib.sha256(raw).hexdigest() != manifest.get(
            "weights_sha256"
        ):
            raise ValueError("Notebook tensor artifact changed or exceeds its budget")
        compute_id = experiment["id"]
        trial = root / "research/compute-jobs" / compute_id
        trial.mkdir(parents=True, exist_ok=False)
        atomic_json(trial / "job.json", experiment)
        shutil.copyfile(kernel, trial / "kernel.py")
        (trial / "kernel.py").chmod(0o444)
        received = trial / "received-notebook"
        received.mkdir()
        (received / "weights.safetensors").write_bytes(raw)
        atomic_json(
            received / "receipt.json",
            {"files": {"weights.safetensors": hashlib.sha256(raw).hexdigest()}},
        )
        atomic_json(
            trial / "state.json",
            {
                "state": "imported",
                "branch": proposal["branch"],
                "attempts": 0,
                "job_sha256": file_hash(trial / "job.json"),
                "result": str(received),
                "weights_promoted": False,
                "detail": "Interactive notebook weights require independent local evaluation",
            },
        )
        from rlm.v100.drones import schedule

        try:
            pending = schedule(root, proposal["branch"], "compute-audit", compute_id, 0)
        except ValueError:
            pending = None  # Full queue defers local validation; no weights promoted.
        if pending:
            state = json.loads((trial / "state.json").read_text())
            atomic_json(trial / "state.json", {**state, "validation_job": pending["id"]})
    receipt = {
        "compute_job_id": compute_id,
        "job_id": identity,
        "path": str(destination),
        "status": "external result imported; independent local reproduction required",
        "weights_promoted": False,
    }
    atomic_json(destination / "receipt.json", receipt)
    return receipt
