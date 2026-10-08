"""Interactive Colab proposals; never remote workers or quota bypass."""

import hashlib
import json
import math
import uuid
import zipfile
from pathlib import Path

from rlm.v100.common import atomic_json


def propose(root: Path, branch: str, steps: int, purpose: str) -> dict:
    if (
        type(steps) is not int
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
        "scope": "Optional tiny separate-model pilot. No automatic account/session, distributed worker, cookie rotation, paid plan or live-model promotion.",
    }
    atomic_json(path, value)
    from rlm.v100.insights import verified_record
    from rlm.v100.protection import file_hash

    records = [
        verified_record(
            {
                "kind": "decimal_calculation",
                "expression": f"{i}*(1+7/1000)*(1-80/10000)-{i}*(1+80/10000)",
            }
        )
        for i in range(101, 181)
    ]
    source = json.loads((root / "research/self-code-source.json").read_text())
    pool = "".join(json.dumps(row) + "\n" for row in records).encode()
    value.update(
        job_id=path.stem,
        source_revision=source["revision"],
        pool_sha256=hashlib.sha256(pool).hexdigest(),
    )
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
        if not allowed <= names or any(info.file_size > 32 * 2**20 for info in archive.infolist()):
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
                not isinstance(row.get(key), (int, float))
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
    receipt = {
        "job_id": identity,
        "path": str(destination),
        "status": "external result imported; independent local reproduction required",
        "weights_promoted": False,
    }
    atomic_json(destination / "receipt.json", receipt)
    return receipt
