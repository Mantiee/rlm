"""Bounded CPU-only integrity experiment with independently recomputed receipts.

This is a data-quality experiment, never a model evaluation or training promotion.
"""

import hashlib
import json
import time
from pathlib import Path

from rlm.v100.common import atomic_json


def _measure(path: Path) -> dict:
    hashes = []
    for line in path.read_bytes().splitlines():
        if line.strip():
            # Compare canonical examples, not whitespace variations in JSON.
            record = json.loads(line)
            canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            hashes.append(hashlib.sha256(canonical.encode()).hexdigest())
    return {
        "records": len(hashes),
        "unique_records": len(set(hashes)),
        "duplicate_records": len(hashes) - len(set(hashes)),
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def run_integrity_experiment(pool: Path, output: Path, cycle: int) -> dict:
    """Execute and independently check a deterministic source-integrity hypothesis.

    Receipt is persisted even when the evidence is inconclusive or invalid.
    This experiment never claims independent source provenance or model quality.
    """
    if type(cycle) is not int or cycle < 1:
        raise ValueError("Cycle must be a positive integer")
    pool, output = pool.resolve(), output.resolve()
    if not pool.is_file() or output == pool or output.is_relative_to(pool.parent / pool.name):
        raise ValueError("Invalid experiment paths")
    output.mkdir(parents=True, exist_ok=True)
    experiment_id = f"cpu-pool-integrity-{cycle:04d}"
    artifact = output / f"{experiment_id}.json"
    if artifact.exists():
        existing = json.loads(artifact.read_text())
        if existing.get("id") != experiment_id or existing.get("input") != str(pool):
            raise ValueError("Existing experiment receipt conflicts with requested input")
        if existing.get("input_sha256") != hashlib.sha256(pool.read_bytes()).hexdigest():
            raise ValueError("Previously evaluated experiment input has changed")
        return existing

    started = time.time()
    report = {
        "schema": "v100-cpu-experiment-v1",
        "id": experiment_id,
        "hypothesis": "The verified training pool has no duplicate canonical records",
        "scope": "CPU data integrity only; not proof of source independence, model improvement, or income",
        "worker": "local-cpu",
        "input": str(pool),
        "started_at": started,
        "transitions": [
            {"state": "PROPOSED", "at": started},
            {"state": "VALIDATED", "at": started},
            {"state": "QUEUED", "at": started},
            {"state": "RUNNING", "at": started},
        ],
    }
    try:
        measurement = _measure(pool)
        report["measurement"] = measurement
        report["input_sha256"] = measurement["dataset_sha256"]
        report["transitions"].append({"state": "EVALUATING", "at": time.time()})
        # Independent second pass over the immutable input. Do not grade model
        # output using its own assertions.
        independent = _measure(pool)
        identical = independent == measurement
        status = (
            "INCONCLUSIVE" if not measurement["records"]
            else "ACCEPTED" if identical and measurement["duplicate_records"] == 0
            else "REJECTED"
        )
        report["evaluation"] = {
            "independent_recalculation_matches": identical,
            "independent_measurement": independent,
            "criterion": "nonempty pool with no duplicate canonical examples",
            "passed": status == "ACCEPTED",
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        status = "FAILED"
        report["error"] = {"type": type(error).__name__, "detail": str(error)[:300]}
    report["status"] = status
    report["completed_at"] = time.time()
    report["transitions"].append({"state": status, "at": report["completed_at"]})
    atomic_json(artifact, report)
    return report
