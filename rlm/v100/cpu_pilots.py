"""Schedule bounded source-disjoint formal CPU pilots without touching the master."""

import json
import time
from pathlib import Path

from rlm.v100.common import atomic_json
from rlm.v100.distributed_compute import inspect, propose

PILOTS = (("A", "gru"), ("B", "transformer"))
RETRY_SECONDS = 300
CUDA_RETRY_SECONDS = 15 * 60
MAX_CUDA_DISPATCHES = 2


def _terminal_cuda_retry(previous: dict, latest: dict, now: float) -> dict:
    """Persist a bounded retry only for a known, diagnosed CUDA reserve failure."""
    state = latest.get("state")
    previous["state"] = state
    previous["updated"] = now
    if state == "cancelled":
        previous.update(retry_at=None, retry_policy="terminal-cancelled")
        return previous
    if state != "failed":
        return previous

    dispatches = previous.get("cuda_dispatches")
    attempts = latest.get("attempts")
    failure = latest.get("last_failure")
    raw_diagnostic = failure.get("resource_failure") if isinstance(failure, dict) else None
    try:
        from rlm.v100.remote_nodes import cuda_resource_failure

        diagnostic = cuda_resource_failure(raw_diagnostic)
    except (ImportError, OSError, TypeError, ValueError):
        diagnostic = None
    if (
        type(dispatches) is not int
        or dispatches < 1
        or dispatches >= MAX_CUDA_DISPATCHES
        or type(attempts) is not int
        or not 1 <= attempts < 3
        or diagnostic is None
    ):
        previous.update(
            retry_at=None,
            retry_policy="blocked-unverified-cuda-failure",
            resource_failure=diagnostic,
        )
        return previous
    previous.update(
        retry_at=now + CUDA_RETRY_SECONDS,
        retry_policy="diagnosed-resource-cooldown",
        resource_failure=diagnostic,
    )
    return previous


def schedule_cpu_pilots(root: Path, output: Path) -> list[dict]:
    """Queue at most two real owned-CPU jobs; persist failure and never fake results.

    Each worker result still requires the existing independent host validator.
    These arithmetic competence trials are not market/goal outcome evidence.
    """
    output.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        available = inspect(root)
        jobs_by_id = {job["id"]: job for job in available["jobs"]}
        gpu_workers = [
            worker
            for worker in available["workers"]
            if worker.get("phase") == "idle"
            and not worker.get("stale", True)
            and worker.get("device") == "cuda"
        ]
    except (OSError, ValueError, KeyError):
        jobs_by_id = {}
        gpu_workers = []
    for branch, architecture in PILOTS:
        receipt = output / f"cpu-pilot-{branch}.json"
        previous = json.loads(receipt.read_text()) if receipt.is_file() else {}
        now = time.time()
        if previous.get("state") == "queued":
            latest = jobs_by_id.get(previous.get("id"), {})
            if latest:
                if previous.get("device") == "cuda" and latest.get("state") in {
                    "failed",
                    "cancelled",
                }:
                    previous = _terminal_cuda_retry(previous, latest, now)
                else:
                    previous["state"] = latest.get("state", "queued")
                    previous["updated"] = now
                atomic_json(receipt, previous)
            results.append(previous)
            continue
        if previous.get("device") == "cuda" and previous.get("state") in {
            "failed",
            "cancelled",
        }:
            if previous.get("retry_policy") != "diagnosed-resource-cooldown":
                results.append(previous)
                continue
            if previous.get("retry_at", 0) > now:
                results.append(previous)
                continue
        elif (
            previous.get("device") == "cuda"
            and previous.get("retry_policy") == "operator-review-required"
        ):
            results.append(previous)
            continue
        elif type(previous.get("retry_at")) in (int, float) and previous["retry_at"] > now:
            results.append(previous)
            continue

        if not (root / "research/compute-config.json").is_file():
            result = {
                "branch": branch,
                "architecture": architecture,
                "state": "blocked",
                "reason": "Owned compute mailbox is not configured",
                "retry_at": time.time() + RETRY_SECONDS,
            }
        else:
            try:
                device = "cuda" if branch == "B" and gpu_workers else "cpu"
                dispatches = previous.get("cuda_dispatches", 0)
                if device == "cuda" and type(dispatches) is not int:
                    raise ValueError(
                        "Malformed persisted CUDA dispatch count; operator review required"
                    )
                queued = propose(
                    root,
                    branch,
                    architecture,
                    40,
                    "Independently validate source-disjoint arithmetic CPU model training",
                    "arithmetic",
                    device=device,
                )
                result = {
                    **queued,
                    "branch": branch,
                    "architecture": architecture,
                    "device": device,
                    "kind": "formal-competence-pilot",
                    "source_disjoint": "host-generated formal proof groups; not market outcomes",
                }
                if device == "cuda":
                    result["cuda_dispatches"] = dispatches + 1
                    result["retry_policy"] = "terminal-after-cancel-or-unverified-failure"
            except (OSError, ValueError, RuntimeError, KeyError) as error:
                result = {
                    "branch": branch,
                    "architecture": architecture,
                    "state": "blocked",
                    "reason": f"{type(error).__name__}: {str(error)[:300]}",
                    "retry_at": None if device == "cuda" else time.time() + RETRY_SECONDS,
                    "device": device,
                    "retry_policy": (
                        "operator-review-required" if device == "cuda" else "dispatch-cooldown"
                    ),
                    "cuda_dispatches": previous.get("cuda_dispatches", 0),
                }
        result["updated"] = time.time()
        atomic_json(receipt, result)
        results.append(result)
    return results
