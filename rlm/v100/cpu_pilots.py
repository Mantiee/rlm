"""Schedule bounded source-disjoint formal CPU pilots without touching the master."""

import json
import time
from pathlib import Path

from rlm.v100.common import atomic_json
from rlm.v100.distributed_compute import propose

PILOTS = (("A", "gru"), ("B", "transformer"))
RETRY_SECONDS = 300


def schedule_cpu_pilots(root: Path, output: Path) -> list[dict]:
    """Queue at most two real owned-CPU jobs; persist failure and never fake results.

    Each worker result still requires the existing independent host validator.
    These arithmetic competence trials are not market/goal outcome evidence.
    """
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for branch, architecture in PILOTS:
        receipt = output / f"cpu-pilot-{branch}.json"
        previous = json.loads(receipt.read_text()) if receipt.is_file() else {}
        if previous.get("state") == "queued":
            results.append(previous)
            continue
        if previous.get("retry_at", 0) > time.time():
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
                queued = propose(
                    root,
                    branch,
                    architecture,
                    40,
                    "Independently validate source-disjoint arithmetic CPU model training",
                    "arithmetic",
                )
                result = {
                    **queued,
                    "branch": branch,
                    "architecture": architecture,
                    "device": "cpu",
                    "kind": "formal-competence-pilot",
                    "source_disjoint": "host-generated formal proof groups; not market outcomes",
                }
            except (OSError, ValueError, RuntimeError, KeyError) as error:
                result = {
                    "branch": branch,
                    "architecture": architecture,
                    "state": "blocked",
                    "reason": f"{type(error).__name__}: {str(error)[:300]}",
                    "retry_at": time.time() + RETRY_SECONDS,
                }
        result["updated"] = time.time()
        atomic_json(receipt, result)
        results.append(result)
    return results
