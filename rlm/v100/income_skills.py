"""Host-verified decimal cost exercises; never fabricated trading profits."""

import json
import random
from pathlib import Path

from rlm.v100.common import atomic_json
from rlm.v100.insights import InsightQueue, verified_record
from rlm.v100.protection import file_hash, reserve_audit_sources


def prepare(root: Path, profile: dict) -> dict:
    folder = root / "research/income-cost-skills-v1"
    marker = folder / "manifest.json"
    if marker.exists():
        manifest = json.loads(marker.read_text())
        if any(file_hash(folder / k) != v for k, v in manifest["files"].items()):
            raise ValueError("Verified cost curriculum changed")
        return manifest
    rng = random.Random(20261008)
    records = []
    for _ in range(160):
        capital, movement, entry, exit_fee = (
            rng.randint(100, 5000),
            rng.randint(-100, 100),
            rng.randint(5, 120),
            rng.randint(5, 120),
        )
        records.append(
            verified_record(
                {
                    "kind": "decimal_calculation",
                    "expression": f"{capital}*(1+{movement}/1000)*(1-{exit_fee}/10000)-{capital}*(1+{entry}/10000)",
                }
            )
        )
    train, heldout = records[:128], records[128:]
    reserve_audit_sources(Path(profile["training"]["split_ledger"]), [r["group"] for r in heldout])
    queue = InsightQueue(root)
    try:
        for i, record in enumerate(train):
            queue.add("A" if i % 2 else "B", record["verification"]["task"], admitted=True)
    finally:
        queue.close()
    folder.mkdir(parents=True)
    pool, suite = folder / "pool.jsonl", folder / "development.jsonl"
    pool.write_text("".join(json.dumps(r) + "\n" for r in train))
    suite.write_text(
        "".join(
            json.dumps(
                {
                    "id": r["group"],
                    "skill": "decimal_costs",
                    "messages": r["messages"][:-1],
                    "expected": r["messages"][-1]["content"],
                    "match": "exact",
                }
            )
            + "\n"
            for r in heldout
        )
    )
    manifest = {
        "training_records": 128,
        "reserved_heldout_cases": 32,
        "files": {p.name: file_hash(p) for p in (pool, suite)},
        "weights_changed": False,
        "scope": "Explicit accounting; no forward-market reward labels",
    }
    atomic_json(marker, manifest)
    return manifest
