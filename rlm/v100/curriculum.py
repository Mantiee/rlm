"""Fresh training tasks from trusted calculators, independently of sealed audits."""

import json
import secrets
import sqlite3
from pathlib import Path

from rlm.v100.insights import PROOF_DOMAINS, InsightQueue, verified_record


def draw_task(domain: str, rng) -> dict:
    if domain not in PROOF_DOMAINS:
        raise ValueError("Unsupported independent proof domain")
    a, b, c = (rng.randint(2, 999) for _ in range(3))
    if domain == "sequence_transform":
        payload = {
            "operation": rng.choice(("sort", "reverse", "unique_sorted")),
            "values": [rng.randint(-999, 999) for _ in range(rng.randint(4, 8))],
        }
        expression = json.dumps(payload, separators=(",", ":"))
    elif domain == "structured_extraction":
        data = {"label": "item-" + str(a), "count": b, "code": c, "enabled": bool(rng.randrange(2))}
        expression = json.dumps(
            {"data": data, "field": rng.choice(tuple(data))}, separators=(",", ":")
        )
    else:
        expression = (
            f"({a}+{b})*{c}"
            if domain == "arithmetic"
            else f"{a}*x+{b}={c}"
            if domain == "linear_equation"
            else f"{a}.{b:03d}*{c}/10000"
        )
    return {"kind": domain, "expression": expression}


def request(root: Path, branch: str, domain: str, count: int) -> dict:
    if domain not in PROOF_DOMAINS or type(count) is not int or not 1 <= count <= 32:
        raise ValueError("Choose a supported proof domain and 1-32 new examples")
    queue = InsightQueue(root)
    rng, added = secrets.SystemRandom(), 0
    try:
        for _ in range(count * 10):
            task = draw_task(domain, rng)
            row = verified_record(task)
            ledger = root / "research/state/splits.sqlite3"
            if ledger.exists():
                with sqlite3.connect(ledger) as db:
                    if db.execute(
                        "SELECT role FROM roles WHERE source=?", (row["group"],)
                    ).fetchone():
                        continue
            added += queue.add(branch, task, admitted=True)
            if added == count:
                break
    finally:
        queue.close()
    return {
        "new_verified_examples": added,
        "domain": domain,
        "weights_changed": False,
        "scope": "New independent proof-verified training examples; no benchmark or sealed audit answers. Extraction/sequence fixtures are synthetic, not income labels.",
    }
