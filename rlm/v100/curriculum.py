"""Fresh training tasks from trusted calculators, independently of sealed audits."""

import secrets
import sqlite3
from pathlib import Path

from rlm.v100.insights import InsightQueue, verified_record


def request(root: Path, branch: str, domain: str, count: int) -> dict:
    if (
        domain not in ("arithmetic", "linear_equation", "decimal_calculation")
        or type(count) is not int
        or not 1 <= count <= 32
    ):
        raise ValueError("Choose a supported proof domain and 1-32 new examples")
    queue = InsightQueue(root)
    rng, added = secrets.SystemRandom(), 0
    try:
        for _ in range(count * 10):
            a, b, c = (rng.randint(2, 999) for _ in range(3))
            expression = (
                f"({a}+{b})*{c}"
                if domain == "arithmetic"
                else f"{a}*x+{b}={c}"
                if domain == "linear_equation"
                else f"{a}.{b:03d}*{c}/10000"
            )
            task = {"kind": domain, "expression": expression}
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
        "scope": "New calculator-verified training examples; no benchmark or sealed audit answers",
    }
