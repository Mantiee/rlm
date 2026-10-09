"""Model-proposed formal exercises, checked by a fixed host reference."""

import ast
import hashlib
import json
import operator
import re
import sqlite3
from fractions import Fraction
from pathlib import Path

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

PROOF_DOMAINS = (
    "arithmetic",
    "linear_equation",
    "decimal_calculation",
    "sequence_transform",
    "structured_extraction",
)


def reference(task: dict) -> tuple[dict, str]:
    if not isinstance(task, dict) or set(task) != {"kind", "expression"}:
        raise ValueError("Formal task needs kind and expression")
    expression = task["expression"]
    if not isinstance(expression, str) or not 1 <= len(expression) <= 160:
        raise ValueError("Formal expression exceeds its budget")
    if task["kind"] in ("sequence_transform", "structured_extraction"):
        payload = json.loads(expression)
        if not isinstance(payload, dict):
            raise ValueError("Structured proof task must be a JSON object")
        if task["kind"] == "sequence_transform":
            if set(payload) != {"operation", "values"} or payload["operation"] not in (
                "sort",
                "reverse",
                "unique_sorted",
            ):
                raise ValueError("Unsupported sequence operation")
            values = payload["values"]
            if (
                not isinstance(values, list)
                or not 1 <= len(values) <= 16
                or any(type(value) is not int or abs(value) > 10000 for value in values)
            ):
                raise ValueError("Sequence values exceed the proof budget")
            answer = (
                sorted(set(values))
                if payload["operation"] == "unique_sorted"
                else sorted(values)
                if payload["operation"] == "sort"
                else values[::-1]
            )
        else:
            if (
                set(payload) != {"data", "field"}
                or not isinstance(payload["data"], dict)
                or not isinstance(payload["field"], str)
                or payload["field"] not in payload["data"]
            ):
                raise ValueError("Extraction requires an existing exact field")
            if len(payload["data"]) > 8 or any(
                type(value) not in (str, int, bool, type(None))
                or type(value) is int
                and abs(value) > 10000
                or isinstance(value, str)
                and len(value) > 64
                for value in payload["data"].values()
            ):
                raise ValueError("Extraction supports bounded scalar fields only")
            answer = payload["data"][payload["field"]]
        answer = json.dumps(answer, ensure_ascii=False, separators=(",", ":"))
        normalized = {
            "kind": task["kind"],
            "expression": json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        }
    elif task["kind"] == "decimal_calculation":
        from rlm.v100.calculator import calculate

        answer = calculate(expression)["result"]
        normalized = {"kind": task["kind"], "expression": re.sub(r"\s+", "", expression)}
    elif task["kind"] == "arithmetic":
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 40:
            raise ValueError("Formal expression is too complex")

        def calculate(node):
            if (
                isinstance(node, ast.Constant)
                and type(node.value) is int
                and abs(node.value) <= 10000
            ):
                return node.value
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                return calculate(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
            if isinstance(node, ast.BinOp) and type(node.op) in OPS:
                value = OPS[type(node.op)](calculate(node.left), calculate(node.right))
                if abs(value) > 10**12:
                    raise ValueError("Formal result exceeds its numeric budget")
                return value
            raise ValueError(
                "Only bounded integer arithmetic is supported; no calls or code execution"
            )

        answer = str(calculate(tree.body))
        normalized = {"kind": "arithmetic", "expression": ast.unparse(tree.body)}
    elif task["kind"] == "linear_equation":
        match = re.fullmatch(
            r"\s*([+-]?\d{1,4})\s*\*\s*x\s*([+-])\s*(\d{1,4})\s*=\s*([+-]?\d{1,4})\s*", expression
        )
        if not match:
            raise ValueError("Expected a*x+b=c or a*x-b=c with small integers")
        a, sign, b, c = match.groups()
        a, b, c = int(a), int(b) * (-1 if sign == "-" else 1), int(c)
        if a == 0:
            raise ValueError("Linear coefficient cannot be zero")
        answer = str(Fraction(c - b, a))
        normalized = {
            "kind": "linear_equation",
            "expression": f"{a}*x{'+' if b >= 0 else '-'}{abs(b)}={c}",
        }
    else:
        raise ValueError(
            "Unsupported proof domain; keep this hypothesis outside automatic training"
        )
    return normalized, answer


def verified_record(task: dict) -> dict:
    task, answer = reference(task)
    identity = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()
    prompt = (
        "Apply operation to values. unique_sorted removes duplicates and sorts. Output a compact JSON array."
        if task["kind"] == "sequence_transform"
        else "Return the exact data[field] as a compact JSON scalar; quote strings. No prose."
        if task["kind"] == "structured_extraction"
        else "Calculate these explicit decimal costs. Market assumptions are not validated."
        if task["kind"] == "decimal_calculation"
        else "Calculate this integer expression. // is floor division and % is modulo."
        if task["kind"] == "arithmetic"
        else "Solve this linear equation for x. Use an integer or a reduced fraction, e.g. 2/3."
    )
    return {
        "group": "formal-" + identity,
        "document_ids": ["formal-" + identity],
        "messages": [
            {
                "role": "user",
                "content": prompt
                + (
                    "\n"
                    if task["kind"] in ("sequence_transform", "structured_extraction")
                    else " Return only the result.\n"
                )
                + task["expression"],
            },
            {"role": "assistant", "content": answer},
        ],
        "verification": {"kind": "deterministic_reference", "accepted": True, "task": task},
    }


def verify_record(record: dict) -> None:
    if record != verified_record(record["verification"]["task"]):
        raise ValueError(
            "Automatic training example differs from independently calculated reference"
        )


class InsightQueue:
    def __init__(self, root: Path):
        path = root / "research/state/verified-insights.sqlite3"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS insights(id TEXT PRIMARY KEY, branch TEXT NOT NULL, record TEXT NOT NULL, admitted INTEGER NOT NULL DEFAULT 0)"
        )
        if "admitted" not in {row[1] for row in self.db.execute("PRAGMA table_info(insights)")}:
            self.db.execute("ALTER TABLE insights ADD COLUMN admitted INTEGER NOT NULL DEFAULT 0")
        self.db.commit()

    def add(self, branch: str, task: dict, admitted: bool = False) -> bool:
        if branch not in ("A", "B"):
            raise ValueError("Unknown insight origin")
        record = verified_record(task)
        with self.db:
            result = self.db.execute(
                "INSERT OR IGNORE INTO insights VALUES(?,?,?,?)",
                (record["group"], branch, json.dumps(record, ensure_ascii=False), int(admitted)),
            )
        return result.rowcount == 1

    def admit(self, task: dict) -> None:
        record = verified_record(task)
        with self.db:
            self.db.execute("UPDATE insights SET admitted=1 WHERE id=?", (record["group"],))

    def records(self) -> list[dict]:
        records = [
            json.loads(row[0])
            for row in self.db.execute(
                "SELECT record FROM insights WHERE admitted=1 ORDER BY rowid"
            )
        ]
        for record in records:
            verify_record(record)
        return records

    def close(self):
        self.db.close()


def extend_pool(pool: Path, root: Path, destination: Path, profile: dict | None = None) -> bool:
    from rlm.v100.experiments import record_id

    original = [json.loads(line) for line in pool.read_text().splitlines() if line.strip()]
    queue = InsightQueue(root)
    try:
        extra = queue.records()
    finally:
        queue.close()
    from rlm.v100.paper_outcomes import records as outcome_records

    extra.extend(outcome_records(root))
    from rlm.v100.reward_training import records as policy_records

    extra.extend(policy_records(root))
    from rlm.v100.backtest_learning import records as historical_records

    extra.extend(historical_records(root))
    from rlm.v100.goal_learning import admit
    from rlm.v100.goal_learning import records as goal_records

    goal_examples = goal_records(root)
    extra.extend(admit(root, goal_examples, profile) if profile is not None else goal_examples)
    combined = {record_id(record): record for record in original}
    before = len(combined)
    combined.update({record_id(record): record for record in extra})
    if len(combined) == before:
        return False
    with destination.open("x") as output:
        output.write(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in combined.values())
        )
    return True
