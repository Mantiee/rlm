"""Fixed finite quality suites, independent from training loss and model self-grading."""

import json
from pathlib import Path

from rlm.v100.common import atomic_json
from rlm.v100.protection import execution_hash, file_hash


def evaluate_suite(client, profile: dict, suite: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite quality report: {output}")
    rows = [json.loads(line) for line in suite.read_text().splitlines() if line.strip()]
    identifiers = [row["id"] for row in rows]
    if not rows or len(set(identifiers)) != len(rows):
        raise ValueError("Quality suite requires unique cases")
    for row in rows:
        if (
            row.get("match") not in ("exact", "contains")
            or not row.get("expected")
            or not row.get("skill")
        ):
            raise ValueError("Each case needs skill, expected and exact/contains match")
        if not row.get("messages") or any(
            message["role"] == "assistant" for message in row["messages"]
        ):
            raise ValueError("Quality prompts must not contain the expected assistant answer")
    report = {
        "schema": "v100-quality-v1",
        "suite_sha256": file_hash(suite),
        "model_sha256": file_hash(Path(profile["server"]["model"])),
        "execution_sha256": execution_hash(profile),
        "model_version": profile["runtime"]["model_version"],
        "generation": {
            "temperature": 0.0,
            "seed": 42,
            "max_tokens": profile["runtime"]["max_output_tokens"],
            "context_window": profile["runtime"]["context_window"],
            "thinking": False,
        },
        "memory_mode": "fixed prompt fixtures; no live retrieval",
        "cases": [],
        "scope": "Finite suite only; exact/contains validators do not assess all aspects of quality",
    }
    for row in rows:
        answer = client.completion(row["messages"])
        passed = (
            answer.strip() == row["expected"].strip()
            if row["match"] == "exact"
            else row["expected"] in answer
        )
        report["cases"].append(
            {
                "id": row["id"],
                "skill": row["skill"],
                "passed": passed,
                "answer": answer,
                "match": row["match"],
            }
        )
        print(json.dumps({"case": row["id"], "skill": row["skill"], "passed": passed}), flush=True)
    atomic_json(output, report)
    return report
