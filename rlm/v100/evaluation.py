"""Fixed finite quality suites, independent from training loss and model self-grading."""

import json
import time
from pathlib import Path

from rlm.v100.common import atomic_json
from rlm.v100.inference import assert_client_conditions, generation_conditions
from rlm.v100.protection import execution_hash, file_hash


def evaluate_suite(client, profile: dict, suite: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite quality report: {output}")
    assert_client_conditions(client, profile)
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
        if "tool_fixture" in row:
            from rlm.v100.fixture_tools import visible_sources

            visible_sources(row["tool_fixture"])
    report = {
        "schema": "v100-quality-v1",
        "suite_sha256": file_hash(suite),
        "model_sha256": file_hash(Path(profile["server"]["model"])),
        "execution_sha256": execution_hash(profile),
        "model_version": profile["runtime"]["model_version"],
        "generation": generation_conditions(profile),
        "memory_mode": "fixed prompt fixtures; no live retrieval",
        "cases": [],
        "scope": "Finite suite only; exact/contains validators do not assess all aspects of quality",
    }
    progress_path = output.with_suffix(".progress.json")
    for index, row in enumerate(rows, start=1):
        progress = {
            "schema": "v100-evaluation-progress-v1",
            "state": "running",
            "total": len(rows),
            "completed": index - 1,
            "passed": sum(case["passed"] for case in report["cases"]),
            "current_case": row["id"],
            "case_index": index,
        }
        atomic_json(progress_path, progress)
        print(json.dumps({"phase": "evaluation", **progress}), flush=True)
        error = None
        trace = []
        response_info = {}
        started = time.perf_counter()
        try:
            if "tool_fixture" in row:
                from rlm.v100.fixture_tools import fixture_answer

                result = fixture_answer(client, row)
                answer, trace = result["answer"], result["trace"]
                response_info = result["response_info"]
            else:
                answer = client.completion(row["messages"])
        except ValueError as failure:
            # Missing final answers and context errors are failed cases, not gold labels.
            answer, error = "", str(failure)[:400]
        if "tool_fixture" not in row and hasattr(client, "get_response_info"):
            response_info = client.get_response_info()
        passed = (
            answer.strip() == row["expected"].strip()
            if row["match"] == "exact"
            else row["expected"] in answer
        )
        if error is not None or response_info.get("finish_reason") == "length":
            passed = False
        if "tool_fixture" in row:
            if not set(row["tool_fixture"]["required_tools"]) <= {step["tool"] for step in trace}:
                passed = False
                error = error or "Required fixture tool was not used"
            if "calculate" in row["tool_fixture"]["required_tools"] and not any(
                step["tool"] == "calculate" and step["result"]["answer"] == answer.strip()
                for step in trace
            ):
                passed = False
                error = error or "Final answer is not supported by a calculator result"
        report["cases"].append(
            {
                "id": row["id"],
                "skill": row["skill"],
                "passed": passed,
                "answer": answer,
                "match": row["match"],
                "finish_reason": response_info.get("finish_reason"),
                "reasoning_chars": response_info.get("reasoning_chars", 0),
                "error": error,
                "seconds": time.perf_counter() - started,
                "tool_trace": trace,
            }
        )
        progress.update(
            completed=index,
            passed=sum(case["passed"] for case in report["cases"]),
            state="finished" if index == len(rows) else "running",
        )
        atomic_json(progress_path, progress)
        print(
            json.dumps(
                {
                    "case": row["id"],
                    "skill": row["skill"],
                    "passed": passed,
                    "completed": index,
                    "total": len(rows),
                    "seconds": round(time.perf_counter() - started, 2),
                }
            ),
            flush=True,
        )
    atomic_json(output, report)
    return report
