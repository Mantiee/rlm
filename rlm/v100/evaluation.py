"""Fixed finite quality suites, independent from training loss and model self-grading."""

import hashlib
import json
import math
import time
from pathlib import Path

from rlm.v100.common import atomic_json
from rlm.v100.inference import assert_client_conditions, generation_conditions
from rlm.v100.protection import execution_hash, file_hash


def report_identity(profile: dict, suite: Path) -> dict:
    return {
        "schema": "v100-quality-v1",
        "suite_sha256": file_hash(suite),
        "model_sha256": file_hash(Path(profile["server"]["model"])),
        "execution_sha256": execution_hash(profile),
        "model_version": profile["runtime"]["model_version"],
        "generation": generation_conditions(profile),
        "memory_mode": "fixed prompt fixtures; no live retrieval",
        "scope": "Finite suite only; exact/contains validators do not assess all aspects of quality",
    }


def checkpoint_digest(report: dict) -> str:
    return hashlib.sha256(json.dumps(report, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_checkpoint(path: Path, expected: dict, rows: list[dict]) -> dict:
    if path.is_symlink() or path.stat().st_size > 5 * 2**20:
        raise ValueError("Unsafe evaluation checkpoint")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"schema", "report", "sha256"}:
        raise ValueError("Incomplete evaluation checkpoint")
    report = value["report"]
    if (
        value["schema"] != "v100-evaluation-checkpoint-v1"
        or not isinstance(report, dict)
        or set(report) != set(expected) | {"cases"}
        or any(report.get(key) != entry for key, entry in expected.items())
        or value["sha256"] != checkpoint_digest(report)
    ):
        raise ValueError("Evaluation checkpoint identity or checksum mismatch")
    cases = report["cases"]
    if not isinstance(cases, list) or not cases or len(cases) > len(rows):
        raise ValueError("Evaluation checkpoint needs completed cases")
    fields = {
        "id",
        "skill",
        "passed",
        "answer",
        "match",
        "finish_reason",
        "reasoning_chars",
        "error",
        "seconds",
        "tool_trace",
    }
    for case, row in zip(cases, rows, strict=False):
        if (
            not isinstance(case, dict)
            or set(case) != fields
            or case["id"] != row["id"]
            or case["skill"] != row["skill"]
            or case["match"] != row["match"]
            or type(case["passed"]) is not bool
            or not isinstance(case["answer"], str)
            or case["error"] is not None
            and not isinstance(case["error"], str)
            or case["finish_reason"] is not None
            and not isinstance(case["finish_reason"], str)
            or type(case["reasoning_chars"]) is not int
            or case["reasoning_chars"] < 0
            or type(case["seconds"]) not in (int, float)
            or not math.isfinite(case["seconds"])
            or case["seconds"] < 0
            or not isinstance(case["tool_trace"], list)
        ):
            raise ValueError("Invalid evaluation checkpoint case or order")
        answer, trace = case["answer"], case["tool_trace"]
        passed = (
            answer.strip() == row["expected"].strip()
            if row["match"] == "exact"
            else row["expected"] in answer
        )
        if case["error"] is not None or case["finish_reason"] == "length":
            passed = False
        if "tool_fixture" in row:
            if any(
                not isinstance(step, dict) or not isinstance(step.get("tool"), str)
                for step in trace
            ):
                raise ValueError("Invalid checkpoint tool trace")
            required = set(row["tool_fixture"]["required_tools"])
            if not required <= {step["tool"] for step in trace}:
                passed = False
            if "calculate" in required and not any(
                step["tool"] == "calculate"
                and isinstance(step.get("result"), dict)
                and step["result"].get("answer") == answer.strip()
                for step in trace
            ):
                passed = False
        elif trace:
            raise ValueError("Unexpected checkpoint tool trace")
        if case["passed"] != passed:
            raise ValueError("Checkpoint grade does not match retained answer")
    return report


def resume_baseline(root: Path, profile: dict, suite: Path, output: Path) -> Path | None:
    """Carry newest exact original partial baseline, retaining failures, never selecting by score."""
    destination = output.with_suffix(".partial.json")
    if output.exists() or destination.exists():
        return None
    expected = report_identity(profile, suite)
    rows = [
        json.loads(line) for line in suite.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    paths = root.glob(f"research/mission/run-*/{destination.name}")
    for source in sorted(paths, key=lambda item: item.stat().st_mtime_ns, reverse=True)[:128]:
        if (
            source.parent.resolve() == output.parent.resolve()
            or not source.resolve().is_relative_to((root / "research/mission").resolve())
        ):
            continue
        try:
            report = read_checkpoint(source, expected, rows)
        except (OSError, ValueError, TypeError, KeyError):
            continue
        atomic_json(
            destination,
            {
                "schema": "v100-evaluation-checkpoint-v1",
                "report": report,
                "sha256": checkpoint_digest(report),
            },
        )
        atomic_json(
            output.with_suffix(".resume.json"),
            {
                "source": str(source),
                "source_sha256": file_hash(source),
                "completed": len(report["cases"]),
                "total": len(rows),
                "scope": "Newest matching baseline prefix; retained failures; no score selection",
            },
        )
        return source
    return None


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
    report = {**report_identity(profile, suite), "cases": []}
    checkpoint = output.with_suffix(".partial.json")
    if checkpoint.exists():
        report = read_checkpoint(checkpoint, report_identity(profile, suite), rows)
    completed = len(report["cases"])
    progress_path = output.with_suffix(".progress.json")
    for index, row in enumerate(rows[completed:], start=completed + 1):
        progress = {
            "schema": "v100-evaluation-progress-v1",
            "state": "running",
            "total": len(rows),
            "completed": index - 1,
            "passed": sum(case["passed"] for case in report["cases"]),
            "current_case": row["id"],
            "case_index": index,
            "updated": time.time(),
        }
        if report["cases"]:
            previous = report["cases"][-1]
            progress["last_case"] = {
                key: previous.get(key)
                for key in ("id", "passed", "finish_reason", "error", "seconds")
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
        atomic_json(
            checkpoint,
            {
                "schema": "v100-evaluation-checkpoint-v1",
                "report": report,
                "sha256": checkpoint_digest(report),
            },
        )
        progress.update(
            completed=index,
            passed=sum(case["passed"] for case in report["cases"]),
            state="finished" if index == len(rows) else "running",
            updated=time.time(),
            last_case={
                "id": row["id"],
                "passed": passed,
                "finish_reason": response_info.get("finish_reason"),
                "error": error,
                "seconds": round(time.perf_counter() - started, 2),
            },
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
                    "finish_reason": response_info.get("finish_reason"),
                    "error": error,
                }
            ),
            flush=True,
        )
    if completed == len(rows):
        atomic_json(
            progress_path,
            {
                "schema": "v100-evaluation-progress-v1",
                "state": "finished",
                "total": len(rows),
                "completed": len(rows),
                "passed": sum(case["passed"] for case in report["cases"]),
                "updated": time.time(),
                "resumed": True,
            },
        )
    atomic_json(output, report)
    return report
