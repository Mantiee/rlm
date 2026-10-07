"""Host-timed experiment costs, subordinate to independently measured quality."""

import json
import math
from pathlib import Path

from rlm.v100.protection import file_hash


def observed_training(path: Path) -> dict:
    """Learner-reported diagnostics are advisory, including for self-code trials."""
    rows = (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.exists()
        else []
    )
    summary = {}
    for row in rows:
        for name in (
            "train_runtime",
            "train_steps_per_second",
            "train_samples_per_second",
            "num_input_tokens_seen",
            "peak_vram_gib",
        ):
            value = row.get(name)
            if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                summary[name] = (
                    max(summary.get(name, 0), value) if name == "peak_vram_gib" else value
                )
    runtime, tokens = summary.get("train_runtime", 0), summary.get("num_input_tokens_seen", 0)
    if runtime > 0 and tokens > 0:
        summary["input_tokens_per_second"] = tokens / runtime
    return summary


def load_performance(directory: Path, model: Path, report: dict) -> dict | None:
    path = directory / "performance.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if (
        data.get("schema") != "v100-performance-v1"
        or data["model_sha256"] != file_hash(model)
        or data["quality_report_sha256"] != file_hash(directory / "development-quality.json")
    ):
        raise ValueError("Performance record belongs to different model or quality results")
    if data["model_sha256"] != report["model_sha256"]:
        raise ValueError("Performance record and supplied model report differ")
    for name in (
        "training_wall_seconds",
        "export_wall_seconds",
        "evaluation_wall_seconds",
        "total_wall_seconds",
    ):
        value = data[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid measured experiment duration")
    total = sum(
        data[name]
        for name in ("training_wall_seconds", "export_wall_seconds", "evaluation_wall_seconds")
    )
    if not math.isclose(total, data["total_wall_seconds"], rel_tol=1e-8):
        raise ValueError("Inconsistent experiment timings")
    return data


def continuation(winner: str | None, performance: dict[str, dict | None]) -> tuple[str | None, str]:
    if winner is None:
        return None, "No branch passed the quality gates"
    if winner != "tie":
        return winner, "Higher fixed-suite quality takes precedence over runtime"
    if performance.get("A") is not None and performance.get("B") is not None:
        times = {branch: performance[branch]["total_wall_seconds"] for branch in ("A", "B")}
        faster = min(times, key=times.get)
        slower = "B" if faster == "A" else "A"
        if times[faster] <= 0.95 * times[slower]:
            return (
                faster,
                "Equal task scores; at least 5% lower observed total experiment time in this trial",
            )
    return (
        "A",
        "Equal task scores; missing timings or less than 5% runtime difference; deterministic fallback",
    )
