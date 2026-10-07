"""Best validation-loss selection; latest complete checkpoint remains resumable."""

import math
from pathlib import Path

from rlm.v100.common import atomic_json


def best_model_arguments(settings: dict) -> dict:
    evaluate, save, steps = (settings[name] for name in ("eval_steps", "save_steps", "max_steps"))
    if any(type(value) is not int or value < 1 for value in (evaluate, save, steps)):
        raise ValueError("Evaluation/save intervals and max_steps must be positive integers")
    if save != evaluate or steps < evaluate:
        raise ValueError("Best selection requires save_steps == eval_steps <= max_steps")
    return {
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": 3,
    }


def record_best(output: Path, state, manifest_sha256: str) -> None:
    if state.best_model_checkpoint is None:
        return
    checkpoint = Path(state.best_model_checkpoint).resolve()
    if (
        not checkpoint.is_relative_to(output.resolve())
        or not (checkpoint / "complete.json").is_file()
    ):
        raise ValueError("Best checkpoint must be complete and inside its training round")
    metric = float(state.best_metric)
    if not math.isfinite(metric):
        raise ValueError("Best checkpoint has a nonfinite validation metric")
    atomic_json(
        output / "best.json",
        {
            "checkpoint": str(checkpoint),
            "eval_loss": metric,
            "observed_at_step": state.global_step,
            "manifest_sha256": manifest_sha256,
            "scope": "Lowest observed development eval_loss, not a task-quality or audit verdict",
        },
    )
