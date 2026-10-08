"""Strict JSON metrics and explicit AMP overflow accounting."""

import math
from dataclasses import dataclass


def finite_metrics(metrics: dict) -> dict:
    """Keep non-finite values visible as named failures, never as invalid JSON."""
    invalid = []

    def clean(value, path):
        if isinstance(value, float) and not math.isfinite(value):
            invalid.append(path)
            return None
        if isinstance(value, dict):
            return {
                key: clean(item, f"{path}.{key}" if path else key) for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [clean(item, f"{path}[{index}]") for index, item in enumerate(value)]
        return value

    row = clean(metrics, "")
    if invalid:
        row["nonfinite_fields"] = invalid
    return row


def initialize_amp(scaler) -> float:
    """Lower the initial FP16 scale while preserving public dynamic-scaler settings."""
    if scaler is None or not scaler.is_enabled():
        raise ValueError("V100 FP16 training requires an enabled dynamic GradScaler")
    state = scaler.state_dict()
    state.update(scale=128.0, _growth_tracker=0)
    scaler.load_state_dict(state)
    return scaler.get_scale()


@dataclass
class TrainingHealth:
    attempted_steps: int = 0
    optimizer_updates: int = 0
    skipped_steps: int = 0
    consecutive_skips: int = 0
    last_step: int = -1
    last_skipped: bool = False
    finite_gradient_steps: int = 0

    def step(self, step: int, skipped: bool) -> None:
        if step <= self.last_step:
            raise ValueError("Optimizer progress must increase")
        self.last_step, self.last_skipped = step, skipped
        self.attempted_steps += 1
        self.skipped_steps += int(skipped)
        self.optimizer_updates += int(not skipped)
        self.consecutive_skips = self.consecutive_skips + 1 if skipped else 0

    def check_metrics(self, row: dict) -> None:
        invalid = set(row.get("nonfinite_fields", []))
        if invalid - {"grad_norm"}:
            raise FloatingPointError(f"Non-finite training metrics: {sorted(invalid)}")
        if "grad_norm" in row:
            if row["step"] != self.last_step:
                raise ValueError("Gradient metrics do not match optimizer progress")
            if row["grad_norm"] is None and not self.last_skipped:
                raise FloatingPointError("Non-finite gradient on a non-skipped optimizer step")
            if row["grad_norm"] is not None and not self.last_skipped:
                self.finite_gradient_steps += 1
        if self.consecutive_skips > 8:
            raise FloatingPointError(
                "More than eight consecutive AMP overflows; candidate rejected"
            )

    def report(self) -> dict:
        eligible = (
            self.optimizer_updates > 0
            and self.finite_gradient_steps == self.optimizer_updates
            and not self.last_skipped
            and self.skipped_steps <= self.attempted_steps / 4
        )
        return {
            "schema": "v100-training-health-v1",
            "attempted_steps": self.attempted_steps,
            "optimizer_updates": self.optimizer_updates,
            "amp_skipped_steps": self.skipped_steps,
            "finite_gradient_steps": self.finite_gradient_steps,
            "last_step_skipped": self.last_skipped,
            "eligible": eligible,
            "scope": "Numerical training gate; independent task-quality checks still required",
        }
