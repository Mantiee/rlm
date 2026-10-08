import json
import subprocess
import sys

import pytest

from rlm.v100.training_health import TrainingHealth, finite_metrics, initialize_amp


def test_nonfinite_metrics_are_strict_json_and_explicit():
    row = finite_metrics({"loss": 2.0, "grad_norm": float("nan"), "nested": [float("inf")]})
    assert json.loads(json.dumps(row, allow_nan=False))["grad_norm"] is None
    assert row["nonfinite_fields"] == ["grad_norm", "nested[0]"]
    assert row["loss"] == 2.0


def test_initial_amp_scale_keeps_dynamic_backoff_and_growth():
    class Scaler:
        state = {
            "scale": 65536.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
            "_growth_tracker": 7,
        }

        def is_enabled(self):
            return True

        def state_dict(self):
            return dict(self.state)

        def load_state_dict(self, state):
            self.state = state

        def get_scale(self):
            return self.state["scale"]

    scaler = Scaler()
    assert initialize_amp(scaler) == 128.0
    assert scaler.state == {
        "scale": 128.0,
        "growth_factor": 2.0,
        "backoff_factor": 0.5,
        "growth_interval": 2000,
        "_growth_tracker": 0,
    }
    with pytest.raises(ValueError, match="GradScaler"):
        initialize_amp(None)


def test_recoverable_amp_overflow_is_not_counted_as_learning():
    health = TrainingHealth()
    health.step(1, True)
    health.check_metrics(finite_metrics({"step": 1, "loss": 2.0, "grad_norm": float("nan")}))
    assert not health.report()["eligible"]
    for step in range(2, 5):
        health.step(step, False)
        health.check_metrics({"step": step, "loss": 1.9, "grad_norm": 0.7})
    assert health.report()["eligible"]
    assert health.report()["optimizer_updates"] == 3
    assert health.report()["amp_skipped_steps"] == 1


@pytest.mark.parametrize(
    "skipped,field", [(False, "grad_norm"), (True, "loss"), (True, "eval_loss")]
)
def test_nonfinite_loss_or_unskipped_gradient_fails(skipped, field):
    health = TrainingHealth()
    health.step(1, skipped)
    with pytest.raises(FloatingPointError):
        health.check_metrics(finite_metrics({"step": 1, field: float("nan")}))


def test_persistent_or_excessive_skips_reject_candidate():
    health = TrainingHealth()
    for step in range(1, 10):
        health.step(step, True)
    with pytest.raises(FloatingPointError, match="consecutive"):
        health.check_metrics(finite_metrics({"step": 9, "grad_norm": float("nan")}))
    health.step(10, False)
    health.check_metrics({"step": 10, "grad_norm": 1.0})
    assert not health.report()["eligible"]


def test_pilot_streams_to_same_console_and_retains_log(tmp_path, capsys):
    from rlm.v100.training_calibration import run_pilot

    log = tmp_path / "pilot.log"
    code = run_pilot([sys.executable, "-u", "-c", 'print("real pilot output")'], log)
    assert code == 0
    assert capsys.readouterr().out == "real pilot output\n"
    assert log.read_text() == "real pilot output\n"


def test_pilot_timeout_returns_failure_and_cleans_reader(tmp_path, monkeypatch):
    from rlm.v100 import training_calibration

    def timeout(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 1)

    monkeypatch.setattr(training_calibration.subprocess, "run", timeout)
    assert training_calibration.run_pilot(["unused"], tmp_path / "pilot.log", 1) is None
