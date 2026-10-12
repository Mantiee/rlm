from rlm.v100 import inference_benchmark as benchmark


def test_cooldown_waits_for_transient_pressure(monkeypatch):
    reasons = iter(["busy", None])
    waits = []
    monkeypatch.setattr(benchmark, "pressure_reason", lambda **kwargs: next(reasons))
    monkeypatch.setattr(benchmark.time, "sleep", waits.append)
    assert benchmark.wait_admission(60) is None
    assert waits


def test_cooldown_budget_is_bounded(monkeypatch):
    clock = iter([0, 61])
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(benchmark, "pressure_reason", lambda **kwargs: "temperature")
    assert benchmark.wait_admission(60) == "temperature"
