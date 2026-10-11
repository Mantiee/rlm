import copy
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlm.v100 import backtesting, mission_chat, resource_optimization
from rlm.v100 import inference_benchmark as bench
from rlm.v100.cli import server_command
from rlm.v100.common import load_profile
from tests.test_v100_duel import planned


def profile(tmp_path):
    return load_profile(Path(__file__).parents[1] / "profiles/v100.toml", tmp_path)


def test_ngram_has_no_draft_weights_and_retains_legacy_disabled_draft(tmp_path):
    data = profile(tmp_path)
    assert "--spec-type" not in server_command(data)
    original = copy.deepcopy(data)
    data["server"].update(spec_type="ngram-simple", draft_tokens=4)
    command = server_command(data)
    assert command[command.index("--spec-type") + 1] == "ngram-simple"
    assert (
        "--spec-draft-model" not in command
        and data["server"]["model"] == original["server"]["model"]
    )
    data["server"]["draft_tokens"] = True
    with pytest.raises(ValueError, match="N-gram"):
        server_command(data)


def test_runtime_help_gates_ngram_and_variants_do_not_resize_model_or_context(tmp_path):
    data = profile(tmp_path)
    data["server"]["flash_attention"] = "on"
    old = copy.deepcopy(data)
    assert "ngram" not in [row["name"] for row in bench.variants(data, "legacy native help")]
    rows = bench.variants(data, "--spec-type ngram-simple --spec-draft-n-max")
    assert {row["name"] for row in rows} == {
        "baseline",
        "ngram",
        "q8-cache",
        "ngram-q8-cache",
        "cpu-offload",
    }
    assert data == old
    for row in rows:
        for key in ("model", "context_per_slot", "slots"):
            assert row["server"][key] == data["server"][key]
    assert rows[-1]["server"]["gpu_layers"] == 16


def measured(name, seconds, content="same"):
    return {
        "name": name,
        "state": "measured",
        "server": {},
        "measurement": {
            "seconds": seconds,
            "probes": [{"output": {"content": content}} for _ in range(8)],
        },
    }


def test_faster_wrong_outputs_do_not_win_and_small_speed_difference_keeps_baseline():
    rows = [measured("baseline", 10), measured("ngram", 1, "changed"), measured("cpu-offload", 9.5)]
    assert bench.select(rows)["winner"] == "baseline"
    rows.append(measured("ngram-q8-cache", 8))
    assert bench.select(rows)["winner"] == "ngram-q8-cache"
    assert not rows[1]["same_probe_outputs"]
    assert bench.select([{"name": "baseline", "state": "blocked"}])["state"] == "blocked"


@pytest.mark.parametrize(
    "finish,content,tokens,error",
    [
        ("length", "437", 128, "incomplete"),
        ("stop", "438", 2, "correctness"),
        ("stop", "437", None, "token count"),
    ],
)
def test_incomplete_incorrect_or_unmeasured_output_cannot_be_promoted(
    monkeypatch, finish, content, tokens, error
):
    clock = iter([1.0, 2.0])
    monkeypatch.setattr(bench.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        bench,
        "request",
        lambda *a, **k: {
            "choices": [{"finish_reason": finish, "message": {"content": content}}],
            "usage": {"completion_tokens": tokens},
        },
    )
    with pytest.raises(ValueError, match=error):
        bench.measure(8097, threading.Event())


def test_pressure_stops_only_the_owned_process(monkeypatch):
    stopped = []
    process = SimpleNamespace(poll=lambda: None, terminate=lambda: stopped.append("owned"))
    monkeypatch.setattr(bench, "pressure_reason", lambda: "GPU hot")
    interrupted = threading.Event()
    finished = SimpleNamespace(wait=lambda timeout: False)
    bench.pressure_stop(process, interrupted, finished)
    assert interrupted.is_set() and stopped == ["owned"]


def test_missing_sensors_block_owned_process(monkeypatch):
    monkeypatch.setattr(
        bench, "pressure_reason", lambda: (_ for _ in ()).throw(OSError("sensor failed"))
    )
    interrupted = threading.Event()
    stopped = []
    bench.pressure_stop(
        SimpleNamespace(poll=lambda: None, terminate=lambda: stopped.append(1)),
        interrupted,
        SimpleNamespace(wait=lambda timeout: False),
    )
    assert interrupted.is_set() and stopped == [1]


def test_spot_buy_and_exit_costs_reconcile_independently_on_constant_prices():
    bars = [{"open_time": i, "available_at": i + 1, "open": 100, "close": 100} for i in range(5)]
    result = backtesting.simulate(bars, 2, 5, "buy_hold", 1, 50, 10, capture_executions=True)
    entry, exit_receipt = result["executions"]
    expected_units = 1 / (100 * 1.001 * 1.005)
    expected_cash = expected_units * 100 * 0.999 * 0.995
    assert result["net_return"] == pytest.approx(expected_cash - 1)
    assert entry["execution_friction_percent"] == pytest.approx(100 * (1 - 1 / (1.001 * 1.005)))
    assert exit_receipt["execution_friction_percent"] == pytest.approx(100 * (1 - 0.999 * 0.995))
    assert result["total_costs"] == pytest.approx(1 - expected_cash)
    assert all(row["accounting_verified"] for row in result["executions"])
    zero = backtesting.simulate(bars, 2, 5, "buy_hold", 1, 0, 0, capture_executions=True)
    assert zero["net_return"] == 0 and zero["total_costs"] == 0


def test_combined_requests_do_not_reset_html_or_hijack_ngram_with_cpu_policy(tmp_path):
    for message in (
        "Napraw dashboard i sprawdz koszty przy kupowaniu",
        "Optimize CPU and GPU using n-gram and offload",
        "Zrob lepszy model na RTX, naucz go zarabiac",
    ):
        assert not resource_optimization.requested(message)
    assert (
        mission_chat.direct_facts(tmp_path, "Napraw dashboard i sprawdz koszty przy kupowaniu")
        is None
    )


def test_curriculum_reports_selected_exposure_without_reassigning_validation(tmp_path, monkeypatch):
    _, _, _, duel = planned(tmp_path, monkeypatch)
    for item in duel["branches"].values():
        curriculum = item["curriculum"]
        assert (
            curriculum["available_training_records"] > curriculum["selected_training_records"] == 1
        )
        assert curriculum["validation_records"] > 0
        assert curriculum["approximate_example_passes"] == 200
        assert curriculum["planned_optimizer_steps"] == 25
        assert curriculum["task_kinds"] == ["human_feedback"]


def test_native_failure_reaps_owned_child_and_records_blocked_variants(tmp_path, monkeypatch):
    import hashlib

    data = profile(tmp_path)
    binary, model = tmp_path / "server", tmp_path / "model"
    binary.write_bytes(b"not executed")
    model.write_bytes(b"owned weights")
    data["server"].update(binary=str(binary), model=str(model))
    monkeypatch.setattr(bench, "pressure_reason", lambda **k: None)
    monkeypatch.setattr(bench.subprocess, "check_output", lambda *a, **k: "")
    monkeypatch.setattr(
        bench.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="old help", stderr="")
    )
    children = []

    class Child:
        def __init__(self, *args, **kwargs):
            self.stopped = False
            self.waited = False
            children.append(self)

        def poll(self):
            return 1 if self.stopped else None

        def terminate(self):
            self.stopped = True

        def wait(self, **kwargs):
            self.waited = True

    monkeypatch.setattr(bench.subprocess, "Popen", Child)
    monkeypatch.setattr(bench, "request", lambda *a, **k: {"status": "ok"})
    monkeypatch.setattr(
        bench, "measure", lambda *a: (_ for _ in ()).throw(RuntimeError("probe blocked"))
    )
    result = bench.run(data, tmp_path / "experiment")
    assert result["model_sha256"] == hashlib.sha256(b"owned weights").hexdigest()
    assert result["selection"]["state"] == "blocked"
    assert any("does not advertise" in row.get("error", "") for row in result["variants"])
    assert children and all(child.stopped and child.waited for child in children)


def test_wrong_owned_model_checksum_never_launches_native_process(tmp_path, monkeypatch):
    data = profile(tmp_path)
    binary, model = tmp_path / "server", tmp_path / "model"
    binary.write_bytes(b"not executed")
    model.write_bytes(b"changed weights")
    data["server"].update(binary=str(binary), model=str(model), model_sha256="0" * 64)
    monkeypatch.setattr(bench, "pressure_reason", lambda **k: None)
    monkeypatch.setattr(bench.subprocess, "check_output", lambda *a, **k: "")
    monkeypatch.setattr(
        bench.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="", stderr="")
    )
    monkeypatch.setattr(
        bench.subprocess, "Popen", lambda *a, **k: pytest.fail("Unverified model launched")
    )
    with pytest.raises(ValueError, match="checksum"):
        bench.run(data, tmp_path / "experiment")


def test_master_resource_tool_sees_recorded_speed_proposals_without_claiming_active_service(
    tmp_path,
):
    from rlm.v100.chat_resources import inference_performance
    from rlm.v100.common import atomic_json

    assert inference_performance(tmp_path)["state"] == "not recorded"
    parent = tmp_path / "research/inference-recovery-update"
    folder = parent / "one"
    atomic_json(parent / "current.json", {"profile": str(folder / "profile.json")})
    atomic_json(folder / "profile.json", {"server": {"spec_type": "ngram-simple"}})
    atomic_json(
        folder / "inference-benchmark/results.json",
        {
            "selection": {"state": "measured", "winner": "ngram"},
            "variants": [
                {"name": "ngram", "state": "measured", "speedup": 1.3, "same_probe_outputs": True}
            ],
        },
    )
    receipt = inference_performance(tmp_path)
    assert receipt["configured_spec_type"] == "ngram-simple"
    assert receipt["variants"][0]["speedup"] == 1.3
    assert "not proof" in receipt["scope"]
    atomic_json(parent / "current.json", {"profile": str(tmp_path / "outside.json")})
    with pytest.raises(ValueError, match="escaped"):
        inference_performance(tmp_path)
