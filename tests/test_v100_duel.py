import json
import subprocess
import threading
from concurrent.futures import Future
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlm.v100 import code_lab, competition, experiments, researchers
from rlm.v100.checkpointing import best_model_arguments, record_best
from rlm.v100.common import atomic_json, load_profile
from rlm.v100.protection import execution_hash, file_hash
from rlm.v100.training import completed_checkpoint, load_records


def profile(root):
    chosen = load_profile(Path(__file__).parents[1] / "profiles/v100.toml", root)
    binary = root / "native-server"
    binary.write_bytes(b"native test binary")
    chosen["server"]["binary"] = str(binary)
    return chosen


def records():
    return [
        {
            "group": f"source-{i}",
            "document_ids": [f"source-{i}"],
            "messages": [
                {"role": "user", "content": f"Question {i}"},
                {"role": "assistant", "content": f"Verified answer {i}"},
            ],
            "verification": {"kind": "human_feedback", "accepted": True},
        }
        for i in range(20)
    ]


def decision(row):
    return {
        "selected_ids": [experiments.record_id(row)],
        "parameters": {
            "learning_rate": 0.00005,
            "rank": 16,
            "max_length": 1024,
            "gradient_accumulation": 8,
            "max_steps": 25,
            "distillation_weight": 0.1,
        },
        "rationale": "test a curriculum",
        "message_to_peer": "try a distinct source",
        "research_jobs": [{"role": "tester", "brief": "Find loss anomalies"}],
    }


def planned(tmp_path, monkeypatch):
    data = profile(tmp_path)
    pool = tmp_path / "pool.jsonl"
    pool.write_text("".join(json.dumps(row) + "\n" for row in records()))
    monkeypatch.setattr(
        experiments,
        "choose_experiment",
        lambda c, b, rows, p, h: decision(rows[0 if b == "A" else 1]),
    )
    output = tmp_path / "duel"
    bundle = experiments.plan_duel(None, data, pool, output, tmp_path)
    return data, pool, output, bundle


def test_planner_shared_messages_fixed_validation_and_mandatory_replay(tmp_path, monkeypatch):
    data, pool, first, bundle = planned(tmp_path, monkeypatch)
    datasets = [
        load_records(Path(item["dataset"]), Path(data["training"]["split_ledger"]))
        for item in bundle["branches"].values()
    ]
    assert datasets[0][1] == datasets[1][1]
    assert datasets[0][0] != datasets[1][0]
    shared = experiments.SharedLab(Path(bundle["shared_memory"]))
    assert {row["branch"] for row in shared.recent() if row["kind"] == "message"} == {"A", "B"}
    shared.close()
    second = tmp_path / "duel-2"
    newer = experiments.plan_duel(
        None, data, pool, second, tmp_path, replay=first / "verified-pool.jsonl"
    )
    old_train, old_validation = load_records(pool, Path(data["training"]["split_ledger"]))
    for item in newer["branches"].values():
        train, validation = load_records(
            Path(item["dataset"]), Path(data["training"]["split_ledger"])
        )
        assert train == old_train and validation == old_validation
    Path(newer["branches"]["A"]["dataset"]).write_text("tampered")
    with pytest.raises(ValueError, match="changed"):
        experiments.load_duel(second)


@pytest.mark.parametrize(
    "field,value",
    [
        ("rank", 1024),
        ("max_steps", 100000),
        ("learning_rate", float("nan")),
        ("gradient_accumulation", True),
    ],
)
def test_model_cannot_exceed_parameter_budgets(tmp_path, field, value):
    params = decision(records()[0])["parameters"]
    params[field] = value
    with pytest.raises(ValueError):
        experiments.validate_parameters(params, experiments.parameter_schema(profile(tmp_path)))


def test_native_planner_rejects_unknown_examples_and_does_not_expose_answers(tmp_path, monkeypatch):
    seen = []
    proposed = decision(records()[0])

    def turn(client, messages, **kwargs):
        seen.append(messages)
        return {"content": json.dumps(proposed)}

    monkeypatch.setattr(experiments, "native_turn", turn)
    assert (
        experiments.choose_experiment(SimpleNamespace(), "A", records()[:2], profile(tmp_path), [])
        == proposed
    )
    assert "Verified answer" not in seen[0][1]["content"]
    proposed["selected_ids"] = ["unknown"]
    with pytest.raises(ValueError, match="unknown"):
        experiments.choose_experiment(SimpleNamespace(), "A", records()[:2], profile(tmp_path), [])


def test_planner_repairs_out_of_budget_choice_before_any_training(tmp_path, monkeypatch):
    valid = decision(records()[0])
    invalid = deepcopy(valid)
    invalid["parameters"]["learning_rate"] = 0.001
    responses = iter([invalid, valid])
    calls = []
    client = SimpleNamespace(activity_root=tmp_path, enable_thinking=True, sampling_args={})

    def turn(selected, messages, **kwargs):
        calls.append(deepcopy(messages))
        assert (
            kwargs["response_format"]["schema"]["properties"]["parameters"]["properties"][
                "learning_rate"
            ]["enum"]
            == experiments.parameter_schema(profile(tmp_path))["learning_rate"]["enum"]
        )
        return {"content": json.dumps(next(responses))}

    monkeypatch.setattr(experiments, "native_turn", turn)
    result = experiments.choose_experiment(client, "A", records()[:2], profile(tmp_path), [])
    assert result == valid
    feedback = json.loads(calls[1][-1]["content"])
    assert "learning_rate=0.001" in feedback["host_validation_error"]
    assert feedback["budget"]["learning_rate"]["maximum"] == 0.0002
    assert "Verified answer" not in json.dumps(calls)
    assert client.sampling_args == {} and client.enable_thinking is True
    journals = list((tmp_path / "research/logs/activity").glob("*/A/model/errors.jsonl"))
    event = json.loads(journals[0].read_text())
    assert event["kind"] == "experiment-plan-rejected"
    assert event["payload"]["accepted"] is False
    assert not (tmp_path / "research/state/competition.sqlite3").exists()


def test_invalid_plan_retry_is_bounded_and_never_clamps_values(tmp_path, monkeypatch):
    proposed = decision(records()[0])
    proposed["parameters"]["learning_rate"] = 0.001
    calls = []

    def turn(client, messages, **kwargs):
        calls.append(deepcopy(messages))
        return {"content": json.dumps(proposed)}

    monkeypatch.setattr(experiments, "native_turn", turn)
    with pytest.raises(ValueError, match="learning_rate=0.001"):
        experiments.choose_experiment(SimpleNamespace(), "B", records()[:2], profile(tmp_path), [])
    assert len(calls) == 2
    assert proposed["parameters"]["learning_rate"] == 0.001


def test_all_numeric_choices_are_finite_and_preserve_adapter_and_step_limits(tmp_path):
    chosen = profile(tmp_path)
    chosen["training"]["max_steps"] = 63
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r":32}')
    chosen["training"]["init_adapter"] = str(adapter)
    schema = experiments.parameter_schema(chosen)
    assert schema["rank"]["enum"] == [32]
    assert schema["max_steps"]["enum"] == [25, 50, 63]
    base = {key: rule["enum"][0] for key, rule in schema.items()}
    for key, rule in schema.items():
        for value in rule["enum"]:
            experiments.validate_parameters({**base, key: value}, schema)
    with pytest.raises(ValueError, match="Invalid experiment choice: learning_rate"):
        experiments.validate_parameters({**base, "learning_rate": 0.0000123}, schema)


def test_helpers_can_be_discarded_and_failures_are_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(
        researchers,
        "native_turn",
        lambda *a, **k: {
            "content": json.dumps({"useful_indices": [], "conclusion": "No evidence"})
        },
    )
    result = researchers.review_research(
        SimpleNamespace(), "A", [{"hypothesis": "maybe"}], tmp_path
    )
    assert result["useful_indices"] == []
    future = Future()
    future.set_exception(TimeoutError("assistant too slow"))
    assert competition.collect_worker(future, "A", tmp_path)["status"] == "failed"
    shared = experiments.SharedLab(tmp_path / "research/state/competition.sqlite3")
    assert shared.recent()[-1]["payload"]["error"] == "TimeoutError"
    shared.close()


def test_cpu_worker_really_runs_while_training_process_is_alive(tmp_path, monkeypatch):
    _, _, output, bundle = planned(tmp_path, monkeypatch)
    item = bundle["branches"]["A"]
    training = Path(json.loads(Path(item["profile"]).read_text())["training"]["output"])
    training.mkdir()
    (training / "metrics.jsonl").write_text('{"step":1,"loss":2.1}\n')
    called = threading.Event()

    class FakeTraining:
        returncode = None
        alive = True

        def poll(self):
            if called.is_set():
                self.alive = False
                self.returncode = 0
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

    process = FakeTraining()
    monkeypatch.setattr(competition.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(competition, "available_ram_gib", lambda: 12)
    monkeypatch.setattr(competition, "helper_client", lambda p, *a: None)

    def observe(*args):
        assert process.alive and process.returncode is None
        assert args[3][0]["loss"] == 2.1
        called.set()
        return {"status": "unverified hypothesis"}

    monkeypatch.setattr(competition, "research_task", observe)
    result = competition.train_branch(
        tmp_path, output, "A", item, {"resources": {"min_available_ram_gib": 6}}, train_timeout=5
    )
    assert called.is_set() and result[0]["status"] == "unverified hypothesis"
    workers = json.loads((output / "A/workers.json").read_text())
    assert workers["submitted"] == 1 and workers["unsubmitted"] == 0


def test_partial_metrics_not_treated_as_complete_and_invalid_rows_fail(tmp_path):
    path = tmp_path / "metrics"
    path.write_text('{"step":1}\n{"step":')
    assert competition.complete_metrics(path) == [{"step": 1}]
    path.write_text("corrupt\n")
    with pytest.raises(json.JSONDecodeError):
        competition.complete_metrics(path)


def test_judge_rejects_regressions_against_every_parent(tmp_path, monkeypatch):
    _, _, output, bundle = planned(tmp_path, monkeypatch)
    reports = {}
    for branch, item in bundle["branches"].items():
        chosen = json.loads(Path(item["profile"]).read_text())
        model = Path(chosen["training"]["output"]) / "export-Q6_K.gguf"
        model.parent.mkdir()
        model.write_bytes(branch.encode())
        chosen["server"]["model"] = str(model)
        atomic_json(output / branch / "serving.json", chosen)
        reports[branch] = {
            "schema": "v100-quality-v1",
            "suite_sha256": "fixed",
            "generation": {},
            "memory_mode": "fixed",
            "model_sha256": file_hash(model),
            "execution_sha256": execution_hash(chosen),
            "cases": [
                {"id": "old", "passed": branch == "A"},
                {"id": "new", "passed": branch == "B"},
            ],
        }
    parents = [deepcopy(reports["A"]), deepcopy(reports["B"])]
    verdict = experiments.judge_duel(output, parents, reports)
    assert verdict["winner"] is None
    assert all(not row["eligible"] for row in verdict["branches"].values())


def test_bad_suite_is_rejected_before_starting_any_process(tmp_path, monkeypatch):
    _, _, output, _ = planned(tmp_path, monkeypatch)
    suite = tmp_path / "suite"
    suite.write_text("fixed")
    monkeypatch.setattr(
        competition, "require_idle_gpu", lambda: pytest.fail("started resources too early")
    )
    with pytest.raises(ValueError, match="fixed development suite"):
        competition.run_duel(output, tmp_path, suite, {"suite_sha256": "wrong"})


def test_best_checkpoint_is_kept_loaded_and_latest_is_resumable(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    from safetensors.torch import load_file

    torch.set_num_threads(1)
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(vocab_size=16, n_positions=8, n_embd=8, n_layer=1, n_head=1)
    )
    rows = [{"input_ids": [1, 2, 3, 4], "labels": [1, 2, 3, 4]}] * 6

    class Progress(transformers.TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            atomic_json(
                tmp_path / f"checkpoint-{state.global_step}" / "complete.json",
                {"step": state.global_step},
            )
            record_best(tmp_path, state, "fixed-manifest")

    class NonMonotonic(transformers.Trainer):
        def evaluate(self, *args, **kwargs):
            report = super().evaluate(*args, **kwargs)
            report["eval_loss"] = float(self.state.global_step)
            return report

    args = transformers.TrainingArguments(
        output_dir=str(tmp_path),
        max_steps=5,
        eval_strategy="steps",
        eval_steps=1,
        save_steps=1,
        use_cpu=True,
        per_device_train_batch_size=1,
        learning_rate=0.01,
        report_to=[],
        disable_tqdm=True,
        **best_model_arguments({"eval_steps": 1, "save_steps": 1, "max_steps": 5}),
    )
    trainer = NonMonotonic(
        model=model, args=args, train_dataset=rows, eval_dataset=rows, callbacks=[Progress()]
    )
    trainer.train()
    assert Path(trainer.state.best_model_checkpoint).name == "checkpoint-1"
    assert (tmp_path / "checkpoint-1/complete.json").is_file()
    assert completed_checkpoint(tmp_path).name == "checkpoint-5"
    best = load_file(tmp_path / "checkpoint-1/model.safetensors")
    assert all(torch.equal(model.state_dict()[name], value) for name, value in best.items())
    latest = load_file(tmp_path / "checkpoint-5/model.safetensors")
    assert any(not torch.equal(best[name], value) for name, value in latest.items())
    assert json.loads((tmp_path / "best.json").read_text())["eval_loss"] == 1.0


def source_repo(tmp_path):
    source = tmp_path / "repository"
    (source / "rlm/v100").mkdir(parents=True)
    (source / "rlm/v100/algorithm.py").write_text("def score():\n    return 1\n")
    (source / "rlm/v100/agent.py").write_text("PROMPT='protected'\n")
    (source / "tests").mkdir()
    (source / "tests/test_algorithm.py").write_text("def test_fixed():\n    assert True\n")
    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(source), "add", "rlm", "tests"], check=True)
    return source


def code_candidate(tmp_path, monkeypatch):
    source = source_repo(tmp_path)
    monkeypatch.setattr(
        code_lab,
        "native_turn",
        lambda *a, **k: {
            "content": json.dumps(
                {"find": "return 1", "replace": "return 2", "hypothesis": "Test new behavior"}
            )
        },
    )
    output = tmp_path / "code"
    code_lab.propose_code(None, source, "rlm/v100/algorithm.py", output)
    return source, output


def test_code_candidate_preserves_parent_and_prompts_and_detects_extra_files(tmp_path, monkeypatch):
    source, output = code_candidate(tmp_path, monkeypatch)
    assert "return 1" in (source / "rlm/v100/algorithm.py").read_text()
    assert "return 2" in (output / "source/rlm/v100/algorithm.py").read_text()
    code_lab.verify_code(output)
    with pytest.raises(ValueError, match="read-only"):
        code_lab.propose_code(None, source, "rlm/v100/agent.py", tmp_path / "blocked")
    (output / "source/hidden.py").write_text("unexpected")
    with pytest.raises(ValueError, match="Unexpected"):
        code_lab.verify_code(output)


def test_code_never_runs_on_host_when_namespace_missing(tmp_path, monkeypatch):
    _, output = code_candidate(tmp_path, monkeypatch)
    monkeypatch.setattr(code_lab.shutil, "which", lambda _: None)
    monkeypatch.setattr(
        code_lab.subprocess, "run", lambda *a, **k: pytest.fail("host code execution")
    )
    with pytest.raises(FileNotFoundError, match="never falls back"):
        code_lab.check_code(output)


def test_code_namespace_command_drops_network_and_mounts_sources_readonly(tmp_path, monkeypatch):
    _, output = code_candidate(tmp_path, monkeypatch)
    monkeypatch.setattr(code_lab.shutil, "which", lambda _: "/usr/bin/bwrap")
    command = code_lab.sandbox_command(output, ["python", "-I", "-c", "pass"])
    assert "--unshare-all" in command and "--share-net" not in command
    assert ["--ro-bind", str(output / "source"), "/work"] == command[
        command.index(str(output / "source")) - 1 : command.index(str(output / "source")) + 2
    ]
    assert "--bind" not in command
    assert code_lab.sandbox_environment()["CUDA_VISIBLE_DEVICES"] == ""
    assert "HF_TOKEN" not in code_lab.sandbox_environment(gpu=True)


@pytest.mark.parametrize(
    "settings",
    [
        {"eval_steps": 25, "save_steps": 50, "max_steps": 100},
        {"eval_steps": 25, "save_steps": 25, "max_steps": 10},
    ],
)
def test_best_requires_real_validation_checkpoints(settings):
    with pytest.raises(ValueError):
        best_model_arguments(settings)


def test_cpu_profile_uses_cpu_in_addition_to_gpu(tmp_path):
    from rlm.v100.cli import server_command

    chosen = profile(tmp_path)
    chosen["server"].update(gpu_layers=0, draft_model="", threads=4)
    argv = server_command(chosen)
    assert argv[argv.index("--gpu-layers") + 1] == "0"
    assert argv[argv.index("--threads") + 1] == "4"


def test_record_best_rejects_outside_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="inside"):
        record_best(
            tmp_path,
            SimpleNamespace(best_model_checkpoint="/tmp/outside", best_metric=1, global_step=1),
            "manifest",
        )


def test_evolution_preserves_all_eligible_reports_replays_and_stops_on_failure(
    tmp_path, monkeypatch
):
    data, pool, _, _ = planned(tmp_path, monkeypatch)
    suite = tmp_path / "suite.jsonl"
    suite.write_text("fixed development")
    baseline = {
        "schema": "v100-quality-v1",
        "suite_sha256": file_hash(suite),
        "generation": {},
        "memory_mode": "fixed",
        "cases": [{"id": "old", "passed": True}],
    }

    @contextmanager
    def server(*args):
        yield None

    monkeypatch.setattr(competition, "managed_server", server)
    monkeypatch.setattr(competition, "helper_client", lambda p, *a: None)
    monkeypatch.setattr(competition, "require_idle_gpu", lambda: None)
    plans, gate_counts = [], []
    real_plan = experiments.plan_duel

    def plan(client, settings, pool, output, root, replay=None, recent=False, **kwargs):
        plans.append(
            {
                "init": settings["training"]["init_adapter"],
                "teacher": settings["training"]["teacher_adapter"],
                "replay": replay,
            }
        )
        return real_plan(client, settings, pool, output, root, replay, recent=recent, **kwargs)

    monkeypatch.setattr(competition, "plan_duel", plan)

    def duel(output, root, suite, gates, researcher, timeout):
        gate_counts.append(len(gates))
        for branch in ("A", "B"):
            atomic_json(output / branch / "development-quality.json", baseline)
            parent = json.loads((output / branch / "profile.json").read_text())
            atomic_json(output / branch / "serving.json", parent)
            adapter = output / branch / "training/candidate"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text('{"r":16}')
        return {
            "winner": "tie" if len(gate_counts) == 1 else None,
            "branches": {"A": {"eligible": True}, "B": {"eligible": True}},
        }

    monkeypatch.setattr(competition, "run_duel", duel)
    result = competition.evolve(
        data, tmp_path, pool, tmp_path / "evolution", suite, [baseline], generations=4
    )
    assert gate_counts == [1, 3]
    assert len(result["rounds"]) == 2
    assert plans[1]["init"].endswith("generation-01/A/training/candidate")
    assert plans[1]["init"] == plans[1]["teacher"]
    assert plans[1]["replay"].name == "verified-pool.jsonl"


def test_code_training_only_writes_private_ledger_and_new_round(tmp_path, monkeypatch):
    _, output = code_candidate(tmp_path, monkeypatch)
    report = code_lab.verify_code(output)
    atomic_json(
        output / "check-result.json", {"passed": True, "source_files": report["source_files"]}
    )
    data = profile(tmp_path)
    ledger = Path(data["training"]["split_ledger"])
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in records()))
    load_records(dataset, ledger)
    before = file_hash(ledger)
    destination = tmp_path / "private"
    data["training"]["output"] = str(tmp_path / "training")
    seen = {}

    def sandbox(output, task, mounts, gpu):
        seen.update(mounts=mounts, gpu=gpu)
        return task

    monkeypatch.setattr(code_lab, "sandbox_command", sandbox)
    code_lab.code_training_command(output, data, dataset, tmp_path, destination)
    chosen = json.loads((destination / "worker-profile.json").read_text())
    assert chosen["training"]["output"] == data["training"]["output"]
    assert chosen["training"]["split_ledger"] == str(
        destination / "private-root/research/state/splits.sqlite3"
    )
    assert file_hash(ledger) == before
    assert (ledger, True) not in seen["mounts"]
    assert (Path(data["training"]["base_model"]), False) in seen["mounts"]
    assert seen["gpu"] is True
