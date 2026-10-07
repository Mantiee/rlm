import copy
import json
import sys

import pytest

from rlm.v100 import architecture_worker, architectures, goals, research_tools
from rlm.v100.common import atomic_json
from rlm.v100.insights import verified_record


def workload(tmp_path):
    pool = tmp_path / "pool.jsonl"
    pool.write_text(
        "".join(
            json.dumps(verified_record({"kind": "arithmetic", "expression": expr})) + "\n"
            for expr in ("2+3", "7*4", "13+19")
        )
    )
    suite = tmp_path / "suite.jsonl"
    suite.write_text(
        json.dumps(
            {
                "id": "held-out",
                "skill": "arithmetic",
                "messages": [{"role": "user", "content": "8+9"}],
                "expected": "17",
                "match": "exact",
            }
        )
        + "\n"
    )
    return pool, suite


def test_user_goal_binds_workload_and_detects_edits(tmp_path):
    _, suite = workload(tmp_path)
    goal = goals.set_goal(tmp_path, "Improve arithmetic and cooperate when useful", suite)
    assert goals.load_goal(tmp_path, suite) == goal
    assert (tmp_path / "research/goals" / (goal["id"] + ".json")).exists()
    suite.write_text(suite.read_text().replace('"17"', '"18"'))
    with pytest.raises(ValueError, match="differs"):
        goals.load_goal(tmp_path, suite)
    path = tmp_path / "research/goal.json"
    edited = json.loads(path.read_text())
    edited["text"] = "New objective"
    atomic_json(path, edited)
    with pytest.raises(ValueError, match="changed"):
        goals.load_goal(tmp_path)


def test_submodel_workload_snapshots_are_checked(tmp_path):
    pool, suite = workload(tmp_path)
    architectures.prepare_inputs(tmp_path, pool, suite)
    a, b = architectures.prepared_inputs(tmp_path)
    assert a.read_bytes() == pool.read_bytes() and b.read_bytes() == suite.read_bytes()
    a.write_text("forged pool")
    with pytest.raises(ValueError, match="changed"):
        architectures.prepared_inputs(tmp_path)


def test_new_architecture_is_not_imported_on_host_and_source_is_bound(tmp_path):
    code = "raise RuntimeError('must never execute here')\ndef build(config): return None\n"
    report = architectures.create_candidate(
        tmp_path, "shared", "joint", code, "Test a different network"
    )
    assert report["owner"] == "shared"
    output = architectures.candidate_path(tmp_path, "joint")
    architectures.verify_candidate(output)
    (output / "source/model.py").write_text(code + "# edit\n")
    with pytest.raises(ValueError, match="changed"):
        architectures.verify_candidate(output)
    with pytest.raises(ValueError, match="identifier"):
        architectures.candidate_path(tmp_path, "../../host")


@pytest.mark.parametrize(
    "key,value",
    [
        ("threads", 32),
        ("ram_gib", 32),
        ("vram_gib", 32),
        ("steps", True),
        ("learning_rate", float("nan")),
        ("device", "auto"),
    ],
)
def test_architecture_resource_requests_must_fit_envelope(key, value):
    budget = copy.deepcopy(architectures.DEFAULT_BUDGET)
    budget[key] = value
    with pytest.raises(ValueError):
        architectures.validate_budget(budget)


def test_two_architecture_trials_cannot_share_one_lease(tmp_path):
    with architectures.resource_lease(tmp_path, "cpu"):
        with pytest.raises(RuntimeError, match="owns"):
            with architectures.resource_lease(tmp_path, "cpu"):
                pytest.fail("Must not acquire the same resource twice")
    with architectures.resource_lease(tmp_path, "cpu"):
        pass


def test_shared_trial_requires_matching_agreement_from_both_parents(tmp_path):
    pool, suite = workload(tmp_path)
    architectures.create_candidate(
        tmp_path, "A", "joint", "def build(config): return None\n", "Joint alternative", joint=True
    )
    with pytest.raises(ValueError, match="both A and B"):
        architectures.run_candidate(tmp_path, "joint", pool, suite)
    consent = architectures.support_candidate(tmp_path, "B", "joint")
    assert consent["both_agree"]
    changed = copy.deepcopy(architectures.DEFAULT_BUDGET)
    changed["steps"] += 1
    with pytest.raises(ValueError, match="both A and B"):
        architectures.run_candidate(tmp_path, "joint", pool, suite, changed)


def test_submodel_prediction_scores_ignore_self_reported_claims(tmp_path):
    _, suite = workload(tmp_path)
    rows = [json.loads(suite.read_text())]
    assert not architectures.score_predictions(rows, [{"id": "held-out", "answer": "wrong"}])[0][
        "passed"
    ]
    assert architectures.score_predictions(rows, [{"id": "held-out", "answer": "17"}])[0]["passed"]
    with pytest.raises(ValueError, match="predictions"):
        architectures.score_predictions(
            rows, [{"id": "held-out", "answer": "wrong", "passed": True}]
        )


def test_autonomous_submodels_remain_cpu_bound_and_require_ram(tmp_path, monkeypatch):
    from rlm.v100 import researchers

    monkeypatch.setattr(researchers, "available_ram_gib", lambda: 7)
    with pytest.raises(RuntimeError, match="deferred"):
        research_tools.ResearchTools(tmp_path, {}, "B").execute(
            "test_submodel", {"candidate_id": "new"}
        )


def test_scratch_worker_really_updates_full_network_weights(tmp_path, monkeypatch, capsys):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file

    # This is a trusted fixture, not execution of a generated architecture on
    # the host. Production imports happen exclusively inside bubblewrap.
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(257, 8)
            self.output = torch.nn.Linear(8, 257)

        def forward(self, tokens):
            return self.output(self.embedding(tokens))

    torch.manual_seed(42)
    initial = {k: v.detach().clone() for k, v in Toy().state_dict().items()}
    monkeypatch.setattr(architecture_worker, "load_model", lambda config: Toy())
    budget = copy.deepcopy(architectures.DEFAULT_BUDGET)
    budget.update(steps=3, threads=1)
    config, data, output = tmp_path / "config.json", tmp_path / "data.json", tmp_path / "weights"
    output.mkdir()
    atomic_json(config, budget)
    record = verified_record({"kind": "arithmetic", "expression": "2+3"})
    atomic_json(data, [record])
    monkeypatch.setattr(sys, "argv", ["runner", "train", str(config), str(data), str(output)])
    architecture_worker.main()
    saved = load_file(str(output / "weights.safetensors"))
    changed = {key for key in initial if not torch.equal(initial[key], saved[key])}
    assert changed == set(initial) and all("lora" not in key for key in changed)
    logs = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(logs) == 3 and all(row["loss"] > 0 for row in logs)
    atomic_json(data, [{"id": "probe", "messages": record["messages"][:-1]}])
    monkeypatch.setattr(sys, "argv", ["runner", "predict", str(config), str(data), str(output)])
    architecture_worker.main()
    assert json.loads(capsys.readouterr().out)["id"] == "probe"


def test_host_evaluates_architecture_without_exposing_gold_to_worker(tmp_path, monkeypatch):
    pool, suite = workload(tmp_path)
    architectures.create_candidate(
        tmp_path, "A", "new", "def build(config): return None\n", "Protocol test"
    )
    seen = []

    def phase(output, run, inputs, budget, mode):
        data = json.loads((inputs / "data.json").read_text())
        seen.append(mode)
        if mode == "train":
            assert all(row["messages"][-1]["role"] == "assistant" for row in data)
            (run / "weights/weights.safetensors").write_bytes(b"bound weights")
        else:
            assert data == [{"id": "held-out", "messages": [{"role": "user", "content": "8+9"}]}]
            assert not any("expected" in row for row in data)
            (run / "predict.log").write_text(json.dumps({"id": "held-out", "answer": "17"}) + "\n")
        return 1.0

    monkeypatch.setattr(architectures, "phase", phase)
    report = architectures.run_candidate(tmp_path, "new", pool, suite)
    assert seen == ["train", "predict"] and report["passed_cases"] == 1
    assert "not replaced" in report["status"]


def test_architecture_never_falls_back_without_bubblewrap(tmp_path, monkeypatch):
    from rlm.v100 import code_lab

    pool, suite = workload(tmp_path)
    architectures.create_candidate(
        tmp_path, "B", "blocked", "def build(config): return None\n", "Needs isolation"
    )
    monkeypatch.setattr(code_lab.shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="bubblewrap"):
        architectures.run_candidate(tmp_path, "blocked", pool, suite)
