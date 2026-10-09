import pytest

from rlm.v100 import architectures, goals, morphology
from tests.test_v100_architectures import workload


def propose(root, name, **changes):
    values = dict(
        architecture="gru",
        width=16,
        layers=1,
        heads=1,
        residual_layers=0,
        bottleneck=8,
        parent_candidate_id="",
        hypothesis="Test goal-specific capacity",
    )
    values.update(changes)
    return morphology.propose(root, "A", name, **values)


def fixture_model(root, name):
    # Only our fixed generated source is executed in this test process.
    namespace = {}
    exec((architectures.candidate_path(root, name) / "source/model.py").read_text(), namespace)
    return namespace["build"]({})


@pytest.mark.parametrize("architecture", ["gru", "transformer"])
def test_causal_models_and_warm_growth_preserve_old_weights(tmp_path, architecture):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file, save_file

    torch.set_num_threads(1)
    _, suite = workload(tmp_path)
    goals.set_goal(tmp_path, "Learn the recorded task", suite)
    propose(tmp_path, "parent", architecture=architecture)
    parent = fixture_model(tmp_path, "parent").eval()
    tokens = torch.tensor([[1, 2, 3, 4]])
    altered = torch.tensor([[1, 2, 200, 201]])
    assert torch.allclose(parent(tokens)[:, :2], parent(altered)[:, :2], atol=1e-6)
    weights = architectures.candidate_path(tmp_path, "parent") / "trial/weights/weights.safetensors"
    weights.parent.mkdir(parents=True)
    save_file(parent.state_dict(), str(weights))
    result = propose(
        tmp_path,
        "grown",
        architecture=architecture,
        residual_layers=2,
        parent_candidate_id="parent",
    )
    assert not result["weights_changed"]
    initial, growth = morphology.initialization(tmp_path, "grown")
    assert initial == weights and growth
    child = fixture_model(tmp_path, "grown")
    child.load_parent_state(load_file(str(initial)))
    assert torch.equal(parent(tokens), child(tokens))
    before = {k: v.clone() for k, v in child.state_dict().items()}
    optimizer = torch.optim.AdamW([p for p in child.parameters() if p.requires_grad], lr=0.01)
    for _ in range(3):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(child(tokens).reshape(-1, 257), tokens.reshape(-1))
        loss.backward()
        optimizer.step()
    for k, v in child.state_dict().items():
        if not k.startswith("residual."):
            assert torch.equal(before[k], v)
    assert any(
        not torch.equal(before[k], v)
        for k, v in child.state_dict().items()
        if k.startswith("residual.")
    )
    output = tmp_path / "grown.safetensors"
    save_file(child.state_dict(), str(output))
    reloaded = fixture_model(tmp_path, "grown")
    reloaded.load_state_dict(load_file(str(output)))
    assert torch.equal(child(tokens), reloaded(tokens))
    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        morphology.initialization(tmp_path, "grown")


@pytest.mark.parametrize(
    "changes",
    [
        {"width": True},
        {"width": 512},
        {"heads": 3},
        {"residual_layers": 5},
        {"width": 256, "layers": 4},
    ],
)
def test_dimension_budget_rejected_before_allocation(tmp_path, changes):
    _, suite = workload(tmp_path)
    goals.set_goal(tmp_path, "Task", suite)
    with pytest.raises(ValueError):
        propose(tmp_path, "invalid", **changes)
    assert not architectures.candidate_path(tmp_path, "invalid").exists()


def test_requires_goal_and_exposes_tools(tmp_path):
    from rlm.v100.research_tools import TOOLS

    with pytest.raises(ValueError, match="goal"):
        propose(tmp_path, "no-goal")
    names = {tool["function"]["name"] for tool in TOOLS}
    assert {"morph_model", "morph_model_status"} <= names


def test_morphology_activation_requires_strict_goal_proof(tmp_path, monkeypatch):
    from rlm.v100 import architecture_goal_gate, architecture_promotion
    from rlm.v100.protection import ExpertRegistry
    from tests.test_v100_architecture_activation import ready

    parent, result, gates, registry = ready(tmp_path, monkeypatch)
    entry = registry.get(result["expert_id"])
    entry["profile"].setdefault("resources", {})["require_goal_improvement"] = True
    monkeypatch.setattr(ExpertRegistry, "get", lambda *args: entry)
    with pytest.raises(ValueError, match="goal improvement"):
        architecture_promotion.activate(tmp_path, parent, result, gates)
    result["goal_gate"] = str(tmp_path / "goal-proof")
    monkeypatch.setattr(
        architecture_goal_gate, "verify", lambda *args: {"passed": True, "improvements": []}
    )
    with pytest.raises(ValueError, match="goal-linked"):
        architecture_promotion.activate(tmp_path, parent, result, gates)


def test_missing_goal_outcomes_stop_master_before_training(tmp_path, monkeypatch):
    from rlm.v100 import architecture_goal_gate, scratch_master, serving
    from rlm.v100.common import atomic_json
    from tests.test_v100_continual import profile

    monkeypatch.setattr(serving, "process_identity", lambda pid: "test-process")
    pool, suite = workload(tmp_path)
    goals.set_goal(tmp_path, "Task", suite)
    propose(tmp_path, "new")
    parent = profile(tmp_path)
    budget = {
        **architectures.DEFAULT_BUDGET,
        "context_window": parent["runtime"]["context_window"],
        "max_new_tokens": parent["runtime"]["max_output_tokens"],
    }
    path = tmp_path / "proposal.json"
    atomic_json(path, {"candidate_id": "new", "budget": budget})
    monkeypatch.setattr(architecture_goal_gate, "prepare", lambda *args: None)

    def forbidden(*args, **kwargs):
        raise AssertionError("must not train without goal outcomes")

    monkeypatch.setattr(scratch_master, "run_candidate", forbidden)
    result = scratch_master.trial(tmp_path, parent, pool, suite, [], path)
    assert result["state"] == "failed; predecessor retained"
    assert "held-out goal" in result["detail"]


def test_worker_growth_train_and_predict_does_not_need_parent_mount(tmp_path, monkeypatch):
    import json
    import sys

    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file, save_file

    from rlm.v100 import architecture_worker
    from rlm.v100.protection import file_hash

    _, suite = workload(tmp_path)
    goals.set_goal(tmp_path, "Task", suite)
    propose(tmp_path, "parent")
    parent = fixture_model(tmp_path, "parent")
    weights = architectures.candidate_path(tmp_path, "parent") / "trial/weights/weights.safetensors"
    weights.parent.mkdir(parents=True)
    save_file(parent.state_dict(), str(weights))
    propose(tmp_path, "grown", residual_layers=1, parent_candidate_id="parent")
    configuration = {
        **architectures.DEFAULT_BUDGET,
        "steps": 2,
        "morph_growth": True,
        "init_weights": str(weights),
        "init_weights_sha256": file_hash(weights),
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps(configuration))
    data = tmp_path / "data.json"
    data.write_text(
        json.dumps(
            [
                {
                    "messages": [
                        {"role": "user", "content": "2+3"},
                        {"role": "assistant", "content": "5"},
                    ]
                }
            ]
        )
    )
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(
        architecture_worker, "load_model", lambda config: fixture_model(tmp_path, "grown")
    )
    monkeypatch.setattr(sys, "argv", ["worker", "train", str(config), str(data), str(output)])
    architecture_worker.main()
    learned = load_file(str(output / "weights.safetensors"))
    assert all(torch.equal(value, learned[key]) for key, value in parent.state_dict().items())
    assert torch.count_nonzero(learned["residual.0.2.weight"]) > 0
    weights.unlink()  # prediction mounts only the child, never the parent
    data.write_text(json.dumps([{"id": "test", "messages": [{"role": "user", "content": "2+3"}]}]))
    monkeypatch.setattr(sys, "argv", ["worker", "predict", str(config), str(data), str(output)])
    architecture_worker.main()
