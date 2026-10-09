import copy
import json
from types import SimpleNamespace

import pytest

from rlm.v100.breeding import record_adapter
from rlm.v100.capacity_growth import embedding_extension, residual_column
from rlm.v100.common import atomic_json
from rlm.v100.retention import (
    AdapterRetention,
    artifact_hashes,
    expand_adapter,
    options,
    select_reference,
)

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")


class TinyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(3, 2, bias=False)})
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 1, bias=False)})

    def forward(self, x, y):
        predicted = self.lora_B["default"](self.lora_A["default"](x))
        return SimpleNamespace(loss=(predicted - y).square().mean())


@pytest.mark.parametrize("mode", [1, 2, 3, 4])
def test_penalty_zero_at_anchor_and_positive_after_change(mode, tmp_path):
    torch.manual_seed(7)
    model = TinyAdapter()
    state = AdapterRetention(model, options({"retention_mode": mode}))
    inputs = {"x": torch.ones(1, 3), "y": torch.ones(1, 1)}
    if mode in (2, 4):
        state.estimate(model, [inputs])
        assert model.training
        assert state.samples == 1
        assert all(p.grad is None for p in model.parameters())
    assert state.penalty().item() == 0
    state.save(tmp_path, ["historical-train"])
    hashes = artifact_hashes(tmp_path)
    assert set(hashes) == {"retention.json", "retention-anchor.safetensors"}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1)
    loss = state.penalty()
    assert loss.item() > 0
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    restored = AdapterRetention(model, state.configuration)
    restored.restore(tmp_path, ["historical-train"])
    assert restored.penalty().item() == pytest.approx(loss.item())
    with pytest.raises(ValueError, match="changed"):
        restored.restore(tmp_path, ["audit-label"])
    path = tmp_path / "retention-anchor.safetensors"
    path.write_bytes(path.read_bytes() + b"changed")
    assert artifact_hashes(tmp_path) != hashes
    with pytest.raises(ValueError, match="changed"):
        restored.restore(tmp_path, ["historical-train"])


def test_fisher_requires_history_and_finite_samples():
    model = TinyAdapter()
    state = AdapterRetention(model, options({"retention_mode": 2}))
    with pytest.raises(ValueError, match="historical"):
        state.penalty()
    with pytest.raises(ValueError, match="eight"):
        state.estimate(model, [])
    with pytest.raises(FloatingPointError):
        state.estimate(model, [{"x": torch.ones(1, 3), "y": torch.full((1, 1), float("nan"))}])
    assert model.training


def test_reference_allocation_stops_before_exhausting_host_ram(monkeypatch):
    import psutil

    model = torch.nn.Module()
    model.lora_A = torch.nn.Linear(4096, 2048, bias=False, device="meta")
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=2**30))
    with pytest.raises(ValueError, match="remaining host RAM"):
        AdapterRetention(model, options({"retention_mode": 1}))


def test_projection_removes_conflict_without_overwriting_loss_gradients():
    model = TinyAdapter()
    state = AdapterRetention(model, options({"retention_mode": 5}))
    batch = {"x": torch.ones(1, 3), "y": torch.ones(1, 1)}
    reference = torch.autograd.grad(model(**batch).loss, tuple(model.parameters()))
    for p, gradient in zip(model.parameters(), reference, strict=True):
        p.grad = -gradient.clone()
    result = state.project(model, batch)
    assert result["projected"]
    assert result["gradient_dot_reference"] < 0
    dot = sum((p.grad * g).sum().item() for p, g in zip(model.parameters(), reference, strict=True))
    assert dot >= -1e-6
    for p, gradient in zip(model.parameters(), reference, strict=True):
        p.grad = gradient.clone()
    before = [p.grad.clone() for p in model.parameters()]
    assert not state.project(model, batch)["projected"]
    assert all(torch.equal(p.grad, old) for p, old in zip(model.parameters(), before, strict=True))


def test_rank_growth_preserves_scaling_delta_and_original_files(tmp_path):
    import hashlib

    from safetensors.torch import load_file, save_file

    source = tmp_path / "previous"
    source.mkdir()
    config = {"peft_type": "LORA", "bias": "none", "r": 2, "lora_alpha": 4}
    atomic_json(source / "adapter_config.json", config)
    weights = {"layer.lora_A.weight": torch.randn(2, 3), "layer.lora_B.weight": torch.randn(4, 2)}
    save_file(weights, str(source / "adapter_model.safetensors"))
    signature = {"files": {"config.json": "example"}}
    signature["sha256"] = hashlib.sha256(
        json.dumps(signature["files"], sort_keys=True).encode()
    ).hexdigest()
    record_adapter(source, signature)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    destination = tmp_path / "expanded"
    assert expand_adapter(source, destination, 2)["initial_delta_preserved"]
    grown = load_file(str(destination / "adapter_model.safetensors"))
    assert torch.allclose(
        grown["layer.lora_B.weight"] @ grown["layer.lora_A.weight"],
        weights["layer.lora_B.weight"] @ weights["layer.lora_A.weight"],
    )
    new_config = json.loads((destination / "adapter_config.json").read_text())
    assert new_config["lora_alpha"] / new_config["r"] == 2
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    with pytest.raises(FileExistsError):
        expand_adapter(source, destination, 2)


def test_frozen_column_old_route_survives_new_layer_training():
    torch.manual_seed(42)
    previous = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(0.5))
    x = torch.randn(5, 3)
    previous.eval()
    original = previous(x).detach().clone()
    grown = residual_column(copy.deepcopy(previous), 3, layers=2, bottleneck=2)
    assert torch.equal(grown(x, "expanded"), original)
    optimizer = torch.optim.SGD([p for p in grown.parameters() if p.requires_grad], lr=0.1)
    for _ in range(5):
        grown.train()
        optimizer.zero_grad()
        grown(x, "expanded").square().mean().backward()
        optimizer.step()
    assert torch.equal(grown(x, "previous"), original)
    assert not torch.equal(grown(x, "expanded"), original)
    assert all(p.grad is None for p in grown.predecessor.parameters())


def test_embedding_extension_never_changes_old_ids():
    previous = torch.nn.Embedding(8, 3)
    grown = embedding_extension(previous, 2)
    ids = torch.arange(10)
    original = previous(torch.arange(8)).detach().clone()
    optimizer = torch.optim.SGD(grown.parameters(), lr=0.1)
    for _ in range(3):
        optimizer.zero_grad()
        grown(ids).square().mean().backward()
        optimizer.step()
    assert torch.equal(grown(torch.arange(8)), original)
    assert torch.equal(previous.weight, original)
    with pytest.raises(ValueError):
        grown(torch.tensor([10]))


def test_stratified_references_are_bounded_and_stable():
    rows = [
        {"group": str(i), "verification": {"domain": "common" if i < 19 else "rare"}}
        for i in range(20)
    ]
    selected = select_reference(rows)
    assert len(selected) == 8
    assert any(row["group"] == "19" for row in selected)
    assert selected == select_reference(list(reversed(rows)))


@pytest.mark.parametrize(
    "settings",
    [
        {"retention_mode": True},
        {"retention_mode": 7},
        {"retention_strength": float("nan")},
        {"retention_rank_growth": 3},
    ],
)
def test_invalid_retention_configuration(settings):
    with pytest.raises(ValueError):
        options(settings)


def test_experiment_schema_exposes_measured_history_and_capacity_only(tmp_path):
    from rlm.v100.experiments import parameter_schema, validate_parameters

    profile = {
        "training": {"init_adapter": "", "max_steps": 25},
        "resources": {"retention_experiments": True},
    }
    schema = parameter_schema(profile)
    assert schema["retention_mode"]["enum"] == [0, 1]
    assert schema["retention_rank_growth"]["enum"] == [1]
    adapter = tmp_path / "accepted"
    adapter.mkdir()
    atomic_json(adapter / "adapter_config.json", {"r": 16})
    profile["training"].update(init_adapter=str(adapter), retention_has_history=True)
    schema = parameter_schema(profile)
    assert 5 in schema["retention_mode"]["enum"]
    assert schema["retention_rank_growth"]["enum"] == [1, 2]
    values = {key: rule["enum"][0] for key, rule in schema.items()}
    validate_parameters(values, schema)
    values["retention_mode"] = 100
    with pytest.raises(ValueError):
        validate_parameters(values, schema)


def test_expanded_adapter_peft_load_train_and_reload_preserves_initial_logits(tmp_path):
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    import hashlib

    torch.manual_seed(21)
    config = transformers.GPT2Config(n_layer=1, n_embd=16, n_head=2, vocab_size=32, n_positions=32)
    base = transformers.GPT2LMHeadModel(config)
    untouched = copy.deepcopy(base)
    model = peft.get_peft_model(
        base, peft.LoraConfig(r=2, lora_alpha=4, target_modules=["c_attn"], task_type="CAUSAL_LM")
    )
    with torch.no_grad():
        for name, value in model.named_parameters():
            if "lora_B" in name:
                value.normal_(std=0.01)
    model.eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        original = model(tokens).logits.clone()
    source = tmp_path / "previous"
    model.save_pretrained(source)
    signature = {"files": {"config.json": "test-base"}}
    signature["sha256"] = hashlib.sha256(
        json.dumps(signature["files"], sort_keys=True).encode()
    ).hexdigest()
    record_adapter(source, signature)
    expanded = tmp_path / "expanded"
    expand_adapter(source, expanded, 2)
    candidate = peft.PeftModel.from_pretrained(
        copy.deepcopy(untouched), expanded, is_trainable=True
    )
    candidate.load_adapter(source, adapter_name="teacher", is_trainable=False)
    candidate.set_adapter("default")
    for name, value in candidate.named_parameters():
        if ".teacher." in name:
            value.requires_grad_(False)
    candidate.eval()
    with torch.no_grad():
        assert torch.allclose(candidate(tokens).logits, original, atol=1e-6)
    protected = AdapterRetention(candidate, options({"retention_mode": 2}))
    protected.estimate(candidate, [{"input_ids": tokens, "labels": tokens}])
    candidate.train()
    optimizer = torch.optim.SGD([p for p in candidate.parameters() if p.requires_grad], lr=0.01)
    from rlm.v100.distillation import masked_kl, teacher_mode

    with teacher_mode(candidate, True), torch.no_grad():
        teacher = candidate(tokens).logits.detach()
    prediction = candidate(input_ids=tokens, labels=tokens)
    loss = (
        prediction.loss + 0.1 * masked_kl(prediction.logits, teacher, tokens) + protected.penalty()
    )
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert protected.penalty().item() >= 0
    destination = tmp_path / "trained"
    candidate.save_pretrained(destination)
    reloaded = peft.PeftModel.from_pretrained(copy.deepcopy(untouched), destination)
    candidate.eval()
    reloaded.eval()
    with torch.no_grad():
        assert torch.allclose(candidate(tokens).logits, reloaded(tokens).logits, atol=1e-6)
