import json

import pytest

from rlm.v100.breeding import base_signature, breed_adapters, record_adapter


def parent(path, base, rank, alpha, seed):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    path.mkdir()
    config = {
        "peft_type": "LORA",
        "r": rank,
        "lora_alpha": alpha,
        "bias": "none",
        "target_modules": ["q_proj"],
        "task_type": "CAUSAL_LM",
    }
    (path / "adapter_config.json").write_text(json.dumps(config))
    generator = torch.Generator().manual_seed(seed)
    tensors = {
        "base_model.model.layer.q_proj.lora_A.weight": torch.randn(rank, 4, generator=generator),
        "base_model.model.layer.q_proj.lora_B.weight": torch.randn(3, rank, generator=generator),
    }
    save_file(tensors, str(path / "adapter_model.safetensors"))
    record_adapter(path, base_signature(base))
    return tensors


def test_exact_delta_breeding_with_different_ranks_and_unchanged_parents(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file

    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"same base")
    first, second = tmp_path / "parent-a", tmp_path / "parent-b"
    left, right = parent(first, base, 2, 4, 1), parent(second, base, 3, 9, 2)
    originals = {path: path.read_bytes() for folder in (first, second) for path in folder.iterdir()}
    result = breed_adapters(first, second, tmp_path / "child", 0.25, base, tmp_path)
    child = load_file(str(tmp_path / "child/candidate/adapter_model.safetensors"))
    a, b = sorted(left)
    expected = 0.25 * (4 / 2) * (left[b] @ left[a]) + 0.75 * (9 / 3) * (right[b] @ right[a])
    assert torch.allclose(child[b] @ child[a], expected, atol=1e-6)
    assert result["rank"] == 5
    assert all(path.read_bytes() == content for path, content in originals.items())
    with pytest.raises(FileExistsError):
        breed_adapters(first, second, tmp_path / "child", 0.5, base, tmp_path)
    with pytest.raises(ValueError, match="overlaps"):
        breed_adapters(first, second, first / "child", 0.5, base, tmp_path)
    (base / "model.safetensors").write_bytes(b"different base")
    with pytest.raises(ValueError, match="identical"):
        breed_adapters(first, second, tmp_path / "other-child", 0.5, base, tmp_path)


def test_parent_tampering_and_incompatible_target_modules_rejected(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"base")
    first, second = tmp_path / "parent-a", tmp_path / "parent-b"
    parent(first, base, 2, 4, 1)
    parent(second, base, 2, 4, 2)
    path = second / "adapter_config.json"
    config = json.loads(path.read_text())
    config["target_modules"] = ["v_proj"]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="changed"):
        breed_adapters(first, second, tmp_path / "child", 0.5, base, tmp_path)
    record_adapter(second, base_signature(base))
    with pytest.raises(ValueError, match="incompatible"):
        breed_adapters(first, second, tmp_path / "child", 0.5, base, tmp_path)


def test_real_cpu_trainer_updates_only_child_with_checkpointed_backward(tmp_path):
    torch = pytest.importorskip("torch")
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments

    from rlm.v100.distillation import preserving_trainer

    torch.manual_seed(42)
    model = get_peft_model(
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                use_cache=False,
            )
        ),
        LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )
    predecessor = tmp_path / "teacher"
    model.save_pretrained(predecessor)
    model.load_adapter(predecessor, adapter_name="teacher", is_trainable=False)
    model.set_adapter("default")
    for name, parameter in model.named_parameters():
        if ".teacher." in name:
            parameter.requires_grad_(False)
    originals = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    trainer_type = preserving_trainer(Trainer, 0.1, 1.0, True)
    args = TrainingArguments(
        output_dir=str(tmp_path / "round"),
        max_steps=3,
        use_cpu=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=0.01,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="no",
        report_to=[],
        disable_tqdm=True,
    )
    examples = [
        {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1], "labels": [-100, -100, 3, 4]}
    ] * 8
    trainer = trainer_type(model=model, args=args, train_dataset=examples)
    trainer.model_accepts_loss_kwargs = False
    trainer.train()
    changes = [
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(originals[name], parameter)
    ]
    assert changes and all(".default." in name for name in changes)
    assert model.active_adapter == "default"
    assert any("supervised_loss" in row for row in trainer.state.log_history)
    assert any(row.get("preservation_kl", 0) > 0 for row in trainer.state.log_history)


def test_bred_adapter_loads_into_real_peft_model(tmp_path):
    torch = pytest.importorskip("torch")
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    base = tmp_path / "base"
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    LlamaForCausalLM(config).save_pretrained(base)
    parents = [tmp_path / "a", tmp_path / "b"]
    for index, directory in enumerate(parents):
        model = get_peft_model(
            LlamaForCausalLM.from_pretrained(base, local_files_only=True),
            LoraConfig(r=index + 2, lora_alpha=4, target_modules=["q_proj"], task_type="CAUSAL_LM"),
        )
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.fill_(0.03 * (index + 1))
        model.save_pretrained(directory)
        record_adapter(directory, base_signature(base))
    breed_adapters(*parents, tmp_path / "child", 0.5, base, tmp_path)
    child = PeftModel.from_pretrained(
        LlamaForCausalLM.from_pretrained(base, local_files_only=True), tmp_path / "child/candidate"
    )
    assert child.peft_config["default"].r == 5
    with torch.inference_mode():
        assert torch.isfinite(child(input_ids=torch.tensor([[1, 2, 3]])).logits).all()
