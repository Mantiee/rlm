"""Resumable LoRA candidates; never automatically replaces the serving model."""

import hashlib
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

from rlm.v100.common import atomic_json


def load_records(path: Path) -> tuple[list[dict], list[dict]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError("Empty dataset")
    for record in records:
        verification = record.get("verification", {})
        if verification.get("kind") != "human_feedback" or verification.get("accepted") is not True:
            raise ValueError("Only explicit verified feedback is supported in this release")
        if not record.get("group") or record["messages"][-1]["role"] != "assistant":
            raise ValueError("Each record needs group and final assistant answer")
    # Connected source groups stay together, avoiding leakage when one answer uses multiple docs.
    groups: list[set[str]] = []
    for record in records:
        sources = set(record.get("document_ids") or [record["group"]])
        overlaps = [group for group in groups if group & sources]
        for group in overlaps:
            sources |= group
            groups.remove(group)
        groups.append(sources)
    if len(groups) < 2:
        raise ValueError("Need at least two independent source groups for held-out validation")
    groups.sort(key=lambda g: hashlib.sha256(json.dumps(sorted(g)).encode()).hexdigest())
    held_out = set().union(*groups[: max(1, len(groups) // 5)])
    train, evaluation = [], []
    for record in records:
        sources = set(record.get("document_ids") or [record["group"]])
        (evaluation if sources & held_out else train).append(record)
    return train, evaluation


def encode_record(record: dict, tokenizer, max_length: int) -> dict:
    messages = record["messages"]
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if full[: len(prefix)] != prefix:
        raise ValueError(
            "Chat template prefix mismatch; assistant-only masking requires explicit template support"
        )
    if len(full) > max_length:
        raise ValueError(
            f"Training example has {len(full)} tokens > {max_length}; shorten sources or raise max_length"
        )
    labels = [-100] * len(prefix) + full[len(prefix) :]
    if not any(label != -100 for label in labels):
        raise ValueError("No supervised answer tokens")
    return {"input_ids": full, "attention_mask": [1] * len(full), "labels": labels}


def completed_checkpoint(output: Path) -> Path:
    candidates = [p for p in output.glob("checkpoint-*") if (p / "complete.json").is_file()]
    if not candidates:
        raise ValueError("No complete checkpoint available")
    checkpoint = max(candidates, key=lambda p: int(p.name.split("-")[-1]))
    for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        if not (checkpoint / name).is_file():
            raise ValueError(f"Incomplete checkpoint: missing {name}")
    return checkpoint


def train_model(profile: dict, dataset_path: Path, resume: bool) -> None:
    # Lazy imports keep the controller free of CUDA/PyTorch dependencies.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Gemma4UnifiedForConditionalGeneration,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise ValueError("This training profile requires a CUDA sm_70 V100")
    free, _ = torch.cuda.mem_get_info()
    if free < 28 * 2**30:
        raise ValueError("Stop inference server before training: need at least 28 GiB free VRAM")
    settings = profile["training"]
    output = Path(settings["output"])
    output.mkdir(parents=True, exist_ok=True)
    train, evaluation = load_records(dataset_path)
    base = Path(settings["base_model"])
    indices = sorted(base.glob("*.safetensors"))
    if not indices:
        raise ValueError("Missing local base safetensors")
    manifest = {
        "data_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "settings": settings,
        "base_config_sha256": hashlib.sha256((base / "config.json").read_bytes()).hexdigest(),
        "base_files": [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in indices],
        "torch": torch.__version__,
        "packages": {
            name: version(name)
            for name in ("transformers", "peft", "accelerate", "bitsandbytes", "datasets")
        },
        "training_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "train_records": len(train),
        "eval_records": len(evaluation),
    }
    # JSON round-trip makes tuples identical after reading the saved manifest.
    manifest = json.loads(json.dumps(manifest))
    if settings["init_adapter"]:
        adapter_path = Path(settings["init_adapter"])
        manifest["initial_adapter"] = {
            name: hashlib.sha256((adapter_path / name).read_bytes()).hexdigest()
            for name in ("adapter_config.json", "adapter_model.safetensors")
        }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(
                "Dataset/base/settings changed. Use a new output directory, not resume."
            )
        if not resume:
            raise ValueError("Output already exists; use --resume or a new output directory")
    elif resume:
        raise ValueError("Cannot resume without manifest")
    atomic_json(manifest_path, manifest)
    checkpoint = completed_checkpoint(output) if resume else None
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_dataset = Dataset.from_list(
        [encode_record(r, tokenizer, settings["max_length"]) for r in train]
    )
    eval_dataset = Dataset.from_list(
        [encode_record(r, tokenizer, settings["max_length"]) for r in evaluation]
    )
    if settings["precision"] not in ("nf4", "fp16"):
        raise ValueError("precision must be nf4 or fp16")
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    kwargs = {"quantization_config": quant} if settings["precision"] == "nf4" else {}
    model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
        base,
        local_files_only=True,
        dtype=torch.float16,
        device_map={"": 0},
        attn_implementation="sdpa",
        **kwargs,
    )
    if settings["precision"] == "nf4":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    if settings["init_adapter"]:
        model = PeftModel.from_pretrained(model, settings["init_adapter"], is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=settings["rank"],
                lora_alpha=2 * settings["rank"],
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
            ),
        )
    model.print_trainable_parameters()

    class Progress(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            row = {
                "step": state.global_step,
                **(logs or {}),
                "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            with (output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

        def on_save(self, args, state, control, **kwargs):
            atomic_json(
                output / f"checkpoint-{state.global_step}" / "complete.json",
                {
                    "step": state.global_step,
                    "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                },
            )

    args = TrainingArguments(
        output_dir=str(output),
        per_device_train_batch_size=settings["microbatch"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=settings["gradient_accumulation"],
        max_steps=settings["max_steps"],
        learning_rate=settings["learning_rate"],
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=settings["eval_steps"],
        save_steps=settings["save_steps"],
        save_total_limit=3,
        save_only_model=False,
        report_to=[],
        seed=42,
        dataloader_num_workers=0,
        include_num_input_tokens_seen=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
        callbacks=[Progress()],
    )
    if not resume:
        atomic_json(output / "baseline_eval.json", trainer.evaluate())
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    candidate = output / "candidate"
    trainer.save_model(str(candidate))
    tokenizer.save_pretrained(candidate)
    atomic_json(output / "candidate_eval.json", trainer.evaluate())
    print(
        "Candidate saved. Serving model unchanged. Compare held-out task quality before merging/exporting."
    )


def export_candidate(profile: dict, root: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, Gemma4UnifiedForConditionalGeneration

    settings = profile["training"]
    output = Path(settings["output"])
    adapter = output / "candidate"
    destination = output / "export-fp16"
    converter_python = root / "venvs/convert/bin/python"
    llama_source = Path(profile["server"]["binary"]).parents[2]
    converter = llama_source / "convert_hf_to_gguf.py"
    quantizer = Path(profile["server"]["binary"]).with_name("llama-quantize")
    for path in (adapter / "adapter_model.safetensors", converter_python, converter, quantizer):
        if not path.is_file():
            raise FileNotFoundError(path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite export: {destination}")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise ValueError("Export profile requires a CUDA sm_70 V100")
    if torch.cuda.mem_get_info()[0] < 28 * 2**30:
        raise ValueError("Stop inference server before exporting: need 28 GiB free VRAM")
    model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
        settings["base_model"],
        dtype=torch.float16,
        device_map={"": 0},
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload(safe_merge=True)
    model.save_pretrained(destination, safe_serialization=True, max_shard_size="1GB")
    AutoTokenizer.from_pretrained(settings["base_model"], local_files_only=True).save_pretrained(
        destination
    )
    atomic_json(
        destination / "v100-export.json",
        {
            "base": settings["base_model"],
            "adapter_sha256": hashlib.sha256(
                (adapter / "adapter_model.safetensors").read_bytes()
            ).hexdigest(),
            "source_candidate": str(adapter),
        },
    )
    del model
    torch.cuda.empty_cache()
    full_gguf = output / "export-F16.gguf"
    quant_gguf = output / "export-Q6_K.gguf"
    for path in (full_gguf, quant_gguf):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite GGUF: {path}")
    subprocess.run(
        [
            str(converter_python),
            str(converter),
            str(destination),
            "--outfile",
            str(full_gguf),
            "--outtype",
            "f16",
        ],
        check=True,
    )
    subprocess.run(
        [
            str(quantizer),
            str(full_gguf),
            str(quant_gguf),
            "Q6_K",
            str(profile["server"]["threads"]),
        ],
        check=True,
    )
    print("Export ready:", quant_gguf)
    print(
        "Compare quality, then set server.model and a NEW runtime.model_version in research/v100.toml. Restart server."
    )
