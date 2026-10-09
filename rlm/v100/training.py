"""Resumable LoRA candidates; never automatically replaces the serving model."""

import hashlib
import json
import subprocess
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path

from rlm.v100.activity import ActivityLog
from rlm.v100.breeding import base_signature, parent_exports, record_adapter, verified_adapter
from rlm.v100.checkpointing import best_model_arguments, record_best
from rlm.v100.common import atomic_json
from rlm.v100.protection import assert_candidate_output, file_hash, fixed_split
from rlm.v100.retention import (
    AdapterRetention,
    artifact_hashes,
    expand_adapter,
    options,
    select_reference,
)
from rlm.v100.training_health import TrainingHealth, finite_metrics, initialize_amp


def load_records(path: Path, ledger: Path | None = None) -> tuple[list[dict], list[dict]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError("Empty dataset")
    paper_labels = None
    policy_labels = None
    goal_labels = None
    for record in records:
        verification = record.get("verification", {})
        if verification.get("kind") == "deterministic_reference":
            from rlm.v100.insights import verify_record

            verify_record(record)
        elif verification.get("kind") == "goal_observation":
            from rlm.v100.goal_learning import records as goal_records

            if ledger is None:
                raise ValueError("Goal observations require a host split ledger")
            if goal_labels is None:
                goal_labels = {row["id"]: row for row in goal_records(ledger.resolve().parents[2])}
            if record != goal_labels.get(record.get("id")):
                raise ValueError("Goal label differs from the host-observed future outcome")
        elif verification.get("kind") == "historical_postmortem":
            from rlm.v100.backtest_learning import verified

            if ledger is None:
                raise ValueError("Historical reviews require the host split ledger")
            report = Path(verification["report"]).resolve()
            root = ledger.resolve().parents[2]
            if not report.is_relative_to(root / "research/backtests") or record != verified(report):
                raise ValueError(
                    "Historical review differs from the independently rerun experiment"
                )
        elif verification.get("kind") == "paper_policy_preference":
            if ledger is None:
                raise ValueError("Policy preferences require the host ledger")
            from rlm.v100.reward_training import records as policy_records

            if policy_labels is None:
                policy_labels = {
                    row["group"]: row for row in policy_records(ledger.resolve().parents[2])
                }
            if record != policy_labels.get(record.get("group")):
                raise ValueError("Policy preference differs from the audited realized outcome")
        elif verification.get("kind") == "paper_outcome":
            if ledger is None:
                raise ValueError("Paper labels require a host split ledger and audited paper book")
            from rlm.v100.paper_outcomes import records as outcome_records

            if paper_labels is None:
                paper_labels = {
                    row["group"]: row for row in outcome_records(ledger.resolve().parents[2])
                }
            if record != paper_labels.get(record.get("group")):
                raise ValueError("Paper training label differs from the audited realized outcome")
        elif (
            verification.get("kind") != "human_feedback" or verification.get("accepted") is not True
        ):
            raise ValueError(
                "Only explicit verified feedback, checked formal tasks or audited paper outcomes are supported"
            )
        if not record.get("group") or record["messages"][-1]["role"] != "assistant":
            raise ValueError("Each record needs group and final assistant answer")
    if ledger is not None:
        return fixed_split(records, ledger)
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


def chat_token_ids(value) -> list[int]:
    """Normalize single-conversation tokenization, including Transformers 5 mappings."""
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or not value or any(type(token) is not int for token in value):
        raise ValueError("Expected one nonempty token-ID sequence from the chat template")
    return value


def encode_record(record: dict, tokenizer, max_length: int) -> dict:
    messages = record["messages"]
    prefix = chat_token_ids(
        tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    )
    full = chat_token_ids(
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    )
    if full[: len(prefix)] != prefix and hasattr(tokenizer, "encode"):
        # Gemma's inference prompt suppresses reasoning with an empty thought
        # channel. Complete answer-only training turns omit that inference marker.
        # Remove only this exact suffix after the explicit model-turn header;
        # never guess a boundary using a longest-common-prefix heuristic.
        marker = chat_token_ids(
            tokenizer.encode("<|channel>thought\n<channel|>", add_special_tokens=False)
        )
        header = chat_token_ids(tokenizer.encode("<|turn>model\n", add_special_tokens=False))
        if prefix[-len(marker) :] == marker:
            candidate = prefix[: -len(marker)]
            if candidate[-len(header) :] == header and full[: len(candidate)] == candidate:
                prefix = candidate
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


def train_model(profile: dict, dataset_path: Path, resume: bool, root: Path | None = None) -> None:
    # Lazy imports keep the controller free of CUDA/PyTorch dependencies.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise ValueError("This training profile requires a CUDA sm_70 V100")
    free, _ = torch.cuda.mem_get_info()
    if free < 28 * 2**30:
        raise ValueError("Stop inference server before training: need at least 28 GiB free VRAM")
    settings = profile["training"]
    retention_options = options(settings)
    if retention_options["mode"] in (2, 3, 4, 5) and not settings.get("init_adapter"):
        raise ValueError("Historical retention modes require a pinned predecessor adapter")
    best_arguments = best_model_arguments(settings)
    seed = settings.get("seed", 42)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid training seed")
    set_seed(seed)
    output = Path(settings["output"])
    base = Path(settings["base_model"])
    root = root or output.parents[2]
    adapter = Path(settings["init_adapter"]) if settings["init_adapter"] else None
    assert_candidate_output(output, base, adapter, root)
    if settings.get("teacher_adapter"):
        assert_candidate_output(output, base, Path(settings["teacher_adapter"]), root)
    ledger = Path(settings.get("split_ledger", root / "research/state/splits.sqlite3"))
    train, evaluation = load_records(dataset_path, ledger)
    print(json.dumps({"training_stage": "tokenizing-verified-examples"}), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    from rlm.v100.reward_training import encode as encode_with_reward

    has_preferences = any(
        row.get("verification", {}).get("kind") == "paper_policy_preference"
        for row in train + evaluation
    )
    encoder = encode_with_reward if has_preferences else encode_record
    encoded_train = [encoder(r, tokenizer, settings["max_length"]) for r in train]
    encoded_evaluation = [encoder(r, tokenizer, settings["max_length"]) for r in evaluation]
    print(
        json.dumps(
            {
                "training_stage": "verifying-base-file-hashes",
                "train_records": len(encoded_train),
                "validation_records": len(encoded_evaluation),
            }
        ),
        flush=True,
    )
    signature = base_signature(base)
    print(json.dumps({"training_stage": "base-file-hashes-verified"}), flush=True)
    parent_metadata = None
    if adapter is not None:
        parent_metadata, _ = verified_adapter(adapter)
        if parent_metadata["base"] != signature:
            raise ValueError("Initial adapter was trained on a different base")
    if not resume and output.exists() and any(output.iterdir()):
        raise FileExistsError("New training round requires an empty, separate output directory")
    output.mkdir(parents=True, exist_ok=True)
    journal = ActivityLog(root, profile["runtime"].get("activity_branch", "controller"), "trainer")
    journal.write(
        "training",
        "training-start",
        {"output": str(output), "resume": resume, "settings": settings},
    )
    indices = sorted(base.glob("*.safetensors"))
    if not indices:
        raise ValueError("Missing local base safetensors")
    manifest = {
        "data_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "settings": settings,
        "base_config_sha256": hashlib.sha256((base / "config.json").read_bytes()).hexdigest(),
        "base_signature": signature,
        "base_files": [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in indices],
        "torch": torch.__version__,
        "packages": {
            name: version(name)
            for name in ("transformers", "peft", "accelerate", "bitsandbytes", "datasets")
        },
        "training_code_sha256": {
            name: file_hash(Path(__file__).with_name(name))
            for name in (
                "training.py",
                "reward_training.py",
                "foundation.py",
                "distillation.py",
                "protection.py",
                "breeding.py",
                "checkpointing.py",
                "insights.py",
                "paper_outcomes.py",
                "backtest_learning.py",
                "backtesting.py",
                "retention.py",
                "goal_learning.py",
            )
        },
        "train_records": len(train),
        "retention_artifacts": artifact_hashes(output),
        "eval_records": len(evaluation),
        "split_roles": sorted(
            {
                (source, role)
                for role, records in (("train", train), ("validation", evaluation))
                for record in records
                for source in record.get("document_ids") or [record["group"]]
            }
        ),
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
    if settings.get("teacher_adapter"):
        teacher_path = Path(settings["teacher_adapter"])
        teacher_metadata, _ = verified_adapter(teacher_path)
        if teacher_metadata["base"] != signature:
            raise ValueError("Teacher was trained on a different base")
        manifest["teacher_files"] = teacher_metadata["files"]
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
    train_dataset = Dataset.from_list(encoded_train)
    eval_dataset = Dataset.from_list(encoded_evaluation)
    if settings["precision"] not in ("nf4", "fp16"):
        raise ValueError("precision must be nf4 or fp16")
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    kwargs = {"quantization_config": quant} if settings["precision"] == "nf4" else {}
    print(
        json.dumps({"training_stage": "loading-model", "precision": settings["precision"]}),
        flush=True,
    )
    from rlm.v100.foundation import model_class

    model = model_class(Path(settings["base_model"])).from_pretrained(
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
    initial_adapter = settings["init_adapter"]
    if retention_options["growth"] > 1:
        if not initial_adapter:
            raise ValueError("Capacity growth requires an accepted predecessor adapter")
        expanded = output / "expanded-initial"
        if not resume:
            growth_report = expand_adapter(
                Path(initial_adapter), expanded, retention_options["growth"]
            )
            atomic_json(output / "capacity-growth.json", growth_report)
        else:
            verified_adapter(expanded)
        initial_adapter = str(expanded)
    if settings["init_adapter"]:
        model = PeftModel.from_pretrained(model, initial_adapter, is_trainable=True)
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
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise ValueError("Only candidate LoRA parameters may be trainable; base must be frozen")
    strength = float(settings.get("distillation_weight", 0.0))
    temperature = float(settings.get("distillation_temperature", 1.0))
    if not 0 <= strength <= 10 or not 0 < temperature <= 10:
        raise ValueError("Invalid distillation weight or temperature")
    teacher_adapter = settings.get("teacher_adapter", "") or settings["init_adapter"]
    if (strength or has_preferences) and teacher_adapter:
        model.load_adapter(teacher_adapter, adapter_name="teacher", is_trainable=False)
        model.set_adapter("default")
        for name, parameter in model.named_parameters():
            if ".teacher." in name:
                parameter.requires_grad_(False)

    health = TrainingHealth()
    retention = None
    reference_batches = []
    if retention_options["mode"]:
        retention = AdapterRetention(model, retention_options)
        historical_groups = set(settings.get("retention_reference_groups", []))
        reference = select_reference([row for row in train if row["group"] in historical_groups])
        source_groups = [row["group"] for row in reference]
        if retention_options["mode"] in (2, 4, 5):
            if not settings["init_adapter"] or not reference:
                raise ValueError(
                    "EWC/projection needs a predecessor and historical TRAINING references"
                )
            reference_batches = [
                {
                    key: torch.tensor([value], device=next(model.parameters()).device)
                    for key, value in encode_record(row, tokenizer, settings["max_length"]).items()
                }
                for row in reference
            ]
        if resume:
            retention.restore(output, source_groups)
        else:
            if retention_options["mode"] in (2, 4):
                print(
                    json.dumps({"training_stage": "historical-fisher", "examples": len(reference)}),
                    flush=True,
                )
                retention.estimate(model, reference_batches)
            retention.save(output, source_groups)
        journal.write(
            "training",
            "retention-reference",
            {
                "configuration": retention_options,
                "fisher_samples": retention.samples,
                "source_groups": source_groups,
            },
        )

    if not resume:
        manifest["retention_artifacts"] = artifact_hashes(output)
        atomic_json(manifest_path, manifest)

    class Progress(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            if retention is not None and retention_options["mode"] == 5:
                # Trainer has unscaled and clipped gradients at this callback.
                # Skip nonfinite AMP-overflow attempts; the numerical gate tracks them.
                if all(
                    p.grad is None or torch.isfinite(p.grad).all()
                    for p in retention.parameters.values()
                ):
                    projected = retention.project(
                        model, reference_batches[state.global_step % len(reference_batches)]
                    )
                    journal.write(
                        "training",
                        "replay-gradient-projection",
                        {"step": state.global_step, **projected},
                    )

        def on_step_end(self, args, state, control, **kwargs):
            health.step(state.global_step, trainer.accelerator.optimizer_step_was_skipped)
            atomic_json(output / "training_health.json", health.report())

        def on_log(self, args, state, control, logs=None, **kwargs):
            row = finite_metrics(
                {
                    "step": state.global_step,
                    **(logs or {}),
                    "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "amp_scale": trainer.accelerator.scaler.get_scale(),
                }
            )
            if "grad_norm" in row:
                row.update(
                    optimizer_step_skipped=health.last_skipped,
                    optimizer_updates=health.optimizer_updates,
                    amp_skipped_steps=health.skipped_steps,
                )
            with (output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row, allow_nan=False), flush=True)
            journal.write("metrics", "training-progress", row, output=str(output))
            health.check_metrics(row)
            atomic_json(output / "training_health.json", health.report())

        def on_save(self, args, state, control, **kwargs):
            manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            atomic_json(
                output / f"checkpoint-{state.global_step}" / "complete.json",
                {
                    "step": state.global_step,
                    "manifest_sha256": manifest_sha256,
                },
            )
            journal.write(
                "training",
                "checkpoint-complete",
                {
                    "step": state.global_step,
                    "checkpoint": str(output / f"checkpoint-{state.global_step}"),
                },
            )
            record_best(output, state, manifest_sha256)

        def on_train_end(self, args, state, control, **kwargs):
            journal.write(
                "training",
                "training-finished",
                {
                    "step": state.global_step,
                    "best_checkpoint": state.best_model_checkpoint,
                    "best_metric": state.best_metric,
                    "output": str(output),
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
        **best_arguments,
        save_only_model=False,
        report_to=[],
        seed=seed,
        dataloader_num_workers=0,
        include_num_input_tokens_seen=True,
        logging_nan_inf_filter=False,
        remove_unused_columns=not has_preferences,
    )
    from rlm.v100.distillation import preserving_trainer

    trainer_type = preserving_trainer(Trainer, strength, temperature, bool(teacher_adapter))
    if has_preferences:
        from rlm.v100.reward_training import trainer as outcome_trainer

        trainer_type = outcome_trainer(trainer_type, bool(teacher_adapter))

    class FiniteLossTrainer(trainer_type):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            result = super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )
            if retention is not None and model.training:
                penalty = retention.penalty()
                if return_outputs:
                    result = (result[0] + penalty, result[1])
                else:
                    result = result + penalty
            loss = result[0] if return_outputs else result
            if not torch.isfinite(loss.detach()).all().item():
                journal.write("training", "nonfinite-loss", {"step": self.state.global_step})
                raise FloatingPointError("Non-finite loss before backward; candidate rejected")
            return result

    data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)
    if has_preferences:
        from rlm.v100.reward_training import collator

        data_collator = collator(tokenizer, data_collator)
    trainer = FiniteLossTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[Progress()],
    )
    scale = initialize_amp(trainer.accelerator.scaler)
    print(json.dumps({"training_stage": "amp-configured", "initial_amp_scale": scale}), flush=True)
    # compute_loss uses each microbatch's mean loss, not a num_items_in_batch
    # denominator. Ask Trainer to apply gradient-accumulation scaling itself.
    if strength or has_preferences or retention is not None:
        trainer.model_accepts_loss_kwargs = False
    if not resume:
        print(json.dumps({"training_stage": "baseline-validation"}), flush=True)
        atomic_json(output / "baseline_eval.json", trainer.evaluate())
    print(json.dumps({"training_stage": "optimizer-steps"}), flush=True)
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    report = health.report()
    atomic_json(output / "training_health.json", report)
    if not report["eligible"]:
        raise FloatingPointError("Numerical training gate failed; candidate not exported")
    record_best(output, trainer.state, file_hash(manifest_path))
    candidate = output / "candidate"
    trainer.save_model(str(candidate))
    tokenizer.save_pretrained(candidate)
    record_adapter(candidate, signature, parent_metadata["parents"] if parent_metadata else None)
    atomic_json(output / "candidate_eval.json", trainer.evaluate())
    print(
        "Candidate saved. Serving model unchanged. Compare held-out task quality before merging/exporting."
    )


def export_candidate(profile: dict, root: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    settings = profile["training"]
    output = Path(settings["output"])
    assert_candidate_output(
        output,
        Path(settings["base_model"]),
        Path(settings["init_adapter"]) if settings["init_adapter"] else None,
        root,
    )
    adapter = output / "candidate"
    metadata, _ = verified_adapter(adapter)
    parent_model_hashes = parent_exports(metadata)
    if metadata["base"] != base_signature(Path(settings["base_model"])):
        raise ValueError("Candidate base changed before export")
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
    from rlm.v100.foundation import model_class

    model = model_class(Path(settings["base_model"])).from_pretrained(
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
    atomic_json(
        quant_gguf.with_suffix(".provenance.json"),
        {
            "schema": "v100-gguf-v1",
            "model_sha256": file_hash(quant_gguf),
            "adapter": metadata,
            "parent_model_sha256": parent_model_hashes,
        },
    )
    atomic_json(
        adapter / "v100-export.json",
        {
            "model_path": str(quant_gguf),
            "model_sha256": file_hash(quant_gguf),
            "adapter_sha256": metadata["files"]["adapter_model.safetensors"],
        },
    )
    print("Export ready:", quant_gguf)
    print(
        "Compare quality, then set server.model and a NEW runtime.model_version in research/v100.toml. Restart server."
    )
