"""Trusted launcher copied into an isolated experiment, never used as its judge.

The candidate implements build(config) -> nn.Module with byte-token logits
[batch, length, 257]. This launcher supports full-weight training of new models.
Candidate code is arbitrary and must only be imported inside bubblewrap.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def prompt_bytes(messages: list[dict]) -> list[int]:
    text = "".join(m["role"] + ":" + m["content"] + "\n" for m in messages)
    return [256, *list((text + "assistant:").encode())]


def encoded(record: dict, length: int) -> tuple[list[int], list[int]]:
    prefix = prompt_bytes(record["messages"][:-1])
    answer = list((record["messages"][-1]["content"] + "\n").encode())
    tokens = prefix + answer
    if len(tokens) > length:
        raise ValueError("Scratch-model training record exceeds its byte context")
    labels = [-100] * len(prefix) + answer
    return tokens[:-1], labels[1:]


def load_model(config: dict):
    # Only called by the production worker inside its isolated namespace.
    spec = importlib.util.spec_from_file_location("candidate_model", "/work/model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build(config)


def main() -> None:
    import torch
    from safetensors.torch import load_file, save_file

    mode, config_path, data_path, output_path = sys.argv[1:]
    config = json.loads(Path(config_path).read_text())
    output = Path(output_path)
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    device = config["device"]
    if device == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(config["vram_gib"] * 2**30 / total, 0.94))
    model = load_model(config)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Architecture must return a torch.nn.Module")
    parameters = sum(p.numel() for p in model.parameters())
    if not 1 <= parameters <= config["max_parameters"]:
        raise ValueError("Architecture exceeds its parameter budget")
    model.to(device)
    if mode == "train" and config.get("init_weights"):
        initial = Path(config["init_weights"])
        if hashlib.sha256(initial.read_bytes()).hexdigest() != config["init_weights_sha256"]:
            raise ValueError("Scratch continuation weights changed")
        state = load_file(str(initial))
        if config.get("morph_growth"):
            model.load_parent_state(state)
        else:
            model.load_state_dict(state, strict=True)
    # All parameters of this new network may learn; the Gemma parent is absent.
    if not config.get("morph_growth"):
        model.requires_grad_(True)
    if mode == "train":
        records = json.loads(Path(data_path).read_text())
        examples = [encoded(row, config["context_window"]) for row in records]
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=config["learning_rate"]
        )
        scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda", init_scale=128)
        validation = []
        if config.get("validation_file"):
            validation = [
                encoded(row, config["context_window"])
                for row in json.loads(Path(config["validation_file"]).read_text())
            ]
        model.train()
        best = float("inf")
        for step in range(config["steps"]):
            tokens, labels = examples[step % len(examples)]
            x = torch.tensor([tokens], device=device)
            y = torch.tensor([labels], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(x)
                if logits.shape != (*x.shape, 257):
                    raise ValueError("Architecture must return byte-token logits [B,T,257]")
                loss = torch.nn.functional.cross_entropy(
                    logits.float().reshape(-1, 257), y.reshape(-1)
                )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite scratch-model loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            value = float(loss.detach())
            score = value
            if validation:
                model.eval()
                values = []
                with torch.no_grad():
                    for tokens, labels in validation:
                        x = torch.tensor([tokens], device=device)
                        y = torch.tensor([labels], device=device)
                        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                            logits = model(x)
                        measured = torch.nn.functional.cross_entropy(
                            logits.float().reshape(-1, 257), y.reshape(-1)
                        )
                        if not torch.isfinite(measured):
                            raise ValueError("Nonfinite scratch validation loss")
                        values.append(float(measured))
                score = sum(values) / len(values)
                model.train()
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "loss": value,
                        "validation_loss": score if validation else None,
                        "parameters": parameters,
                    }
                ),
                flush=True,
            )
            if score < best:
                best = score
                save_file(
                    {
                        k: v.detach().cpu().contiguous().clone()
                        for k, v in model.state_dict().items()
                    },
                    str(output / "weights.safetensors"),
                )
        return
    if mode not in ("predict", "infer"):
        raise ValueError("Unknown architecture phase")
    model.load_state_dict(load_file(str(output / "weights.safetensors")))
    model.eval()
    with torch.no_grad():
        for case in json.loads(Path(data_path).read_text()):
            tokens = prompt_bytes(case["messages"])
            if len(tokens) > config["context_window"]:
                raise ValueError("Evaluation prompt exceeds byte context")
            generated = []
            finish = "length"
            sampling = case.get("sampling", {}) if mode == "infer" else {}
            torch.manual_seed(sampling.get("seed", config["seed"]))
            for _ in range(config["max_new_tokens"]):
                if len(tokens) >= config["context_window"]:
                    break
                with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                    logits = model(torch.tensor([tokens], device=device))
                scores = logits[0, -1].float()
                if not torch.isfinite(scores).all():
                    raise ValueError("Nonfinite scratch-model logits")
                temperature = sampling.get("temperature", 0)
                if temperature > 0:
                    scores = scores / temperature
                    top_k = min(sampling.get("top_k", 257), 257)
                    scores[scores < torch.topk(scores, top_k).values[-1]] = -float("inf")
                    ordered, ids = scores.sort(descending=True)
                    probability = ordered.softmax(-1)
                    remove = probability.cumsum(-1) - probability > sampling.get("top_p", 1)
                    ordered[remove] = -float("inf")
                    next_token = int(ids[torch.multinomial(ordered.softmax(-1), 1)])
                else:
                    next_token = int(scores.argmax())
                if next_token in (10, 256):
                    finish = "stop"
                    break
                tokens.append(next_token)
                generated.append(next_token)
            row = {"id": case["id"], "answer": bytes(generated).decode("utf-8", errors="replace")}
            if mode == "infer":
                row.update(finish_reason=finish, completion_tokens=len(generated))
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
