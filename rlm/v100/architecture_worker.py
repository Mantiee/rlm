"""Trusted launcher copied into an isolated experiment, never used as its judge.

The candidate implements build(config) -> nn.Module with byte-token logits
[batch, length, 257]. This launcher supports full-weight training of new models.
Candidate code is arbitrary and must only be imported inside bubblewrap.
"""

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
    # All parameters of this new network may learn; the Gemma parent is absent.
    model.requires_grad_(True)
    if mode == "train":
        records = json.loads(Path(data_path).read_text())
        examples = [encoded(row, config["context_window"]) for row in records]
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
        scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            value = float(loss.detach())
            print(
                json.dumps({"step": step + 1, "loss": value, "parameters": parameters}), flush=True
            )
            if value < best:
                best = value
                save_file(
                    {
                        k: v.detach().cpu().contiguous().clone()
                        for k, v in model.state_dict().items()
                    },
                    str(output / "weights.safetensors"),
                )
        return
    if mode != "predict":
        raise ValueError("Unknown architecture phase")
    model.load_state_dict(load_file(str(output / "weights.safetensors")))
    model.eval()
    with torch.no_grad():
        for case in json.loads(Path(data_path).read_text()):
            tokens = prompt_bytes(case["messages"])
            if len(tokens) > config["context_window"]:
                raise ValueError("Evaluation prompt exceeds byte context")
            generated = []
            for _ in range(config["max_new_tokens"]):
                if len(tokens) >= config["context_window"]:
                    break
                with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                    logits = model(torch.tensor([tokens], device=device))
                next_token = int(logits[0, -1].argmax())
                if next_token in (10, 256):
                    break
                tokens.append(next_token)
                generated.append(next_token)
            print(
                json.dumps(
                    {"id": case["id"], "answer": bytes(generated).decode("utf-8", errors="replace")}
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
