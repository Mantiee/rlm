"""Trusted tiny byte language-model kernel for owned CPU workers and notebook pilots.

Run as a file, without importing the RLM controller or any received Python code.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path


def validate(job: dict) -> None:
    if job.get("schema") != "v100-compute-job-v1" or job.get("architecture") not in (
        "gru",
        "transformer",
    ):
        raise ValueError("Unknown built-in compute job")
    for name, low, high in (
        ("steps", 10, 100),
        ("width", 32, 128),
        ("layers", 1, 2),
        ("threads", 1, 2),
        ("context", 128, 256),
        ("seconds", 10, 120),
    ):
        if type(job.get(name)) is not int or not low <= job[name] <= high:
            raise ValueError("Compute parameter outside its budget: " + name)
    if job["width"] % 4 or job.get("device") != "cpu":
        raise ValueError("Owned compute workers use CPU, with width divisible by four")
    rate = job.get("learning_rate")
    if type(rate) not in (int, float) or not math.isfinite(rate) or not 1e-5 <= rate <= 1e-2:
        raise ValueError("Invalid compute learning rate")
    for key in ("train", "validation"):
        if not isinstance(job.get(key), list) or not 1 <= len(job[key]) <= 128:
            raise ValueError("Invalid compute dataset size")
        for row in job[key]:
            if set(row) != {"group", "prompt", "answer"} or not all(
                isinstance(v, str) and 1 <= len(v.encode()) <= 1024 for v in row.values()
            ):
                raise ValueError("Invalid compute example")
            if len((row["prompt"] + row["answer"]).encode()) + 2 > job["context"]:
                raise ValueError("Compute example exceeds working context")
    if {r["group"] for r in job["train"]} & {r["group"] for r in job["validation"]}:
        raise ValueError("Compute training/validation sources overlap")


def make_model(job: dict):
    import torch
    from torch import nn

    class ByteModel(nn.Module):
        def __init__(self):
            super().__init__()
            width = job["width"]
            self.embedding = nn.Embedding(257, width)
            if job["architecture"] == "gru":
                self.body = nn.GRU(width, width, job["layers"], batch_first=True)
            else:
                self.position = nn.Embedding(job["context"], width)
                self.body = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(width, 4, width * 2, dropout=0, batch_first=True),
                    job["layers"],
                    enable_nested_tensor=False,
                )
            self.head = nn.Linear(width, 257)

        def forward(self, tokens):
            value = self.embedding(tokens)
            if job["architecture"] == "gru":
                value, _ = self.body(value)
            else:
                value = value + self.position(torch.arange(tokens.shape[1], device=tokens.device))
                mask = torch.triu(
                    torch.ones(
                        tokens.shape[1], tokens.shape[1], device=tokens.device, dtype=torch.bool
                    ),
                    1,
                )
                value = self.body(value, mask=mask)
            return self.head(value)

    model = ByteModel()
    if sum(p.numel() for p in model.parameters()) > 2_000_000:
        raise ValueError("Built-in compute model exceeds two million parameters")
    return model


def batch(rows: list[dict]):
    import torch

    values, labels = [], []
    for row in rows:
        prefix = [256] + list(row["prompt"].encode())
        tokens = prefix + list(row["answer"].encode()) + [256]
        values.append(tokens[:-1])
        labels.append([-100] * (len(prefix) - 1) + tokens[len(prefix) :])
    size = max(map(len, values))
    return (
        torch.tensor([r + [256] * (size - len(r)) for r in values]),
        torch.tensor([r + [-100] * (size - len(r)) for r in labels]),
    )


def loss(model, rows: list[dict]):
    from torch.nn import functional as F

    inputs, targets = batch(rows)
    device = next(model.parameters()).device
    inputs, targets = inputs.to(device), targets.to(device)
    return F.cross_entropy(model(inputs).reshape(-1, 257), targets.reshape(-1))


def evaluate(model, rows: list[dict]) -> float:
    import torch

    model.eval()
    with torch.no_grad():
        values = [float(loss(model, rows[i : i + 4])) for i in range(0, len(rows), 4)]
    value = sum(values) / len(values)
    if not math.isfinite(value):
        raise ValueError("Nonfinite compute validation loss")
    return value


def run(job: dict, output: Path, weights: Path | None = None, device: str = "cpu") -> dict:
    validate(job)
    import torch
    from safetensors.torch import load_file, save_file

    if device not in ("cpu", "cuda") or device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested interactive experiment device is unavailable")
    if device == "cuda":
        # Only an explicitly interactive notebook requests CUDA. Owned workers
        # never pass this argument and their environment hides all GPUs.
        torch.cuda.set_per_process_memory_fraction(0.25)
    torch.set_num_threads(job["threads"])
    torch.manual_seed(42)
    model = make_model(job).to(device)
    output.mkdir(parents=True, exist_ok=False)
    initial = evaluate(model, job["validation"])
    started, metrics = time.monotonic(), []
    if weights is not None:
        model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
        final = evaluate(model, job["validation"])
        report = {"initial_loss": initial, "heldout_loss": final, "weights_promoted": False}
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=job["learning_rate"])
        best = initial
        for step in range(1, job["steps"] + 1):
            if time.monotonic() - started >= job["seconds"]:
                break
            model.train()
            offset = ((step - 1) * 4) % len(job["train"])
            rows = [job["train"][(offset + i) % len(job["train"])] for i in range(4)]
            optimizer.zero_grad(set_to_none=True)
            value = loss(model, rows)
            if not torch.isfinite(value):
                raise ValueError("Nonfinite compute training loss")
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            heldout = evaluate(model, job["validation"])
            row = {
                "step": step,
                "train_loss": float(value.detach()),
                "heldout_loss": heldout,
                "seconds": time.monotonic() - started,
            }
            metrics.append(row)
            print(json.dumps({"phase": "compute-training", **row}), flush=True)
            if heldout <= best:
                best = heldout
                save_file(
                    {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
                    str(output / "weights.safetensors"),
                )
        if not metrics or not (output / "weights.safetensors").exists():
            raise ValueError("Compute trial made no validated improvement")
        report = {
            "initial_loss": initial,
            "heldout_loss": best,
            "metrics": metrics,
            "parameters": sum(p.numel() for p in model.parameters()),
            "weights_promoted": False,
        }
    report["seconds"] = time.monotonic() - started
    (output / "report.json").write_text(json.dumps(report, allow_nan=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args()
    raw = args.job.read_bytes()
    if len(raw) > 2 * 2**20:
        raise ValueError("Compute job exceeds two MiB")
    job = json.loads(raw)
    expected = job["kernel_sha256"]
    if expected != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("Compute kernel differs from the pinned job")
    print(json.dumps(run(job, args.output, args.weights)), flush=True)
