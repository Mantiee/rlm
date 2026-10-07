"""Exact same-base LoRA delta combinations, always saved as unpromoted children."""

import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

from rlm.v100.common import atomic_json
from rlm.v100.protection import assert_candidate_output, file_hash


def base_signature(base: Path) -> dict:
    weights = sorted(base.glob("*.safetensors"))
    if not weights:
        raise ValueError("Base signature requires local safetensors")
    files = {
        path.name: file_hash(path)
        for path in sorted(base.iterdir())
        if path.is_file() and path.suffix in (".json", ".safetensors", ".model", ".jinja", ".txt")
    }
    return {
        "files": files,
        "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
    }


def record_adapter(directory: Path, signature: dict, parents: list[dict] | None = None) -> dict:
    data = {
        "schema": "v100-adapter-v1",
        "base": signature,
        "files": {
            name: file_hash(directory / name)
            for name in ("adapter_config.json", "adapter_model.safetensors")
        },
        "parents": parents or [],
        "status": "candidate only; no automatic promotion",
    }
    atomic_json(directory / "v100-adapter.json", data)
    return data


def verified_adapter(directory: Path) -> tuple[dict, dict]:
    path = directory / "v100-adapter.json"
    if not path.is_file():
        raise ValueError("Adapter lacks verified base provenance: v100-adapter.json")
    data = json.loads(path.read_text())
    if data["schema"] != "v100-adapter-v1":
        raise ValueError("Unsupported adapter provenance")
    signature = hashlib.sha256(
        json.dumps(data["base"]["files"], sort_keys=True).encode()
    ).hexdigest()
    if signature != data["base"]["sha256"]:
        raise ValueError("Invalid base signature")
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if file_hash(directory / name) != data["files"][name]:
            raise ValueError("Parent adapter changed after recording its provenance")
    config = json.loads((directory / "adapter_config.json").read_text())
    if config["peft_type"] != "LORA" or config.get("bias", "none") != "none":
        raise ValueError("Breeding supports bias-free standard LoRA only")
    for name in (
        "use_dora",
        "use_rslora",
        "rank_pattern",
        "alpha_pattern",
        "modules_to_save",
        "trainable_token_indices",
        "layer_replication",
        "lora_bias",
        "target_parameters",
    ):
        if config.get(name):
            raise ValueError(f"Unsupported breeding setting: {name}")
    if type(config["r"]) is not int or config["r"] < 1:
        raise ValueError("Invalid parent rank")
    if not math.isfinite(config["lora_alpha"]) or config["lora_alpha"] <= 0:
        raise ValueError("Invalid parent scaling")
    return data, config


def parent_exports(metadata: dict) -> list[str]:
    """Bind each bred parent to its own exported model before promotion evaluation."""
    hashes = []
    for parent in metadata.get("parents", []):
        path = Path(parent["path"])
        actual, _ = verified_adapter(path)
        if actual["files"]["adapter_model.safetensors"] != parent["adapter_sha256"]:
            raise ValueError("Bred parent changed before child export")
        receipt = path / "v100-export.json"
        if not receipt.exists():
            raise ValueError("Export both parents before exporting a bred child")
        export = json.loads(receipt.read_text())
        if export["adapter_sha256"] != parent["adapter_sha256"]:
            raise ValueError("Parent export belongs to a different adapter")
        if file_hash(Path(export["model_path"])) != export["model_sha256"]:
            raise ValueError("Parent exported model changed")
        hashes.append(export["model_sha256"])
    return hashes


def breed_adapters(
    first: Path, second: Path, output: Path, alpha: float, base: Path, root: Path
) -> dict:
    import torch
    from safetensors.torch import load_file, save_file

    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and between 0 and 1")
    first, second, output = first.resolve(), second.resolve(), output.resolve()
    if first == second:
        raise ValueError("Select two distinct parent adapters")
    for parent in (first, second):
        assert_candidate_output(output, base, parent, root)
    if output.exists():
        raise FileExistsError("Child requires a new, separate output directory")
    left_meta, left_config = verified_adapter(first)
    right_meta, right_config = verified_adapter(second)
    signature = base_signature(base)
    if left_meta["base"] != right_meta["base"] or left_meta["base"] != signature:
        raise ValueError("Parents and configured base must have identical verified base files")
    # Compare all semantic configuration fields, allowing only rank/scaling and provenance
    # differences. This rejects mixtures with different target modules or layer patterns.
    ignored = {"r", "lora_alpha", "base_model_name_or_path", "revision", "inference_mode"}
    left_semantics = {key: value for key, value in left_config.items() if key not in ignored}
    right_semantics = {key: value for key, value in right_config.items() if key not in ignored}
    for semantics in (left_semantics, right_semantics):
        for key in ("target_modules", "exclude_modules"):
            if isinstance(semantics.get(key), list):
                semantics[key] = sorted(semantics[key])
    if left_semantics != right_semantics:
        raise ValueError("Parent LoRA configurations are incompatible")
    left = load_file(str(first / "adapter_model.safetensors"), device="cpu")
    right = load_file(str(second / "adapter_model.safetensors"), device="cpu")
    if not left or left.keys() != right.keys():
        raise ValueError("Parent tensor sets differ")
    a_names = sorted(name for name in left if name.endswith(".lora_A.weight"))
    expected = set(a_names) | {name.replace(".lora_A.weight", ".lora_B.weight") for name in a_names}
    if expected != left.keys():
        raise ValueError("Only paired linear LoRA A/B tensors can be bred")
    child = {}
    for a_name in a_names:
        b_name = a_name.replace(".lora_A.weight", ".lora_B.weight")
        for tensors, config in ((left, left_config), (right, right_config)):
            a, b = tensors[a_name], tensors[b_name]
            if (
                a.ndim != 2
                or b.ndim != 2
                or a.shape[0] != config["r"]
                or b.shape[1] != config["r"]
                or not a.is_floating_point()
                or not b.is_floating_point()
                or not torch.isfinite(a).all()
                or not torch.isfinite(b).all()
            ):
                raise ValueError("Invalid parent LoRA tensor")
        if (
            left[a_name].shape[1] != right[a_name].shape[1]
            or left[b_name].shape[0] != right[b_name].shape[0]
        ):
            raise ValueError("Parent LoRA matrix dimensions differ")
        child[a_name] = torch.cat((left[a_name].float(), right[a_name].float()), dim=0)
        child[b_name] = torch.cat(
            (
                left[b_name].float() * alpha * left_config["lora_alpha"] / left_config["r"],
                right[b_name].float()
                * (1 - alpha)
                * right_config["lora_alpha"]
                / right_config["r"],
            ),
            dim=1,
        )
    config = copy.deepcopy(left_config)
    config.update(
        r=left_config["r"] + right_config["r"],
        lora_alpha=left_config["r"] + right_config["r"],
        base_model_name_or_path=str(base.resolve()),
        inference_mode=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".breed-", dir=output.parent))
    try:
        candidate = staging / "candidate"
        candidate.mkdir()
        atomic_json(candidate / "adapter_config.json", config)
        save_file(child, str(candidate / "adapter_model.safetensors"))
        parents = [
            {
                "path": str(path),
                "adapter_sha256": meta["files"]["adapter_model.safetensors"],
                "weight": weight,
            }
            for path, meta, weight in ((first, left_meta, alpha), (second, right_meta, 1 - alpha))
        ]
        data = record_adapter(candidate, signature, parents)
        atomic_json(
            staging / "breeding.json", {"method": "exact weighted delta concatenation", **data}
        )
        # Recheck parents before committing the child, without modifying either parent.
        verified_adapter(first)
        verified_adapter(second)
        output.mkdir()
        for path in staging.iterdir():
            os.rename(path, output / path.name)
        return {
            "output": str(output),
            "rank": config["r"],
            "parents": parents,
            "status": data["status"],
        }
    finally:
        shutil.rmtree(staging)
