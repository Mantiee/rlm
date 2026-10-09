"""Bounded adapter retention experiments; finite evidence, never universal guarantees.

EWC uses empirical diagonal Fisher on verified historical TRAINING examples.
Delta-A orthogonality is an experimental regularizer, not a reproduction of O-LoRA.
"""

import copy
import hashlib
import json
import math
from pathlib import Path

from rlm.v100.common import atomic_json
from rlm.v100.protection import file_hash


def options(settings: dict) -> dict:
    mode = settings.get("retention_mode", 0)
    strength = settings.get("retention_strength", 0.01)
    growth = settings.get("retention_rank_growth", 1)
    if type(mode) is not int or mode not in range(7):
        raise ValueError("Unknown retention mode")
    if type(growth) is not int or growth not in (1, 2):
        raise ValueError("Retention rank growth must be 1 or 2")
    if type(strength) not in (int, float) or not math.isfinite(strength) or not 0 < strength <= 1:
        raise ValueError("Invalid retention regularization strength")
    return {"mode": mode, "strength": strength, "growth": growth}


def artifact_hashes(directory: Path) -> dict:
    paths = [
        directory / "retention.json",
        directory / "retention-anchor.safetensors",
        directory / "capacity-growth.json",
        directory / "expanded-initial/adapter_config.json",
        directory / "expanded-initial/adapter_model.safetensors",
        directory / "expanded-initial/v100-adapter.json",
    ]
    return {str(path.relative_to(directory)): file_hash(path) for path in paths if path.exists()}


def expand_adapter(source: Path, destination: Path, factor: int) -> dict:
    """Increase standard LoRA rank without changing its initial B@A delta."""
    import torch
    from safetensors.torch import load_file, save_file

    from rlm.v100.breeding import record_adapter, verified_adapter

    metadata, config = verified_adapter(source)
    if factor != 2 or config["r"] * factor > 64:
        raise ValueError("Expansion supports doubling standard LoRA rank up to 64")
    if destination.exists():
        raise FileExistsError("Expansion requires a separate new directory")
    weights = load_file(str(source / "adapter_model.safetensors"))
    if sum(v.numel() for v in weights.values()) * factor > 160_000_000:
        raise ValueError("Expanded adapter exceeds 160 million parameters")
    expanded, rank = {}, config["r"]
    for name, value in weights.items():
        if name.endswith("lora_A.weight") and value.ndim == 2 and value.shape[0] == rank:
            # New A rows start random, B columns zero: new capacity receives gradients
            # while its initial contribution is exactly zero. Do not change old rows.
            extra = torch.randn(rank, value.shape[1], dtype=torch.float32) / math.sqrt(
                value.shape[1]
            )
            expanded[name] = torch.cat((value, extra.to(value.dtype)), dim=0)
        elif name.endswith("lora_B.weight") and value.ndim == 2 and value.shape[1] == rank:
            expanded[name] = torch.cat((value, value.new_zeros(value.shape[0], rank)), dim=1)
        else:
            raise ValueError("Expansion requires only standard paired LoRA A/B tensors")
    paired = set()
    for name, value in weights.items():
        if name.endswith("lora_A.weight"):
            peer = name.replace("lora_A.weight", "lora_B.weight")
            if peer not in weights:
                raise ValueError("Unpaired adapter tensors")
            paired.update((name, peer))
            # Structural equality avoids allocating huge dense transformer deltas.
            if not (
                torch.equal(expanded[name][:rank], value)
                and torch.equal(expanded[peer][:, :rank], weights[peer])
                and torch.count_nonzero(expanded[peer][:, rank:]).item() == 0
            ):
                raise ValueError("Expansion changed the predecessor delta")
    if paired != set(weights):
        raise ValueError("Unpaired adapter tensors")
    # Keep alpha/r invariant. Config and tensors export through normal PEFT/GGUF.
    grown = copy.deepcopy(config)
    grown.update(r=rank * factor, lora_alpha=config["lora_alpha"] * factor)
    destination.mkdir(parents=True)
    atomic_json(destination / "adapter_config.json", grown)
    save_file(expanded, str(destination / "adapter_model.safetensors"))
    record_adapter(destination, metadata["base"], metadata.get("parents"))
    return {"old_rank": rank, "new_rank": rank * factor, "initial_delta_preserved": True}


class AdapterRetention:
    def __init__(self, model, configuration: dict):
        import torch

        self.configuration = configuration
        self.parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if not self.parameters or any("lora_" not in n for n in self.parameters):
            raise ValueError("Retention trainer protects LoRA parameters only")
        count = sum(p.numel() for p in self.parameters.values())
        if count > 160_000_000:
            raise ValueError("Retention anchors exceed 160 million adapter parameters")
        if count * 8 > 32 * 2**20:
            import psutil

            if psutil.virtual_memory().available < count * 8 + 4 * 2**30:
                raise ValueError("Retention anchors need four GiB remaining host RAM")
        self.anchor = {n: p.detach().float().cpu().clone() for n, p in self.parameters.items()}
        self.fisher = {n: torch.zeros_like(v) for n, v in self.anchor.items()}
        self.samples = 0

    def estimate(self, model, batches: list[dict]) -> None:
        import torch

        if not 1 <= len(batches) <= 8:
            raise ValueError("Fisher needs one to eight historical training examples")
        training = model.training
        model.eval()
        try:
            for batch in batches:
                loss = model(**batch).loss
                if not torch.isfinite(loss).all():
                    raise FloatingPointError("Nonfinite Fisher reference loss")
                gradients = torch.autograd.grad(
                    loss, tuple(self.parameters.values()), allow_unused=True
                )
                for (name, _), gradient in zip(self.parameters.items(), gradients, strict=True):
                    if gradient is not None:
                        if not torch.isfinite(gradient).all():
                            raise FloatingPointError("Nonfinite Fisher reference gradient")
                        self.fisher[name] += gradient.detach().float().cpu().square() / len(batches)
                self.samples += 1
        finally:
            model.train(training)

    def penalty(self):
        import torch

        mode = self.configuration["mode"]
        first = next(iter(self.parameters.values()))
        total = first.new_zeros((), dtype=torch.float32)
        if mode in (0, 5):
            return total
        if mode in (2, 4, 6) and not self.samples:
            raise ValueError("EWC requires measured historical Fisher information")
        for name, parameter in self.parameters.items():
            anchor = self.anchor[name].to(parameter.device)
            delta = parameter.float() - anchor
            importance = self.fisher[name].to(parameter.device) if mode in (2, 4, 6) else 1
            total = total + (importance * delta.square()).sum() / 2
            if mode in (3, 4, 6) and "lora_A." in name:
                # Penalize only CHANGES along previous row directions; initialization
                # has zero penalty even when resuming a nonzero accepted adapter.
                norm = anchor.norm(dim=1, keepdim=True).clamp_min(1e-8)
                overlap = delta @ (anchor / norm).T
                total = total + overlap.square().mean()
        return total * self.configuration["strength"]

    def project(self, model, batch: dict) -> dict:
        """A-GEM half-space projection of the unscaled optimizer gradient.

        Only first-order loss protection on this replay batch, not a guarantee
        after finite steps, optimizer momentum or on unseen tasks.
        """
        import torch

        training = model.training
        model.eval()
        try:
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite replay projection loss")
            reference = torch.autograd.grad(
                loss, tuple(self.parameters.values()), allow_unused=True
            )
        finally:
            model.train(training)
        pairs = [
            (p.grad, ref)
            for p, ref in zip(self.parameters.values(), reference, strict=True)
            if p.grad is not None and ref is not None
        ]
        if not pairs or any(
            not torch.isfinite(g).all() or not torch.isfinite(r).all() for g, r in pairs
        ):
            raise FloatingPointError("Replay projection needs finite optimizer/reference gradients")
        dot = sum((g.float() * r.float()).sum() for g, r in pairs)
        norm = sum(r.float().square().sum() for _, r in pairs)
        projected = dot.item() < 0 and norm.item() > 1e-12
        if projected:
            with torch.no_grad():
                for gradient, ref in pairs:
                    gradient.sub_((dot / norm * ref.float()).to(gradient.dtype))
        return {
            "projected": projected,
            "gradient_dot_reference": dot.item(),
            "reference_norm_sq": norm.item(),
        }

    def save(self, directory: Path, source_groups: list[str]) -> dict:
        from safetensors.torch import save_file

        path = directory / "retention-anchor.safetensors"
        save_file(
            {
                **{"anchor." + n: v for n, v in self.anchor.items()},
                **{"fisher." + n: v for n, v in self.fisher.items()},
            },
            str(path),
        )
        report = {
            "schema": "v100-retention-v1",
            "configuration": self.configuration,
            "anchor_sha256": file_hash(path),
            "fisher_samples": self.samples,
            "source_groups": source_groups,
            "scope": "Historical training only; no audit/test labels",
        }
        atomic_json(directory / "retention.json", report)
        return report

    def restore(self, directory: Path, source_groups: list[str]) -> None:
        import torch
        from safetensors.torch import load_file

        report = json.loads((directory / "retention.json").read_text())
        path = directory / "retention-anchor.safetensors"
        if (
            report["configuration"] != self.configuration
            or report["source_groups"] != source_groups
            or file_hash(path) != report["anchor_sha256"]
        ):
            raise ValueError("Retention reference changed; cannot resume")
        tensors = load_file(str(path))
        expected = {prefix + n for n in self.parameters for prefix in ("anchor.", "fisher.")}
        if set(tensors) != expected:
            raise ValueError("Retention reference parameter set differs")
        for name, parameter in self.parameters.items():
            for prefix, collection in (("anchor.", self.anchor), ("fisher.", self.fisher)):
                value = tensors[prefix + name]
                if value.shape != parameter.shape or not torch.isfinite(value).all():
                    raise ValueError("Invalid retention reference tensor")
                if prefix == "fisher." and (value < 0).any():
                    raise ValueError("Negative Fisher information")
                collection[name] = value
        self.samples = report["fisher_samples"]


def select_reference(records: list[dict], maximum: int = 8) -> list[dict]:
    """Stable round-robin across verified historical domains, never heldout records."""
    if not 1 <= maximum <= 8:
        raise ValueError("Reference sample budget exceeds eight")
    groups = {}
    for row in records:
        verification = row.get("verification", {})
        domain = verification.get(
            "domain", verification.get("task", {}).get("kind", verification.get("kind", "unknown"))
        )
        groups.setdefault(domain, []).append(row)
    for rows in groups.values():
        rows.sort(
            key=lambda row: hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        )
    selected = []
    while groups and len(selected) < maximum:
        for domain in sorted(list(groups)):
            selected.append(groups[domain].pop(0))
            if not groups[domain]:
                del groups[domain]
            if len(selected) == maximum:
                break
    return selected
