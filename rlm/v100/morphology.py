"""Typed structural proposals for isolated goal-gated master candidates.

No source is imported on the controller. Native GGUF weights are never relabeled
as another architecture. Warm growth is limited to compatible typed predecessors.
"""

import ast
import json
from pathlib import Path

from rlm.v100.activity import ActivityLog
from rlm.v100.architectures import candidate_path, create_candidate, verify_candidate
from rlm.v100.goals import load_goal
from rlm.v100.protection import file_hash

MODEL_SOURCE = """
import torch
from torch import nn

class MorphModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        s = MORPH_SHAPE
        w = s['width']
        self.embedding = nn.Embedding(257, w)
        if s['architecture'] == 'gru':
            self.body = nn.GRU(w, w, s['layers'], batch_first=True)
        else:
            self.body = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(w, s['heads'], w * 2, dropout=0, batch_first=True),
                s['layers'], enable_nested_tensor=False)
        self.head = nn.Linear(w, 257)
        self.residual = nn.ModuleList([
            nn.Sequential(nn.Linear(257, s['bottleneck']), nn.GELU(), nn.Linear(s['bottleneck'], 257))
            for _ in range(s['residual_layers'])])
        for block in self.residual:
            nn.init.zeros_(block[-1].weight)
            nn.init.zeros_(block[-1].bias)

    def load_parent_state(self, state):
        old = MORPH_SHAPE['parent_shape']
        if old is None:
            raise ValueError('No compatible morphology predecessor')
        expected = self.state_dict()
        old_blocks = old['residual_layers']
        allowed_missing = {k for k in expected if k.startswith('residual.') and int(k.split('.')[1]) >= old_blocks}
        if set(state) != set(expected) - allowed_missing:
            raise ValueError('Parent keys differ from the frozen growth contract')
        if any(expected[k].shape != value.shape for k, value in state.items()):
            raise ValueError('Parent shape differs from the frozen growth contract')
        self.load_state_dict(state, strict=False)
        self.requires_grad_(False)
        for block in list(self.residual)[old_blocks:]:
            block.requires_grad_(True)
        self.embedding.eval()
        self.body.eval()
        self.head.eval()

    def forward(self, tokens):
        value = self.embedding(tokens)
        if MORPH_SHAPE['architecture'] == 'gru':
            value, _ = self.body(value)
        else:
            # Parameter-free positions preserve the vocabulary and avoid a huge
            # learned positional table when the parent has a long context.
            positions = torch.arange(tokens.shape[1], device=tokens.device).float()
            scale = torch.arange(value.shape[-1], device=tokens.device).float() + 1
            value = value + torch.sin(positions[:, None] / scale[None, :]).to(value.dtype)
            mask = torch.triu(torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device, dtype=torch.bool), 1)
            value = self.body(value, mask=mask)
        value = self.head(value)
        for block in self.residual:
            value = value + block(value)
        return value

def build(config):
    return MorphModel(config)
"""


def read_shape(source: Path) -> dict | None:
    if source.is_symlink() or source.stat().st_size > 12000:
        raise ValueError("Morphology source exceeds its frozen budget")
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "MORPH_SHAPE" for t in node.targets
        ):
            value = ast.literal_eval(node.value)
            if not isinstance(value, dict) or value.get("schema") != "synta-morphology-v1":
                raise ValueError("Invalid morphology shape contract")
            return value
    return None


def propose(
    root: Path,
    branch: str,
    candidate_id: str,
    architecture: str,
    width: int,
    layers: int,
    heads: int,
    residual_layers: int,
    bottleneck: int,
    parent_candidate_id: str,
    hypothesis: str,
) -> dict:
    goal = load_goal(root)
    if not goal:
        raise ValueError("Morphology needs the recorded operator goal")
    if architecture not in ("gru", "transformer") or any(
        type(v) is not int for v in (width, layers, heads, residual_layers, bottleneck)
    ):
        raise ValueError("Invalid morphology dimensions")
    if not (
        16 <= width <= 256
        and 1 <= layers <= 4
        and heads in (1, 2, 4, 8)
        and width % heads == 0
        and 0 <= residual_layers <= 4
        and 1 <= bottleneck <= 256
    ):
        raise ValueError("Morphology dimensions exceed the bounded search space")
    # Conservative bound, before allocating a tensor on any worker.
    estimate = (
        257 * width * 2
        + layers * width * width * 12
        + residual_layers * (514 * bottleneck + 257 + bottleneck)
    )
    if estimate > 2_000_000:
        raise ValueError("Morphology exceeds two million pilot parameters")
    parent_shape, parent_weights_hash = None, None
    if parent_candidate_id:
        parent = candidate_path(root, parent_candidate_id)
        manifest = verify_candidate(parent)
        parent_shape = read_shape(parent / "source/model.py")
        if manifest["goal"] != goal or parent_shape is None:
            raise ValueError("Warm growth requires a typed predecessor for the same goal")
        if (
            any(
                parent_shape[key] != value
                for key, value in {
                    "architecture": architecture,
                    "width": width,
                    "layers": layers,
                    "heads": heads,
                    "bottleneck": bottleneck,
                }.items()
            )
            or residual_layers <= parent_shape["residual_layers"]
        ):
            raise ValueError(
                "Warm growth can append residual layers; other shape changes need an independent candidate"
            )
        weights = parent / "trial/weights/weights.safetensors"
        if weights.is_symlink() or not weights.is_file() or weights.stat().st_size > 16 * 2**20:
            raise ValueError("Warm growth requires bounded trained predecessor weights")
        parent_weights_hash = file_hash(weights)
        parent_shape = {key: value for key, value in parent_shape.items() if key != "parent_shape"}
    shape = {
        "schema": "synta-morphology-v1",
        "goal_id": goal["id"],
        "architecture": architecture,
        "width": width,
        "layers": layers,
        "heads": heads,
        "residual_layers": residual_layers,
        "bottleneck": bottleneck,
        "parent_candidate_id": parent_candidate_id,
        "parent_weights_sha256": parent_weights_hash,
        "parent_shape": parent_shape,
    }
    result = create_candidate(
        root, branch, candidate_id, "MORPH_SHAPE = " + repr(shape) + "\n" + MODEL_SOURCE, hypothesis
    )
    result.update(
        shape=shape,
        estimated_parameters=estimate,
        weights_changed=False,
        next_tools=["test_submodel", "propose_scratch_master"],
        scope="Isolated structural candidate; master activation requires strict measured goal improvement and all retained/fresh/public gates",
    )
    ActivityLog(root, branch, "morphology").write("decisions", "morphology-proposed", result)
    return result


def initialization(root: Path, candidate_id: str) -> tuple[Path | None, bool]:
    folder = candidate_path(root, candidate_id)
    verify_candidate(folder)
    shape = read_shape(folder / "source/model.py")
    if shape is None or not shape["parent_candidate_id"]:
        return None, False
    parent = candidate_path(root, shape["parent_candidate_id"])
    verify_candidate(parent)
    weights = parent / "trial/weights/weights.safetensors"
    if weights.is_symlink() or file_hash(weights) != shape["parent_weights_sha256"]:
        raise ValueError("Morphology predecessor weights changed")
    return weights, True


def status(root: Path) -> dict:
    values = []
    for folder in sorted((root / "research/architecture-candidates").glob("*"))[-64:]:
        verify_candidate(folder)
        shape = read_shape(folder / "source/model.py")
        if shape is not None:
            report = folder / "trial/quality.json"
            values.append(
                {
                    "candidate_id": folder.name,
                    "shape": shape,
                    "trial": json.loads(report.read_text()) if report.exists() else None,
                }
            )
    return {
        "candidates": values,
        "scope": "Shape proposals and tested artifacts; not deployment claims",
    }
