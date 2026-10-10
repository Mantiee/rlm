"""Bounded Engram-inspired causal memory pilot, not a pretrained GGUF retrofit."""

from pathlib import Path

from rlm.v100.architectures import create_candidate
from rlm.v100.goals import load_goal

MODEL_SOURCE = """
import torch
from torch import nn

class NgramModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        s = MORPH_SHAPE
        width = s['width']
        self.embedding = nn.Embedding(257, width)
        self.body = nn.GRU(width, width, s['layers'], batch_first=True)
        self.head = nn.Linear(width, 257)
        self.tables = nn.ModuleList([nn.Embedding(s['memory_slots'], width) for _ in range(s['memory_order'] - 1)])
        self.key = nn.Linear(width, width, bias=False)
        self.query = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.value.weight)
        self.norm = nn.LayerNorm(width)
        self.memory_enabled = not config.get('ablate_ngram_memory', False)

    def memory_indices(self, tokens, order):
        # Each address uses only the current and preceding byte tokens.
        shifted = torch.nn.functional.pad(tokens + 1, (order - 1, 0))
        hashed = torch.zeros_like(tokens)
        for offset in range(order):
            hashed = (hashed * 263 + shifted[:, offset:offset + tokens.shape[1]]) % MORPH_SHAPE['memory_slots']
        return hashed

    def forward(self, tokens):
        value = self.embedding(tokens)
        if self.memory_enabled:
            memory = sum(table(self.memory_indices(tokens, order)) for order, table in enumerate(self.tables, 2)) / len(self.tables)
            gate = torch.sigmoid((self.query(self.norm(value)) * self.key(self.norm(memory))).sum(-1, keepdim=True) / value.shape[-1] ** .5)
            value = value + gate * self.value(memory)
        value, _ = self.body(value)
        return self.head(value)

def build(config):
    return NgramModel(config)
"""


def propose(
    root: Path,
    branch: str,
    candidate_id: str,
    width: int,
    layers: int,
    memory_slots: int,
    memory_order: int,
    hypothesis: str,
) -> dict:
    goal = load_goal(root)
    if not goal:
        raise ValueError("N-gram memory needs the current recorded goal")
    if any(type(v) is not int for v in (width, layers, memory_slots, memory_order)) or not (
        16 <= width <= 64
        and 1 <= layers <= 2
        and memory_order in (2, 3, 4)
        and 256 <= memory_slots <= 8192
        and memory_slots & (memory_slots - 1) == 0
    ):
        raise ValueError("N-gram memory exceeds bounded pilot dimensions")
    parameters = (
        (memory_order - 1) * memory_slots * width
        + 514 * width
        + 12 * layers * width * width
        + 3 * width * width
        + 4096
    )
    if parameters > 2_000_000:
        raise ValueError("N-gram memory exceeds two million pilot parameters")
    shape = {
        "schema": "synta-morphology-v1",
        "goal_id": goal["id"],
        "architecture": "gru",
        "width": width,
        "layers": layers,
        "heads": 1,
        "residual_layers": 0,
        "bottleneck": 1,
        "parent_candidate_id": "",
        "parent_shape": None,
        "parent_weights_sha256": None,
        "memory_slots": memory_slots,
        "memory_order": memory_order,
        "memory_kind": "causal-hashed-ngram-v1",
    }
    result = create_candidate(
        root, branch, candidate_id, "MORPH_SHAPE = " + repr(shape) + "\n" + MODEL_SOURCE, hypothesis
    )
    result.update(
        shape=shape,
        estimated_parameters=parameters,
        memory_table_mib=(memory_order - 1) * memory_slots * width * 4 / 2**20,
        weights_changed=False,
        windows_rtx_started=False,
        next_tools=["test_submodel", "propose_scratch_master"],
        scope="Independent byte-token pilot, not the full DeepSeek Engram implementation. CPU default. NVMe stores checkpoints; no disk memory expansion claimed. No Gemma/GGUF modification. Serving promotion requires held-out goal improvement and retention/fresh/public gates. Compare memory_enabled=False on the same frozen workload.",
    )
    return result
