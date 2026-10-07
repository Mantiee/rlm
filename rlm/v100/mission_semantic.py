"""CPU hybrid mission memory with bounded incremental indexing."""

import json
import threading
from pathlib import Path

from rlm.v100.common import atomic_json
from rlm.v100.protection import file_hash
from rlm.v100.semantic import Encoder, hybrid_retrieve, index_memory, prepare_encoder

LOCK = threading.Lock()
ENCODERS = {}


def prepare(root: Path) -> dict:
    from rlm.v100.mission import status

    if status(root)["running"]:
        raise ValueError("Prepare the encoder while the mission is stopped")
    directory = root / "models/memory-encoder"
    prepare_encoder(directory)
    value = {"encoder": str(directory), "device": "cpu", "new_nodes_per_query": 4}
    atomic_json(root / "research/mission-semantic.json", value)
    return value


def retrieve(root: Path, memory, question: str, count: int) -> list[dict]:
    marker = root / "research/mission-semantic.json"
    if not marker.exists():
        return memory.search(question, count)
    config = json.loads(marker.read_text())
    directory = Path(config["encoder"])
    with LOCK:
        key = (str(directory.resolve()), file_hash(directory / "encoder.json"))
        if key not in ENCODERS:
            import torch

            torch.set_num_threads(2)
            ENCODERS[key] = Encoder(directory, device="cpu")
        encoder = ENCODERS[key]
        index_memory(memory, encoder, max_nodes=4)
        try:
            return hybrid_retrieve(memory, encoder, question, count)
        except ValueError as error:
            if (
                str(error)
                != "Semantic index is incomplete; run index-memory after importing sources"
            ):
                raise
            from rlm.v100.activity import ActivityLog

            ActivityLog(root, "controller", "memory").write(
                "tools",
                "semantic-index-pending",
                {"scope": "Explicit temporary lexical retrieval while bounded index catches up"},
            )
            return memory.search(question, count)
