"""Versioned exact cosine search; no FAISS/AVX2 dependency, no silent truncation."""

import array
import json
import math
from pathlib import Path

from rlm.v100.common import atomic_json
from rlm.v100.memory import split_text
from rlm.v100.protection import file_hash

ENCODER_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def prepare_encoder(directory: Path) -> dict:
    from huggingface_hub import HfApi, snapshot_download

    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "encoder.json"
    if manifest.exists():
        return verify_encoder(directory)
    revision_path = directory / "revision.json"
    if revision_path.exists():
        revision = json.loads(revision_path.read_text())["revision"]
    else:
        revision = HfApi().model_info(ENCODER_ID).sha
        atomic_json(revision_path, {"model": ENCODER_ID, "revision": revision})
    snapshot_download(
        ENCODER_ID,
        revision=revision,
        local_dir=directory,
        allow_patterns=["*.json", "*.safetensors", "*.txt"],
        max_workers=2,
    )
    names = [
        path
        for path in directory.rglob("*")
        if path.is_file()
        and ".cache" not in path.parts
        and path.name not in ("revision.json", "encoder.json")
    ]
    data = {
        "model": ENCODER_ID,
        "revision": revision,
        "pooling": "mean; windowed tokens; max cosine per source; normalized-f32-v1",
        "files": {str(path.relative_to(directory)): file_hash(path) for path in names},
    }
    atomic_json(manifest, data)
    return data


def verify_encoder(directory: Path) -> dict:
    data = json.loads((directory / "encoder.json").read_text())
    if data["model"] != ENCODER_ID or not data["files"]:
        raise ValueError("Invalid semantic encoder manifest")
    for name, expected in data["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or file_hash(path) != expected:
            raise ValueError("Semantic encoder changed; build a new index version")
    return data


class Encoder:
    def __init__(self, directory: Path, device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        manifest = verify_encoder(directory)
        self.identity = file_hash(directory / "encoder.json")
        self.model_id = manifest["model"]
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        if not self.tokenizer.is_fast:
            raise ValueError("Windowed embeddings require a fast tokenizer with overflow mapping")
        self.model = AutoModel.from_pretrained(directory, local_files_only=True).to(device).eval()
        self.torch = torch
        self.window = min(
            int(self.tokenizer.model_max_length), int(self.model.config.max_position_embeddings)
        )
        sentence_config = directory / "sentence_bert_config.json"
        if sentence_config.exists():
            self.window = min(
                self.window, int(json.loads(sentence_config.read_text())["max_seq_length"])
            )
        self.budget = self.window - self.tokenizer.num_special_tokens_to_add(pair=False)
        if self.budget < 16:
            raise ValueError("Invalid semantic encoder window")

    def encode(self, text: str) -> list[list[float]]:
        if not text.strip():
            raise ValueError("Cannot embed empty text")

        def count_tokens(piece: str) -> int:
            return len(
                self.tokenizer.encode(
                    piece, add_special_tokens=True, truncation=False, verbose=False
                )
            )

        # Preserve source characters and count each window explicitly. Do not rely
        # on overflow bindings: tokenizers 0.23.2 may discard overflow encodings.
        windows = split_text(text, count_tokens, self.window)
        vectors = []
        for _, _, piece in windows:
            values = self.tokenizer(piece, truncation=False, return_tensors="pt")
            inputs = {
                key: value.to(self.device)
                for key, value in values.items()
                if key in self.tokenizer.model_input_names
            }
            if inputs["input_ids"].shape[1] > self.window:
                raise ValueError("Semantic window exceeds its token budget")
            with self.torch.inference_mode():
                hidden = self.model(**inputs).last_hidden_state.float()
                mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                pooled = self.torch.nn.functional.normalize(pooled, dim=-1)
            vectors.append(pooled[0].cpu().tolist())
        return vectors


def checked_vector(vector: list[float]) -> list[float]:
    if not vector or not all(
        type(value) in (float, int) and math.isfinite(value) for value in vector
    ):
        raise ValueError("Embedding contains invalid values")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise ValueError("Embedding has zero norm")
    return [value / norm for value in vector]


def ensure_schema(memory) -> None:
    memory.db.execute("""CREATE TABLE IF NOT EXISTS embeddings(
        encoder TEXT NOT NULL, node_id TEXT NOT NULL, part INTEGER NOT NULL,
        dimensions INTEGER NOT NULL, vector BLOB NOT NULL,
        PRIMARY KEY(encoder, node_id, part))""")


def index_memory(memory, encoder, max_nodes: int = 4096) -> int:
    ensure_schema(memory)
    if type(max_nodes) is not int or not 1 <= max_nodes <= 4096:
        raise ValueError("Index batch must have 1-4096 nodes")
    count = 0
    rows = memory.db.execute(
        """SELECT n.id,n.text FROM nodes n JOIN documents d ON d.id=n.document_id
        WHERE n.level=0 AND (?=0 OR (d.source NOT LIKE 'worker:%' AND d.source NOT LIKE 'source-error:%'))
        AND NOT EXISTS(SELECT 1 FROM embeddings e WHERE e.node_id=n.id AND e.encoder=?)
        ORDER BY n.id LIMIT ?""",
        (int(getattr(memory, "exclude_self_transcripts", False)), encoder.identity, max_nodes),
    ).fetchall()
    for row in rows:
        if memory.db.execute(
            "SELECT 1 FROM embeddings WHERE encoder=? AND node_id=?", (encoder.identity, row["id"])
        ).fetchone():
            continue
        vectors = [checked_vector(vector) for vector in encoder.encode(row["text"])]
        if not vectors or len({len(vector) for vector in vectors}) != 1:
            raise ValueError("Invalid embedding dimensions")
        with memory.db:
            for part, vector in enumerate(vectors):
                blob = array.array("f", vector).tobytes()
                memory.db.execute(
                    "INSERT INTO embeddings VALUES(?,?,?,?,?)",
                    (encoder.identity, row["id"], part, len(vector), blob),
                )
        count += 1
    return count


def hybrid_retrieve(memory, encoder, question: str, count: int) -> list[dict]:
    import torch

    if count < 1:
        raise ValueError("Retrieval count must be positive")
    ensure_schema(memory)
    missing = memory.db.execute(
        """SELECT COUNT(*) FROM nodes n JOIN documents d ON d.id=n.document_id WHERE level=0
        AND (?=0 OR (d.source NOT LIKE 'worker:%' AND d.source NOT LIKE 'source-error:%')) AND NOT EXISTS(
        SELECT 1 FROM embeddings e WHERE e.node_id=n.id AND e.encoder=?)""",
        (int(getattr(memory, "exclude_self_transcripts", False)), encoder.identity),
    ).fetchone()[0]
    if missing:
        raise ValueError("Semantic index is incomplete; run index-memory after importing sources")
    queries = [checked_vector(vector) for vector in encoder.encode(question)]
    scores = {}
    cursor = memory.db.execute(
        """SELECT e.node_id,e.dimensions,e.vector FROM embeddings e JOIN nodes n ON n.id=e.node_id
        JOIN documents d ON d.id=n.document_id WHERE e.encoder=?
        AND (?=0 OR (d.source NOT LIKE 'worker:%' AND d.source NOT LIKE 'source-error:%'))""",
        (encoder.identity, int(getattr(memory, "exclude_self_transcripts", False))),
    )
    # Bounded matrix batches avoid loading the entire vector index into RAM/GPU.
    while rows := cursor.fetchmany(1024):
        vectors = []
        for row in rows:
            vector = list(array.array("f", row["vector"]))
            if row["dimensions"] != len(queries[0]) or len(vector) != row["dimensions"]:
                raise ValueError("Semantic index dimensions differ from query encoder")
            vectors.append(checked_vector(vector))
        similarities = (torch.tensor(vectors) @ torch.tensor(queries).T).max(dim=1).values.tolist()
        for row, score in zip(rows, similarities, strict=True):
            scores[row["node_id"]] = max(score, scores.get(row["node_id"], -2.0))
    semantic = sorted(scores, key=lambda key: (-scores[key], key))[: count * 4]
    lexical = [node["id"] for node in memory.retrieve(question, count * 4)]
    fused = {}
    for order in (semantic, lexical):
        for rank, node_id in enumerate(order):
            fused[node_id] = fused.get(node_id, 0.0) + 1 / (60 + rank + 1)
    return [
        memory.node(node_id)
        for node_id in sorted(fused, key=lambda key: (-fused[key], key))[:count]
    ]
