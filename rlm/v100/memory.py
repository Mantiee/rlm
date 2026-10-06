"""Persistent source-preserving memory, with SQLite FTS instead of AVX2 vector binaries."""

import hashlib
import json
import re
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def split_text(
    text: str, count_tokens: Callable[[str], int], budget: int
) -> list[tuple[int, int, str]]:
    """Preserve every source character; never exceed the native tokenizer budget."""
    if budget < 16:
        raise ValueError("chunk budget must be at least 16 tokens")
    result, start = [], 0
    while start < len(text):
        end = min(len(text), start + budget * 4)
        while count_tokens(text[start:end]) > budget:
            end = start + (end - start) // 2
            if end == start:
                raise ValueError("Cannot fit a source character into token budget")
        boundary = text.rfind("\n", start + (end - start) // 2, end)
        if boundary > start and end < len(text):
            end = boundary + 1
        result.append((start, end, text[start:end]))
        start = end
    return result


class Memory:
    def __init__(self, database: Path, model_version: str):
        database.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(database)
        self.db.row_factory = sqlite3.Row
        self.model_version = model_version
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS documents(
                id TEXT PRIMARY KEY, source TEXT NOT NULL, text TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS nodes(
                id TEXT PRIMARY KEY, document_id TEXT NOT NULL, level INTEGER NOT NULL,
                ordinal INTEGER NOT NULL, start INTEGER, end INTEGER, text TEXT NOT NULL,
                children TEXT NOT NULL, model_version TEXT NOT NULL);
            CREATE VIRTUAL TABLE IF NOT EXISTS node_search USING fts5(id UNINDEXED, text);
            CREATE TABLE IF NOT EXISTS experiences(
                id TEXT PRIMARY KEY, record TEXT NOT NULL);
        """)

    def close(self) -> None:
        self.db.close()

    def node(self, node_id: str) -> dict[str, Any]:
        row = self.db.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
        if row is None:
            raise KeyError(node_id)
        value = dict(row)
        value["children"] = json.loads(value["children"])
        return value

    def insert_node(self, node: dict[str, Any]) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO nodes VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    node["id"],
                    node["document_id"],
                    node["level"],
                    node["ordinal"],
                    node.get("start"),
                    node.get("end"),
                    node["text"],
                    json.dumps(node["children"]),
                    node["model_version"],
                ),
            )
            self.db.execute("INSERT INTO node_search VALUES(?,?)", (node["id"], node["text"]))

    def ingest(
        self, source: str, text: str, count_tokens: Callable[[str], int], budget: int
    ) -> str:
        document_id = digest(source + "\0" + text)
        if self.db.execute("SELECT 1 FROM documents WHERE id=?", (document_id,)).fetchone():
            return document_id
        chunks = split_text(text, count_tokens, budget)
        # One transaction: an interrupted import cannot leave a half-imported document.
        with self.db:
            self.db.execute("INSERT INTO documents VALUES(?,?,?)", (document_id, source, text))
            for ordinal, (start, end, chunk) in enumerate(chunks):
                node_id = digest(f"{document_id}:{start}:{end}")
                self.db.execute(
                    "INSERT INTO nodes VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        node_id,
                        document_id,
                        0,
                        ordinal,
                        start,
                        end,
                        chunk,
                        "[]",
                        "source",
                    ),
                )
                self.db.execute("INSERT INTO node_search VALUES(?,?)", (node_id, chunk))
        return document_id

    def build_tree(self, document_id: str, summarize: Callable[[str], str], fanout: int = 4) -> str:
        if fanout < 2:
            raise ValueError("fanout must be at least 2")
        children = [
            r["id"]
            for r in self.db.execute(
                "SELECT id FROM nodes WHERE document_id=? AND level=0 ORDER BY ordinal",
                (document_id,),
            )
        ]
        if not children:
            raise ValueError("Document has no chunks")
        level = 1
        while len(children) > 1:
            parents = []
            for offset in range(0, len(children), fanout):
                group = children[offset : offset + fanout]
                node_id = digest(json.dumps(["summary-v1", self.model_version, group]))
                if self.db.execute("SELECT 1 FROM nodes WHERE id=?", (node_id,)).fetchone() is None:
                    material = "\n\n".join(f"[{i}]\n{self.node(i)['text']}" for i in group)
                    summary = summarize(material)
                    self.insert_node(
                        {
                            "id": node_id,
                            "document_id": document_id,
                            "level": level,
                            "ordinal": offset // fanout,
                            "text": summary,
                            "children": group,
                            "model_version": self.model_version,
                        }
                    )
                parents.append(node_id)
            children, level = parents, level + 1
        return children[0]

    def search(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        words = re.findall(r"\w+", query.lower())[:32]
        if not words or limit < 1:
            raise ValueError("Search needs words and a positive limit")
        expression = " OR ".join('"' + word + '"' for word in words)
        rows = self.db.execute(
            """
            SELECT n.id FROM node_search s JOIN nodes n ON n.id=s.id
            WHERE node_search MATCH ? AND (n.level=0 OR n.model_version=?)
            ORDER BY bm25(node_search), n.id LIMIT ?
        """,
            (expression, self.model_version, limit),
        )
        return [self.node(r["id"]) for r in rows]

    def leaves(self, node_id: str) -> list[dict[str, Any]]:
        node = self.node(node_id)
        if node["level"] == 0:
            return [node]
        return [leaf for child in node["children"] for leaf in self.leaves(child)]

    def retrieve(self, question: str, count: int) -> list[dict[str, Any]]:
        found: dict[str, dict[str, Any]] = {}
        words = set(re.findall(r"\w+", question.lower()))
        for hit in self.search(question, count * 4):
            for leaf in self.leaves(hit["id"]):
                found[leaf["id"]] = leaf
        ranked = sorted(
            found.values(),
            key=lambda n: (-len(words & set(re.findall(r"\w+", n["text"].lower()))), n["id"]),
        )
        return ranked[:count]

    def add_feedback(self, run: dict[str, Any], correct_answer: str) -> str:
        if not correct_answer.strip():
            raise ValueError("Feedback must supply a nonempty corrected or confirmed answer")
        record = {
            "messages": run["messages"] + [{"role": "assistant", "content": correct_answer}],
            "group": run["group"],
            "source_ids": run["source_ids"],
            "document_ids": run.get("document_ids", []),
            "verification": {"kind": "human_feedback", "accepted": True},
            "model_version": run["model_version"],
        }
        experience_id = digest(json.dumps(record, sort_keys=True, ensure_ascii=False))
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO experiences VALUES(?,?)",
                (experience_id, json.dumps(record, ensure_ascii=False)),
            )
        return experience_id

    def export_feedback(self, output: Path) -> int:
        rows = self.db.execute("SELECT record FROM experiences ORDER BY id").fetchall()
        if not rows:
            raise ValueError("No verified feedback yet; benchmark answers are not training data")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as handle:
            for row in rows:
                handle.write(row[0] + "\n")
        return len(rows)
