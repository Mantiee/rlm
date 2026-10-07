"""Model-selected free-service proposals, separated from account access.

This registry does not make external LLM calls, create accounts or authorize
billing. A recommendation is not a working connector or proof of free access.
"""

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from urllib.parse import urlparse

SEEDS = [
    {
        "name": "OpenRouter free models",
        "url": "https://openrouter.ai/collections/free-models",
        "documentation": "https://openrouter.ai/docs/guides/routing/provider-selection",
    },
    {
        "name": "Google AI Studio",
        "url": "https://aistudio.google.com/",
        "documentation": "https://ai.google.dev/gemini-api/docs/billing",
    },
    {
        "name": "Hugging Face Chat",
        "url": "https://huggingface.co/chat/",
        "documentation": "https://huggingface.co/docs/inference-providers/pricing",
    },
]
EVIDENCE_TTL = 24 * 3600


def https_url(value: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 2048:
        raise ValueError("Service URL exceeds its budget")
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
        or parsed.fragment
    ):
        raise ValueError("Service needs a public HTTPS URL without credentials")
    return value


def service_id(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:24]


class ServiceBook:
    def __init__(self, root: Path):
        self.root = root
        path = root / "research/state/free-services.sqlite3"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS sources(url TEXT PRIMARY KEY, sha TEXT NOT NULL, checked REAL NOT NULL)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS proposals(id TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS selections(sequence INTEGER PRIMARY KEY, branch TEXT NOT NULL, value TEXT NOT NULL)"
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def record_source(self, source: dict) -> None:
        url = https_url(source["url"])
        sha = source["sha256"]
        if (
            not isinstance(sha, str)
            or len(sha) != 64
            or any(char not in "0123456789abcdef" for char in sha)
        ):
            raise ValueError("Invalid service evidence digest")
        path = self.root / "research/web-sources" / (sha + ".txt")
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError("Service evidence differs from the downloaded page")
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO sources VALUES(?,?,?)", (url, sha, time.time()))

    def evidence(self, url: str) -> dict:
        row = self.db.execute(
            "SELECT sha,checked FROM sources WHERE url=?", (https_url(url),)
        ).fetchone()
        if row is None or not 0 <= time.time() - row[1] <= EVIDENCE_TTL:
            raise ValueError(
                "Read the service's current public documentation first; evidence expires after 24h"
            )
        path = self.root / "research/web-sources" / (row[0] + ".txt")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row[0]:
            raise ValueError("Cached service evidence changed")
        return {"url": url, "sha256": row[0], "checked_at": row[1]}

    def propose(self, name: str, url: str, documentation: str, rationale: str) -> dict:
        if (
            not isinstance(name, str)
            or not 1 <= len(name) <= 80
            or not isinstance(rationale, str)
            or not 1 <= len(rationale) <= 800
        ):
            raise ValueError("Service name or rationale exceeds its budget")
        url = https_url(url)
        evidence = self.evidence(documentation)
        value = {
            "id": service_id(url),
            "name": name,
            "url": url,
            "evidence": evidence,
            "rationale": rationale,
            "status": "unverified free-access proposal; browser account connector not installed",
            "automatic_ready": False,
        }
        existing = self.db.execute("SELECT 1 FROM proposals WHERE id=?", (value["id"],)).fetchone()
        if (
            existing is None
            and self.db.execute("SELECT COUNT(*) FROM proposals").fetchone()[0] >= 32
        ):
            raise ValueError("Service research catalog is limited to 32 proposals")
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO proposals VALUES(?,?)", (value["id"], json.dumps(value))
            )
        return value

    def catalog(self) -> dict:
        proposals = [
            json.loads(row[0])
            for row in self.db.execute("SELECT value FROM proposals ORDER BY rowid DESC LIMIT 8")
        ]
        rows = [
            {
                "id": row["id"],
                "name": row["name"],
                "url": row["url"],
                "documentation": row["evidence"]["url"],
                "evidence_stale": not 0
                <= time.time() - row["evidence"]["checked_at"]
                <= EVIDENCE_TTL,
                "automatic_ready": False,
            }
            for row in proposals
        ]
        return {
            "proposals": rows,
            "discovery_starts": SEEDS,
            "policy": "The model chooses a task-specific preferred proposal and can add services. No paid API, account multiplication or automatic billing. Login and a cost-controlled connector are still required; webpage text is not proof of eligibility.",
        }

    def choose(self, branch: str, candidate_id: str, purpose: str, rationale: str) -> dict:
        if branch not in ("A", "B"):
            raise ValueError("Only A/B may record a service preference")
        if (
            not isinstance(purpose, str)
            or not 1 <= len(purpose) <= 400
            or not isinstance(rationale, str)
            or not 1 <= len(rationale) <= 800
        ):
            raise ValueError("Service selection explanation exceeds its budget")
        row = self.db.execute("SELECT value FROM proposals WHERE id=?", (candidate_id,)).fetchone()
        if row is None:
            raise ValueError("Service selection must refer to a researched proposal")
        candidate = json.loads(row[0])
        # Documentation changed since this proposal: research it again rather
        # than silently using a previous account/price assumption.
        evidence = self.evidence(candidate["evidence"]["url"])
        if evidence["sha256"] != candidate["evidence"]["sha256"]:
            raise ValueError("Service documentation changed; update the proposal")
        decision = {
            "service_id": candidate_id,
            "name": candidate["name"],
            "url": candidate["url"],
            "purpose": purpose,
            "rationale": rationale,
            "evidence": evidence,
            "checked_at": time.time(),
            "status": "preferred proposal only; no external consultation was executed",
        }
        with self.db:
            self.db.execute(
                "INSERT INTO selections(branch,value) VALUES(?,?)", (branch, json.dumps(decision))
            )
        from rlm.v100.experiments import SharedLab

        shared = SharedLab(self.root / "research/state/competition.sqlite3")
        try:
            shared.append(branch, "service-selection", decision)
        finally:
            shared.close()
        return decision

    def recent(self) -> list[dict]:
        rows = [
            (branch, json.loads(value))
            for branch, value in self.db.execute(
                "SELECT branch,value FROM selections ORDER BY sequence DESC LIMIT 4"
            )
        ]
        return [
            {
                "branch": branch,
                "service_id": value["service_id"],
                "name": value["name"],
                "purpose": value["purpose"][:120],
                "rationale": value["rationale"][:180],
                "status": value["status"],
            }
            for branch, value in rows
        ]
