"""
Source influence measurement — roadmap item: public_candidate_research.

Records which verified public sources were used in a hypothesis, and
measures their influence on subsequent resolved outcomes versus a
source-free control group.  All comparisons are chronologically safe:
source attribution is recorded at hypothesis time; outcomes are read
only after they resolve.  No look-ahead bias.  No profit claim.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UTC = timezone.utc
_SCHEMA = "synta-source-influence-v1"
_MAX_EXCERPT = 512
_MAX_SOURCES_PER_HYPOTHESIS = 8


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _db(root: Path) -> sqlite3.Connection:
    folder = root / "research/source-influence"
    folder.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(folder / "ledger.sqlite3", timeout=10)
    db.row_factory = sqlite3.Row
    with db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS hypothesis_sources (
                hypothesis_id TEXT NOT NULL,
                source_id     TEXT NOT NULL,
                url           TEXT NOT NULL,
                sha256        TEXT NOT NULL,
                excerpt       TEXT,
                model_score   REAL,
                acquired_at   TEXT NOT NULL,
                recorded_at   TEXT NOT NULL,
                PRIMARY KEY (hypothesis_id, source_id)
            );
            CREATE TABLE IF NOT EXISTS outcome_receipts (
                hypothesis_id  TEXT PRIMARY KEY,
                outcome        REAL NOT NULL,
                resolved_at    TEXT NOT NULL,
                source_used    INTEGER NOT NULL DEFAULT 0,
                cohort_id      TEXT
            );
        """)
    return db


def record_hypothesis_sources(
    root: Path,
    hypothesis_id: str,
    sources: list[dict],
) -> dict:
    """
    Record which verified public sources were used in a hypothesis.

    Each source dict must contain:
      source_id (str), url (str), sha256 (str), excerpt (str ≤512),
      model_score (float 0..1, the model's stated relevance),
      acquired_at (ISO-8601 str).

    Rejects unknown or empty source lists silently rather than
    fabricating attribution.  Returns a receipt.
    """
    if not isinstance(hypothesis_id, str) or not hypothesis_id:
        raise ValueError("hypothesis_id must be a non-empty string")
    if not isinstance(sources, list) or not sources:
        return {
            "schema": _SCHEMA,
            "hypothesis_id": hypothesis_id,
            "sources_recorded": 0,
            "note": "No sources provided; source influence will not be measured for this hypothesis.",
        }
    if len(sources) > _MAX_SOURCES_PER_HYPOTHESIS:
        sources = sources[:_MAX_SOURCES_PER_HYPOTHESIS]

    recorded_at = _now_iso()
    rows: list[dict] = []
    skipped = 0
    for src in sources:
        if not isinstance(src, dict):
            skipped += 1
            continue
        sid = src.get("source_id", "")
        url = src.get("url", "")
        sha = src.get("sha256", "")
        excerpt = str(src.get("excerpt", ""))[:_MAX_EXCERPT]
        score = src.get("model_score")
        acq = src.get("acquired_at", recorded_at)
        if not (isinstance(sid, str) and sid and isinstance(url, str) and url
                and isinstance(sha, str) and len(sha) == 64):
            skipped += 1
            continue
        if score is not None:
            if not isinstance(score, (int, float)) or not (0.0 <= float(score) <= 1.0):
                skipped += 1
                continue
            score = float(score)
        rows.append({
            "hypothesis_id": hypothesis_id,
            "source_id": sid,
            "url": url,
            "sha256": sha,
            "excerpt": excerpt,
            "model_score": score,
            "acquired_at": acq,
            "recorded_at": recorded_at,
        })

    if not rows:
        return {
            "schema": _SCHEMA,
            "hypothesis_id": hypothesis_id,
            "sources_recorded": 0,
            "skipped": skipped,
            "note": "No valid sources passed validation; check source_id, url, sha256 (64 hex), model_score (0..1).",
        }

    with _db(root) as db:
        for r in rows:
            db.execute(
                """INSERT OR REPLACE INTO hypothesis_sources
                   (hypothesis_id, source_id, url, sha256, excerpt, model_score, acquired_at, recorded_at)
                   VALUES (:hypothesis_id, :source_id, :url, :sha256, :excerpt, :model_score, :acquired_at, :recorded_at)""",
                r,
            )

    return {
        "schema": _SCHEMA,
        "hypothesis_id": hypothesis_id,
        "sources_recorded": len(rows),
        "skipped": skipped,
        "source_ids": [r["source_id"] for r in rows],
        "recorded_at": recorded_at,
        "note": "Source attribution recorded at hypothesis time; outcome is compared after resolution only.",
    }


def record_outcome(
    root: Path,
    hypothesis_id: str,
    outcome: float,
    resolved_at: str | None = None,
    cohort_id: str | None = None,
) -> dict:
    """
    Record a resolved outcome for a hypothesis.

    outcome: paper return or Brier score delta (no real-money amounts).
    Sets source_used=1 if any source was recorded for this hypothesis.
    """
    if not isinstance(hypothesis_id, str) or not hypothesis_id:
        raise ValueError("hypothesis_id must be a non-empty string")
    if not isinstance(outcome, (int, float)) or not (-1e9 < outcome < 1e9):
        raise ValueError("outcome must be a finite number")

    resolved_at = resolved_at or _now_iso()
    db = _db(root)
    with db:
        count = db.execute(
            "SELECT COUNT(*) FROM hypothesis_sources WHERE hypothesis_id=?",
            (hypothesis_id,),
        ).fetchone()[0]
        db.execute(
            """INSERT OR REPLACE INTO outcome_receipts
               (hypothesis_id, outcome, resolved_at, source_used, cohort_id)
               VALUES (?, ?, ?, ?, ?)""",
            (hypothesis_id, float(outcome), resolved_at, 1 if count > 0 else 0, cohort_id),
        )
    return {
        "schema": _SCHEMA,
        "hypothesis_id": hypothesis_id,
        "outcome": outcome,
        "source_used": count > 0,
        "resolved_at": resolved_at,
    }


def source_influence_report(root: Path, *, limit: int = 100) -> dict:
    """
    Compare source-used vs source-free outcomes chronologically.

    Only uses resolved outcomes.  Splits by source_used flag set at
    hypothesis time — no look-ahead.  Returns:
      - recent_sources: latest hypothesis-source attributions with
        source_id, url, sha256, excerpt, model_score, acquired_at
      - used_mean / control_mean: average outcome for each group
      - used_count / control_count: sample sizes
      - influence_delta: used_mean - control_mean (no causal claim)
      - note: plain-text caveat
    """
    db = _db(root)

    recent = db.execute(
        """SELECT hs.hypothesis_id, hs.source_id, hs.url, hs.sha256,
                  hs.excerpt, hs.model_score, hs.acquired_at, hs.recorded_at
           FROM hypothesis_sources hs
           ORDER BY hs.recorded_at DESC LIMIT ?""",
        (min(limit, 200),),
    ).fetchall()

    rows = db.execute(
        """SELECT source_used, outcome
           FROM outcome_receipts
           ORDER BY resolved_at ASC LIMIT ?""",
        (limit,),
    ).fetchall()

    used = [r["outcome"] for r in rows if r["source_used"]]
    ctrl = [r["outcome"] for r in rows if not r["source_used"]]

    used_mean = sum(used) / len(used) if used else None
    ctrl_mean = sum(ctrl) / len(ctrl) if ctrl else None
    delta = (round(used_mean - ctrl_mean, 6) if used_mean is not None and ctrl_mean is not None else None)

    return {
        "schema": _SCHEMA,
        "recent_sources": [
            {
                "hypothesis_id": r["hypothesis_id"],
                "source_id": r["source_id"],
                "url": r["url"],
                "sha256": r["sha256"],
                "excerpt": r["excerpt"],
                "model_score": r["model_score"],
                "acquired_at": r["acquired_at"],
                "recorded_at": r["recorded_at"],
            }
            for r in recent
        ],
        "used_count": len(used),
        "control_count": len(ctrl),
        "used_mean_outcome": round(used_mean, 6) if used_mean is not None else None,
        "control_mean_outcome": round(ctrl_mean, 6) if ctrl_mean is not None else None,
        "influence_delta": delta,
        "note": (
            "influence_delta = used_mean - control_mean on resolved paper outcomes. "
            "This is a descriptive difference, not a causal estimate. "
            "Source attribution is recorded before resolution; no look-ahead. "
            "No profit claim. Empty or unverified sources show as zero entries."
        ),
    }
