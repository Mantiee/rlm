"""
Verified adapter for goal-learning canonical records.

Binds each training record's evidence observations to their frozen
body_sha256, validates the archive files on disk, and returns records
that carry explicit source authenticity flags matching the dashboard
contract.

No scoring or training is performed. Disjoint cohort check: records
sharing a group id (same outcome series) must never appear in both
training and evaluation sets.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

_SCHEMA = "synta-goal-learning-adapter-v1"


def _check_archive(root: Path, identity: str, expected_sha256: str) -> dict:
    archive = root / "research/goal-learning/sources" / (identity + ".json")
    if not archive.exists():
        return {"identity": identity, "verified": False, "reason": "archive_missing"}
    try:
        stored = json.loads(archive.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"identity": identity, "verified": False, "reason": "archive_unreadable"}
    body = stored.get("body")
    if not isinstance(body, str):
        return {"identity": identity, "verified": False, "reason": "body_absent"}
    actual = hashlib.sha256(body.encode()).hexdigest()
    if actual != expected_sha256:
        return {
            "identity": identity, "verified": False, "reason": "body_sha256_mismatch",
            "expected": expected_sha256, "actual": actual,
        }
    return {"identity": identity, "verified": True, "body_sha256": actual}


def _observation_sha256(root: Path, identity: str) -> str | None:
    from rlm.v100.goal_learning import database
    with database(root) as db:
        row = db.execute("SELECT data FROM observations WHERE id=?", (identity,)).fetchone()
    if row is None:
        return None
    try:
        data = json.loads(row[0])
        sha = data.get("body_sha256")
        if isinstance(sha, str) and len(sha) == 64:
            return sha
    except (ValueError, TypeError):
        pass
    return None


def verified_canonical_record(root: Path, record: dict) -> dict:
    """
    Return a copy of a training record with source authenticity fields added.

    For each observation identity referenced in the record's messages, the
    adapter verifies the on-disk archive against the stored body_sha256.
    The result carries:
      - evidence_sha256: list of {identity, verified, body_sha256 | reason}
      - source_authenticity_verified: True only when EVERY referenced
        observation passes archive verification.
      - disjoint_group: the record's group id.
      - schema: _SCHEMA

    No scoring or model update is performed.
    """
    if not isinstance(record, dict):
        raise ValueError("record must be a dict")
    messages = record.get("messages", [])
    obs_ids: list[str] = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        try:
            content = json.loads(msg["content"])
        except (KeyError, ValueError, TypeError):
            continue
        target = content.get("target_at_prediction", {})
        if isinstance(target, dict):
            tid = target.get("id") or target.get("identity")
            if isinstance(tid, str) and tid:
                obs_ids.append(tid)
        for ev in content.get("evidence_at_prediction", []):
            if isinstance(ev, dict):
                eid = ev.get("id") or ev.get("identity")
                if isinstance(eid, str) and eid:
                    obs_ids.append(eid)

    evidence_results: list[dict] = []
    all_verified = True
    for identity in obs_ids:
        stored_sha = _observation_sha256(root, identity)
        if stored_sha is None:
            result = {"identity": identity, "verified": False, "reason": "sha256_not_stored"}
            all_verified = False
        else:
            result = _check_archive(root, identity, stored_sha)
            if not result["verified"]:
                all_verified = False
        evidence_results.append(result)

    out = dict(record)
    out["schema"] = _SCHEMA
    out["evidence_sha256"] = evidence_results
    out["source_authenticity_verified"] = all_verified and bool(obs_ids)
    out["disjoint_group"] = record.get("group")
    if not obs_ids:
        out["source_authenticity_verified"] = False
        out["source_authenticity_note"] = (
            "No observation identities found in record messages; "
            "cannot verify source authenticity."
        )
    return out


def verified_training_set(
    root: Path,
    records: list[dict],
    *,
    require_verified: bool = True,
) -> dict:
    """
    Process canonical records and return a training-set summary.

    - Each record is passed through verified_canonical_record.
    - Disjoint cohort check: each group id appears at most once.
    - If require_verified=True (default), unverified records are quarantined.

    Returns dict with admitted, quarantined, group_ids_admitted,
    source_authenticity_ok.  No training performed.
    """
    admitted: list[dict] = []
    quarantined: list[dict] = []
    seen_groups: dict[str, str] = {}

    for rec in records:
        try:
            verified = verified_canonical_record(root, rec)
        except ValueError as exc:
            quarantined.append({"record_id": rec.get("id"), "reason": str(exc)})
            continue
        if require_verified and not verified["source_authenticity_verified"]:
            quarantined.append({
                "record_id": verified.get("id"),
                "reason": "source_authenticity_unverified",
                "evidence_sha256": verified.get("evidence_sha256"),
            })
            continue
        group = verified.get("disjoint_group")
        if group and group in seen_groups:
            quarantined.append({
                "record_id": verified.get("id"),
                "reason": "duplicate_group",
                "group": group,
                "first_record": seen_groups[group],
            })
            continue
        if group:
            seen_groups[group] = verified.get("id", "")
        admitted.append(verified)

    all_ok = all(r.get("source_authenticity_verified") for r in admitted)
    return {
        "schema": _SCHEMA,
        "admitted": admitted,
        "quarantined": quarantined,
        "admitted_count": len(admitted),
        "quarantined_count": len(quarantined),
        "group_ids_admitted": list(seen_groups.keys()),
        "source_authenticity_ok": all_ok,
        "note": (
            "Admitted records have source observations verified against on-disk archives. "
            "Quarantined records are excluded, not deleted. "
            "No training, scoring or model update is performed here."
        ),
    }
