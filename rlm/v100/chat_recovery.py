"""Upgrade-only repair of plans created while answering an informational question."""

import json
import re
from pathlib import Path

from rlm.v100.common import atomic_json


def repair(root: Path) -> dict:
    from rlm.v100.mission import status
    from rlm.v100.mission_chat import connect, conversation_question

    if status(root)["running"]:
        raise ValueError("Stop the mission before recovering chat-created plans")
    directory = root / "research/state/chat-question-repairs"
    repaired = []
    with connect(root) as db:
        rows = db.execute(
            "SELECT * FROM requests WHERE state='completed' ORDER BY rowid DESC LIMIT 500"
        ).fetchall()
        for row in rows:
            if not conversation_question(row["message"]):
                continue
            response = json.loads(row["response"])
            receipts = [
                r for r in response.get("applied", []) if r.get("horizon") in ("short", "mid")
            ]
            if not receipts:
                continue
            if not re.fullmatch(r"[a-f0-9]{32}", row["id"]):
                raise ValueError("Invalid archived question identity")
            archive = directory / (row["id"] + ".json")
            if archive.exists():
                continue
            path = root / "research/plans/current.json"
            plans = json.loads(path.read_text()) if path.exists() else {}
            restored = []
            for receipt in receipts:
                horizon, identity = receipt["horizon"], receipt.get("id", "")
                if (
                    not re.fullmatch(r"[a-f0-9]{32}", identity)
                    or plans.get(horizon, {}).get("id") != identity
                ):
                    continue
                version = json.loads((root / "research/plans" / (identity + ".json")).read_text())
                previous = version.get("previous") or {}
                # A later goal change must not resurrect an unrelated earlier goal's plan.
                if previous.get("goal_id") not in (None, plans[horizon].get("goal_id")):
                    continue
                plans[horizon] = previous
                restored.append(horizon)
            atomic_json(
                archive,
                {
                    "request": dict(row),
                    "restored_plans": restored,
                    "scope": "Question-created plan reverted only if its exact ID is still active; later plans and original evidence retained.",
                },
            )
            if restored:
                atomic_json(
                    archive.with_name(row["id"] + "-plans-before.json"),
                    json.loads(path.read_text()),
                )
                atomic_json(path, plans)
            db.execute(
                "UPDATE requests SET state='queued',response=NULL,attempts=0,error=NULL,updated=0,wait_started=0 WHERE id=?",
                (row["id"],),
            )
            repaired.append({"id": row["id"], "restored_plans": restored, "archive": str(archive)})
    return {"requeued_questions": repaired, "long_term_goal_changed": False}
