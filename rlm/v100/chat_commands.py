"""Gateway-owned commands, independent of trainer package upgrades.

Never adopt old queued requests or replay an interrupted actionful turn.
"""

import fcntl
import json
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from rlm.v100.activity import ActivityLog
from rlm.v100.mission_chat import BackendNotReady, connect

PENDING = "gateway-pending"
RUNNING = "gateway-running"


def submit(root: Path, message: str) -> str:
    from rlm.v100.mission_chat import submit as legacy_submit

    if message.strip().startswith("/cancel-chat "):
        return legacy_submit(root, message)
    identity = uuid.uuid4().hex
    with connect(root) as db:
        db.execute("BEGIN IMMEDIATE")
        existing = db.execute(
            "SELECT id FROM requests WHERE state IN ('queued','readonly-pending','readonly-running',?,?) AND message=? LIMIT 1",
            (PENDING, RUNNING, message),
        ).fetchone()
        if existing:
            return existing["id"]
        if (
            db.execute(
                "SELECT count(*) FROM requests WHERE state IN ('queued','readonly-pending','readonly-running',?,?)",
                (PENDING, RUNNING),
            ).fetchone()[0]
            >= 32
        ):
            raise ValueError("Chat queue has 32 pending requests; let them complete first")
        db.execute(
            "INSERT INTO requests(id,created,message,state) VALUES(?,?,?,?)",
            (identity, datetime.now(UTC).isoformat(), message, PENDING),
        )
    return identity


def response(root: Path, row: dict) -> dict:
    from rlm.v100.mission import status
    from rlm.v100.mission_chat import respond

    mission = status(root)
    directory = Path(mission.get("run", root / "research/mission"))
    row["deadline"] = time.monotonic() + 180
    return respond(root, directory, row)


def work(root: Path, stop: threading.Event) -> None:
    folder = root / "research/state"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "gateway-commands.lock").open("a") as lease:
        while not stop.is_set():
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                stop.wait(0.2)
        else:
            return
        while not (folder / "user-chat.sqlite3").is_file():
            if stop.wait(0.2):
                return
        with connect(root) as db:
            db.execute(
                "UPDATE requests SET state='failed',error=?,updated=? WHERE state=?",
                (
                    "Gateway interrupted during command. Prior action receipts retained; command was not replayed.",
                    time.time(),
                    RUNNING,
                ),
            )
        while not stop.wait(0.2):
            with connect(root) as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute(
                    "SELECT * FROM requests WHERE state=? AND retry_after<=? ORDER BY rowid LIMIT 1",
                    (PENDING, time.time()),
                ).fetchone()
                if row is None:
                    continue
                row = dict(row)
                db.execute(
                    "UPDATE requests SET state=?,updated=? WHERE id=?",
                    (RUNNING, time.time(), row["id"]),
                )
            result = None
            try:
                result = response(root, row)
                if not isinstance(result.get("answer"), str) or not result["answer"].strip():
                    raise ValueError("Empty gateway command reply rejected")
                state = "failed" if result.get("execution_status") == "failed" else "completed"
                error = result["answer"][:500] if state == "failed" else None
            except BackendNotReady as failure:
                # respond raises this only during preflight, before any tools.
                state = PENDING if row["attempts"] < 2 else "failed"
                error = str(failure)[:500]
            except Exception as failure:
                # Inference may follow tools. Never retry a whole actionful turn.
                state, error = "failed", str(failure)[:500]
            with connect(root) as db:
                db.execute(
                    "UPDATE requests SET state=?,response=?,error=?,attempts=attempts+1,updated=?,retry_after=? WHERE id=? AND state=?",
                    (
                        state,
                        json.dumps(result, ensure_ascii=False) if result else None,
                        error,
                        time.time(),
                        time.time() + 15 if state == PENDING else 0,
                        row["id"],
                        RUNNING,
                    ),
                )
            ActivityLog(root, "controller", "chat").write(
                "decisions" if state == "completed" else "errors",
                "gateway-command-result",
                {"id": row["id"], "state": state, "error": error, "response": result},
            )
