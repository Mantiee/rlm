"""Saved, action-free gateway questions on the independently admitted planner."""

import json
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import requests

from rlm.v100.mission_chat import connect, conversation_question, conversational_response
from rlm.v100.model_roles import selected_profile

PENDING = "readonly-pending"
RUNNING = "readonly-running"


def submit(root: Path, message: str) -> str | None:
    from rlm.v100.chat_resources import instruction
    from rlm.v100.mission_chat import preferences

    if preferences(root).get("remote_helper_enabled", True) is False:
        return None

    if not (
        conversation_question(message)
        or message.strip().casefold() in ("hello", "hi", "hej", "cześć", "czesc")
    ) or instruction(message):
        return None
    if not (root / "research/model-roles/active.json").is_file():
        return None
    if selected_profile(root, "planner") is None:
        return None
    identity = uuid.uuid4().hex
    with connect(root) as db:
        db.execute("BEGIN IMMEDIATE")
        existing = db.execute(
            "SELECT id FROM requests WHERE state IN (?,?) AND message=? LIMIT 1",
            (PENDING, RUNNING, message),
        ).fetchone()
        if existing:
            return existing["id"]
        if (
            db.execute(
                "SELECT count(*) FROM requests WHERE state IN ('queued','gateway-pending','gateway-running',?,?)",
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
    from rlm.v100.common import load_profile
    from rlm.v100.competition import helper_client
    from rlm.v100.mission import status
    from rlm.v100.mission_chat import backend_preflight, preferences
    from rlm.v100.planning import read

    if preferences(root).get("remote_helper_enabled", True) is False:
        raise ValueError("Remote GPU disabled by operator")
    path = selected_profile(root, "planner")
    if path is None:
        raise requests.ConnectionError("Waiting for an admitted planner")
    client = helper_client(load_profile(path, root), root)
    client.request_deadline = time.monotonic() + 90
    client.timeout = min(client.timeout, 5)
    client.enable_thinking = False
    client.activity_actor = "chat"
    client.activity_context = {**client.activity_context, "chat_request_id": row["id"]}
    backend_preflight(client)
    client.timeout = 60
    return conversational_response(
        root,
        client,
        row["message"],
        status(root),
        read(root),
        {"model": client.model_name, "gateway_readonly_worker": True, "role": "planner"},
        generation_timeout=60,
    )


def work(root: Path, stop: threading.Event) -> None:
    import fcntl

    folder = root / "research/state"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "gateway-readonly.lock").open("a") as lease:
        while not stop.is_set():
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                stop.wait(0.2)
        else:
            return
        if (folder / "user-chat.sqlite3").is_file():
            with connect(root) as db:
                # A stopped gateway cannot have live threads holding this lease.
                # These requests have never been allowed tools or control actions.
                db.execute("UPDATE requests SET state=? WHERE state=?", (PENDING, RUNNING))
        while not stop.wait(0.2):
            if not (folder / "user-chat.sqlite3").is_file():
                continue
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
            try:
                result = response(root, row)
                if not isinstance(result.get("answer"), str) or not result["answer"].strip():
                    raise ValueError("Empty read-only gateway reply rejected")
                if result.get("actions") or result.get("applied") or result.get("tool_receipts"):
                    raise ValueError("Read-only gateway cannot execute actions or tools")
                state, error = "completed", None
            except Exception as failure:
                result = None
                transient = isinstance(
                    failure, (requests.ConnectionError, requests.Timeout, TimeoutError)
                ) or (
                    isinstance(failure, requests.HTTPError)
                    and failure.response is not None
                    and failure.response.status_code in (502, 503)
                )
                state = PENDING if transient and row["attempts"] < 2 else "failed"
                error = str(failure)[:500]
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
