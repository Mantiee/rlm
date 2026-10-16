"""Persistent user chat/control queue serviced alongside the owned mission.

Accepted weights can be served on CPU during exclusive GPU training when RAM permits.
User directives steer R&D. Only the local user's explicit natural-language request or /goal changes the long-term goal.
"""

import copy
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path

import requests

from rlm.v100.activity import ActivityLog, append_locked
from rlm.v100.common import atomic_json, load_profile
from rlm.v100.serving import assert_served_expert


class BackendNotReady(requests.ConnectionError):
    """Owned accepted serving endpoint is not ready before generation."""


def backend_preflight(client) -> None:
    probe = getattr(client, "preflight", None)
    if callable(probe):
        probe()
    else:
        client.identity()


@contextmanager
def connect(root: Path):
    path = root / "research/state/user-chat.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=2)
    db.row_factory = sqlite3.Row
    db.execute("""CREATE TABLE IF NOT EXISTS requests(
        id TEXT PRIMARY KEY, created TEXT NOT NULL, message TEXT NOT NULL,
        state TEXT NOT NULL, response TEXT, attempts INTEGER NOT NULL DEFAULT 0,
        error TEXT, updated REAL NOT NULL DEFAULT 0)""")
    fields = ("wait_started", "retry_after")
    if any(
        field not in {row[1] for row in db.execute("PRAGMA table_info(requests)")}
        for field in fields
    ):
        db.execute("BEGIN IMMEDIATE")
        for field in fields:
            if field not in {row[1] for row in db.execute("PRAGMA table_info(requests)")}:
                db.execute(f"ALTER TABLE requests ADD COLUMN {field} REAL NOT NULL DEFAULT 0")
    db.commit()
    try:
        with db:
            yield db
    finally:
        db.close()


def submit(root: Path, message: str) -> str:
    if not isinstance(message, str) or not message.strip() or len(message) > 8000:
        raise ValueError("Chat needs 1-8000 characters")
    if re.fullmatch(r"/cancel-chat [a-f0-9]{32}", message.strip()):
        identity = message.strip().split()[1]
        with connect(root) as db:
            row = db.execute("SELECT state FROM requests WHERE id=?", (identity,)).fetchone()
            if row is None:
                raise ValueError("Unknown chat request to cancel")
            if row["state"] not in (
                "queued",
                "readonly-pending",
                "readonly-running",
                "gateway-pending",
                "cancelled",
            ):
                raise ValueError("Only pending chat requests can be cancelled")
            db.execute(
                "UPDATE requests SET state='cancelled',error='Cancelled by operator',updated=? WHERE id=? AND state IN ('queued','readonly-pending','readonly-running','gateway-pending')",
                (time.time(), identity),
            )
        ActivityLog(root, "controller", "chat").write(
            "decisions",
            "user-command-cancelled",
            {
                "id": identity,
                "scope": "Pending attempts cancelled; prior receipts retained. An already in-flight call may finish.",
            },
        )
        return identity
    identity = uuid.uuid4().hex
    with connect(root) as db:
        db.execute("BEGIN IMMEDIATE")
        pending = db.execute(
            "SELECT id FROM requests WHERE state IN ('queued','readonly-pending','readonly-running','gateway-pending','gateway-running') AND message=? ORDER BY rowid LIMIT 1",
            (message,),
        ).fetchone()
        if pending:
            return pending["id"]
        if (
            db.execute(
                "SELECT count(*) FROM requests WHERE state IN ('queued','readonly-pending','readonly-running','gateway-pending','gateway-running')"
            ).fetchone()[0]
            >= 32
        ):
            raise ValueError("Chat queue has 32 pending requests; let them complete first")
        db.execute(
            "INSERT INTO requests(id,created,message,state) VALUES(?,?,?,'queued')",
            (identity, datetime.now(UTC).isoformat(), message),
        )
    return identity


def recent_requests(root: Path, limit: int = 20) -> list[dict]:
    if type(limit) is not int or not 1 <= limit <= 32:
        raise ValueError("Read 1-32 recent chat requests")
    with connect(root) as db:
        rows = db.execute("SELECT * FROM requests ORDER BY rowid DESC LIMIT ?", (limit,)).fetchall()
    return [
        {
            **dict(row),
            "state": "queued"
            if row["state"]
            in ("readonly-pending", "readonly-running", "gateway-pending", "gateway-running")
            else row["state"],
            "response": json.loads(row["response"]) if row["response"] else None,
        }
        for row in reversed(rows)
    ]


def retire_request(root: Path, identity: str, reason: str) -> dict:
    """Retire only the specified request after mission shutdown, retaining its original row."""
    if not re.fullmatch(r"[a-f0-9]{32}", identity):
        raise ValueError("Invalid chat request identity")
    with connect(root) as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM requests WHERE id=?", (identity,)).fetchone()
        if row is None:
            return {"id": identity, "retired": False, "reason": "Request not present"}
        archive = root / "research/state/chat-retired" / (identity + ".json")
        if not archive.exists():
            atomic_json(archive, dict(row))
        if row["state"] != "completed":
            db.execute(
                "UPDATE requests SET state='cancelled',error=?,updated=? WHERE id=?",
                (reason, time.time(), identity),
            )
        return {"id": identity, "retired": row["state"] != "completed", "archive": str(archive)}


def trade_visualization_requested(message: str) -> bool:
    from rlm.v100.income_policy import normalized

    text = normalized(message)
    return bool(
        re.search(r"zrob|dodaj|pokaz|create|show|visualize|wizualizuj", text)
        and not re.search(r"niemozliwe|blad|glitch|spada|fee|spread|koszt|wrong|error", text)
        and re.search(r"wykres|wizualiz|visualiz|chart", text)
        and re.search(r"kup|sprzed|buy|sell|transakc|trad|zwrot", text)
    )


def trade_visualization_reply(root: Path) -> dict:
    from rlm.v100.trade_explorer import snapshot

    receipt = snapshot(root)
    return {
        "answer": "Interaktywny Trade explorer jest częścią renderera hosta: / na porcie 8765. "
        "Wybierz serię paper lub historyczną, zakres czasu i metrykę. Kliknij znacznik kupna/sprzedaży, "
        "aby odczytać cenę, koszty i zapis wykonania. Brak transakcji jest pokazany jawnie. "
        "Dane odświeżają się bez resetowania wybranego zakresu. "
        + ("Błędy danych: " + "; ".join(receipt["errors"]) if receipt["errors"] else ""),
        "actions": [],
        "applied": [],
        "responder": {"model": "controller-trade-explorer"},
        "evidence": {"series": len(receipt["series"]), "updated": receipt["updated"]},
    }


def direct_facts(root: Path, message: str) -> dict | None:
    """Explicit read-only host questions never queue behind GPU work."""
    from rlm.v100 import chat_facts, chat_plan, chat_progress

    if chat_plan.requested(message):
        return chat_plan.respond(root)
    if trade_visualization_requested(message):
        return trade_visualization_reply(root)
    if message.strip() == "/status" or chat_progress.requested(message):
        return chat_progress.respond(root, message)
    if conversation_question(message) and chat_facts.requested(message):
        return chat_facts.response(root, message)
    return None


def save_direct_reply(root: Path, message: str, response: dict) -> dict:
    if not isinstance(message, str) or not message.strip() or len(message) > 8000:
        raise ValueError("Chat needs 1-8000 characters")
    if response.get("actions") or response.get("applied"):
        raise ValueError("Direct replies must be read-only facts")
    if not isinstance(response.get("answer"), str) or not response["answer"].strip():
        raise ValueError("Empty direct chat reply rejected")
    identity = uuid.uuid4().hex
    with connect(root) as db:
        db.execute(
            "INSERT INTO requests(id,created,message,state,response,updated) VALUES(?,?,?,'completed',?,?)",
            (
                identity,
                datetime.now(UTC).isoformat(),
                message,
                json.dumps(response, ensure_ascii=False),
                time.time(),
            ),
        )
    ActivityLog(root, "controller", "chat").write(
        "decisions", "user-command-completed", {"id": identity, **response}
    )
    return inspect(root, identity)


class LateReplies:
    """Deliver eventual results while the operator is back at the input prompt."""

    def __init__(self, root: Path):
        self.root = root
        self.pending = set()
        self.lock = threading.Lock()
        self.stop = threading.Event()

    def track(self, identity: str):
        with self.lock:
            self.pending.add(identity)

    def poll(self):
        with self.lock:
            identities = list(self.pending)
        for identity in identities:
            result = inspect(self.root, identity)
            if result["state"] == "queued":
                continue
            with self.lock:
                self.pending.discard(identity)
            print("\nLATE REPLY | REQUEST: " + identity, flush=True)
            display_reply(result)
            print("\nYOU > ", end="", flush=True)

    def run(self):
        while not self.stop.wait(2):
            try:
                self.poll()
            except (OSError, sqlite3.Error) as error:
                print("\nReply monitor delayed: " + str(error)[:200], flush=True)


def display_reply(result: dict):
    if result["state"] == "completed":
        response = result["response"]
        label = response.get("responder", {}).get("model", "controller")
        print(f"\nSYNTA [{label}]\n\n{response['answer']}\n", flush=True)
        print(
            "ACTION RECEIPTS:",
            json.dumps(response.get("applied", []), ensure_ascii=False),
            flush=True,
        )
    else:
        print(
            waiting_label(result)
            if result["state"] == "queued"
            else "FAILED: " + str(result.get("error")),
            flush=True,
        )


def inspect(root: Path, identity: str) -> dict:
    with connect(root) as db:
        row = db.execute("SELECT * FROM requests WHERE id=?", (identity,)).fetchone()
        position = (
            db.execute(
                "SELECT count(*) FROM requests WHERE state='queued' AND rowid<=(SELECT rowid FROM requests WHERE id=?)",
                (identity,),
            ).fetchone()[0]
            if row is not None and row["state"] == "queued"
            else None
        )
    if row is None:
        raise ValueError("Unknown chat request")
    result = dict(row)
    if result["state"] in (
        "readonly-pending",
        "readonly-running",
        "gateway-pending",
        "gateway-running",
    ):
        result["route"] = (
            "gateway-commands" if result["state"].startswith("gateway-") else "gateway-readonly"
        )
        if result["state"] in ("readonly-running", "gateway-running"):
            result["processing"] = {
                "id": identity,
                "started": result["updated"],
                "time_budget_seconds": 180 if result["state"] == "gateway-running" else 90,
            }
        result["state"] = "queued"
    result["queue_position"] = position
    if result["response"]:
        result["response"] = json.loads(result["response"])
    active_path = root / "research/state/chat-active.json"
    if active_path.exists():
        active = json.loads(active_path.read_text())
        if active.get("id") == identity and result["state"] == "queued":
            from rlm.v100.mission import status

            mission = status(root)
            if mission["running"] and mission.get("pid") == active.get("pid"):
                result["processing"] = active
    return result


def preferences(root: Path) -> dict:
    path = root / "research/user-preferences.json"
    return json.loads(path.read_text()) if path.exists() else {"directive": "", "alerts": False}


def apply_actions(root: Path, actions: list[dict], user_message: str | None = None) -> list[dict]:
    from rlm.v100.research_policy import choose

    if not isinstance(actions, list) or len(actions) > 3:
        raise ValueError("At most three chat actions per reply")
    # Validate the entire plan before executing any control change.
    for action in actions:
        if set(action) != {
            "kind",
            "text",
            "target",
            "thinking",
            "max_tokens",
            "batch_tokens",
            "enabled",
        }:
            raise ValueError("Invalid chat action keys")
        kind = action["kind"]
        allowed = ["directive", "alerts", "budget", "plan_mid", "plan_short", "sandbox"]
        if user_message is not None:
            from rlm.v100.chat_goals import authorizes_long

            if authorizes_long(user_message):
                allowed.append("goal_long")
        if kind not in allowed or type(action["enabled"]) is not bool:
            raise ValueError("Unknown chat control action")
        if not isinstance(action["text"], str) or len(action["text"]) > 2000:
            raise ValueError("Directive exceeds its budget")
        if kind in ("plan_mid", "plan_short", "sandbox") and not action["text"].strip():
            raise ValueError("Goal and plan must not be empty")
        if kind == "goal_long":
            from rlm.v100.chat_goals import validate_long

            validate_long(user_message, action["text"])
        if kind == "budget":
            target = action["target"]
            maximum = 8192 if target == "master" else 4096
            batches = (128, 256, 512) if target == "master" else (16,)
            if (
                target not in ("master", "helper")
                or type(action["thinking"]) is not bool
                or type(action["max_tokens"]) is not int
                or not 256 <= action["max_tokens"] <= maximum
                or type(action["batch_tokens"]) is not int
                or action["batch_tokens"] not in batches
            ):
                raise ValueError("Budget outside measured host limits")
    receipts = []
    chosen = preferences(root)
    for action in actions:
        if action["kind"] == "budget":
            receipts.append(
                choose(
                    root,
                    **{k: action[k] for k in ("target", "thinking", "max_tokens", "batch_tokens")},
                )
            )
        elif action["kind"] == "goal_long":
            from rlm.v100.planning import update

            receipts.append(update(root, "long", action["text"], "user"))
        elif action["kind"] in ("plan_mid", "plan_short"):
            from rlm.v100.planning import update

            receipts.append(update(root, action["kind"].split("_")[1], action["text"], "user"))
        elif action["kind"] == "sandbox":
            from rlm.v100.drones import schedule

            receipts.append(schedule(root, "A", "desktop", action["text"], 0))
        elif action["kind"] == "directive":
            chosen["directive"] = action["text"]
            receipts.append({"directive": action["text"], "effective": "next R&D request"})
        else:
            chosen.update(alerts=action["enabled"], alert_rule=action["text"])
            receipts.append(
                {
                    "alerts": action["enabled"],
                    "rule": action["text"],
                    "scope": "local paper/research alerts; no real orders",
                }
            )
    atomic_json(root / "research/user-preferences.json", chosen)
    return receipts


def emit_alert(root: Path, branch: str, proposal: dict, rejected: bool) -> None:
    settings = preferences(root)
    if not settings["alerts"] or proposal.get("action") not in ("open", "close"):
        return
    event = {
        "time": datetime.now(UTC).isoformat(),
        "branch": branch,
        "proposal": proposal,
        "rejected": rejected,
        "scope": "Paper-model signal, not an independently validated buy/sell recommendation",
    }
    append_locked(
        root / "research/alerts/signals.jsonl", json.dumps(event, ensure_ascii=False) + "\n"
    )
    ActivityLog(root, branch, "alerts").write("decisions", "paper-signal", event)


def schema(allow_long_goal: bool = False) -> dict:
    kinds = ["directive", "alerts", "budget", "plan_mid", "plan_short", "sandbox"]
    if allow_long_goal:
        kinds.append("goal_long")
    fields = {
        "kind": {
            "type": "string",
            "enum": kinds,
        },
        "text": {"type": "string", "maxLength": 2000},
        "target": {"type": "string", "enum": ["master", "helper"]},
        "thinking": {"type": "boolean"},
        "max_tokens": {"type": "integer", "minimum": 256, "maximum": 8192},
        "batch_tokens": {"type": "integer", "enum": [16, 128, 256, 512]},
        "enabled": {"type": "boolean"},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["answer", "actions"],
        "properties": {
            "answer": {"type": "string", "maxLength": 3000},
            "actions": {
                "type": "array",
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": fields,
                    "required": list(fields),
                    "additionalProperties": False,
                },
            },
        },
    }


def verified_plan_response(root: Path, client, responder: dict) -> dict:
    """One model-selected read and concise generated answer under existing admission."""
    from rlm.v100.agent import native_turn, tool_schema
    from rlm.v100.research_tools import ResearchTools

    bounded = copy.copy(client)
    bounded.sampling_args = dict(client.sampling_args) | {"max_tokens": 48}
    bounded.research_token_ceiling = 48
    bounded.enable_thinking = False
    bounded.request_deadline = min(
        getattr(client, "request_deadline", float("inf")), time.monotonic() + 60
    )
    if bounded.request_deadline <= time.monotonic():
        raise TimeoutError("Read-only plan verification deadline exhausted")
    bounded.timeout = min(client.timeout, bounded.request_deadline - time.monotonic())
    selected = native_turn(
        bounded,
        [{"role": "user", "content": "Call get_plan once."}],
        tools=[tool_schema("get_plan", "Read recorded goal.", {})],
    )
    calls = selected.get("tool_calls") or []
    if len(calls) != 1 or selected.get("actions") or selected.get("applied"):
        raise ValueError("Verification requires exactly one model-selected read tool")
    call = calls[0]
    function = call.get("function", {})
    if function.get("name") != "get_plan" or json.loads(function.get("arguments", "null")) != {}:
        raise ValueError("Verification accepts only get_plan with empty arguments")
    if time.monotonic() >= bounded.request_deadline:
        raise TimeoutError("Read-only plan verification expired before tool execution")
    journal = ActivityLog(root, "A", "chat-plan-verification")
    step = journal.write(
        "tools", "tool-start", {"tool": "get_plan", "arguments": {}}, tool_call_id=call.get("id")
    )
    tools = ResearchTools(root, {}, "A")
    tools.allowed_tool_names = {"get_plan"}
    receipt = {"tool": "get_plan", "arguments": {}, "result": tools.execute("get_plan", {})}
    journal.write("tools", "tool-result", receipt, step_id=step)
    if receipt_failed(receipt):
        raise ValueError("Read-only plan tool failed; no generated success accepted")
    goal = receipt["result"].get("long") or {}
    excerpt = str(goal.get("text", "No recorded goal"))[:120]
    if time.monotonic() >= bounded.request_deadline:
        raise TimeoutError("Read-only plan verification expired after recorded tool receipt")
    bounded.timeout = min(client.timeout, bounded.request_deadline - time.monotonic())
    answer = native_turn(
        bounded,
        [
            {
                "role": "user",
                "content": "Report this recorded goal excerpt briefly: "
                + json.dumps(excerpt, ensure_ascii=False),
            }
        ],
        response_format={
            "type": "json_object",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["answer"],
                "properties": {"answer": {"type": "string", "maxLength": 160}},
            },
        },
    )
    result = json.loads(answer.get("content", "null"))
    if (
        answer.get("tool_calls")
        or answer.get("actions")
        or answer.get("applied")
        or not isinstance(result, dict)
        or set(result) != {"answer"}
        or not isinstance(result["answer"], str)
        or len(result["answer"]) > 160
        or not answer_complete(result["answer"])
    ):
        raise ValueError("Invalid generated read-only plan answer; no extra actions accepted")
    if time.monotonic() >= bounded.request_deadline:
        raise TimeoutError("Read-only plan verification answer arrived after deadline")
    return {
        "answer": result["answer"],
        "actions": [],
        "applied": [],
        "tool_receipts": [receipt],
        "responder": responder,
        "scope": "Model-selected get_plan read and generated goal excerpt; no plan changes or additional tools",
    }


def explicit_no_tools(message: str) -> bool:
    """Current user restrictions only reduce permissions, never grant new ones."""
    return bool(
        re.search(
            r"\b(?:no tools|without tools|do not use tools|don't use tools|"
            r"bez narzędzi|bez narzedzi|nie używaj narzędzi|nie uzywaj narzedzi)\b",
            message.casefold(),
        )
    )


def recorded_context_requested(message: str) -> bool:
    return bool(re.search(r"\b(?:goal|plan|cel|plany|planu|celu)\b", message.casefold()))


def no_tools_response(client, message: str, plans: dict, responder: dict) -> dict:
    """One real generation with strict empty actions and no tool/research context."""
    prompt = (
        "Answer the current user message directly in its language. Return JSON with answer "
        "and actions: []. No tools or changes are allowed. Do not claim execution or income."
    )
    if plans:
        prompt += " Recorded plans: " + json.dumps(
            {
                horizon: str((plans.get(horizon) or {}).get("text", "No recorded plan"))[:500]
                for horizon in ("long", "mid", "short")
            },
            ensure_ascii=False,
        )
    reply = client.request(
        "/v1/chat/completions",
        {
            "model": client.model_name,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": message},
            ],
            "max_tokens": 256,
            "temperature": 0.0,
            "stream": False,
            "response_format": {
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["answer", "actions"],
                    "properties": {
                        "answer": {"type": "string", "maxLength": 2000},
                        "actions": {"type": "array", "maxItems": 0, "items": {}},
                    },
                },
            },
            **client.template_args(),
        },
    )
    choice = reply["choices"][0]
    result = json.loads(choice["message"]["content"])
    if (
        choice.get("finish_reason") not in ("stop", "eos")
        or not isinstance(result, dict)
        or set(result) != {"answer", "actions"}
        or result["actions"] != []
        or not isinstance(result["answer"], str)
        or not answer_complete(result["answer"])
    ):
        raise ValueError("Invalid explicit no-tools response; no actions accepted")
    return {
        **result,
        "applied": [],
        "tool_receipts": [],
        "responder": responder,
        "scope": "Explicit no-tools question; model answer only, no execution",
    }


def conversation_question(message: str) -> bool:
    """Read-only questions take one generation; explicit work retains tools."""
    from rlm.v100.chat_goals import normalized

    value = normalized(message).strip()
    return bool(
        re.match(
            r"(?:(?:hejka|hej|czesc|hello|hi)[,! ]+)?(?:(?:a|to|no|ale)\s+)*(?:co|czego|czy|jak|dlaczego|czemu|gdzie|kiedy|what|why|where|when|ej zaczniesz)\b|(?:dalej|znowu|nadal).*\b(?:tracisz|przegrywasz)\b",
            value,
        )
    ) and not bool(
        re.search(
            r"\b(?:zrób|zrob|napraw|popraw|dodaj|edyt|zmień|zmien|uruchom|sprawdź|sprawdz|zbadaj|zaimplement|chcę|chce|możesz|mozesz)\w*",
            value,
        )
    )


def answer_complete(answer: str) -> bool:
    """Detect long natural-language cutoffs even when JSON grammar reports stop."""
    text = answer.strip()
    return bool(text) and (len(text) < 240 or text[-1] in ".!?…:;)]}`\"'»")


def repair_answer(client, message: str, facts: dict) -> str:
    """One concise recovery generation, with no tools or partial assistant history."""
    if time.monotonic() + 8 >= getattr(client, "request_deadline", float("inf")):
        raise ValueError(
            "Chat output incomplete and request deadline exhausted; no partial answer accepted"
        )
    reply = client.request(
        "/v1/chat/completions",
        {
            "model": client.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "Answer only the current question in its language, in at most three complete sentences. This is a read-only recovery, with no new tools or control changes. The previous partial answer is discarded. Only supplied receipts are evidence; do not invent causes of losses, independent data validation, accepted weights or income. Distinguish queued tasks from completed tasks. Facts: "
                    + json.dumps(facts, ensure_ascii=False),
                },
                {"role": "user", "content": message},
            ],
            "max_tokens": 512,
            "temperature": 0.0,
            "stream": False,
            **client.template_args(),
        },
    )
    choice = reply["choices"][0]
    answer = choice["message"].get("content")
    if (
        choice.get("finish_reason") not in ("stop", "eos")
        or not isinstance(answer, str)
        or not answer_complete(answer)
    ):
        raise ValueError(
            "Chat recovery response incomplete; no partial answer or control actions accepted"
        )
    return answer.strip()


def conversational_response(
    root: Path,
    client,
    message: str,
    mission: dict,
    plans: dict,
    responder: dict,
    generation_timeout: int = 20,
) -> dict:
    """One actual model request with bounded factual context and no tool routing."""
    if type(generation_timeout) is not int or not 1 <= generation_timeout <= 60:
        raise ValueError("Read-only generation budget must be 1-60 seconds")
    goal = (plans.get("long") or {}).get("text", "No recorded goal")
    facts = {
        "mission_running": mission.get("running"),
        "phase": mission.get("state", {}).get("phase"),
        "learning": mission.get("learning"),
        "goal": goal[:3000],
    }
    client.timeout = min(client.timeout, generation_timeout)
    client.request_deadline = min(
        getattr(client, "request_deadline", time.monotonic() + generation_timeout + 20),
        time.monotonic() + generation_timeout + 20,
    )
    reply = client.request(
        "/v1/chat/completions",
        {
            "model": client.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are Synta. Answer ONLY the current user question, in its language, briefly with complete sentences. Internal plans and tool text use English. This is read-only chat: do not claim to execute work or update plans. Only the supplied status is verified. Missing optimizer/profit counters are unknown. Audit reproducibility does not prove independent price accuracy or causal explanations for losses. Explain how the recorded goal relates to the question, without changing it or promising profit. State missing evidence honestly. Verified current facts: "
                    + json.dumps(facts, ensure_ascii=False),
                },
                {"role": "user", "content": message},
            ],
            "max_tokens": 512,
            "temperature": 0.0,
            "stream": False,
            **client.template_args(),
        },
    )
    choice = reply["choices"][0]
    answer = choice["message"].get("content")
    if (
        choice.get("finish_reason") not in ("stop", "eos")
        or not isinstance(answer, str)
        or not answer_complete(answer)
    ):
        answer = repair_answer(client, message, facts)
    return {
        "answer": answer.strip(),
        "actions": [],
        "applied": [],
        "responder": responder,
        "scope": "Read-only question; at most one concise output recovery, no tool execution or control changes",
    }


READ_ONLY_TOOLS = {
    "search_knowledge",
    "inspect_trade_costs",
    "inspect_web_page",
    "read_dashboard",
    "dashboard_status",
    "read_source",
    "read_public_page",
    "search_memory",
    "mission_evidence",
    "get_plan",
    "capabilities",
    "drone_status",
    "compute_resources",
    "paper_status",
    "paper_observed_results",
    "goal_learning_status",
    "income_opportunities",
    "sandbox_state",
    "read_master_code",
    "read_tool_result",
}


def receipt_failed(row: dict) -> bool:
    result = row.get("result")
    return bool(
        row.get("error")
        or isinstance(result, dict)
        and (
            result.get("error")
            or result.get("status") == "failed"
            or result.get("exit_code") not in (None, 0)
        )
    )


def execution_receipts(rows: list[dict]) -> list[dict]:
    receipts = []
    for row in rows:
        result = row.get("result") or {}
        state = (
            "failed"
            if receipt_failed(row)
            else "read-only"
            if row.get("tool") in READ_ONLY_TOOLS
            else "operation recorded; task outcome unverified"
        )
        if isinstance(result, dict) and not receipt_failed(row):
            if result.get("status") in ("queued", "already scheduled"):
                state = "queued; not execution"
            elif result.get("written") and not result.get("published"):
                state = "saved; publication unconfirmed"
        receipts.append({"tool": row.get("tool"), "state": state, "result": result})
    return receipts


def operator_context(root: Path, message: str, state: dict) -> dict:
    """Keep authenticated instructions verbatim; archive oversized host snapshots."""
    from rlm.v100.research_tools import bounded_tool_result

    result = {"message": message}
    for key, value in state.items():
        result[key] = json.loads(bounded_tool_result(root, value, 2048))
    return result


def respond(
    root: Path,
    directory: Path,
    request: dict,
    accepted_cpu=None,
    *,
    skip_local=False,
    lifecycle_snapshot=None,
) -> dict:
    from rlm.v100.agent import native_turn
    from rlm.v100.competition import helper_client
    from rlm.v100.mission import status
    from rlm.v100.mission_evidence import collect as collect_evidence
    from rlm.v100.planning import read as read_plans
    from rlm.v100.planning import update as update_plan
    from rlm.v100.research_policy import settings

    # Authorization comes only from the current authenticated local user message,
    # never from a model-generated action or text retrieved from the internet.
    message = request["message"].strip()
    from rlm.v100 import chat_plan

    no_tools = explicit_no_tools(message)
    quick_question = no_tools
    allow_long_goal = False
    command_verification = False
    if not no_tools:
        if chat_plan.requested(message):
            return chat_plan.respond(root)
        command_verification = bool(
            re.fullmatch(
                r"Execute a read-only check: use get_plan once, then report the recorded goal\. Do not change plans, files, preferences, models or schedule work\. Command verification ID: [a-f0-9]{40}",
                message,
            )
        )
        if message.startswith("/approve-web "):
            from rlm.v100.web_operations import approve

            receipt = approve(root, message.removeprefix("/approve-web ").strip())
            return {
                "answer": json.dumps(receipt, ensure_ascii=False),
                "actions": [],
                "applied": [receipt],
                "responder": {"model": "controller-web-approval"},
            }
        from rlm.v100 import resource_optimization

        resource_context = resource_optimization.context(root, request)
        if resource_context is not None:
            return resource_optimization.apply(root, message, resource_context)
        from rlm.v100 import chat_progress

        if chat_progress.requested(message):
            return chat_progress.respond(root, message)
        if trade_visualization_requested(message):
            return trade_visualization_reply(root)
        from rlm.v100 import chat_facts, income_policy, market_research
        from rlm.v100.chat_goals import authorizes_long

        if re.search(
            r"backtest|drawdown|odchyl|obsuni|glitch", income_policy.normalized(message)
        ) and re.search(r"audit|sprawdz|zbadaj|check", income_policy.normalized(message)):
            return chat_facts.audit_response(root, message)
        if (
            not authorizes_long(message)
            and not message.startswith(("/goal ", "/cel "))
            and income_policy.setting(message) is not None
        ):
            receipt = income_policy.update(root, message)
            return {
                "answer": (
                    "Zapisano badanie wzrostu hipotetycznego kapitału w paper. Zlecono porównanie strategii; wynik i koszty muszą być zmierzone."
                    if chat_facts.language(message) == "Polish"
                    else "Hypothetical-capital research saved. Strategy comparison commissioned; returns and costs must be measured."
                ),
                "actions": [],
                "applied": [receipt],
                "responder": {"model": "controller-research-policy"},
            }
        if market_research.requested(message):
            return chat_facts.test_response(root, message)
        if conversation_question(message) and chat_facts.requested(message):
            return chat_facts.response(root, message)
        quick_question = conversation_question(message) or message.casefold() in (
            "hej",
            "cześć",
            "czesc",
            "hello",
            "hi",
            "odpowiedz",
        )
        if re.search(
            r"\b(?:co potrafisz|co umiesz|twoje możliwości|twoje mozliwosci|what can you do)\b",
            message.casefold(),
        ):
            return {
                "answer": "\n".join(
                    [
                        "- Badam publiczne źródła i zapisuję dowody pod aktualny cel.",
                        "- Zlecam ograniczone zadania CPU/RTX i sprawdzam ich wyniki; dostępność zasobów nie oznacza wykonania zadania.",
                        "- Uruchamiam kod w izolowanej VM oraz edytuję i waliduję dashboard.",
                        "- Rejestruję prognozy przed wynikiem i weryfikuję późniejsze etykiety do ML.",
                        "- Trenuję kandydatów z zabezpieczeniami pamięci; nowe wagi wymagają niezależnych testów.",
                        "- Pokazuję akcje, plany, wyniki, blokady i zmierzone tokeny w dashboardzie.",
                        "- Zmieniam cel na wyraźne polecenie operatora. Nie składam realnych zleceń ani nie gwarantuję zysku.",
                        "- Aktualny postęp odczytasz przez /status lub pytanie o postęp; brak danych pokazuję jako nieznany.",
                    ]
                ),
                "actions": [],
                "applied": [],
                "responder": {"model": "controller-capabilities"},
                "scope": "Implemented mechanisms; not proof of current execution or accepted improvement",
            }
        from rlm.v100 import chat_progress

        if chat_progress.requested(message) or chat_progress.continue_requested(message):
            result = chat_progress.respond(root, message)
            if chat_progress.continue_requested(message):
                result["answer"] = (
                    "Kontynuacja dotychczasowego celu. Nie zmieniam planu na edycję dashboardu. "
                    "Działająca misja kontynuuje swoją kolejkę; trening wymaga zweryfikowanych danych i bramek jakości. "
                    "Ta odpowiedź nie uruchamia zatrzymanej misji ani nie potwierdza nowych wag.\n"
                    + result["answer"]
                )
            return result
        if message.strip() == "/reset-dashboard" or re.fullmatch(
            r"(?:reset|restore|przywroc|przywróć) (?:the )?(?:default|domyslny|domyślny) (?:dashboard|raport html)[.! ]*",
            message.strip(),
            re.I,
        ):
            from rlm.v100.dashboard_editor import status as layout_status
            from rlm.v100.dashboard_editor import write as layout_write
            from rlm.v100.dashboard_layout import BASE_TEMPLATE

            written = layout_write(root, BASE_TEMPLATE)
            publication = layout_status(root)
            return {
                "answer": (
                    "Zapisano zweryfikowany widok statusu, agentów, zasobów, raportów i dowodów. "
                    "Poprzedni HTML zachowano w kopii. Dane odświeża renderer hosta. "
                    + (
                        "Publikacja potwierdzona."
                        if publication.get("published")
                        else "Host nie potwierdził jeszcze publikacji; sprawdź dashboard_status."
                    )
                ),
                "actions": [],
                "applied": [written],
                "dashboard": publication,
                "responder": {
                    "model": "controller-dashboard",
                    "delegated_while_master_busy": False,
                },
            }
        if re.fullmatch(
            r"(?:hej|czesc|cześć|hello|hi|witaj)(?:\s+(?:synta|master|v100))?[!.,\s]*",
            message,
            re.I,
        ):
            return {
                "answer": "Cześć! Jestem dostępny. Co mam sprawdzić lub wykonać?"
                if chat_facts.language(message) == "Polish"
                else "Hello. What should I inspect or execute?",
                "actions": [],
                "applied": [],
                "responder": {"model": "controller-chat", "delegated_while_master_busy": False},
            }
        if message.startswith(("/goal ", "/cel ")):
            receipt = update_plan(root, "long", message.split(" ", 1)[1], "user")
            return {
                "answer": "Zmieniono cel długoterminowy. Następna runda przeplanuje zadania.",
                "actions": [],
                "applied": [receipt],
            }

        from rlm.v100.chat_goals import authorizes_long, direct_plan

        direct = direct_plan(message)
        if direct is not None:
            horizon, objective = direct
            receipt = update_plan(root, horizon, objective, "user")
            return {
                "answer": "Zapisano cel/plan: " + objective + ". Następna runda uwzględni zmianę.",
                "actions": [],
                "applied": [receipt],
                "responder": {"model": "controller-goals", "delegated_while_master_busy": False},
            }
        allow_long_goal = authorizes_long(message)

        if message == "/status":
            from rlm.v100.progress import report

            value = report(root)
            return {
                "answer": json.dumps(
                    {
                        key: value[key]
                        for key in (
                            "running",
                            "phase",
                            "completed_learning_cycles",
                            "accepted_weight_updates_this_run",
                            "last_learning_cycle",
                            "drones",
                            "desktop",
                            "official_benchmark",
                        )
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "actions": [],
                "applied": [],
                "responder": {"model": "controller-status", "delegated_while_master_busy": False},
            }

    mission = status(root)
    if lifecycle_snapshot is not None:
        # Use the exact profile snapshots whose endpoints were leased before
        # this turn. Re-reading mutable selection files here could redirect
        # generation to a port that was never included in the lease set.
        profile = copy.deepcopy(lifecycle_snapshot["accepted"])
    else:
        paths = [directory / "serving-active.json", directory / "learning/live.json"]
        current = mission.get("state", {}).get("live_profile")
        if current:
            paths.append(Path(current))
        paths.extend(sorted(directory.glob("profile-*.json")))
        path = next((p for p in paths if p.exists()), None)
        if path is None:
            raise BackendNotReady("Waiting for owned inference server")
        profile = copy.deepcopy(load_profile(path, root))
    accepted_profile = copy.deepcopy(profile)
    if lifecycle_snapshot is not None:
        planner = lifecycle_snapshot.get("planner")
    else:
        from rlm.v100.model_roles import selected_profile

        planner = selected_profile(root, "planner")
    if planner is not None and preferences(root).get("remote_helper_enabled", True):
        try:
            chosen = (
                copy.deepcopy(planner)
                if lifecycle_snapshot is not None
                else load_profile(planner, root)
            )
            if not chat_profile_origin_enabled(root, chosen):
                raise BackendNotReady("Optional planner origin disabled by Synta preferences")
            if skip_local:
                from rlm.v100.remote_helper import remote_profile

                if not remote_profile(chosen):
                    raise BackendNotReady("Local planner endpoint is in a serving transition")
            probe = helper_client(chosen, root)
            probe.timeout = min(probe.timeout, 5)
            backend_preflight(probe)
            profile = copy.deepcopy(chosen)
        except (OSError, ValueError, RuntimeError, requests.RequestException) as error:
            # No generation/tool execution has started; retain accepted local master.
            ActivityLog(root, "controller", "chat").write(
                "errors",
                "planner-preflight-rejected",
                {
                    "error": str(error)[:300],
                    "scope": "Rejected planner unused; accepted backend still requires validation",
                },
            )
    profile["runtime"].update(enable_thinking=False, max_output_tokens=2048, max_timeout=120)
    client = helper_client(profile, root)
    client.activity_actor = "chat"
    # The endpoint is reused for candidate evaluation. Chat waits rather than
    # talking to an unaccepted candidate or opening another GPU server.
    client.request_deadline = request.get("deadline", time.monotonic() + 180)
    generation_timeout = min(getattr(client, "timeout", 120), 60)
    client.timeout = min(generation_timeout, 10)
    delegated = False
    cpu_master = False
    cpu_detail = None
    try:
        from rlm.v100.remote_helper import remote_profile

        if remote_profile(profile):
            backend_preflight(client)
            delegated = True
        elif skip_local:
            raise BackendNotReady("Accepted local endpoint is in a serving transition")
        else:
            # The shared local port can temporarily serve a training candidate.
            # Verify its protected PID, artifact hashes, execution identity and
            # reported model before any user chat or tool call reaches it.
            assert_served_expert(client, profile, root)
    except (requests.RequestException, OSError, ValueError) as serving_error:
        profile = copy.deepcopy(accepted_profile)
        if accepted_cpu is not None and profile.get("resources", {}).get("accepted_cpu_chat"):
            try:
                accepted_cpu.deadline = request.get("deadline", time.monotonic() + 180)
                client = accepted_cpu.get(profile)
                generation_timeout = min(client.timeout, 60)
                cpu_master = True
            except (ValueError, OSError, RuntimeError, requests.RequestException) as error:
                cpu_detail = str(error)[:300]
                atomic_json(
                    directory / "accepted-cpu-chat-status.json",
                    {"available": False, "detail": cpu_detail},
                )
        if not cpu_master:
            if not preferences(root).get("remote_helper_enabled", True):
                raise BackendNotReady(str(serving_error)) from serving_error
            if not profile.get("resources", {}).get("interactive_lab"):
                raise BackendNotReady(str(serving_error)) from serving_error
            from rlm.v100.remote_helper import remote_profile, selected_helper
            from rlm.v100.remote_nodes import selected_node

            candidates = [
                selected_node(root),
                selected_helper(root, "planner"),
                selected_helper(root, "fast"),
            ]
            errors = ["Accepted CPU: " + cpu_detail] if cpu_detail else []
            for candidate in dict.fromkeys(p for p in candidates if p is not None):
                try:
                    helper = load_profile(candidate, root)
                    if not remote_profile(helper) or not chat_profile_origin_enabled(root, helper):
                        continue
                    available = helper_client(helper, root)
                    available.request_deadline = request.get("deadline", time.monotonic() + 180)
                    generation_timeout = min(available.timeout, 60)
                    available.timeout = min(generation_timeout, 10)
                    backend_preflight(available)
                    client = available
                    break
                except (OSError, ValueError, RuntimeError, requests.RequestException) as error:
                    errors.append(str(error)[:240])
            else:
                raise BackendNotReady(
                    "Accepted local backend and qualified helpers unavailable: " + "; ".join(errors)
                )
            client.sampling_args["max_tokens"] = 2048
            client.enable_thinking = False
            client.activity_actor = "chat-delegate"
            delegated = True
    client.request_deadline = request.get("deadline", time.monotonic() + 180)
    client.timeout = generation_timeout
    client.enable_thinking = False
    client.sampling_args = dict(client.sampling_args)
    client.sampling_args["max_tokens"] = min(2048, client.sampling_args.get("max_tokens", 2048))
    client.activity_context = {**client.activity_context, "chat_request_id": request.get("id")}
    from rlm.v100 import chat_resources

    resource_request = None if no_tools else chat_resources.instruction(message)
    question_only = no_tools or (quick_question and not resource_request)
    if command_verification:
        return verified_plan_response(
            root,
            client,
            {
                "model": client.model_name,
                "delegated_while_master_busy": delegated,
                "accepted_master_on_cpu": cpu_master,
                "accepted_model_sha256": accepted_cpu.sha256 if cpu_master else None,
            },
        )
    if no_tools:
        return no_tools_response(
            client,
            message,
            read_plans(root) if recorded_context_requested(message) else {},
            {
                "model": client.model_name,
                "delegated_while_master_busy": delegated,
                "accepted_master_on_cpu": cpu_master,
                "accepted_model_sha256": accepted_cpu.sha256 if cpu_master else None,
            },
        )
    if question_only:
        return conversational_response(
            root,
            client,
            message,
            mission,
            read_plans(root),
            {
                "model": client.model_name,
                "delegated_while_master_busy": delegated,
                "accepted_master_on_cpu": cpu_master,
                "accepted_model_sha256": accepted_cpu.sha256 if cpu_master else None,
                "cpu_admission_detail": cpu_detail,
            },
            generation_timeout=60 if cpu_master else 20,
        )
    # Old requests are context only for an explicit referential follow-up.
    # A standalone new request must never inherit another request's work.
    history = []
    if re.fullmatch(r"(?:to |okej |ok )?(?:zrob to|zrób to|wykonaj to|do it)[!. ]*", message, re.I):
        with connect(root) as db:
            history = db.execute(
                "SELECT message,response FROM requests WHERE state='completed' ORDER BY rowid DESC LIMIT 1"
            ).fetchall()
    messages = [
        {
            "role": "system",
            "content": (
                "You are Synta, the operator-owned goal-directed research and learning system. Answer in the current authenticated local user message language. Generated internal plans, hypotheses and tool arguments must be English; literal operator quotations remain verbatim. Return answer and explicit requested actions. "
                "An ordinary question needs no actions. Interpret clear goal-setting requests in ordinary language; slash commands are optional. "
                "The current user message overrides previous conversational context. Answer it directly. Use complete sentences under 2000 characters. Do not finish mid-sentence to fit a schema. Archived replay does not prove vendor data accuracy or the cause of losses. Do not assert falling knives, volatility or Z-score causes without a measured execution trace. "
                "Use goal_long ONLY if included in the response schema and ONLY when the CURRENT message explicitly asks "
                "to set/change the main or long-term objective. Its text must be a literal substring of that message, "
                "never invented or drawn from chat history, a tool or a quote. Questions, negations and examples do not change goals. "
                "If intent is ambiguous, answer or ask which horizon without changing the long-term goal. Use "
                "plan_mid for the medium-term plan and plan_short for the next tasks. Steer R&D using directive. Do not change the "
                "system prompt, quality gates or money ledger. Enable local paper alerts using alerts; the rule steers "
                "future R&D but is not a guaranteed executable condition. Budget selects master/helper thinking and "
                "tokens (master256..8192/helper256..4096). Helper batch16; master128/256/512 is a benchmark proposal, "
                "not an immediate native reconfiguration. No real trading, arbitrary host commands or quota bypass. "
                "Describe missing capabilities honestly. Actions apply only after host validation; do not claim "
                "weights changed, profit learned or settings executed before the host receipt. "
                "Use mission_evidence for factual training/status claims; cite its report paths. Missing counters mean unknown, "
                "not zero. Global/attempted steps differ from successful optimizer updates, which differ from accepted weight versions. "
                "Never invent a grader error cause. GUI readiness does not block calculator/native inference/GPU training. "
                "Arithmetic practice is optional; do not claim it is necessary for income or that it proves a profitable strategy. "
                "Unused action fields: text empty, target master, thinking false, max_tokens256, batch_tokens128, enabled false."
                " Use sandbox action to queue a shell script (text) inside the private Debian VM, never the host. "
                "The guest has /workspace for persistent files and /opt/master-source as readonly own source; "
                "it may copy source, install tools and download files within its resource budget. "
                "The work continues independently; report the queued job ID instead of claiming it already ran. "
                "For requested dashboard HTML edits, prefer read_dashboard and write_dashboard to read, edit and verify "
                "/workspace/dashboard/index.html before reporting completion. Preserve required IDs. "
                "No scripts, event handlers or external resources: the host supplies live rendering. Removing guest scripts does NOT disable host refresh or counters. Never treat a previous assistant promise as an executed change. "
                "Use dashboard_status before claiming publication. A cat command only reads; it does not edit. GUI observation is unnecessary for HTML edits. "
                "If vision is deferred, continue file work without waiting for an image."
                " For a concrete implementation request, execute allowed tools now and verify their receipts. "
                "A plan alone does not implement the request. If blocked, name the exact error and unfinished work. "
                "read_dashboard returns invalid source for editing; use repair_dashboard to safely repair it with backup. "
                "Never ask the user to paste guest HTML merely because it failed publication validation. "
                "Use capabilities to inspect named operations and their scope; never invent missing operations. Public internet reads and guest browser/shell operations are available in the sandbox. Use inspect_trade_costs to investigate execution losses and separate entry friction from later price changes. Use inspect_web_page/open_web_page to inspect public pages and forms, and prepare_web_submission to create a reviewable POST. Only the operator can approve the exact draft with /approve-web ID. Never claim that sandbox isolation means internet actions are impossible, or that a prepared draft was sent."
            ),
        }
    ]
    for old in reversed([] if resource_request else (history[:1] if question_only else history)):
        messages += [
            {"role": "user", "content": old["message"][:1500]},
            {"role": "assistant", "content": json.loads(old["response"])["answer"][:1500]},
        ]
    if resource_request:
        messages[0]["content"] += (
            " CURRENT REQUEST: allocate operator-owned RTX/Windows CPU resources for the EXISTING goal. "
            "Do not change short/mid/long plans, invent trading strategies, edit HTML, run GUI or submit placeholder scripts. "
            "Read compute_resources and get_plan if needed. Use schedule_drone for a useful concrete bounded researcher/critic/source task, "
            "or propose_compute_trial only for a genuinely useful proof-domain architecture experiment. "
            "Report actual tool job IDs/states; do not claim work completed before independent receipts. "
            "A worker uses its own RAM/disk; it does not enlarge Debian RAM or V100 VRAM."
        )
    messages.append(
        {
            "role": "user",
            "content": json.dumps(
                operator_context(
                    root,
                    request["message"],
                    {
                        "mission": mission,
                        "mission_evidence": collect_evidence(root),
                        "settings": settings(root),
                        "preferences": preferences(root),
                        "plans": read_plans(root),
                    },
                ),
                ensure_ascii=False,
            ),
        }
    )
    dashboard_request = any(
        word in message.casefold() for word in ("dashboard", "raport html", "pulpit html")
    )
    dashboard_before = {}
    if dashboard_request:
        from rlm.v100.dashboard_editor import status as dashboard_status

        dashboard_before = dashboard_status(root)
    # Tool permission belongs to the accepted operator configuration, not the
    # serving role profile (which only declares model/transport resources).
    if accepted_profile.get("resources", {}).get("interactive_lab"):
        from rlm.v100.research_tools import research_turn

        client.research_owner = "A"
        knowledge_task_id = None
        client.research_tool_names = {
            "inspect_trade_costs",
            "inspect_web_page",
            "open_web_page",
            "prepare_web_submission",
            "read_public_page",
            "backtest_prices",
            "audit_backtests",
            "paper_status",
            "paper_observed_results",
            "morph_model",
            "propose_ngram_memory",
            "managed_upgrade_status",
            "request_model_preparation",
            "test_submodel",
            "income_opportunities",
            "register_income_opportunity",
            "capabilities",
            "repair_dashboard",
            "compute_resources",
            "schedule_drone",
            "cancel_drone",
            "propose_colab_trial",
            "propose_compute_trial",
            "compute_trial_status",
            "cancel_compute_trial",
            "dashboard_status",
            "read_dashboard",
            "write_dashboard",
            "goal_learning_status",
            "observe_goal_source",
            "predict_goal_pattern",
            "mission_evidence",
            "read_tool_result",
            "get_plan",
            "drone_status",
            "sandbox_state",
            "sandbox_run",
            "read_master_code",
            "sandbox_gui",
            "search_knowledge",
            "search_memory",
            "read_source",
        }
        if question_only:
            client.research_tool_names = READ_ONLY_TOOLS
        if resource_request:
            client.research_tool_names = chat_resources.RESOURCE_TOOLS
            client.resource_only_chat = True
        if command_verification:
            client.research_tool_names = {"get_plan"}
        research = research_turn(client, messages, schema(allow_long_goal), root)
        knowledge_task_id = research.get("knowledge_task_id")
        tool_receipts = research.get("research_trace", [])
        result = json.loads(research["content"])
    else:
        knowledge_task_id = None
        tool_receipts = []
        result = json.loads(
            native_turn(
                client,
                messages,
                response_format={"type": "json_object", "schema": schema(allow_long_goal)},
            )["content"]
        )
    if (question_only or command_verification) and result.get("actions"):
        raise ValueError("Read-only chat question cannot execute unsolicited actions")
    if set(result) != {"answer", "actions"} or not isinstance(result["answer"], str):
        raise ValueError("Invalid chat response")
    if not answer_complete(result["answer"]):
        result["actions"] = []
        result["answer"] = repair_answer(
            client,
            message,
            {
                "tool_receipts": [
                    {
                        "tool": row.get("tool"),
                        "state": "failed"
                        if receipt_failed(row)
                        else "recorded; outcome unverified",
                        "receipt": {
                            key: (row.get("result") or {}).get(key)
                            for key in ("id", "state", "status", "error", "report", "published")
                        },
                    }
                    for row in tool_receipts
                ],
                "mission": mission,
            },
        )
        result["answer_recovered"] = True
    if resource_request:
        result["actions"], rejected = chat_resources.filter_actions(result["actions"])
        evidence = chat_resources.status(root)
        result["answer"] = chat_resources.answer(evidence, tool_receipts, rejected)
        result["resource_contract"] = {"evidence": evidence, "rejected_actions": rejected}
    implementation_requested = bool(
        re.search(
            r"\b(?:zrob|zrób|wprowadz|wprowadź|wdroz|wdroż|zaimplementuj|napraw)\b",
            message.casefold(),
        )
    )
    if (
        implementation_requested
        and not any(
            row.get("tool") not in READ_ONLY_TOOLS and not receipt_failed(row)
            for row in tool_receipts
        )
        and result["actions"]
        and all(action.get("kind") in ("plan_short", "plan_mid") for action in result["actions"])
    ):
        result["actions"] = []
        result["answer"] = (
            "Nie wykonano żądanej implementacji: model zwrócił tylko plan, bez potwierdzenia narzędzi. "
            "Nie zapisano go jako wykonanego zadania ani nie zmieniono planów."
        )
    result["execution_receipts"] = execution_receipts(tool_receipts)
    if tool_receipts and all(receipt_failed(row) for row in tool_receipts):
        result["actions"] = []
        result["execution_status"] = "failed"
        result["answer"] = "Requested work failed: " + "; ".join(
            str(row.get("result")) for row in tool_receipts
        )
    result["tool_receipts"] = tool_receipts
    if knowledge_task_id:
        result["knowledge_task_id"] = knowledge_task_id
    result["applied"] = apply_actions(root, result["actions"], message)
    if dashboard_request or any(
        row["tool"] in ("write_dashboard", "repair_dashboard") for row in tool_receipts
    ):
        from rlm.v100.dashboard_editor import status as dashboard_status

        receipt = dashboard_status(root)
        result["dashboard"] = receipt
        if not receipt["valid"]:
            result["answer"] = "Zmiana dashboardu nie została zaakceptowana: " + receipt["error"]
        elif not receipt["published"]:
            result["answer"] = (
                "HTML jest poprawny, ale host nie potwierdził jeszcze jego publikacji."
            )
        elif receipt["guest_sha256"] == dashboard_before.get("guest_sha256") and any(
            word in message.casefold()
            for word in ("dod", "zmien", "zmień", "edyt", "uaktual", "zwizual")
        ):
            result["answer"] = (
                "Nie potwierdzono zmiany HTML podczas tej prośby. Host nadal pokazuje poprzedni poprawny układ."
            )
    result["responder"] = {
        "model": client.model_name,
        "delegated_while_master_busy": delegated,
        "accepted_master_on_cpu": cpu_master,
        "accepted_model_sha256": accepted_cpu.sha256 if cpu_master else None,
        "cpu_admission_detail": cpu_detail,
    }
    result["scope"] = (
        "Weights change only through independently tested training; CPU chat uses accepted weights, never a training candidate"
    )
    return result


def resolve_failed_status_questions(root: Path) -> dict:
    """Answer failed read-only status questions from receipts; never replay tools."""
    from rlm.v100 import chat_progress

    with connect(root) as db:
        rows = db.execute(
            "SELECT * FROM requests WHERE state='failed' AND response IS NULL ORDER BY rowid DESC LIMIT 32"
        ).fetchall()
    resolved = []
    for row in rows:
        if not chat_progress.requested(row["message"]):
            continue
        response = chat_progress.respond(root, row["message"])
        response["previous_error"] = row["error"]
        response["scope"] = "Fresh read-only status after failed inference; no actions replayed"
        with connect(root) as db:
            changed = db.execute(
                "UPDATE requests SET state='completed',response=?,error=NULL,updated=? WHERE id=? AND state='failed' AND response IS NULL",
                (json.dumps(response, ensure_ascii=False), time.time(), row["id"]),
            ).rowcount
        if changed:
            resolved.append(row["id"])
            ActivityLog(root, "controller", "chat").write(
                "decisions",
                "status-question-recovered",
                {"id": row["id"], "previous_error": row["error"], "scope": response["scope"]},
            )
        if len(resolved) >= 8:
            break
    return {"resolved_read_only_questions": resolved, "actions_replayed": 0}


def reopen_unavailable_questions(root: Path) -> dict:
    from rlm.v100.chat_goals import normalized

    reopened = []
    with connect(root) as db:
        rows = db.execute(
            "SELECT * FROM requests WHERE state='failed' AND response IS NULL AND error LIKE '%adaptive guard unavailable%' ORDER BY rowid DESC LIMIT 32"
        ).fetchall()
        for row in rows:
            text = normalized(row["message"])
            if "?" not in text or not re.match(r"^(czy|mozesz|can|are|is|do you)\b", text):
                continue
            if re.search(
                r"\b(zrob|wykonaj|kup|sprzedaj|wyslij|send|execute|register|delete|update|ustaw|zmien|trenuj)\b",
                text,
            ):
                continue
            atomic_json(root / "research/chat-recovery" / (row["id"] + ".json"), dict(row))
            db.execute(
                "UPDATE requests SET state='queued',attempts=0,wait_started=0,retry_after=0,error=NULL,updated=0 WHERE id=? AND state='failed'",
                (row["id"],),
            )
            reopened.append(row["id"])
    result = {
        "reopened_questions": reopened,
        "scope": "Only plain read-only questions; prior failure archived. Commands and tool execution are not replayed.",
    }
    atomic_json(root / "research/chat-recovery/status.json", result)
    return result


def chat_profile_origin_enabled(root: Path, profile: dict) -> bool:
    """Skip disabled optional hosts before constructing clients or acquiring leases."""
    from rlm.v100.remote_helper import origin_enabled, remote_profile

    return not remote_profile(profile) or origin_enabled(root, profile["runtime"]["base_url"])


def _chat_lifecycle_profiles(root: Path, directory: Path, accepted_cpu=None) -> dict:
    """Resolve immutable profile snapshots and every local endpoint for a turn."""
    from rlm.v100.mission import status

    mission = status(root)
    paths = [directory / "serving-active.json", directory / "learning/live.json"]
    current = mission.get("state", {}).get("live_profile")
    if current:
        paths.append(Path(current))
    paths.extend(sorted(directory.glob("profile-*.json")))
    selected = next((path for path in paths if path.exists()), None)
    if selected is None:
        return {"accepted": None, "planner": None, "profiles": []}
    from rlm.v100.common import load_profile

    accepted = load_profile(selected, root)
    planner_profile = None
    profiles = [accepted]
    if preferences(root).get("remote_helper_enabled", True):
        from rlm.v100.model_roles import selected_profile

        planner = selected_profile(root, "planner")
        if planner is not None:
            try:
                chosen = load_profile(planner, root)
                if chat_profile_origin_enabled(root, chosen):
                    planner_profile = chosen
                    profiles.append(planner_profile)
            except (OSError, ValueError):
                # respond() records and ignores an unusable optional planner.
                pass
    if accepted_cpu is not None and accepted.get("resources", {}).get("accepted_cpu_chat"):
        from rlm.v100.chat_backend import cpu_profile

        profiles.append(cpu_profile(accepted))
    return {"accepted": accepted, "planner": planner_profile, "profiles": profiles}


@contextmanager
def _chat_endpoint_leases(root: Path, directory: Path, accepted_cpu=None, *, profiles=None):
    """Hold shared endpoint leases from before identity checks through all tools/recovery."""
    from rlm.v100.serving import ServingLeaseUnavailable, endpoint_lease, local_endpoint

    profiles = (
        profiles
        if profiles is not None
        else _chat_lifecycle_profiles(root, directory, accepted_cpu)["profiles"]
    )
    endpoints = {}
    for profile in profiles:
        endpoint = local_endpoint(profile)
        if endpoint is not None:
            endpoints[endpoint] = profile
    stack = ExitStack()
    try:
        # Identical ordering across chat processes prevents multi-port deadlocks.
        for endpoint in sorted(endpoints):
            stack.enter_context(
                endpoint_lease(root, endpoints[endpoint], exclusive=False, timeout=0.25)
            )
    except ServingLeaseUnavailable:
        stack.close()
        yield False
        return
    try:
        yield True
    finally:
        stack.close()


@contextmanager
def _chat_fallback_lease(root: Path, directory: Path, accepted_cpu=None, *, accepted_profile=None):
    """Protect the isolated accepted CPU fallback when the master port is busy."""
    from rlm.v100.serving import endpoint_lease

    if accepted_cpu is None or not accepted_cpu:
        yield
        return
    if accepted_profile is None:
        lifecycle = _chat_lifecycle_profiles(root, directory, accepted_cpu)
        accepted_profile = lifecycle["accepted"]
    profile = accepted_profile
    if not profile.get("resources", {}).get("accepted_cpu_chat"):
        yield
        return
    from rlm.v100.chat_backend import cpu_profile
    from rlm.v100.serving import ServingLeaseUnavailable

    try:
        with endpoint_lease(root, cpu_profile(profile), exclusive=False, timeout=0.25):
            yield
    except ServingLeaseUnavailable as error:
        raise BackendNotReady(str(error)) from error


def _respond_with_lifecycle(root: Path, directory: Path, request: dict, accepted_cpu=None):
    lifecycle = _chat_lifecycle_profiles(root, directory, accepted_cpu)
    if lifecycle["accepted"] is None:
        raise BackendNotReady("No immutable accepted profile was available for this chat turn")
    with _chat_endpoint_leases(
        root, directory, accepted_cpu, profiles=lifecycle["profiles"]
    ) as lease_acquired:
        if lease_acquired:
            return (
                respond(
                    root,
                    directory,
                    request,
                    accepted_cpu,
                    lifecycle_snapshot=lifecycle,
                )
                if accepted_cpu is not None
                else respond(root, directory, request, lifecycle_snapshot=lifecycle)
            )
        with _chat_fallback_lease(
            root, directory, accepted_cpu, accepted_profile=lifecycle["accepted"]
        ):
            return (
                respond(
                    root,
                    directory,
                    request,
                    accepted_cpu,
                    skip_local=True,
                    lifecycle_snapshot=lifecycle,
                )
                if accepted_cpu is not None
                else respond(
                    root,
                    directory,
                    request,
                    skip_local=True,
                    lifecycle_snapshot=lifecycle,
                )
            )


def service(root: Path, directory: Path, stop: threading.Event, accepted_cpu=None) -> None:
    try:
        service_loop(root, directory, stop, accepted_cpu)
    finally:
        if accepted_cpu is not None:
            accepted_cpu.close()


def service_loop(root: Path, directory: Path, stop: threading.Event, accepted_cpu=None) -> None:
    while not stop.wait(1):
        if accepted_cpu is not None:
            accepted_cpu.maintain()
        with connect(root) as db:
            row = db.execute(
                "SELECT * FROM requests WHERE state='queued' AND updated<? AND retry_after<=? ORDER BY rowid LIMIT 1",
                (time.time() - 15, time.time()),
            ).fetchone()
        if row is None:
            continue
        request = dict(row)
        request["deadline"] = time.monotonic() + 180
        atomic_json(
            root / "research/state/chat-active.json",
            {
                "id": request["id"],
                "pid": os.getpid(),
                "started": time.time(),
                "time_budget_seconds": 180,
                "phase": "processing",
            },
        )
        try:
            response = _respond_with_lifecycle(root, directory, request, accepted_cpu)
            if not isinstance(response.get("answer"), str) or not response["answer"].strip():
                raise ValueError("Empty chat reply rejected; request not completed")
            failed = response.get("execution_status") == "failed"
            with connect(root) as db:
                db.execute(
                    "UPDATE requests SET state=?,response=?,error=?,updated=? WHERE id=? AND state='queued'",
                    (
                        "failed" if failed else "completed",
                        json.dumps(response, ensure_ascii=False),
                        response["answer"][:500] if failed else None,
                        time.time(),
                        request["id"],
                    ),
                )
            ActivityLog(root, "controller", "chat").write(
                "errors" if failed else "decisions",
                "user-command-failed" if failed else "user-command-completed",
                {"id": request["id"], **response},
            )
        except BackendNotReady as error:
            started = request.get("wait_started") or time.time()
            retry_seconds = min(60, 15 * 2 ** min(request["attempts"], 2))
            with connect(root) as db:
                db.execute(
                    "UPDATE requests SET state='queued',wait_started=?,attempts=attempts+1,error=?,updated=?,retry_after=? WHERE id=? AND state='queued'",
                    (
                        started,
                        str(error)[:300],
                        time.time(),
                        time.time() + retry_seconds,
                        request["id"],
                    ),
                )
            ActivityLog(root, "controller", "chat").write(
                "errors",
                "serving-wait",
                {
                    "id": request["id"],
                    "error": str(error),
                    "started": started,
                    "retry_seconds": retry_seconds,
                    "scope": "Saved pending request; no tool execution started. Other requests may proceed.",
                },
            )
        except (requests.ConnectionError, requests.Timeout) as error:
            expired = (
                time.monotonic() >= request["deadline"]
                or isinstance(error, requests.Timeout)
                or request["attempts"] >= 1
            )
            with connect(root) as db:
                db.execute(
                    "UPDATE requests SET state=?,attempts=attempts+1,error=?,updated=? WHERE id=? AND state='queued'",
                    (
                        "failed" if expired else "queued",
                        str(error)[:300],
                        time.time(),
                        request["id"],
                    ),
                )
            ActivityLog(root, "controller", "chat").write(
                "errors",
                "user-command-failed" if expired else "user-command-retry",
                {
                    "id": request["id"],
                    "error": str(error)[:500],
                    "attempts": request["attempts"] + 1,
                    "terminal": expired,
                },
            )
        except Exception as error:
            with connect(root) as db:
                db.execute(
                    "UPDATE requests SET state='failed',error=?,updated=? WHERE id=?",
                    (str(error)[:500], time.time(), request["id"]),
                )
            ActivityLog(root, "controller", "chat").write(
                "errors", "user-command-failed", {"id": request["id"], "error": str(error)[:500]}
            )
        finally:
            atomic_json(root / "research/state/chat-active.json", {"phase": "idle"})


@contextmanager
def alongside(root: Path, directory: Path):
    from rlm.v100.chat_backend import AcceptedCPUChat

    stop = threading.Event()
    accepted_cpu = AcceptedCPUChat(root, directory, stop)
    thread = threading.Thread(
        target=service, args=(root, directory, stop, accepted_cpu), daemon=True
    )
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1)
        if not thread.is_alive():
            accepted_cpu.close()


def wait_reply(root: Path, identity: str, timeout: float = 200, progress=None) -> dict:
    deadline = time.monotonic() + timeout
    next_update = time.monotonic()
    while time.monotonic() < deadline:
        result = inspect(root, identity)
        if result["state"] != "queued":
            return result
        if progress is not None and time.monotonic() >= next_update:
            progress(result)
            next_update = time.monotonic() + 15
        time.sleep(0.5)
    return inspect(root, identity)


def waiting_label(result: dict) -> str:
    active = result.get("processing")
    if active:
        elapsed = max(0, round(time.time() - active["started"]))
        return f"Processing: {elapsed}s elapsed | budget {active.get('time_budget_seconds', '?')}s | request {result['id']}"
    text = f"Queued: position {result.get('queue_position', '?')} | request {result['id']}"
    if result.get("error"):
        text += " | last recorded issue: " + result["error"]
    return text


def chat(root: Path, message: str | None = None) -> None:
    from rlm.v100.chat_progress import requested as progress_requested

    print(
        "\nSYNTA MASTER CHAT\n/exit closes chat only. /status reads mission status. /results reads recent replies. /pending reads the queue.\n"
    )
    monitor = LateReplies(root)
    if message is None:
        for row in recent_requests(root, 32):
            if row["state"] == "queued":
                monitor.track(row["id"])
        threading.Thread(target=monitor.run, daemon=True).start()
    while True:
        try:
            text = message if message is not None else input("\nYOU > ")
        except (EOFError, KeyboardInterrupt):
            break
        if text.strip() == "/exit":
            break
        if text.strip() in ("/results", "/pending"):
            for row in recent_requests(root):
                if text.strip() == "/results" or row["state"] == "queued":
                    print("REQUEST: " + row["id"] + " | " + row["message"], flush=True)
                    display_reply(row)
            if message is not None:
                break
            continue
        if text.strip() != "/status" and not progress_requested(text):
            direct = direct_facts(root, text)
            if direct is not None:
                display_reply(save_direct_reply(root, text, direct))
                if message is not None:
                    break
                continue
        if (
            text.strip() == "/status"
            or text.strip().startswith(("/goal ", "/cel "))
            or progress_requested(text)
        ):
            response = respond(root, root, {"message": text})
            print("Controller>", response["answer"], flush=True)
            if message is not None:
                break
            continue
        identity = submit(root, text)
        with monitor.lock:
            monitor.pending.discard(identity)
        print("\nREQUEST:", identity, flush=True)
        try:
            result = wait_reply(
                root, identity, progress=lambda row: print(waiting_label(row), flush=True)
            )
        except KeyboardInterrupt:
            print("\nChat closed. Request remains queued/processing; mission continues.")
            break
        display_reply(result)
        if result["state"] == "queued":
            monitor.track(identity)
            print("Awaiting eventual reply; /results also retrieves saved answers.")
        if message is not None:
            break

    monitor.stop.set()
