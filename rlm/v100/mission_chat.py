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
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import requests

from rlm.v100.activity import ActivityLog, append_locked
from rlm.v100.common import atomic_json, load_profile


class BackendNotReady(requests.ConnectionError):
    """Owned accepted serving endpoint is not ready before generation."""


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
    if "wait_started" not in {row[1] for row in db.execute("PRAGMA table_info(requests)")}:
        db.execute("BEGIN IMMEDIATE")
        if "wait_started" not in {row[1] for row in db.execute("PRAGMA table_info(requests)")}:
            db.execute("ALTER TABLE requests ADD COLUMN wait_started REAL NOT NULL DEFAULT 0")
    db.commit()
    try:
        with db:
            yield db
    finally:
        db.close()


def submit(root: Path, message: str) -> str:
    if not isinstance(message, str) or not message.strip() or len(message) > 8000:
        raise ValueError("Chat needs 1-8000 characters")
    identity = uuid.uuid4().hex
    with connect(root) as db:
        db.execute("BEGIN IMMEDIATE")
        pending = db.execute(
            "SELECT id FROM requests WHERE state='queued' AND message=? ORDER BY rowid LIMIT 1",
            (message,),
        ).fetchone()
        if pending:
            return pending["id"]
        if db.execute("SELECT count(*) FROM requests WHERE state='queued'").fetchone()[0] >= 32:
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
        {**dict(row), "response": json.loads(row["response"]) if row["response"] else None}
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
        re.search(r"wykres|wizualiz|visualiz|chart", text)
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
    from rlm.v100 import chat_facts, chat_progress

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


def conversation_question(message: str) -> bool:
    """Read-only questions take one generation; explicit work retains tools."""
    value = message.casefold().strip()
    return bool(
        re.match(
            r"(?:a\s+)?(?:co|czego|czy|jak|dlaczego|czemu|gdzie|kiedy|what|why|where|when|ej zaczniesz)\b",
            value,
        )
    ) and not bool(
        re.search(
            r"\b(?:zrób|zrob|napraw|popraw|dodaj|edyt|zmień|zmien|uruchom|sprawdź|sprawdz|zbadaj|zaimplement|chcę|chce|możesz|mozesz)\w*",
            value,
        )
    )


def conversational_response(
    root: Path, client, message: str, mission: dict, plans: dict, responder: dict
) -> dict:
    """One actual model request with bounded factual context and no tool routing."""
    goal = (plans.get("long") or {}).get("text", "No recorded goal")
    facts = {
        "mission_running": mission.get("running"),
        "phase": mission.get("state", {}).get("phase"),
        "learning": mission.get("learning"),
        "goal": goal[:3000],
    }
    client.timeout = min(client.timeout, 20)
    client.request_deadline = min(
        getattr(client, "request_deadline", time.monotonic() + 40), time.monotonic() + 40
    )
    reply = client.request(
        "/v1/chat/completions",
        {
            "model": client.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are Synta. Answer in the language of the current user message, briefly. Internal plans and tool text use English. This is read-only chat: do not claim to execute work or update plans. Only the supplied status is verified. Missing optimizer/profit counters are unknown. Explain how the recorded goal relates to the question, without changing it or promising profit. State missing evidence honestly. Verified current facts: "
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
        or not answer.strip()
    ):
        raise ValueError("Conversational model response incomplete; no invented answer accepted")
    return {
        "answer": answer.strip(),
        "actions": [],
        "applied": [],
        "responder": responder,
        "scope": "Single model generation; read-only question, no tool execution or control changes",
    }


READ_ONLY_TOOLS = {
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


def respond(root: Path, directory: Path, request: dict, accepted_cpu=None) -> dict:
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
    if (
        any(word in message.casefold() for word in ("dashboard", "raport html"))
        and re.search(r"\b(?:zrob|zrób|napraw|popraw)\b", message.casefold())
        and not re.search(r"\b(?:nie|bez)\b", message.casefold())
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
            "responder": {"model": "controller-dashboard", "delegated_while_master_busy": False},
        }
    if re.fullmatch(
        r"(?:hej|czesc|cześć|hello|hi|witaj)(?:\s+(?:synta|master|v100))?[!.,\s]*", message, re.I
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
    paths = [directory / "serving-active.json", directory / "learning/live.json"]
    current = mission.get("state", {}).get("live_profile")
    if current:
        paths.append(Path(current))
    paths.extend(sorted(directory.glob("profile-*.json")))
    path = next((p for p in paths if p.exists()), None)
    if path is None:
        raise BackendNotReady("Waiting for owned inference server")
    profile = copy.deepcopy(load_profile(path, root))
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
        actual = client.request("/props")
        if Path(actual["model_path"]).resolve() != Path(profile["server"]["model"]).resolve():
            raise BackendNotReady("Waiting for the accepted serving model")
    except (requests.RequestException, OSError) as serving_error:
        if not preferences(root).get("remote_helper_enabled", True):
            raise BackendNotReady(str(serving_error)) from serving_error
        if not profile.get("resources", {}).get("interactive_lab"):
            raise
        if (
            accepted_cpu is not None
            and not quick_question
            and profile.get("resources", {}).get("accepted_cpu_chat")
        ):
            try:
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
            from rlm.v100.remote_helper import remote_profile, selected_helper

            helper = load_profile(selected_helper(root), root)
            if not remote_profile(helper):
                raise requests.ConnectionError(
                    "Master busy; no independent remote chat helper"
                ) from None
            client = helper_client(helper, root)
            client.request_deadline = request.get("deadline", time.monotonic() + 180)
            generation_timeout = min(client.timeout, 60)
            client.timeout = min(generation_timeout, 20)
            client.identity()
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

    resource_request = chat_resources.instruction(message)
    question_only = quick_question and not resource_request
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
        )
    with connect(root) as db:
        history = db.execute(
            "SELECT message,response FROM requests WHERE state='completed' ORDER BY rowid DESC LIMIT 3"
        ).fetchall()
    messages = [
        {
            "role": "system",
            "content": (
                "You are Synta, the operator-owned goal-directed research and learning system. Answer in the current authenticated local user message language. Generated internal plans, hypotheses and tool arguments must be English; literal operator quotations remain verbatim. Return answer and explicit requested actions. "
                "An ordinary question needs no actions. Interpret clear goal-setting requests in ordinary language; slash commands are optional. "
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
                "Use capabilities to inspect named operations and their scope; never invent missing operations."
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
                {
                    "message": request["message"],
                    "mission": mission,
                    "mission_evidence": collect_evidence(root),
                    "settings": settings(root),
                    "preferences": preferences(root),
                    "plans": read_plans(root),
                },
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
    if profile.get("resources", {}).get("interactive_lab") and not question_only:
        from rlm.v100.research_tools import research_turn

        client.research_owner = "A"
        client.research_tool_names = {
            "read_public_page",
            "backtest_prices",
            "audit_backtests",
            "paper_status",
            "paper_observed_results",
            "morph_model",
            "propose_ngram_memory",
            "test_submodel",
            "income_opportunities",
            "register_income_opportunity",
            "capabilities",
            "repair_dashboard",
            "compute_resources",
            "schedule_drone",
            "cancel_drone",
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
            "search_memory",
            "read_source",
        }
        if resource_request:
            client.research_tool_names = chat_resources.RESOURCE_TOOLS
            client.resource_only_chat = True
        research = research_turn(client, messages, schema(allow_long_goal), root)
        tool_receipts = research.get("research_trace", [])
        result = json.loads(research["content"])
    else:
        tool_receipts = []
        result = json.loads(
            native_turn(
                client,
                messages,
                response_format={"type": "json_object", "schema": schema(allow_long_goal)},
            )["content"]
        )
    if question_only and result.get("actions"):
        raise ValueError("Read-only chat question cannot execute unsolicited actions")
    if set(result) != {"answer", "actions"} or not isinstance(result["answer"], str):
        raise ValueError("Invalid chat response")
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
        result["answer"] = "Requested work failed: " + "; ".join(
            str(row.get("result")) for row in tool_receipts
        )
    result["tool_receipts"] = tool_receipts
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
                "SELECT * FROM requests WHERE state='queued' AND updated<? ORDER BY rowid LIMIT 1",
                (time.time() - 15,),
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
            response = (
                respond(root, directory, request, accepted_cpu)
                if accepted_cpu is not None
                else respond(root, directory, request)
            )
            with connect(root) as db:
                db.execute(
                    "UPDATE requests SET state='completed',response=?,error=NULL,updated=? WHERE id=? AND state='queued'",
                    (json.dumps(response, ensure_ascii=False), time.time(), request["id"]),
                )
            ActivityLog(root, "controller", "chat").write(
                "decisions", "user-command-completed", {"id": request["id"], **response}
            )
        except BackendNotReady as error:
            started = request.get("wait_started") or time.time()
            expired = time.time() - started >= 600 or request["attempts"] >= 19
            with connect(root) as db:
                db.execute(
                    "UPDATE requests SET state=?,wait_started=?,attempts=attempts+1,error=?,updated=? WHERE id=? AND state='queued'",
                    (
                        "failed" if expired else "queued",
                        started,
                        str(error)[:300],
                        time.time(),
                        request["id"],
                    ),
                )
            ActivityLog(root, "controller", "chat").write(
                "errors",
                "serving-wait-expired" if expired else "serving-wait",
                {"id": request["id"], "error": str(error), "started": started},
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
