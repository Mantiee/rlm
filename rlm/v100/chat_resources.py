"""Bind resource requests to real allocation receipts, not unrelated chat history."""

import json
import re
import time
from pathlib import Path

import requests

from rlm.v100.chat_goals import normalized
from rlm.v100.common import atomic_json

RESOURCE_TOOLS = {
    "compute_resources",
    "get_plan",
    "mission_evidence",
    "drone_status",
    "propose_compute_trial",
    "compute_trial_status",
    "cancel_compute_trial",
    "schedule_drone",
    "cancel_drone",
    "calculate",
    "goal_learning_status",
}


def instruction(message: str) -> bool:
    text = normalized(message)
    if "?" in text or re.match(r"^(?:czy|jak|why|how|is|are)\b", text):
        return False
    if re.search(r"dashboard|\bhtml\b|pulpit|(?:ustaw|zmien)\w*\s+(?:cel|plan)", text):
        return False
    resource = re.findall(r"\b(?:rtx|windows\w*|cpu|ram\w*|dysk\w*|worker\w*|dron\w*)\b", text)
    return len(set(resource)) >= 2 and bool(
        re.search(r"uzyw|wykorzyst|przydz|deleg|udostep|wspoldziel|shared", text)
    )


def filter_actions(actions: list[dict]) -> tuple[list[dict], list[dict]]:
    allowed = [row for row in actions if row.get("kind") in ("directive", "budget")]
    rejected = [row for row in actions if row not in allowed]
    return allowed, rejected


def status(root: Path) -> dict:
    from rlm.v100.common import load_profile
    from rlm.v100.competition import helper_client
    from rlm.v100.distributed_compute import inspect
    from rlm.v100.drones import inspect as drone_status
    from rlm.v100.remote_helper import remote_profile, selected_helper

    value = {
        "external_compute": inspect(root),
        "drones": drone_status(root),
        "scope": "Owned job execution only. Windows RAM/disk are worker-local resources, not pooled Debian RAM or shared VRAM. No arbitrary Windows shell is exposed.",
    }
    try:
        profile = load_profile(selected_helper(root), root)
        if not remote_profile(profile):
            raise ValueError("Pinned RTX helper is not configured")
        client = helper_client(profile, root)
        client.timeout = 3
        loaded = client.remote_request("/api/ps").get("models", [])
        loaded = [m for m in loaded if m.get("digest") == client.model_digest]
        value["helper"] = {
            "reachable": True,
            "pinned_model_loaded": bool(loaded),
            "context": profile["runtime"]["context_window"],
            "batch": client.helper_batch_tokens,
            "active_request_target_percent": client.helper_duty_percent,
            "scope": "Inference/vision helper, request pacing; not a hard GPU utilization/power cap",
        }
    except (OSError, ValueError, RuntimeError, requests.RequestException) as error:
        value["helper"] = {"reachable": False, "error": str(error)[:300]}
    return value


def answer(evidence: dict, trace: list[dict], rejected: list[dict]) -> str:
    helper = evidence["helper"]
    external = evidence["external_compute"]
    workers = external.get("workers", [])
    active = [w for w in workers if not w.get("stale", True)]
    text = "RTX helper: " + (
        "osiągalny, przypięty model załadowany."
        if helper.get("pinned_model_loaded")
        else "brak potwierdzenia załadowanego modelu."
    )
    text += f"\nWindows CPU: {len(active)} aktualnych workerów. "
    text += "RAM i dysk Windows są używane lokalnie przez workery; nie powiększają RAM ani VRAM Debiana."
    for worker in active:
        text += f"\n{worker.get('name', 'worker')}: {worker.get('phase', 'unknown')}, {worker.get('threads', '?')} wątki, limit RAM {worker.get('ram_limit_gib', '?')} GiB."
    allocations = [
        row for row in trace if row["tool"] in ("propose_compute_trial", "schedule_drone")
    ]
    for row in allocations:
        result = row["result"]
        text += (
            "\nZlecenie: "
            + str(result.get("id", "brak ID"))
            + " | "
            + str(result.get("state", result.get("status", "unknown")))
        )
        if result.get("error"):
            text += " | " + str(result["error"])[:200]
    if not allocations:
        text += "\nW tej odpowiedzi nie przydzielono nowego zadania. Gotowość workera nie oznacza wykonania."
    if rejected:
        text += "\nOdrzucono działania niezwiązane z przydziałem zasobów. Nie zmieniono nimi planów ani HTML."
    text += "\nZlecenie w kolejce nie potwierdza wykonania, przyspieszenia ani aktualizacji wag."
    return text


def repair_completed(root: Path) -> dict:
    """Upgrade-only recovery of misrouted completed resource commands; never overwrite later plans."""
    from rlm.v100.drones import connect as drone_connect
    from rlm.v100.mission import status as mission_status
    from rlm.v100.mission_chat import connect

    if mission_status(root)["running"]:
        raise ValueError("Stop the mission before recovering misrouted chat plans")
    repaired = []
    with connect(root) as db:
        rows = db.execute("SELECT * FROM requests WHERE state='completed'").fetchall()
        for row in rows:
            if not instruction(row["message"]):
                continue
            response = json.loads(row["response"])
            _, rejected = filter_actions(response.get("actions", []))
            if not rejected or response.get("resource_contract"):
                continue
            if not re.fullmatch("[a-f0-9]{32}", row["id"]):
                raise ValueError("Invalid persisted request identity")
            archive = root / "research/state/chat-contract-repairs" / (row["id"] + ".json")
            atomic_json(archive, dict(row))
            plans_path = root / "research/plans/current.json"
            plans = json.loads(plans_path.read_text()) if plans_path.exists() else {}
            restored = []
            for receipt in response.get("applied", []):
                horizon, identity = receipt.get("horizon"), receipt.get("id", "")
                if horizon in ("short", "mid") and re.fullmatch("[a-f0-9]{32}", identity):
                    if plans.get(horizon, {}).get("id") == identity:
                        version = json.loads(
                            (root / "research/plans" / (identity + ".json")).read_text()
                        )
                        plans[horizon] = version.get("previous") or {}
                        restored.append(horizon)
                elif receipt.get("kind") == "desktop":
                    with drone_connect(root) as jobs:
                        current = jobs.execute(
                            "SELECT payload,state FROM jobs WHERE id=?", (identity,)
                        ).fetchone()
                        payloads = {a["text"] for a in rejected if a.get("kind") == "sandbox"}
                        if (
                            current
                            and current["state"] == "queued"
                            and current["payload"] in payloads
                        ):
                            jobs.execute(
                                "UPDATE jobs SET state='cancelled',updated=? WHERE id=?",
                                (time.time(), identity),
                            )
            if restored:
                atomic_json(
                    archive.with_name(row["id"] + "-plans-before.json"),
                    json.loads(plans_path.read_text()),
                )
                atomic_json(plans_path, plans)
            db.execute(
                "UPDATE requests SET state='queued',response=NULL,attempts=0,error=NULL,updated=? WHERE id=?",
                (time.time(), row["id"]),
            )
            repaired.append({"id": row["id"], "restored_plans": restored, "archive": str(archive)})
    return {"requeued_resource_requests": repaired, "long_term_goal_changed": False}
