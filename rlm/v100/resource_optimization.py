"""Execute bounded resource tuning directly, without an LLM choosing an unrelated experiment."""

import fcntl
import json
import re
import time
from pathlib import Path

from rlm.v100.chat_goals import normalized
from rlm.v100.common import atomic_json


def requested(message: str) -> bool:
    text = normalized(message)
    if re.search(
        r"dashboard|\bhtml\b|strateg|backtest|threshold|n-?gram|offload|\bmodel\w*|\blearn\w*|\buczeni\w*|prog\w* transak|(?:ustaw|zmien|set|change)\w*\s+(?:cel|goal|plan)",
        text,
    ):
        return False
    action = re.search(
        r"zrob|rob dalej|optymalizuj|optimize|zwieksz|przyspiesz|zastosuj|implement|improve|increase",
        text,
    )
    subject = re.search(
        r"cpu|gpu|rtx|ram|zasob|resources?|wydajn|performance|optymalizac|optimization", text
    )
    return bool(action and subject)


def followup(message: str) -> bool:
    return bool(
        re.fullmatch(
            r"(?:(?:to|ok|okej|error|blad)[, ]+)?(?:zrob (?:to(?: wszystko)?|wszystko)|wykonaj to|rob dalej|kontynuuj|do (?:it|all of it)|continue)[!. ]*",
            normalized(message),
        )
    )


def context(root: Path, request: dict) -> str | None:
    """Resolve only explicit follow-ups from prior operator messages, never assistant promises."""
    message = request["message"]
    if requested(message):
        return message
    if not followup(message):
        return None
    from rlm.v100.mission_chat import connect

    with connect(root) as db:
        row = db.execute("SELECT rowid FROM requests WHERE id=?", (request.get("id"),)).fetchone()
        rows = db.execute(
            "SELECT message FROM requests WHERE rowid<? ORDER BY rowid DESC LIMIT 8",
            (row[0] if row else 2**63 - 1,),
        ).fetchall()
    for row in rows:
        if followup(row["message"]):
            continue
        return row["message"] if requested(row["message"]) else None
    return None


def apply(root: Path, message: str, original: str) -> dict:
    from rlm.v100.activity import ActivityLog
    from rlm.v100.chat_facts import language
    from rlm.v100.chat_resources import status
    from rlm.v100.goal_compute import propose
    from rlm.v100.resource_budget import available_cpus, local_budget

    path = root / "research/state/resource-policy.json"
    previous = json.loads(path.read_text()) if path.exists() else {"max_cpu_slots": 2}
    policy = {
        "max_cpu_slots": min(16, max(1, (available_cpus() - 1) // 2)),
        "reserve_ram_gib": 4,
        "ram_per_slot_gib": 3,
        "updated": time.time(),
        "source": "explicit operator resource optimization",
        "request": original,
    }
    archive = root / "research/resource-optimization" / str(time.time_ns())
    atomic_json(archive / "before.json", previous)
    atomic_json(path, policy)
    budget = local_budget(root)
    receipts = [
        {
            "tool": "configure_local_resources",
            "result": {
                "state": "applied",
                "policy": str(path),
                "before": previous,
                "after": policy,
                "current_budget": budget,
            },
        }
    ]
    # Inspect existing Windows policy, never weaken game/thermal/pressure guards or change board settings.
    try:
        resources = status(root)
        external = resources["external_compute"]
        if any(
            j.get("state") in ("queued", "running", "imported") for j in external.get("jobs", [])
        ):
            trial = {"state": "waiting", "reason": "Existing owned CPU jobs retain priority"}
        elif any(
            not w.get("stale", True) and w.get("phase") == "idle"
            for w in external.get("workers", [])
        ):
            folder = root / "research/goal-compute"
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / "dispatch.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    trial = {"state": "waiting", "reason": "Goal CPU dispatcher already active"}
                else:
                    trial = propose(root, "A")
        else:
            trial = {"state": "blocked", "reason": "No fresh idle owned CPU worker"}
        receipts.append({"tool": "propose_goal_compute_trial", "result": trial})
    except (OSError, ValueError, RuntimeError) as error:
        resources = {"error": str(error)}
        receipts.append(
            {"tool": "compute_resources", "result": {"status": "failed", "error": str(error)}}
        )
        trial = {"state": "blocked", "reason": str(error)}
    pl = language(message) == "Polish"
    lines = [
        (
            "Zastosowano optymalizację przydziału zasobów."
            if pl
            else "Resource allocation tuning applied."
        ),
        (
            f"Debian: do {budget['max_cpu_slots']} równoległych zadań CPU/sieci zamiast limitu {previous['max_cpu_slots']}; teraz dopuszczono {budget['cpu_slots']}."
            if pl
            else f"Debian: up to {budget['max_cpu_slots']} CPU/network jobs instead of {previous['max_cpu_slots']}; currently admitted {budget['cpu_slots']}."
        ),
        (
            "Limit uwzględnia affinity i quota CPU, po dwa wątki na slot, pozostawia jeden CPU gdy to możliwe oraz 4 GiB RAM rezerwy i 3 GiB na slot. Obciążenie zmniejsza przydział."
            if pl
            else "Admission respects CPU affinity/quota, two threads per slot, one reserved CPU when possible, 4 GiB RAM reserve and 3 GiB per slot. Pressure reduces admission."
        ),
        (
            "Jedno zadanie modelu naraz; osobne sloty obserwatora i CPU zapobiegają blokowaniu jego odświeżania przez kolejkę."
            if pl
            else "One model job at a time; separate observer/CPU capacity prevents queued jobs exhausting the observer pool."
        ),
        (
            "Windows: zachowano limity CPU/RAM oraz ochronę gry, temperatury i obciążenia RTX. Nie zmieniono limitu mocy całej karty."
            if pl
            else "Windows CPU/RAM and RTX game/temperature/pressure guards retained. Board power settings unchanged."
        ),
        ("Zadanie Windows CPU: " if pl else "Windows CPU job: ")
        + str(trial.get("state", trial.get("status", "unknown")))
        + " | "
        + str(trial.get("reason", trial.get("id", "unknown"))),
        (
            "Nie uruchomiono backtestu ani nie zmieniono celu, wag lub parametrów KV cache. Przyspieszenie wymaga pomiaru na Twoim sprzęcie; gotowość GPU nie oznacza wykonanego zadania."
            if pl
            else "No backtest, goal/weight change or KV cache retuning. Speedup requires measurements on your hardware; GPU readiness is not executed work."
        ),
    ]
    result = {
        "answer": "\n".join(lines),
        "actions": [],
        "applied": [receipts[0]["result"]],
        "execution_receipts": receipts,
        "tool_receipts": receipts,
        "resource_contract": {"evidence": resources, "policy": policy},
        "responder": {"model": "controller-resource-optimization"},
        "scope": "Applied concurrency policy; no measured speedup or hardware crash guarantee",
    }
    if trial.get("state") == "queued" and trial.get("id"):
        result["applied"].append({"tool": "propose_goal_compute_trial", **trial})
    atomic_json(archive / "receipt.json", result)
    ActivityLog(root, "controller", "resources").write(
        "decisions", "resource-policy-applied", result
    )
    return result
