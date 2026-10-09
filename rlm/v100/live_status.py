"""Bounded observed work and device telemetry, never inferred model intentions."""

import json
import time
from pathlib import Path

import requests


def recent_events(root: Path, limit: int = 40) -> list[dict]:
    paths = sorted((root / "research/logs/activity").glob("*/timeline.jsonl"))[-2:]
    events = []
    for path in paths:
        with path.open("rb") as stream:
            offset = max(0, path.stat().st_size - 128 * 1024)
            stream.seek(offset)
            if offset:
                stream.readline()
            for line in stream:
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeError):
                    continue
                if not isinstance(row, dict):
                    continue
                payload = row.get("payload", {})
                if not isinstance(payload, dict):
                    continue
                events.append(
                    {
                        "time": row.get("time"),
                        "actor": row.get("actor"),
                        "branch": row.get("branch"),
                        "kind": row.get("kind"),
                        "tool": payload.get("tool"),
                        "detail": str(payload.get("error", payload.get("status", "")))[:250],
                    }
                )
    return events[-limit:]


def current_chat(root: Path, mission: dict) -> dict:
    path = root / "research/state/chat-active.json"
    if not path.exists():
        return {"phase": "unknown"}
    if path.stat().st_size > 8192:
        raise ValueError("Chat status exceeds read budget")
    value = json.loads(path.read_text())
    if value.get("id") and (not mission.get("running") or value.get("pid") != mission.get("pid")):
        return {"phase": "stale", "detail": "Previous mission's chat status"}
    return value


def snapshot(root: Path, mission: dict, report: dict, gpu: dict) -> dict:
    import psutil

    from rlm.v100.drones import inspect
    from rlm.v100.planning import read

    now = time.time()
    jobs = inspect(root)
    active = [
        {key: row.get(key) for key in ("id", "branch", "kind", "state", "updated", "assignment")}
        for row in jobs
        if row["state"] == "running"
    ]
    drones = report.get("drones", {}) or {}
    drone_age = now - drones.get("updated", 0)
    memory = psutil.virtual_memory()
    cpu = {
        "percent": psutil.cpu_percent(),
        "available_ram_gib": memory.available / 2**30,
        "total_ram_gib": memory.total / 2**30,
        "observed_at": now,
    }
    chat = current_chat(root, mission)
    external = report.get("external_compute", {}) or {}
    workers = external.get("workers", [])
    actors = [
        {
            "id": "master",
            "label": "Master",
            "state": mission.get("state", {}).get("phase", "unknown"),
            "detail": "Current mission phase; chat: " + chat.get("phase", "unknown"),
            "jobs": [chat] if chat.get("id") else [],
            "stale": not mission.get("running"),
        },
        {
            "id": "v100",
            "label": "V100 GPU",
            "state": "measured" if gpu else "unknown",
            "detail": "Entire board telemetry; inference and training are distinct",
            "metrics": gpu,
            "jobs": [],
            "stale": not bool(gpu),
        },
        {
            "id": "cpu",
            "label": "Debian CPU",
            "state": "measured",
            "metrics": cpu,
            "detail": "Entire host telemetry",
            "jobs": [j for j in active if j["kind"] not in ("researcher", "critic", "benchmark")],
            "stale": False,
        },
        {
            "id": "drones",
            "label": "Drony",
            "state": "running" if drones.get("running") else "unknown",
            "detail": f"{len(active)} active jobs; {len([j for j in jobs if j['state'] == 'queued'])} queued",
            "jobs": active,
            "stale": drone_age > 60,
            "age_seconds": round(drone_age),
        },
        {
            "id": "windows-cpu",
            "label": "Windows CPU",
            "state": "reported" if workers else "unknown",
            "detail": "Worker heartbeats, not proof of training or validation",
            "jobs": external.get("jobs", []),
            "workers": workers,
            "stale": not workers or all(w.get("stale", True) for w in workers),
        },
    ]
    rtx = {
        "id": "rtx",
        "label": "RTX helper",
        "state": "unknown",
        "jobs": [j for j in active if j["kind"] in ("researcher", "critic", "benchmark")],
        "detail": "Utilization and power unknown; request pacing is not a hard GPU cap",
        "stale": True,
    }
    from rlm.v100.common import load_profile
    from rlm.v100.competition import helper_client
    from rlm.v100.remote_helper import remote_profile, selected_helper

    try:
        profile = load_profile(selected_helper(root), root)
        if not remote_profile(profile):
            raise ValueError("Remote helper is not configured")
        client = helper_client(profile, root)
        client.timeout = 3
        info = client.remote_request("/api/ps")
        loaded = [
            model for model in info.get("models", []) if model.get("digest") == client.model_digest
        ]
        rtx.update(
            state="loaded" if loaded else "reachable; model not loaded",
            stale=False,
            observed_at=time.time(),
            loaded_models=[
                {k: m.get(k) for k in ("name", "size_vram", "context_length")} for m in loaded
            ],
        )
    except (ValueError, OSError, RuntimeError, requests.RequestException) as error:
        rtx["detail"] += "; " + str(error)[:150]
    if "client" in locals():
        import hashlib

        from rlm.v100.remote_helper import helper_boot_id

        name = hashlib.sha256(client.base_url.encode()).hexdigest()[:16]
        path = root / "research/state" / f"helper-{name}.workload.json"
        if path.exists() and path.stat().st_size < 8192:
            pacing = json.loads(path.read_text())
            remaining = (
                max(0, pacing.get("not_before", 0) - time.monotonic())
                if pacing.get("boot_id") == helper_boot_id()
                else 0
            )
            rtx["cooldown_seconds"] = round(remaining)
            if remaining and not rtx["stale"]:
                rtx["state"] = "cooldown"
    actors.append(rtx)
    return {
        "observed_at": now,
        "actors": actors,
        "goals": read(root),
        "events": recent_events(root),
        "chat": chat,
    }
