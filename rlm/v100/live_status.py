"""Bounded observed work and device telemetry, never inferred model intentions."""

import json
import time
from pathlib import Path

import requests


def public_summary(value: dict | str) -> str:
    """Select public conclusions only; never expose reasoning/thinking token fields."""
    if isinstance(value, str):
        return value[:8000]
    if not isinstance(value, dict):
        return ""
    parts = []
    for key in (
        "answer",
        "hypothesis",
        "rationale",
        "next_test",
        "confidence",
        "status",
        "error",
        "detail",
        "report",
    ):
        if isinstance(value.get(key), (str, int, float)):
            parts.append(f"{key}: {str(value[key])[:2000]}")
    return " | ".join(parts)[:8000]


def recent_events(root: Path, limit: int = 200) -> list[dict]:
    paths = sorted((root / "research/logs/activity").glob("*/timeline.jsonl"))[-2:]
    events = []
    for path in paths:
        with path.open("rb") as stream:
            offset = max(0, path.stat().st_size - 1024 * 1024)
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
                        "summary": public_summary(payload.get("content", {}))
                        or public_summary(payload.get("result", {}))
                        or public_summary(payload),
                        "category": row.get("category"),
                        "request_id": row.get("context", {}).get("request_id"),
                        "usage": {
                            key: value
                            for key, value in (payload.get("usage") or {}).items()
                            if key in ("prompt_tokens", "completion_tokens", "total_tokens")
                            and isinstance(value, int)
                            and not isinstance(value, bool)
                            and value >= 0
                        },
                        "seconds": payload.get("seconds"),
                        "returned_trace": payload.get("returned_trace")
                        if row.get("kind") == "local-model-reasoning"
                        else None,
                        "trace_part": payload.get("part"),
                        "delta_channel": payload.get("channel")
                        if row.get("kind") == "inference-delta"
                        else None,
                        "delta_text": payload.get("text")
                        if row.get("kind") == "inference-delta"
                        else None,
                        "source": str(path),
                        "device": payload.get("device"),
                        "model": payload.get("model"),
                        "task": " | ".join(
                            f"{key}: {str(payload.get('arguments', {}).get(key))[:200]}"
                            for key in ("query", "url", "brief", "kind", "action")
                            if isinstance(payload.get("arguments"), dict)
                            and key in payload["arguments"]
                        ),
                    }
                )
    return events[-limit:]


def agent_views(events: list[dict], jobs: list[dict]) -> list[dict]:
    """Observed agent activity plus labelled declarations, not inferred intentions."""
    views = {}
    for event in events:
        key = (event.get("branch"), event.get("actor"))
        view = views.setdefault(
            key,
            {"label": " / ".join(str(v) for v in key if v), "declaration": "", "result": ""},
        )
        view.update(
            updated=event.get("time"),
            state=event.get("kind"),
            tool=event.get("tool"),
            source=event.get("source"),
        )
        if event.get("task"):
            view["task"] = event["task"]
        if event.get("device"):
            view["device"] = event["device"]
            view["model"] = event.get("model")
        if event.get("delta_text"):
            if view.get("stream_request") != event.get("request_id"):
                view["stream_output"], view["stream_reasoning"] = "", ""
            view["stream_request"] = event.get("request_id")
            key = (
                "stream_reasoning" if event.get("delta_channel") == "reasoning" else "stream_output"
            )
            view[key] = (view.get(key, "") + event["delta_text"])[-16384:]
        if event.get("returned_trace"):
            if event.get("trace_part") == 0:
                view["returned_trace"] = ""
            view["returned_trace"] = view.get("returned_trace", "") + event["returned_trace"]
            view["trace_request"] = event.get("request_id")
        if event.get("usage"):
            view["usage"] = event["usage"]
            view["usage_at"] = event.get("time")
        if event.get("seconds") is not None:
            view["seconds"] = event["seconds"]
        if event.get("category") == "decisions" and event.get("summary"):
            view["declaration"] = event["summary"]
        elif event.get("summary"):
            view["result"] = event["summary"]
    result = list(views.values())[-12:]
    for job in jobs[:16]:
        result.append(
            {
                "label": f"{job['branch']} / {job['kind']} / {job['id']}",
                "state": job["state"],
                "task": job.get("assignment", ""),
                "updated": job.get("updated"),
                "result": public_summary(job.get("result") or {}),
                "declaration": "",
                "source": "research/state/drones.sqlite3",
            }
        )
    return result


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
        "detail": "Jobs eligible for helper; actual inference device is recorded in agent events. Utilization and power unknown; request pacing is not a hard GPU cap",
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
    events = recent_events(root)
    return {
        "observed_at": now,
        "actors": actors,
        "goals": read(root),
        "events": events,
        "agents": agent_views(events, jobs),
        "chat": chat,
    }
