"""Bounded observed work and device telemetry, never inferred model intentions."""

import json
import time
from datetime import datetime
from pathlib import Path

import requests


def public_summary(value: dict | str) -> str:
    """Select public conclusions only; never expose reasoning/thinking token fields."""
    from rlm.v100.activity import redact

    value = redact(value)
    if isinstance(value, str):
        return value[:8000] + (
            " [Preview only; full recorded action is in the archive]" if len(value) > 8000 else ""
        )
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
        "state",
        "id",
        "value",
        "exit_code",
        "stdout",
        "completion_scope",
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
                        "id": row.get("id"),
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
                        "goal_id": row.get("context", {}).get("operator_goal_id_at_event"),
                        "step_id": row.get("context", {}).get("step_id"),
                        "evidence_ref": {
                            name: payload["result"][name]
                            for name in (
                                "id",
                                "url",
                                "sha256",
                                "body_sha256",
                                "fetched_at",
                                "acquired",
                                "retrieval",
                                "path",
                                "report",
                            )
                            if isinstance(payload.get("result"), dict) and name in payload["result"]
                        },
                        "usage": {
                            key: value
                            for key, value in (payload.get("usage") or {}).items()
                            if key in ("prompt_tokens", "completion_tokens", "total_tokens")
                            and isinstance(value, int)
                            and not isinstance(value, bool)
                            and value >= 0
                        },
                        "seconds": payload.get("seconds"),
                        "finish_reason": payload.get("finish_reason"),
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
                            f"{key}: {str(payload.get('arguments', {}).get(key))}"
                            for key in ("query", "url", "brief", "kind", "action")
                            if isinstance(payload.get("arguments"), dict)
                            and key in payload["arguments"]
                        ),
                    }
                )
    # Token deltas must not push actual tool calls/errors out of the timeline.
    meaningful = [
        event
        for event in events
        if event["kind"] not in ("inference-delta", "local-model-reasoning")
    ][-limit:]
    deltas = [
        event for event in events if event["kind"] in ("inference-delta", "local-model-reasoning")
    ][-limit:]
    return sorted(meaningful + deltas, key=lambda event: str(event.get("time", "")))


def action_events(events: list[dict]) -> list[dict]:
    return [
        event
        for event in events
        if event.get("kind") not in ("inference-delta", "local-model-reasoning")
    ][-200:]


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
            event_id=event.get("id"),
            goal_id=event.get("goal_id"),
        )
        if event.get("kind") == "inference-start":
            view["stream_output"], view["stream_reasoning"], view["returned_trace"] = "", "", ""
            for name in (
                "declaration",
                "result",
                "task",
                "evidence_ref",
                "evidence_at",
                "usage",
                "usage_at",
                "seconds",
                "finish_reason",
                "completeness",
            ):
                view.pop(name, None)
        if event.get("finish_reason"):
            view["finish_reason"] = event["finish_reason"]
            view["completeness"] = (
                "Incomplete: output token limit reached"
                if event["finish_reason"] == "length"
                else "Backend finish reason: " + str(event["finish_reason"])
            )
        if event.get("evidence_ref"):
            view["evidence_ref"] = event["evidence_ref"]
            view["evidence_at"] = event.get("time")
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
    result = list(views.values())
    for view in result:
        if view.get("finish_reason") == "length":
            view["state"] = "incomplete response"
        try:
            view["historical"] = (
                time.time() - datetime.fromisoformat(view["updated"]).timestamp() > 600
            )
        except (KeyError, TypeError, ValueError):
            view["historical"] = False
    for job in jobs[:16]:
        result.append(
            {
                "label": f"{job['branch']} / {job['kind']} / {job['id']}",
                "state": "execution completed; outcome unverified"
                if job["kind"] in ("desktop", "python") and job["state"] == "completed"
                else job["state"],
                "task": job.get("assignment", ""),
                "updated": job.get("updated"),
                "result": public_summary(job.get("result") or {})
                if job["state"] in ("completed", "failed")
                else "",
                "previous_result": public_summary(job.get("result") or {})
                if job["state"] in ("queued", "running")
                else "",
                "declaration": "",
                "source": "research/state/drones.sqlite3",
                "event_id": job["id"],
                "historical": job["state"] in ("completed", "failed", "cancelled")
                and isinstance(job.get("updated"), (int, float))
                and time.time() - job["updated"] > 600,
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
    workers = [
        {**worker, "stale": worker.get("stale", True) or now - worker.get("updated", 0) > 300}
        for worker in workers
    ]
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
            "jobs": [
                j
                for j in active
                if j["kind"] not in ("researcher", "critic", "benchmark", "income")
            ],
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
    from rlm.v100.mission_chat import preferences
    from rlm.v100.remote_helper import remote_profile, selected_helper

    enabled = preferences(root).get("remote_helper_enabled", True) is not False
    if not enabled:
        rtx.update(
            state="disabled by operator",
            stale=False,
            jobs=[],
            detail="No Windows GPU requests. Model research uses the accepted V100; CPU mailbox stays enabled.",
        )
        actors[1]["jobs"] = [j for j in active if j["kind"] in ("researcher", "critic", "income")]

    try:
        if not enabled:
            raise RuntimeError("Operator disabled helper probes")
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
        if enabled:
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
        "events": action_events(events),
        "agents": agent_views(events, jobs),
        "chat": chat,
    }
