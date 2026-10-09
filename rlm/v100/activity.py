"""Local execution journals with explicit decisions and operator-selected local model traces."""

import fcntl
import json
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

SECRET_KEYS = {
    "authorization",
    "api_key",
    "password",
    "secret",
    "token",
    "access_token",
    "hf_token",
    "v100_free_router_key",
    "system_prompt",
    "reasoning_content",
    "chain_of_thought",
}
CATEGORIES = {"steps", "decisions", "tools", "research", "training", "metrics", "errors"}


def redact(value, depth: int = 0):
    if depth > 8:
        return "[depth limit]"
    if isinstance(value, dict):
        return {
            str(key): "[omitted]" if str(key).lower() in SECRET_KEYS else redact(item, depth + 1)
            for key, item in list(value.items())[:64]
        }
    if isinstance(value, (list, tuple)):
        return [redact(item, depth + 1) for item in value[:64]]
    if isinstance(value, str):
        if value.lstrip().startswith(("{", "[")):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                pass
            else:
                return redact(decoded, depth + 1)
        value = re.sub(r"(?is)<think>.*?(?:</think>|$)", "[private reasoning omitted]", value)
        value = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [omitted]", value)
        value = re.sub(r"\b(?:sk-[A-Za-z0-9_-]{12,}|hf_[A-Za-z0-9]{12,})\b", "[omitted]", value)
        value = re.sub(
            r"(?i)\b(api[_-]?key|password|access[_-]?token|hf_token|v100_free_router_key)\s*[=:]\s*[^\s,;]+",
            r"\1=[omitted]",
            value,
        )
        return value[:8192] + (" [truncated]" if len(value) > 8192 else "")
    return value


def append_locked(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(text)
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)


class ActivityLog:
    def __init__(self, root: Path, branch: str = "controller", actor: str = "model"):
        if branch not in ("A", "B", "shared", "controller"):
            raise ValueError("Unknown activity branch")
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,47}", actor):
            raise ValueError("Invalid activity actor")
        self.root, self.branch, self.actor = root, branch, actor

    def write(self, category: str, kind: str, payload: dict, **context) -> str:
        if category not in CATEGORIES:
            raise ValueError("Unknown activity category")
        now = datetime.now(UTC)
        event_id = uuid.uuid4().hex
        goal_path = self.root / "research/goal.json"
        if goal_path.exists() and goal_path.stat().st_size <= 8192:
            context["operator_goal_id_at_event"] = json.loads(goal_path.read_text()).get("id")
        event = {
            "time": now.isoformat(timespec="milliseconds"),
            "id": event_id,
            "branch": self.branch,
            "actor": self.actor,
            "category": category,
            "kind": kind,
            "context": redact(context),
            "payload": redact(payload),
        }
        # Bound each event, including deeply nested model output.
        encoded = json.dumps(event, ensure_ascii=False, allow_nan=False)
        if len(encoded.encode()) > 65536:
            event["payload"] = {"excerpt": encoded[:8192], "truncated": True}
            encoded = json.dumps(event, ensure_ascii=False, allow_nan=False)
        daily = self.root / "research/logs/activity" / now.strftime("%Y-%m-%d")
        directory = daily / self.branch / self.actor
        append_locked(directory / f"{category}.jsonl", encoded + "\n")
        # Indented JSON under a heading stays easy to inspect with ordinary editors.
        readable = json.dumps(event["payload"], ensure_ascii=False, indent=2, allow_nan=False)
        append_locked(
            directory / f"{category}.md",
            f"\n## {event['time']} | {kind} | {event_id}\n\n"
            + json.dumps(event["context"], ensure_ascii=False)
            + "\n\n"
            + "```json\n"
            + readable.replace("```", "` ` `")
            + "\n```\n",
        )
        append_locked(daily / "timeline.jsonl", encoded + "\n")
        return event_id


def public_event_log(root: Path, branch: str, kind: str, payload: dict, sequence: int) -> None:
    category = {
        "plan": "decisions",
        "message": "decisions",
        "worker-verdict": "decisions",
        "service-selection": "decisions",
        "training": "training",
        "development-result": "metrics",
        "worker-task": "research",
        "worker-result": "research",
        "free-consultation": "research",
        "architecture-result": "research",
        "architecture-proposal": "research",
        "code-candidate": "research",
    }[kind]
    actor = "tester" if kind.startswith("worker-") and kind != "worker-verdict" else "model"
    if kind == "worker-result" and payload.get("role") in ("researcher", "tester", "critic"):
        actor = payload["role"]
    if kind == "worker-task" and payload.get("job", {}).get("role") in (
        "researcher",
        "tester",
        "critic",
    ):
        actor = payload["job"]["role"]
    ActivityLog(root, branch, actor).write(category, kind, payload, sequence=sequence)
