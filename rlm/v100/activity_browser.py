"""Read-only pagination over exact, already-redacted daily execution journals."""

import json
from datetime import date
from pathlib import Path


def days(root: Path) -> list[str]:
    base = (root / "research/logs/activity").resolve()
    result = []
    for path in sorted(base.glob("*/timeline.jsonl")):
        if path.is_symlink() or path.parent.is_symlink() or not path.resolve().is_relative_to(base):
            continue
        try:
            if date.fromisoformat(path.parent.name).isoformat() != path.parent.name:
                continue
        except ValueError:
            continue
        result.append(path.parent.name)
    return result


def page(root: Path, day: str, cursor: int = 0, limit: int = 100) -> dict:
    if date.fromisoformat(day).isoformat() != day or type(cursor) is not int or cursor < 0:
        raise ValueError("Invalid action archive date or cursor")
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("Action archive page limit must be 1-100")
    base = (root / "research/logs/activity").resolve()
    path = base / day / "timeline.jsonl"
    if path.is_symlink() or path.parent.is_symlink() or path.resolve().parent != base / day:
        raise ValueError("Action archive path is outside its day")
    events = []
    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        if cursor > size:
            raise ValueError("Action cursor exceeds archive size")
        stream.seek(cursor)
        while len(events) < limit and stream.tell() - cursor < 1024 * 1024:
            position = stream.tell()
            line = stream.readline(65537)
            if not line:
                break
            if len(line) > 65536:
                raise ValueError("Action event exceeds journal budget")
            if not line.endswith(b"\n"):
                stream.seek(position)
                break
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("Invalid action event")
            events.append(event)
        following = stream.tell()
    return {
        "day": day,
        "events": events,
        "cursor": cursor,
        "next_cursor": following,
        "has_more": following < size,
        "archive_bytes": size,
        "scope": "Recorded actions only; unlogged operations cannot be reconstructed",
    }
