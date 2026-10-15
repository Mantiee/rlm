"""Optional private control-plane release pin; execution stays on verified public code."""

import json
import re
import time
import urllib.request
from pathlib import Path

SCHEMA = "synta-approved-release-v1"


def validate(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError("Approved release must be a JSON object")
    if value.get("schema") != SCHEMA or value.get("repository") != "Mantiee/rlm":
        raise ValueError("Unknown approved release source")
    if value.get("mode") not in ("follow-published", "pinned"):
        raise ValueError("Unknown release selection mode")
    if value["mode"] == "pinned":
        revision = value.get("revision")
        if not isinstance(revision, str) or not re.fullmatch("[a-f0-9]{40}", revision):
            raise ValueError("Release pin requires an immutable commit")
    return {k: value[k] for k in ("schema", "repository", "mode", "revision") if k in value}


def current(root: Path, peer: str | None = None) -> dict:
    # A dedicated-node pin must never trigger maintenance on the daily-use PC.
    dedicated = root / "research/private-sync/3060-release.json"
    path = (
        dedicated
        if peer == "192.168.0.9" and (dedicated.exists() or dedicated.is_symlink())
        else root / "research/private-sync/approved-release.json"
    )
    if not path.exists() and not path.is_symlink():
        return {
            "schema": SCHEMA,
            "repository": "Mantiee/rlm",
            "mode": "follow-published",
            "configured": False,
        }
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8192:
        raise ValueError("Invalid private release receipt")
    return {**validate(json.loads(path.read_text())), "configured": True}


def selection(root: Path, peer: bool = False) -> str | None:
    value = current(root)
    if peer:
        cache = root / "research/private-sync/controller-release.json"
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open("http://192.168.0.68:8765/api/release", timeout=3) as reply:
                body = reply.read(8193)
            if len(body) > 8192:
                raise ValueError("Controller release receipt too large")
            value = validate(json.loads(body))
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_suffix(".new")
            temporary.write_text(json.dumps({**value, "observed_at": time.time()}))
            temporary.replace(cache)
        except (OSError, ValueError):
            if cache.is_file() and not cache.is_symlink() and cache.stat().st_size <= 8192:
                value = validate(json.loads(cache.read_text()))
    return value.get("revision") if value["mode"] == "pinned" else None
