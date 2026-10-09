"""Independent bounded observation; GUI and GPU phases do not block public feeds."""

import time
from pathlib import Path

import requests

from rlm.v100.activity import ActivityLog
from rlm.v100.common import atomic_json


def tick(root: Path) -> None:
    from rlm.v100.goal_compute import tick as compute_tick
    from rlm.v100.goal_learning import tick as goal_tick
    from rlm.v100.spot_bootstrap import prepare

    try:
        goal_tick(root)
    except (OSError, ValueError, RuntimeError, requests.RequestException) as error:
        ActivityLog(root, "controller", "goal-learning").write(
            "errors", "goal-observer-failed", {"detail": str(error)[:400]}
        )
    compute_tick(root)
    ledger = root / "research/paper/ledger.sqlite3"
    if not ledger.exists():
        return
    path = root / "research/state/paper-observer.json"
    import json

    previous = json.loads(path.read_text()) if path.exists() else {}
    if time.time() < previous.get("next_poll", 0):
        return
    # Claim before network I/O. A failed request backs off rather than retrying each loop.
    atomic_json(path, {"state": "polling", "updated": time.time(), "next_poll": time.time() + 60})
    try:
        result = prepare(root, refresh=True)
        value = {
            "state": "ready",
            "updated": time.time(),
            "next_poll": time.time() + 60,
            "result": result,
        }
    except (OSError, ValueError, RuntimeError, requests.RequestException) as error:
        value = {
            "state": "failed",
            "updated": time.time(),
            "next_poll": time.time() + 300,
            "error": str(error)[:400],
        }
        ActivityLog(root, "controller", "paper-observer").write(
            "errors", "feed-refresh-failed", value
        )
    atomic_json(path, value)
