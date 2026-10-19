"""
Tests for rlm.v100.source_ingest — roadmap item: drone.

Uses importlib to avoid pulling in the full rlm package.
"""
import importlib.util
import json
import threading
import time
from pathlib import Path
import tempfile

spec = importlib.util.spec_from_file_location(
    "source_ingest",
    Path(__file__).parent.parent / "rlm/v100/source_ingest.py",
)
si = importlib.util.module_from_spec(spec)
spec.loader.exec_module(si)


def _tmp_root():
    d = tempfile.mkdtemp()
    root = Path(d)
    (root / "research/providers").mkdir(parents=True)
    (root / "research/paper/sources").mkdir(parents=True)
    return root


# ── 1. _load_state returns defaults for new provider ─────────────────────────
root = _tmp_root()
state = si._load_state(root, "test-feed")
assert state["provider_id"] == "test-feed"
assert state["consecutive_errors"] == 0
assert state["total_fetches"] == 0
assert state["measured_lag_s"] == []
print("check 1 ok: default state")

# ── 2. _save_state / _load_state round-trip ──────────────────────────────────
root2 = _tmp_root()
s = si._load_state(root2, "feed-x")
s["total_fetches"] = 5
s["last_fetched"] = "2026-10-18T12:00:00+00:00"
si._save_state(root2, s)
s2 = si._load_state(root2, "feed-x")
assert s2["total_fetches"] == 5
assert s2["last_fetched"] == "2026-10-18T12:00:00+00:00"
print("check 2 ok: state round-trip")

# ── 3. _backoff_s grows with consecutive errors ───────────────────────────────
b0 = si._backoff_s(0)
b1 = si._backoff_s(1)
b8 = si._backoff_s(8)
assert b0 < b1 < b8
assert b8 <= si._MAX_BACKOFF_S
print("check 3 ok: backoff grows and is bounded")

# ── 4. SourceIngestService starts with no providers → zero drones ─────────────
root3 = _tmp_root()
svc = si.SourceIngestService()
result = svc.start(root3)
assert result["total_live"] == 0
assert result["started_drones"] == []
svc.stop()
print("check 4 ok: empty providers → zero drones started")

# ── 5. status returns empty list when no drones started ───────────────────────
root4 = _tmp_root()
svc2 = si.SourceIngestService()
st = svc2.status(root4)
assert st["total_live"] == 0
assert st["drones"] == []
print("check 5 ok: status is empty before start")

# ── 6. Drone starts and is alive after start() ──────────────────────────────
root5 = _tmp_root()
cfg = {
    "id": "test-drone-1",
    "url": "http://invalid.local/feed.json",
    "poll_interval_s": 9999,
}
(root5 / "research/providers" / "test-drone-1.json").write_text(json.dumps(cfg))
svc3 = si.SourceIngestService()
r = svc3.start(root5)
assert "test-drone-1" in r["started_drones"]
assert r["total_live"] == 1
svc3.stop()
print("check 6 ok: drone starts and registers as live")

# ── 7. ingest_status / start_ingest module functions ─────────────────────────
root6 = _tmp_root()
result_status = si.ingest_status(root6)
assert "drones" in result_status
result_start = si.start_ingest(root6)
assert "started_drones" in result_start
print("check 7 ok: module-level ingest_status and start_ingest callable")

# ── 8. get_service returns singleton ─────────────────────────────────────────
s_a = si.get_service()
s_b = si.get_service()
assert s_a is s_b
print("check 8 ok: get_service is singleton")

print("\nAll 8 checks passed.")
