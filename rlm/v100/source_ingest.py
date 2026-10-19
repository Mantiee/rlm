"""
Parallel public source ingest — roadmap item: drone.

Runs one bounded worker thread per registered provider, with per-drone
state, exponential backoff, and failure isolation.  One flaky provider
never stalls others.  Every fetch records published_at (claimed by source)
and acquired_at (local wall clock) separately; lag is measured but not
guaranteed.

No data is invented.  No login-gated or rate-limit-bypassed endpoints.
No real trading orders.  Provider terms and per-source rate limits bound
throughput.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path
from typing import Any

_SCHEMA = "synta-source-ingest-v1"
_MAX_DRONES = 16
_MAX_BYTES = 2**20  # 1 MiB per fetch
_DEFAULT_INTERVAL_S = 60
_MAX_BACKOFF_S = 3600
_MAX_CONSECUTIVE_ERRORS = 8


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _state_path(root: Path, provider_id: str) -> Path:
    p = root / "research/source-ingest" / (provider_id + ".json")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_state(root: Path, provider_id: str) -> dict:
    path = _state_path(root, provider_id)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {
        "schema": _SCHEMA,
        "provider_id": provider_id,
        "last_fetched": None,
        "last_success": None,
        "consecutive_errors": 0,
        "backoff_until": None,
        "total_fetches": 0,
        "total_errors": 0,
        "total_rows": 0,
        "measured_lag_s": [],
    }


def _save_state(root: Path, state: dict) -> None:
    path = _state_path(root, state["provider_id"])
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _backoff_s(consecutive: int) -> float:
    return min(_MAX_BACKOFF_S, 10 * (2 ** min(consecutive, 8)))


def _fetch_once(root: Path, configuration: dict) -> dict:
    """
    Fetch one provider URL, archive the body, and return a fetch receipt.

    Returns:
      {ok: bool, digest, acquired_at, rows_ingested, published_at_claimed?,
       error?, lag_s?}
    """
    from rlm.v100.research_tools import download_page

    provider_id = configuration["id"]
    url = configuration["url"]
    acquired_at = _now_iso()

    try:
        fetched_url, body = download_page(url, max_bytes=_MAX_BYTES)
        if fetched_url != url:
            return {
                "ok": False,
                "provider_id": provider_id,
                "acquired_at": acquired_at,
                "error": "redirect",
                "detail": f"Expected {url!r}, got {fetched_url!r}",
            }
        digest = hashlib.sha256(body.encode()).hexdigest()
        folder = root / "research/paper/sources"
        folder.mkdir(parents=True, exist_ok=True)
        archive = folder / (digest + ".json")
        if not archive.exists():
            archive.write_text(body, encoding="utf-8")
        if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
            return {
                "ok": False,
                "provider_id": provider_id,
                "acquired_at": acquired_at,
                "error": "archive_mismatch",
            }

        from rlm.v100.provider_registry import normalize
        from rlm.v100.paper import PaperBook

        try:
            rows = normalize(configuration, json.loads(body), digest, acquired_at)
        except Exception as exc:
            return {
                "ok": False,
                "provider_id": provider_id,
                "acquired_at": acquired_at,
                "error": "normalize_error",
                "detail": str(exc)[:400],
            }

        rows_ingested = 0
        published_at_claimed: str | None = None
        lags: list[float] = []
        for row in rows:
            pa = row.get("published_at") or row.get("available_at")
            if pa and published_at_claimed is None:
                published_at_claimed = pa
            if pa:
                try:
                    from datetime import timezone
                    from dateutil.parser import isoparse
                    pa_ts = isoparse(pa).timestamp()
                    aq_ts = datetime.fromisoformat(acquired_at.replace("Z", "+00:00")).timestamp()
                    lag = aq_ts - pa_ts
                    if 0 <= lag < 86400:
                        lags.append(lag)
                except Exception:
                    pass
            rows_ingested += 1

        receipt: dict[str, Any] = {
            "ok": True,
            "provider_id": provider_id,
            "acquired_at": acquired_at,
            "digest": digest,
            "rows_ingested": rows_ingested,
        }
        if published_at_claimed:
            receipt["published_at_claimed"] = published_at_claimed
            receipt["published_at_note"] = "publisher-declared; unverified"
        if lags:
            receipt["lag_s_sample"] = round(sum(lags) / len(lags), 2)
        return receipt

    except Exception as exc:
        return {
            "ok": False,
            "provider_id": provider_id,
            "acquired_at": acquired_at,
            "error": type(exc).__name__,
            "detail": str(exc)[:400],
        }


def _drone_worker(
    root: Path,
    configuration: dict,
    stop: threading.Event,
    interval_s: int,
) -> None:
    """Single-provider drone thread.  Runs until stop is set."""
    provider_id = configuration["id"]
    state = _load_state(root, provider_id)

    while not stop.is_set():
        now = time.monotonic()
        backoff_until = state.get("backoff_until")
        if backoff_until:
            remaining = float(backoff_until) - time.time()
            if remaining > 0:
                stop.wait(min(remaining, 30))
                continue

        receipt = _fetch_once(root, configuration)
        state["last_fetched"] = _now_iso()
        state["total_fetches"] = state.get("total_fetches", 0) + 1

        if receipt["ok"]:
            state["consecutive_errors"] = 0
            state["backoff_until"] = None
            state["last_success"] = state["last_fetched"]
            state["total_rows"] = state.get("total_rows", 0) + receipt.get("rows_ingested", 0)
            lag = receipt.get("lag_s_sample")
            if lag is not None:
                lags = state.get("measured_lag_s", [])
                lags.append(lag)
                state["measured_lag_s"] = lags[-20:]
        else:
            state["consecutive_errors"] = state.get("consecutive_errors", 0) + 1
            state["total_errors"] = state.get("total_errors", 0) + 1
            if state["consecutive_errors"] >= _MAX_CONSECUTIVE_ERRORS:
                delay = _backoff_s(state["consecutive_errors"])
                state["backoff_until"] = time.time() + delay
            from rlm.v100.activity import ActivityLog
            ActivityLog(root, "controller", "source_ingest").write(
                "errors", "fetch_failed",
                {"provider_id": provider_id, **receipt},
            )

        _save_state(root, state)
        stop.wait(interval_s)


class SourceIngestService:
    """
    Manages one thread per registered provider.

    Start with .start(root), stop with .stop().  Threads are daemon threads
    so they do not block process exit.
    """

    def __init__(self) -> None:
        self._threads: dict[str, threading.Thread] = {}
        self._stop = threading.Event()

    def start(self, root: Path) -> dict:
        providers = []
        for path in sorted((root / "research/providers").glob("*.json"))[:_MAX_DRONES]:
            try:
                cfg = json.loads(path.read_text(encoding="utf-8"))
                providers.append(cfg)
            except (OSError, ValueError):
                continue

        started = []
        for cfg in providers:
            pid = cfg["id"]
            if pid not in self._threads or not self._threads[pid].is_alive():
                interval = int(cfg.get("poll_interval_s", _DEFAULT_INTERVAL_S))
                t = threading.Thread(
                    target=_drone_worker,
                    args=(root, cfg, self._stop, interval),
                    name=f"source-drone-{pid}",
                    daemon=True,
                )
                t.start()
                self._threads[pid] = t
                started.append(pid)

        return {
            "schema": _SCHEMA,
            "started_drones": started,
            "total_live": sum(1 for t in self._threads.values() if t.is_alive()),
            "note": (
                "Each drone fetches its provider independently.  "
                "Failure of one drone does not stall others.  "
                "Throughput is bounded by provider rate limits, "
                "terms, and available CPU/IO, not by a fixed multiplier."
            ),
        }

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads.values():
            t.join(timeout=5)

    def status(self, root: Path) -> dict:
        drones = []
        for pid, t in self._threads.items():
            state = _load_state(root, pid)
            drones.append({
                "provider_id": pid,
                "alive": t.is_alive(),
                "last_fetched": state.get("last_fetched"),
                "last_success": state.get("last_success"),
                "consecutive_errors": state.get("consecutive_errors", 0),
                "total_fetches": state.get("total_fetches", 0),
                "total_errors": state.get("total_errors", 0),
                "total_rows": state.get("total_rows", 0),
                "avg_lag_s": (
                    round(sum(state["measured_lag_s"]) / len(state["measured_lag_s"]), 2)
                    if state.get("measured_lag_s") else None
                ),
                "backoff_active": bool(
                    state.get("backoff_until") and state["backoff_until"] > time.time()
                ),
            })
        return {
            "schema": _SCHEMA,
            "drones": drones,
            "total_live": sum(1 for d in drones if d["alive"]),
        }


# Module-level singleton used by the drones service and research_tools.
_service: SourceIngestService | None = None
_lock = threading.Lock()


def get_service() -> SourceIngestService:
    global _service
    with _lock:
        if _service is None:
            _service = SourceIngestService()
        return _service


def ingest_status(root: Path) -> dict:
    """Tool handler: return status of all running source drones."""
    return get_service().status(root)


def start_ingest(root: Path) -> dict:
    """Start parallel source ingest drones for all registered providers."""
    return get_service().start(root)
