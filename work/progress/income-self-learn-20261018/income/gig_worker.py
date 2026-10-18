"""
Synta GigWorker — polls free-tier gig/freelance task sources and executes them autonomously.

Supported sources (free, no payment required to list):
  - Freelancer.com public contest/project API
  - Reddit r/forhire (RSS)
  - Hacker News "Who is hiring?" / "Freelancer?" threads (Algolia API)
  - MTurk-style micro-tasks via Amazon Mechanical Turk (requester side)

All results appended to income_results.json.
No live orders, no revenue claims without reconciled receipts.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import hashlib

try:
    import urllib.request as urlreq
    import urllib.parse as urlparse
except ImportError:
    raise

logger = logging.getLogger("synta.gig_worker")

RESULTS_FILE = Path(os.environ.get("SYNTA_INCOME_RESULTS", "income_results.json"))
STATE_FILE = Path(os.environ.get("SYNTA_GIG_STATE", "gig_worker_state.json"))
POLL_INTERVAL_S = int(os.environ.get("SYNTA_GIG_POLL_INTERVAL", "300"))  # 5 min default


@dataclass
class GigTask:
    task_id: str
    source: str
    title: str
    description: str
    category: str
    url: str
    budget_usd: float
    fetched_at: str
    status: str = "pending"  # pending | executed | skipped | error
    result: Optional[str] = None
    error: Optional[str] = None
    receipt_sha: Optional[str] = None


@dataclass
class IncomeRecord:
    record_id: str
    source: str
    task_id: str
    amount_usd: float
    currency: str
    timestamp: str
    confirmed: bool
    note: str
    activation_uncertain: bool = True  # always true until external confirmation


def _fetch_url(url: str, timeout: int = 10) -> bytes:
    req = urlreq.Request(url, headers={"User-Agent": "Synta/1.0 (autonomous research; contact: synta-bot)"})
    with urlreq.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Task sources
# ---------------------------------------------------------------------------

def fetch_hn_freelance_tasks() -> list[GigTask]:
    """HN 'Freelancer? Seeking freelancer?' monthly threads via Algolia."""
    tasks: list[GigTask] = []
    try:
        query = urlparse.urlencode({"query": "freelancer seeking", "tags": "ask_hn", "hitsPerPage": "20"})
        url = f"https://hn.algolia.com/api/v1/search?{query}"
        raw = _fetch_url(url)
        data = json.loads(raw)
        for hit in data.get("hits", [])[:10]:
            title = hit.get("title", "")
            if not title:
                continue
            task_id = f"hn_{hit.get('objectID', _sha256(title)[:12])}"
            tasks.append(GigTask(
                task_id=task_id,
                source="hackernews",
                title=title,
                description=hit.get("story_text") or "",
                category="freelance",
                url=f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}",
                budget_usd=0.0,
                fetched_at=_now_iso(),
            ))
    except Exception as exc:
        logger.warning("HN fetch failed: %s", exc)
    return tasks


def fetch_reddit_forhire_tasks() -> list[GigTask]:
    """Reddit r/forhire RSS — [HIRING] posts."""
    tasks: list[GigTask] = []
    try:
        url = "https://www.reddit.com/r/forhire/search.json?q=hiring&sort=new&limit=25&restrict_sr=1"
        raw = _fetch_url(url)
        data = json.loads(raw)
        posts = data.get("data", {}).get("children", [])
        for post in posts[:10]:
            d = post.get("data", {})
            title = d.get("title", "")
            if "[HIRING]" not in title.upper():
                continue
            task_id = f"reddit_{d.get('id', _sha256(title)[:12])}"
            tasks.append(GigTask(
                task_id=task_id,
                source="reddit_forhire",
                title=title,
                description=d.get("selftext", "")[:500],
                category="freelance",
                url=f"https://reddit.com{d.get('permalink', '')}",
                budget_usd=0.0,
                fetched_at=_now_iso(),
            ))
    except Exception as exc:
        logger.warning("Reddit forhire fetch failed: %s", exc)
    return tasks


def fetch_freelancer_contests() -> list[GigTask]:
    """Freelancer.com public contest API (no auth required for listing)."""
    tasks: list[GigTask] = []
    try:
        url = (
            "https://www.freelancer.com/api/contests/0.1/contests/"
            "?status[]=active&limit=10&compact=true"
        )
        raw = _fetch_url(url)
        data = json.loads(raw)
        results = data.get("result", {}).get("contests", {})
        contests = results if isinstance(results, list) else list(results.values())
        for c in contests[:10]:
            title = c.get("title", "")
            prize = float(c.get("prize", {}).get("amount", 0) or 0)
            cid = str(c.get("id", _sha256(title)[:12]))
            tasks.append(GigTask(
                task_id=f"freelancer_contest_{cid}",
                source="freelancer_contest",
                title=title,
                description=c.get("description", "")[:500],
                category="contest",
                url=f"https://www.freelancer.com/contest/{c.get('seo_url', cid)}",
                budget_usd=prize,
                fetched_at=_now_iso(),
            ))
    except Exception as exc:
        logger.warning("Freelancer contests fetch failed: %s", exc)
    return tasks


# ---------------------------------------------------------------------------
# Task execution (stub — real execution requires domain-specific logic)
# ---------------------------------------------------------------------------

def _is_executable(task: GigTask) -> bool:
    """Heuristic: mark tasks as executable based on category and keywords."""
    keywords = ["python", "data", "script", "automation", "analysis", "scraping", "api"]
    text = (task.title + " " + task.description).lower()
    return any(k in text for k in keywords)


def execute_task(task: GigTask) -> GigTask:
    """
    Stub executor — logs the task as 'skipped' (no real execution without
    confirmed platform credentials and payment setup).
    Real execution requires: platform auth, ability to submit deliverables,
    and reconciled income receipts before claiming any revenue.
    """
    task.status = "skipped"
    task.result = "Execution deferred: no confirmed platform credentials or payment reconciliation."
    task.receipt_sha = None
    logger.info("Task %s skipped (no executor configured): %s", task.task_id, task.title[:60])
    return task


# ---------------------------------------------------------------------------
# State & results persistence
# ---------------------------------------------------------------------------

def _load_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"seen_ids": [], "last_poll": None}


def _save_state(state: dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _append_result(task: GigTask) -> None:
    records: list[dict] = []
    if RESULTS_FILE.exists():
        try:
            records = json.loads(RESULTS_FILE.read_text())
        except Exception:
            pass
    records.append(asdict(task))
    RESULTS_FILE.write_text(json.dumps(records, indent=2))


# ---------------------------------------------------------------------------
# Main polling loop
# ---------------------------------------------------------------------------

def poll_once() -> list[GigTask]:
    state = _load_state()
    seen: set[str] = set(state.get("seen_ids", []))

    all_tasks: list[GigTask] = []
    all_tasks.extend(fetch_hn_freelance_tasks())
    all_tasks.extend(fetch_reddit_forhire_tasks())
    all_tasks.extend(fetch_freelancer_contests())

    new_tasks = [t for t in all_tasks if t.task_id not in seen]
    logger.info("Fetched %d tasks, %d new", len(all_tasks), len(new_tasks))

    processed: list[GigTask] = []
    for task in new_tasks:
        if _is_executable(task):
            task = execute_task(task)
        else:
            task.status = "skipped"
            task.result = "Not in executable categories."
        _append_result(task)
        seen.add(task.task_id)
        processed.append(task)

    state["seen_ids"] = list(seen)[-2000:]  # cap history
    state["last_poll"] = _now_iso()
    _save_state(state)
    return processed


def run_forever() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info("GigWorker starting, poll interval=%ds", POLL_INTERVAL_S)
    while True:
        try:
            tasks = poll_once()
            logger.info("Poll complete: %d tasks processed", len(tasks))
        except Exception as exc:
            logger.error("Poll error: %s", exc)
        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    run_forever()
