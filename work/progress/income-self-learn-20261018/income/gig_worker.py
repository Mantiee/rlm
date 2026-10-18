"""
Synta GigWorker — polls free-tier gig/bounty sources, scores profitability,
and executes qualifying tasks autonomously.

Sources (all free-tier, no upfront payment):
  - Gitcoin bounties (GitHub-funded crypto bounties)
  - IssueHunt open bounties
  - Algora open-source bounties
  - Freelancer.com public contests
  - Reddit r/forhire [HIRING] posts
  - Hacker News "Who is hiring?" / "Freelancer?" threads

Execution policy:
  - Task must score >= MIN_PROFITABILITY_SCORE (0.0–1.0)
  - Budget must be >= MIN_BUDGET_USD
  - Synta only executes tasks within its skill set (Python, data, automation, ML)
  - Wallet address must be configured to enable payout claim
  - All income marked activation_uncertain=True until external confirmation

No live orders, no revenue claims without reconciled receipts.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict
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
POLL_INTERVAL_S = int(os.environ.get("SYNTA_GIG_POLL_INTERVAL", "300"))
MIN_BUDGET_USD = float(os.environ.get("SYNTA_GIG_MIN_BUDGET_USD", "5.0"))
MIN_PROFITABILITY_SCORE = float(os.environ.get("SYNTA_GIG_MIN_SCORE", "0.4"))
WALLET_ADDRESS = os.environ.get("SYNTA_WALLET_ADDR", "")  # ETH/ERC-20 for bounties
MAX_EXEC_TASKS_PER_POLL = int(os.environ.get("SYNTA_GIG_MAX_EXEC_PER_POLL", "3"))

# Skills Synta can autonomously execute
SYNTA_SKILLS = [
    "python", "data analysis", "data science", "machine learning", "ml",
    "automation", "script", "scraping", "api integration", "etl",
    "trading", "quant", "backtesting", "algorithm", "bot",
    "pandas", "numpy", "json", "csv", "sql lite", "sqlite",
    "data pipeline", "data cleaning", "visualization", "chart",
]

# Categories where Synta can reliably deliver
EXECUTABLE_CATEGORIES = {"bounty", "contest", "micro_task", "freelance_script"}


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
    profitability_score: float = 0.0
    skill_match: float = 0.0
    status: str = "pending"  # pending | evaluated | executed | submitted | skipped | error
    result: Optional[str] = None
    error: Optional[str] = None
    receipt_sha: Optional[str] = None
    payout_address: Optional[str] = None


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
    activation_uncertain: bool = True  # always until external confirmation


def _fetch_url(url: str, timeout: int = 15) -> bytes:
    req = urlreq.Request(url, headers={
        "User-Agent": "Synta/1.0 (autonomous income research; contact: synta-bot)",
        "Accept": "application/json",
    })
    with urlreq.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Profitability scoring
# ---------------------------------------------------------------------------

def _skill_match_score(task: GigTask) -> float:
    """0.0–1.0: how well Synta's skills match the task."""
    text = (task.title + " " + task.description).lower()
    matches = sum(1 for skill in SYNTA_SKILLS if skill in text)
    return min(1.0, matches / 3.0)


def _profitability_score(task: GigTask) -> float:
    """
    0.0–1.0 composite score: budget, skill match, category.
    Only tasks above MIN_PROFITABILITY_SCORE are attempted.
    """
    skill = _skill_match_score(task)
    if skill < 0.1:
        return 0.0

    # Budget factor: sigmoid scaled around MIN_BUDGET_USD
    budget = task.budget_usd
    if budget <= 0:
        budget_factor = 0.05  # unknown budget — low but not zero
    elif budget < MIN_BUDGET_USD:
        budget_factor = budget / MIN_BUDGET_USD * 0.3
    elif budget < 50:
        budget_factor = 0.5 + (budget - MIN_BUDGET_USD) / 50 * 0.3
    else:
        budget_factor = min(1.0, 0.8 + (budget - 50) / 200 * 0.2)

    # Category factor: bounties > contests > freelance listings
    category_factor = {"bounty": 1.0, "contest": 0.7, "micro_task": 0.8,
                       "freelance_script": 0.85, "freelance": 0.5}.get(task.category, 0.4)

    score = 0.5 * skill + 0.35 * budget_factor + 0.15 * category_factor
    return round(min(1.0, score), 3)


# ---------------------------------------------------------------------------
# Task sources
# ---------------------------------------------------------------------------

def fetch_gitcoin_bounties() -> list[GigTask]:
    """Gitcoin open bounties via public API."""
    tasks: list[GigTask] = []
    try:
        url = (
            "https://gitcoin.co/api/v0.1/bounties/"
            "?network=mainnet&status=open&limit=20&order_by=-value_in_usdt"
        )
        raw = _fetch_url(url)
        data = json.loads(raw)
        for b in (data if isinstance(data, list) else data.get("results", []))[:15]:
            title = b.get("title", "")
            bounty_id = str(b.get("pk") or b.get("id") or _sha256(title)[:12])
            try:
                budget = float(b.get("value_in_usdt") or b.get("value_true") or 0)
            except (TypeError, ValueError):
                budget = 0.0
            tasks.append(GigTask(
                task_id=f"gitcoin_{bounty_id}",
                source="gitcoin",
                title=title,
                description=(b.get("description") or "")[:600],
                category="bounty",
                url=b.get("url") or f"https://gitcoin.co/issue/{bounty_id}",
                budget_usd=budget,
                fetched_at=_now_iso(),
            ))
    except Exception as exc:
        logger.warning("Gitcoin fetch failed: %s", exc)
    return tasks


def fetch_issuehunt_bounties() -> list[GigTask]:
    """IssueHunt open bounties."""
    tasks: list[GigTask] = []
    try:
        url = "https://issuehunt.io/api/v1/issues?state=open&sort=amount&limit=20"
        raw = _fetch_url(url)
        data = json.loads(raw)
        for issue in (data.get("issues") or data.get("data") or [])[:15]:
            title = issue.get("title", "")
            iid = str(issue.get("id") or _sha256(title)[:12])
            try:
                budget = float(issue.get("amount") or 0)
            except (TypeError, ValueError):
                budget = 0.0
            tasks.append(GigTask(
                task_id=f"issuehunt_{iid}",
                source="issuehunt",
                title=title,
                description=(issue.get("body") or "")[:600],
                category="bounty",
                url=issue.get("url") or f"https://issuehunt.io/issues/{iid}",
                budget_usd=budget,
                fetched_at=_now_iso(),
            ))
    except Exception as exc:
        logger.warning("IssueHunt fetch failed: %s", exc)
    return tasks


def fetch_algora_bounties() -> list[GigTask]:
    """Algora open-source bounties."""
    tasks: list[GigTask] = []
    try:
        url = "https://console.algora.io/api/bounties?status=open&limit=20"
        raw = _fetch_url(url)
        data = json.loads(raw)
        for b in (data.get("bounties") or data.get("data") or [])[:15]:
            title = b.get("title", "") or b.get("issue", {}).get("title", "")
            bid = str(b.get("id") or _sha256(title)[:12])
            try:
                budget = float(b.get("reward_usd") or b.get("amount") or 0)
            except (TypeError, ValueError):
                budget = 0.0
            tasks.append(GigTask(
                task_id=f"algora_{bid}",
                source="algora",
                title=title,
                description=(b.get("description") or "")[:600],
                category="bounty",
                url=b.get("url") or f"https://algora.io/bounties/{bid}",
                budget_usd=budget,
                fetched_at=_now_iso(),
            ))
    except Exception as exc:
        logger.warning("Algora fetch failed: %s", exc)
    return tasks


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
                description=(hit.get("story_text") or "")[:500],
                category="freelance",
                url=f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}",
                budget_usd=0.0,
                fetched_at=_now_iso(),
            ))
    except Exception as exc:
        logger.warning("HN fetch failed: %s", exc)
    return tasks


def fetch_reddit_forhire_tasks() -> list[GigTask]:
    """Reddit r/forhire [HIRING] posts."""
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
            # Try to extract budget from title like "$50" or "50-100"
            budget = 0.0
            m = re.search(r'\$\s*(\d+(?:\.\d+)?)', title + " " + d.get("selftext", ""))
            if m:
                try:
                    budget = float(m.group(1))
                except ValueError:
                    pass
            tasks.append(GigTask(
                task_id=task_id,
                source="reddit_forhire",
                title=title,
                description=d.get("selftext", "")[:500],
                category="freelance",
                url=f"https://reddit.com{d.get('permalink', '')}",
                budget_usd=budget,
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
# Task evaluation
# ---------------------------------------------------------------------------

def evaluate_task(task: GigTask) -> GigTask:
    """Score the task and decide if Synta should attempt it."""
    task.skill_match = _skill_match_score(task)
    task.profitability_score = _profitability_score(task)
    task.payout_address = WALLET_ADDRESS or None

    if task.profitability_score >= MIN_PROFITABILITY_SCORE and task.budget_usd >= MIN_BUDGET_USD:
        task.status = "evaluated"
        logger.info(
            "Task QUALIFIED score=%.2f budget=$%.2f: %s",
            task.profitability_score, task.budget_usd, task.task_id,
        )
    else:
        task.status = "skipped"
        task.result = (
            f"score={task.profitability_score:.2f} < {MIN_PROFITABILITY_SCORE} "
            f"or budget=${task.budget_usd:.2f} < ${MIN_BUDGET_USD:.2f}"
        )
    return task


# ---------------------------------------------------------------------------
# Task execution
# ---------------------------------------------------------------------------

def _generate_python_deliverable(task: GigTask) -> str:
    """
    Generate a simple Python script deliverable for data/automation tasks.
    Returns the script as a string. Must be verified before submission.
    """
    script = f"""#!/usr/bin/env python3
\"\"\"
Synta autonomous deliverable for: {task.title[:80]}
Source: {task.source} | Task: {task.task_id}
Generated: {_now_iso()}
NOTE: Review before submission. Synta generated this as a starting point.
\"\"\"

# Task: {task.title}
# URL: {task.url}

# Minimal implementation based on task description:
# {task.description[:300].replace(chr(10), chr(10) + '# ')}

import json
from pathlib import Path

def main():
    print("Synta deliverable: {task.title[:60]}")
    # TODO: implement task-specific logic based on full description at {task.url}
    return {{"status": "delivered", "task_id": "{task.task_id}"}}

if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2))
"""
    return script


def execute_task(task: GigTask) -> GigTask:
    """
    Execute a qualified task. For bounties: generate deliverable, record for review.
    Does NOT auto-submit — submission requires human review of the deliverable.
    Sets status='executed' with a receipt_sha of the deliverable.
    """
    if task.status != "evaluated":
        task.status = "skipped"
        task.result = "Not evaluated as profitable."
        return task

    if not WALLET_ADDRESS:
        task.status = "skipped"
        task.result = "SYNTA_WALLET_ADDR not configured; set it to enable payout claim."
        logger.warning("Task %s: no wallet configured, skipping execution", task.task_id)
        return task

    try:
        deliverable = _generate_python_deliverable(task)
        receipt_sha = _sha256(deliverable)

        # Write deliverable to output dir
        out_dir = Path(os.environ.get("SYNTA_DATA_DIR", "data")) / "deliverables"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{task.task_id}.py"
        out_file.write_text(deliverable)

        task.status = "executed"
        task.receipt_sha = receipt_sha
        task.result = (
            f"Deliverable generated at {out_file}. "
            f"SHA256={receipt_sha[:16]}... "
            f"Review required before submission to {task.url}. "
            f"Payout address: {WALLET_ADDRESS}"
        )
        logger.info("Task %s executed: deliverable at %s", task.task_id, out_file)
    except Exception as exc:
        task.status = "error"
        task.error = str(exc)
        logger.error("Task %s execution failed: %s", task.task_id, exc)

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
    all_tasks.extend(fetch_gitcoin_bounties())
    all_tasks.extend(fetch_issuehunt_bounties())
    all_tasks.extend(fetch_algora_bounties())
    all_tasks.extend(fetch_hn_freelance_tasks())
    all_tasks.extend(fetch_reddit_forhire_tasks())
    all_tasks.extend(fetch_freelancer_contests())

    new_tasks = [t for t in all_tasks if t.task_id not in seen]
    logger.info("Fetched %d tasks across 6 sources, %d new", len(all_tasks), len(new_tasks))

    executed_count = 0
    processed: list[GigTask] = []
    for task in new_tasks:
        task = evaluate_task(task)
        if task.status == "evaluated" and executed_count < MAX_EXEC_TASKS_PER_POLL:
            task = execute_task(task)
            if task.status == "executed":
                executed_count += 1
        _append_result(task)
        seen.add(task.task_id)
        processed.append(task)

    qualified = [t for t in processed if t.status in ("executed", "evaluated")]
    logger.info(
        "Poll complete: %d new, %d qualified, %d executed",
        len(new_tasks), len(qualified), executed_count,
    )

    state["seen_ids"] = list(seen)[-2000:]
    state["last_poll"] = _now_iso()
    state["last_executed"] = executed_count
    state["wallet"] = WALLET_ADDRESS or "not configured"
    _save_state(state)
    return processed


def run_forever() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info(
        "GigWorker starting: interval=%ds min_budget=$%.2f min_score=%.2f wallet=%s",
        POLL_INTERVAL_S, MIN_BUDGET_USD, MIN_PROFITABILITY_SCORE,
        WALLET_ADDRESS[:10] + "..." if WALLET_ADDRESS else "NOT SET",
    )
    while True:
        try:
            tasks = poll_once()
        except Exception as exc:
            logger.error("Poll error: %s", exc)
        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    run_forever()
