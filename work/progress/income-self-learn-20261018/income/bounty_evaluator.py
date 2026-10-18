"""
Synta BountyEvaluator — autonomously scans and evaluates bounty/gig tasks.

Sources:
  - Gitcoin Grants (public GraphQL API)
  - Dework (public GraphQL API)
  - Layer3 (public API)
  - Freelancer.com public contest listings
  - HN 'Who is hiring?' / 'Freelancer?' threads

Evaluation criteria (ALL must pass):
  - Estimated duration < 4 hours
  - Expected payout > $5 USD
  - Executable by AI without human interaction, meetings, or accounts with history
  - Category: code, data, analysis, text, automation

Earnings logged to earnings_log.json (shared with paper P&L).
Wallet placeholder: set SYNTA_WALLET_ETH env var when Mantie provides address.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import hashlib
import urllib.request as urlreq
import urllib.parse as urlparse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("synta.bounty_evaluator")

EARNINGS_LOG = Path(os.environ.get("SYNTA_EARNINGS_LOG", "earnings_log.json"))
STATE_FILE = Path(os.environ.get("SYNTA_BOUNTY_STATE", "bounty_evaluator_state.json"))
POLL_INTERVAL_S = int(os.environ.get("SYNTA_BOUNTY_POLL_INTERVAL", "1800"))  # 30 min

MIN_PAYOUT_USD: float = float(os.environ.get("SYNTA_BOUNTY_MIN_USD", "5.0"))
MAX_DURATION_H: float = float(os.environ.get("SYNTA_BOUNTY_MAX_HOURS", "4.0"))

WALLET_ETH: Optional[str] = os.environ.get("SYNTA_WALLET_ETH")  # placeholder; set when provided

# Keywords that disqualify a task (requires human interaction, history, etc.)
DISQUALIFY_KEYWORDS = [
    "interview", "meeting", "call", "zoom", "skype", "phone",
    "agency", "team", "years of experience", "portfolio required",
    "references required", "account history", "verified account",
    "ongoing", "retainer", "full-time", "part-time", "contract long",
    "kyc", "identity verification",
]

# Keywords that qualify a task (AI-executable)
QUALIFY_KEYWORDS = [
    "python", "script", "automation", "data", "analysis", "scraping",
    "api", "json", "csv", "text", "writing", "translation", "summarize",
    "classify", "extract", "parse", "report", "research", "coding",
    "bug fix", "unit test", "documentation", "readme", "refactor",
    "smart contract", "solidity", "rust", "go", "javascript", "typescript",
    "web3", "defi", "bot", "crawler", "etl", "ml", "model", "nlp",
]


@dataclass
class BountyTask:
    task_id: str
    source: str
    title: str
    description: str
    category: str
    url: str
    payout_usd: float
    estimated_hours: float
    fetched_at: str
    status: str = "pending"  # pending | evaluated | skipped | accepted | completed | error
    profitable: bool = False
    rejection_reason: Optional[str] = None
    wallet_address: Optional[str] = None


@dataclass
class EarningsRecord:
    timestamp: str
    source: str
    amount_usd: float
    task_id: str
    status: str  # claimed | pending | confirmed | rejected
    wallet_address: Optional[str] = None
    activation_uncertain: bool = True


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def _fetch_url(url: str, timeout: int = 15) -> bytes:
    req = urlreq.Request(
        url,
        headers={"User-Agent": "Synta/1.0 (autonomous research; contact: synta-bot)"},
    )
    with urlreq.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _fetch_graphql(url: str, query: str, variables: dict | None = None, timeout: int = 15) -> dict:
    payload = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urlreq.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "Synta/1.0",
        },
    )
    with urlreq.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _estimate_hours_from_text(text: str) -> float:
    """Rough estimate: look for explicit time mentions, else default to 2h."""
    t = text.lower()
    patterns = [
        (r"(\d+)\s*hour", 1.0),
        (r"(\d+)\s*hr", 1.0),
        (r"(\d+)\s*h\b", 1.0),
        (r"(\d+)\s*day", 8.0),
        (r"(\d+)\s*week", 40.0),
    ]
    for pattern, mult in patterns:
        m = re.search(pattern, t)
        if m:
            val = float(m.group(1)) * mult
            return val
    return 2.0  # default assumption: 2 hours


# ---------------------------------------------------------------------------
# Source fetchers
# ---------------------------------------------------------------------------

def fetch_gitcoin_bounties() -> list[BountyTask]:
    """Gitcoin v2 public bounties API."""
    tasks: list[BountyTask] = []
    try:
        url = "https://gitcoin.co/api/v0.1/bounties/?network=mainnet&status=open&limit=20&order_by=-_val_in_usdt"
        raw = _fetch_url(url)
        data = json.loads(raw)
        for item in data[:20]:
            value = float(item.get("value_in_usdt") or item.get("_val_in_usdt") or 0)
            title = item.get("title", "")
            desc = (item.get("description") or "")[:500]
            task_id = f"gitcoin_{item.get('pk', _sha256(title)[:12])}"
            est_h = _estimate_hours_from_text(title + " " + desc)
            tasks.append(BountyTask(
                task_id=task_id,
                source="gitcoin",
                title=title,
                description=desc,
                category=item.get("project_type", "bounty"),
                url=item.get("url", ""),
                payout_usd=value,
                estimated_hours=est_h,
                fetched_at=_now_iso(),
            ))
    except Exception as exc:
        logger.warning("Gitcoin fetch failed: %s", exc)
    return tasks


def fetch_layer3_bounties() -> list[BountyTask]:
    """Layer3 public GraphQL API."""
    tasks: list[BountyTask] = []
    query = """
    query {
      quests(filter: {status: ACTIVE}, first: 20, orderBy: REWARD_DESC) {
        edges {
          node {
            id title description rewardUSD estimatedTime
            url
          }
        }
      }
    }
    """
    try:
        data = _fetch_graphql("https://api.layer3.xyz/graphql", query)
        edges = data.get("data", {}).get("quests", {}).get("edges", [])
        for edge in edges:
            node = edge.get("node", {})
            task_id = f"layer3_{node.get('id', _sha256(node.get('title', ''))[:12])}"
            est_h = float(node.get("estimatedTime") or 2.0)
            tasks.append(BountyTask(
                task_id=task_id,
                source="layer3",
                title=node.get("title", ""),
                description=(node.get("description") or "")[:500],
                category="web3_quest",
                url=node.get("url", "https://app.layer3.xyz"),
                payout_usd=float(node.get("rewardUSD") or 0),
                estimated_hours=est_h,
                fetched_at=_now_iso(),
            ))
    except Exception as exc:
        logger.warning("Layer3 fetch failed: %s", exc)
    return tasks


def fetch_dework_bounties() -> list[BountyTask]:
    """Dework public GraphQL API."""
    tasks: list[BountyTask] = []
    query = """
    query GetTasks($input: GetTasksInput!) {
      getTasks(input: $input) {
        id name description reward { amount currency { symbol } }
        permalink status dueDate
      }
    }
    """
    variables = {"input": {"statuses": ["TODO"], "limit": 20}}
    try:
        data = _fetch_graphql("https://api.dework.xyz/graphql", query, variables)
        items = data.get("data", {}).get("getTasks", []) or []
        for item in items:
            reward = item.get("reward") or {}
            amount = float(reward.get("amount") or 0)
            symbol = (reward.get("currency") or {}).get("symbol", "USD")
            if symbol not in ("USD", "USDC", "USDT"):
                amount = amount * 0.0  # non-stable: treat as $0 until price feed
            title = item.get("name", "")
            desc = (item.get("description") or "")[:500]
            task_id = f"dework_{item.get('id', _sha256(title)[:12])}"
            est_h = _estimate_hours_from_text(title + " " + desc)
            tasks.append(BountyTask(
                task_id=task_id,
                source="dework",
                title=title,
                description=desc,
                category="bounty",
                url=item.get("permalink", "https://app.dework.xyz"),
                payout_usd=amount,
                estimated_hours=est_h,
                fetched_at=_now_iso(),
            ))
    except Exception as exc:
        logger.warning("Dework fetch failed: %s", exc)
    return tasks


def fetch_freelancer_small_tasks() -> list[BountyTask]:
    """Freelancer.com small fixed-price projects (< $50, newest first)."""
    tasks: list[BountyTask] = []
    try:
        url = (
            "https://www.freelancer.com/api/projects/0.1/projects/"
            "?job_details=true&full_details=true&project_types[]=fixed"
            "&min_avg_price=5&max_avg_price=50&limit=20&sort_field=time_updated&compact=true"
        )
        raw = _fetch_url(url)
        data = json.loads(raw)
        projects = data.get("result", {}).get("projects", []) or []
        for p in projects[:15]:
            budget = p.get("budget") or {}
            min_b = float(budget.get("minimum") or 0)
            max_b = float(budget.get("maximum") or 0)
            payout = (min_b + max_b) / 2.0 if max_b > 0 else min_b
            title = p.get("title", "")
            desc = (p.get("description") or "")[:500]
            task_id = f"freelancer_{p.get('id', _sha256(title)[:12])}"
            est_h = _estimate_hours_from_text(title + " " + desc)
            tasks.append(BountyTask(
                task_id=task_id,
                source="freelancer",
                title=title,
                description=desc,
                category=p.get("type", "project"),
                url=f"https://www.freelancer.com/projects/{p.get('seo_url', p.get('id', ''))}",
                payout_usd=payout,
                estimated_hours=est_h,
                fetched_at=_now_iso(),
            ))
    except Exception as exc:
        logger.warning("Freelancer small tasks fetch failed: %s", exc)
    return tasks


# ---------------------------------------------------------------------------
# Profitability filter
# ---------------------------------------------------------------------------

def evaluate_profitability(task: BountyTask) -> BountyTask:
    """Apply profitability rules; sets task.profitable and task.rejection_reason."""
    text = (task.title + " " + task.description).lower()

    # Hard disqualifiers
    for kw in DISQUALIFY_KEYWORDS:
        if kw in text:
            task.profitable = False
            task.rejection_reason = f"disqualifier_keyword:{kw}"
            task.status = "skipped"
            return task

    # Must have qualifying AI-executable keyword
    if not any(kw in text for kw in QUALIFY_KEYWORDS):
        task.profitable = False
        task.rejection_reason = "no_qualifying_keyword"
        task.status = "skipped"
        return task

    # Payout threshold
    if task.payout_usd < MIN_PAYOUT_USD:
        task.profitable = False
        task.rejection_reason = f"payout_too_low:{task.payout_usd:.2f}"
        task.status = "skipped"
        return task

    # Duration threshold
    if task.estimated_hours > MAX_DURATION_H:
        task.profitable = False
        task.rejection_reason = f"duration_too_long:{task.estimated_hours:.1f}h"
        task.status = "skipped"
        return task

    task.profitable = True
    task.status = "accepted"
    task.wallet_address = WALLET_ETH  # may be None until Mantie provides
    return task


# ---------------------------------------------------------------------------
# Earnings log
# ---------------------------------------------------------------------------

def log_earnings(task: BountyTask) -> None:
    """Append accepted task to earnings_log.json."""
    records: list[dict] = []
    if EARNINGS_LOG.exists():
        try:
            records = json.loads(EARNINGS_LOG.read_text())
        except Exception:
            pass
    record = EarningsRecord(
        timestamp=_now_iso(),
        source=task.source,
        amount_usd=task.payout_usd,
        task_id=task.task_id,
        status="claimed",
        wallet_address=task.wallet_address,
        activation_uncertain=True,  # always uncertain until external confirmation
    )
    records.append(asdict(record))
    EARNINGS_LOG.write_text(json.dumps(records, indent=2))
    logger.info(
        "Logged earnings: %s $%.2f source=%s wallet=%s",
        task.task_id, task.payout_usd, task.source, task.wallet_address or "NOT_SET",
    )


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def _load_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"seen_ids": [], "last_poll": None, "total_accepted": 0, "total_skipped": 0}


def _save_state(state: dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_once() -> list[BountyTask]:
    state = _load_state()
    seen: set[str] = set(state.get("seen_ids", []))

    all_tasks: list[BountyTask] = []
    all_tasks.extend(fetch_gitcoin_bounties())
    all_tasks.extend(fetch_layer3_bounties())
    all_tasks.extend(fetch_dework_bounties())
    all_tasks.extend(fetch_freelancer_small_tasks())

    new_tasks = [t for t in all_tasks if t.task_id not in seen]
    logger.info("Fetched %d tasks total, %d new", len(all_tasks), len(new_tasks))

    accepted: list[BountyTask] = []
    skipped = 0

    for task in new_tasks:
        task = evaluate_profitability(task)
        if task.profitable:
            log_earnings(task)
            accepted.append(task)
            state["total_accepted"] = state.get("total_accepted", 0) + 1
            logger.info(
                "ACCEPTED: %s $%.2f est=%.1fh src=%s url=%s",
                task.task_id, task.payout_usd, task.estimated_hours, task.source, task.url,
            )
        else:
            skipped += 1
            state["total_skipped"] = state.get("total_skipped", 0) + 1
            logger.debug("skipped %s reason=%s", task.task_id, task.rejection_reason)
        seen.add(task.task_id)

    state["seen_ids"] = list(seen)[-5000:]
    state["last_poll"] = _now_iso()
    _save_state(state)

    logger.info("Evaluation complete: %d accepted, %d skipped", len(accepted), skipped)
    return accepted


def run_forever() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info(
        "BountyEvaluator starting — poll=%ds min_usd=%.2f max_h=%.1f wallet=%s",
        POLL_INTERVAL_S, MIN_PAYOUT_USD, MAX_DURATION_H, WALLET_ETH or "NOT_SET",
    )
    while True:
        try:
            accepted = evaluate_once()
            if accepted:
                logger.info("Poll result: %d profitable tasks found", len(accepted))
        except Exception as exc:
            logger.error("Poll error: %s", exc)
        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    run_forever()
