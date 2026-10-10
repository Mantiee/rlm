"""Count actual public retrievals, distinguishing unchanged content from new evidence."""

import sqlite3
import time
from pathlib import Path

from rlm.v100.common import atomic_json
from rlm.v100.goals import load_goal


def record(root: Path, url: str, sha256: str) -> dict:
    goal_id = (load_goal(root) or {}).get("id", "unattributed")
    folder = root / "research/source-acquisition"
    folder.mkdir(parents=True, exist_ok=True)
    now = time.time()
    db = sqlite3.connect(folder / "ledger.sqlite3", timeout=5)
    try:
        with db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS reads(goal TEXT,url TEXT,sha TEXT,first REAL,last REAL,count INTEGER,PRIMARY KEY(goal,url,sha))"
            )
            db.execute("BEGIN IMMEDIATE")
            now = time.time()
            previous = db.execute(
                "SELECT count FROM reads WHERE goal=? AND url=? AND sha=?", (goal_id, url, sha256)
            ).fetchone()
            if previous:
                db.execute(
                    "UPDATE reads SET last=?,count=count+1 WHERE goal=? AND url=? AND sha=?",
                    (now, goal_id, url, sha256),
                )
            else:
                db.execute(
                    "INSERT INTO reads VALUES(?,?,?,?,?,1)", (goal_id, url, sha256, now, now)
                )
            count, versions, reads, urls = db.execute(
                "SELECT count(DISTINCT sha),count(*),coalesce(sum(count),0),count(DISTINCT url) FROM reads WHERE goal=?",
                (goal_id,),
            ).fetchone()
            recent = [
                {
                    "url": row[0],
                    "sha256": row[1],
                    "first_fetched": row[2],
                    "last_fetched": row[3],
                    "fetch_count": row[4],
                }
                for row in db.execute(
                    "SELECT url,sha,first,max(last),count FROM reads WHERE goal=? GROUP BY url ORDER BY max(last) DESC LIMIT 12",
                    (goal_id,),
                )
            ]
            value = {
                "goal_id": goal_id,
                "updated": now,
                "distinct_contents": count,
                "url_content_versions": versions,
                "distinct_urls": urls,
                "total_fetches": reads,
                "unchanged_rereads": reads - versions,
                "latest": recent,
                "scope": "Actual retrieval receipts since instrumentation. New content is not verified truth, task success, learning or income.",
            }
            atomic_json(folder / "status.json", value)
    finally:
        db.close()
    return {
        "content_state": "unchanged reread" if previous else "new content for this goal",
        "sha256": sha256,
        "fetched_at": now,
    }
