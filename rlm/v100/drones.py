"""Persistent, bounded source/CPU/RTX jobs independent of the V100 training phase."""

import json
import re
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from rlm.v100.activity import ActivityLog
from rlm.v100.common import atomic_json, load_profile


@contextmanager
def connect(root: Path):
    path = root / "research/state/drones.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute(
            "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, branch TEXT, kind TEXT, payload TEXT, interval INTEGER, due REAL, state TEXT, updated REAL, result TEXT)"
        )
        db.commit()
        with db:
            yield db
    finally:
        db.close()


def schedule(root: Path, branch: str, kind: str, payload: str, interval: int) -> dict:
    if branch not in ("A", "B") or kind not in (
        "source",
        "python",
        "researcher",
        "critic",
        "desktop",
        "benchmark",
        "compute-audit",
    ):
        raise ValueError("Choose A/B and source/python/researcher/critic/desktop")
    if kind == "compute-audit" and (
        not re.fullmatch(r"[a-f0-9]{24}", payload)
        or not (root / "research/compute-jobs" / payload / "state.json").is_file()
    ):
        raise ValueError("Compute audit requires a registered host job")
    maximum = 12000 if kind in ("python", "desktop") else 1500 if kind == "source" else 400
    if not isinstance(payload, str) or not 1 <= len(payload) <= maximum:
        raise ValueError("Drone payload exceeds its budget")
    if type(interval) is not int or interval != 0 and not 300 <= interval <= 86400:
        raise ValueError("Drone interval: 0 once, or 300-86400 seconds")
    if kind == "source":
        from rlm.v100.research_tools import public_origin

        public_origin(payload)
    if kind == "python":
        compile(payload, "drone.py", "exec")
    with connect(root) as db:
        db.execute("BEGIN IMMEDIATE")
        old = db.execute(
            "SELECT id FROM jobs WHERE branch=? AND kind=? AND payload=? AND state!='cancelled'",
            (branch, kind, payload),
        ).fetchone()
        if old:
            return {"id": old["id"], "status": "already scheduled"}
        if (
            db.execute("SELECT count(*) FROM jobs WHERE state IN ('queued','running')").fetchone()[
                0
            ]
            >= 16
        ):
            raise ValueError("Sixteen active jobs already scheduled; cancel or finish some first")
        identity = uuid.uuid4().hex[:16]
        db.execute(
            "INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?)",
            (identity, branch, kind, payload, interval, time.time(), "queued", time.time(), None),
        )
    return {"id": identity, "status": "queued", "kind": kind, "interval": interval}


def inspect(root: Path) -> list[dict]:
    with connect(root) as db:
        rows = db.execute(
            "SELECT id,branch,kind,interval,state,updated,result FROM jobs ORDER BY updated DESC LIMIT 32"
        ).fetchall()
    return [
        dict(row) | {"result": json.loads(row["result"]) if row["result"] else None} for row in rows
    ]


def cancel(root: Path, identity: str) -> dict:
    with connect(root) as db:
        row = db.execute("SELECT state FROM jobs WHERE id=?", (identity,)).fetchone()
        if row is None:
            raise ValueError("Unknown drone")
        db.execute(
            "UPDATE jobs SET state='cancelled',updated=? WHERE id=?", (time.time(), identity)
        )
    return {"id": identity, "status": "cancelled; an in-flight bounded task may finish"}


def execute(root: Path, job: dict) -> dict:
    if job["kind"] == "compute-audit":
        from rlm.v100.distributed_compute import validate_locally

        return validate_locally(root, job["payload"])
    if job["kind"] == "benchmark":
        from rlm.v100.public_benchmarks import evaluate
        from rlm.v100.remote_helper import selected_helper

        profile = load_profile(selected_helper(root), root)
        destination = root / "research/public-benchmarks/results" / f"rtx-{time.time_ns()}.json"
        result = evaluate(root, profile, destination)
        return {
            "report": str(destination),
            "panel_macro_mean": result["panel_macro_mean"],
            "scope": result["scope"],
        }
    if job["kind"] == "desktop":
        from rlm.v100.desktop import run

        return run(root, job["payload"])
    if job["kind"] == "python":
        from rlm.v100.research_sandbox import run

        return run(root, job["branch"], job["payload"])
    if job["kind"] == "source":
        from rlm.v100.research_tools import ResearchTools

        return ResearchTools(root, {}, job["branch"]).execute(
            "read_public_page", {"url": job["payload"]}
        )
    from rlm.v100.competition import helper_client
    from rlm.v100.goals import load_goal
    from rlm.v100.remote_helper import remote_profile, selected_helper
    from rlm.v100.researchers import research_task

    profile = load_profile(selected_helper(root), root)
    if not remote_profile(profile):
        raise ValueError(
            "Resident LLM drones require the external RTX; no extra local model is loaded"
        )
    profile["runtime"]["max_timeout"] = min(120, profile["runtime"]["max_timeout"])
    client = helper_client(profile, root, job["branch"])
    client.identity()
    client.activity_actor = "resident-" + job["kind"]
    client.research_tool_names = {
        "mission_evidence",
        "read_tool_result",
        "read_public_page",
        "parallel_source_research",
        "search_memory",
        "read_source",
        "calculate",
        "backtest_prices",
        "paper_status",
        "paper_observed_results",
        "schedule_drone",
        "drone_status",
        "propose_foundation_trial",
        "foundation_trial_status",
        "request_fresh_curriculum",
        "public_feed_status",
        "register_public_feed",
        "reward_policy_status",
        "train_reward_policy",
        "consult_browser_model",
        "sandbox_gui",
        "get_plan",
        "set_plan",
    }
    return research_task(
        client,
        job["branch"],
        {"role": job["kind"], "brief": job["payload"]},
        [{"goal": load_goal(root)}],
        root,
    )


def finish(root: Path, identity: str, value: dict) -> None:
    with connect(root) as db:
        row = db.execute("SELECT * FROM jobs WHERE id=?", (identity,)).fetchone()
        interval = row["interval"]
        state = (
            "cancelled" if row["state"] == "cancelled" else "queued" if interval else "completed"
        )
        if (
            row["kind"] == "compute-audit"
            and value.get("status") == "failed"
            and state != "cancelled"
        ):
            source = root / "research/compute-jobs" / row["payload"] / "state.json"
            job_state = json.loads(source.read_text())
            retries = job_state.get("validation_retries", 0) + 1
            job_state["validation_retries"] = retries
            job_state["validation_error"] = value.get("detail", "Local validation failed")
            if retries < 3:
                state, interval = "queued", 60
            else:
                job_state["state"] = "validation-deferred"
            atomic_json(source, job_state)
        db.execute(
            "UPDATE jobs SET state=?,due=?,updated=?,result=? WHERE id=?",
            (state, time.time() + interval, time.time(), json.dumps(value), identity),
        )
    ActivityLog(root, row["branch"], "drone").write(
        "steps", "drone-result", {"id": identity, **value}
    )


def service(root: Path, stop: threading.Event) -> None:
    # A single RTX job plus two CPU/network jobs. RTX requests share the global
    # existing helper pacing lock with foreground researchers and benchmark clients.
    with ThreadPoolExecutor(max_workers=3) as workers:
        active = {}
        while not stop.is_set():
            from rlm.v100.distributed_compute import tick

            try:
                tick(root)
            except (ValueError, OSError, RuntimeError) as error:
                ActivityLog(root, "controller", "compute").write(
                    "errors", "compute-mailbox-unavailable", {"detail": str(error)[:400]}
                )
            for identity, (future, _kind) in list(active.items()):
                if future.done():
                    try:
                        result = future.result()
                    except Exception as error:
                        result = {
                            "status": "failed",
                            "error": type(error).__name__,
                            "detail": str(error)[:400],
                        }
                    finish(root, identity, result)
                    del active[identity]
            with connect(root) as db:
                jobs = db.execute(
                    "SELECT * FROM jobs WHERE state='queued' AND due<=? ORDER BY due LIMIT 16",
                    (time.time(),),
                ).fetchall()
                for row in jobs:
                    llm = row["kind"] in ("researcher", "critic", "benchmark")
                    used = sum(
                        (kind in ("researcher", "critic", "benchmark")) == llm
                        for _, kind in active.values()
                    )
                    if used >= (1 if llm else 2):
                        continue
                    updated = db.execute(
                        "UPDATE jobs SET state='running',updated=? WHERE id=? AND state='queued'",
                        (time.time(), row["id"]),
                    )
                    if updated.rowcount:
                        active[row["id"]] = (workers.submit(execute, root, dict(row)), row["kind"])
            atomic_json(
                root / "research/drones-status.json",
                {
                    "updated": time.time(),
                    "running": True,
                    "active": list(active),
                    "cpu_slots": 2,
                    "rtx_slots": 1,
                },
            )
            stop.wait(2)
        for identity, (future, _) in active.items():
            try:
                finish(root, identity, future.result())
            except Exception as error:
                finish(root, identity, {"status": "failed", "detail": str(error)[:400]})


@contextmanager
def alongside(root: Path):
    # Only one owned mission creates resident workers. Interrupted read-only jobs
    # are explicitly retried, with their prior results retained in activity logs.
    with connect(root) as db:
        db.execute("UPDATE jobs SET state='queued',due=? WHERE state='running'", (time.time(),))
    for branch, role in (("A", "researcher"), ("B", "critic")):
        seed(
            root,
            branch,
            role,
            "Investigate the operator-owned goal and useful self-upgrades using fresh public evidence. Read previous findings, avoid repeated hypotheses, select one concrete next experiment and schedule a useful source or CPU task. Check net costs and falsify weak claims.",
            900,
        )
    if (root / "research/public-benchmarks/current.json").exists():
        seed(root, "B", "benchmark", "Pinned public panel for RTX helper", 86400)
    log = root / "research/logs/resident-drones.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as handle:
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", "rlm.v100.drones", str(root)],
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        stopping = threading.Event()

        def monitor():
            nonlocal process
            while not stopping.wait(2):
                if process.poll() is None:
                    continue
                atomic_json(
                    root / "research/drones-status.json",
                    {
                        "updated": time.time(),
                        "running": False,
                        "state": "worker exited; retry in 30s",
                        "exit_code": process.returncode,
                    },
                )
                if stopping.wait(30):
                    return
                try:
                    with connect(root) as db:
                        db.execute(
                            "UPDATE jobs SET state='queued',due=? WHERE state='running'",
                            (time.time(),),
                        )
                    process = subprocess.Popen(
                        [sys.executable, "-u", "-m", "rlm.v100.drones", str(root)],
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                    )
                except OSError as error:
                    ActivityLog(root, "controller", "drone").write(
                        "errors", "drone-restart-failed", {"detail": str(error)[:300]}
                    )

        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
        try:
            yield
        finally:
            stopping.set()
            watcher.join(timeout=5)
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            atomic_json(
                root / "research/drones-status.json", {"updated": time.time(), "running": False}
            )


def seed(root: Path, branch: str, kind: str, payload: str, interval: int) -> None:
    try:
        schedule(root, branch, kind, payload, interval)
    except ValueError as error:
        if "Sixteen active jobs" not in str(error):
            raise
        ActivityLog(root, branch, "drone").write(
            "steps",
            "default-drone-deferred",
            {"kind": kind, "reason": "Persistent queue full; existing jobs continue"},
        )


if __name__ == "__main__":
    service(Path(sys.argv[1]), threading.Event())
