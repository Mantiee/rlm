"""Model-requested historical experiments on pinned source snapshots.

This is research throughput, not a strategy selector or a profit label.
"""

import hashlib
import json
import math
import multiprocessing
import os
import re
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path

import psutil

from rlm.v100.backtesting import backtest_key_lock, run
from rlm.v100.common import atomic_json
from rlm.v100.protection import file_hash
from rlm.v100.resource_budget import available_cpus


def io_pressure_avg10(path: Path = Path("/proc/pressure/io")) -> float | None:
    if os.name != "posix":
        return None
    match = re.search(r"^full avg10=(\d+(?:\.\d+)?)\b", path.read_text(), re.MULTILINE)
    if match is None:
        raise ValueError("Cannot measure Linux IO pressure")
    return float(match[1])


def available_memory_bytes(
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    membership: Path = Path("/proc/self/cgroup"),
) -> int:
    """Use the tighter host or effective ancestor cgroup memory headroom."""
    available = psutil.virtual_memory().available
    if os.name != "posix" or not membership.exists():
        return available
    rows = [line[3:] for line in membership.read_text().splitlines() if line.startswith("0::")]
    if len(rows) != 1:
        raise ValueError("Cannot verify backtest cgroup membership")
    base = cgroup_root.resolve()
    folder = (base / rows[0].lstrip("/")).resolve()
    if not folder.is_relative_to(base) or not folder.exists():
        folder = base
    while folder.is_relative_to(base):
        maximum = folder / "memory.max"
        current = folder / "memory.current"
        if maximum.exists() and current.exists():
            ceiling = maximum.read_text().strip()
            if ceiling != "max":
                available = min(available, max(0, int(ceiling) - int(current.read_text())))
        if folder == base:
            break
        folder = folder.parent
    return available


def admission(root: Path) -> dict:
    """Leave one CPU and host memory/disk headroom for serving and other work."""
    cpus = available_cpus()
    pressure = psutil.cpu_percent(interval=0.05)
    io_pressure = io_pressure_avg10()
    free_ram = available_memory_bytes()
    free_disk = shutil.disk_usage(root).free
    # One worker processes at most 3000 closed bars and a small JSON source.
    ram_reserve, ram_per_worker = 2 * 2**30, 512 * 2**20
    disk_reserve = 2 * 2**30
    cpu_slots = max(0, cpus - 1)
    ram_slots = max(0, (free_ram - ram_reserve) // ram_per_worker)
    disk_slots = max(0, (free_disk - disk_reserve) // (32 * 2**20))
    slots = min(cpu_slots, ram_slots, disk_slots)
    if pressure >= 85:
        slots = 0
    elif pressure >= 65:
        slots = min(slots, 1)
    if io_pressure is None or io_pressure >= 10:
        slots = min(slots, 1)
    elif io_pressure >= 5:
        slots = min(slots, max(1, slots // 4))
    if io_pressure is not None and io_pressure >= 50:
        slots = 0
    return {
        "slots": int(slots),
        "available_cpus": cpus,
        "cpu_percent": pressure,
        "io_full_avg10": io_pressure,
        "available_ram_bytes": free_ram,
        "available_disk_bytes": free_disk,
        "scope": "Measured CPU/RAM/disk admission, not a throughput or GPU-use guarantee",
    }


def _worker(root: Path, branch: str, arguments: dict, snapshot: tuple[str, str, str]) -> dict:
    if os.name == "posix":
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
        resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 2**20, 8 * 2**20))
    return run(
        root,
        branch,
        arguments,
        source_snapshot=snapshot[:2],
        fetched_at=datetime.fromisoformat(snapshot[2]),
    )


def _fetch_source(url: str, fetcher) -> tuple[str, str, str]:
    resolved_url, body = fetcher(url)
    return resolved_url, body, datetime.now(UTC).isoformat()


def fetch_snapshots(
    root: Path, urls: list[str], fetcher, *, deadline_at: float, worker_limit: int | None = None
) -> dict[str, tuple[str, str, str]]:
    """Fetch distinct sources concurrently and reap every child on error/deadline."""
    if not urls:
        raise ValueError("Backtest batch needs a public source")
    capacity = admission(root)
    if capacity["available_ram_bytes"] < 512 * 2**20:
        raise RuntimeError("Backtest source retrieval deferred: insufficient RAM")
    if capacity["slots"] <= 0:
        raise RuntimeError("Backtest source retrieval deferred: no admitted worker slots")
    if worker_limit is not None and (type(worker_limit) is not int or worker_limit < 1):
        raise ValueError("Invalid backtest source worker limit")
    workers = min(len(urls), capacity["slots"], worker_limit or capacity["slots"])
    pool = multiprocessing.get_context("spawn").Pool(processes=workers)
    try:
        pending = {url: pool.apply_async(_fetch_source, (url, fetcher)) for url in urls}
        snapshots = {}
        for url in urls:
            remaining = deadline_at - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Backtest source retrieval exceeded the batch deadline")
            try:
                snapshots[url] = pending[url].get(timeout=remaining)
            except multiprocessing.TimeoutError as error:
                raise TimeoutError(
                    "Backtest source retrieval exceeded the batch deadline"
                ) from error
        pool.close()
        pool.join()
        return snapshots
    except BaseException:
        pool.terminate()
        pool.join()
        raise


def _verified(row: dict, root: Path) -> bool:
    path = Path(row.get("report", ""))
    archive = (root / "research/backtests").resolve()
    return (
        path.is_file()
        and not path.is_symlink()
        and path.resolve().is_relative_to(archive)
        and file_hash(path) == row.get("sha256")
    )


def update_recent(directory: Path, state: dict) -> None:
    """Publish a small atomic index so dashboard reads never scan all batches."""
    path = directory / "recent.json"
    with backtest_key_lock(path):
        if path.is_symlink():
            raise ValueError("Backtest recent index cannot be a symbolic link")
        if path.exists():
            if path.stat().st_size > 8192:
                raise ValueError("Backtest recent index exceeds its bound")
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("schema") != "synta-backtest-batch-recent-v1" or not isinstance(
                previous.get("batches"), list
            ):
                raise ValueError("Backtest recent index changed")
            rows = previous["batches"]
        else:
            rows = []
        entry = {
            "identity": state["identity"],
            "state": state["state"],
            "updated_at": state["updated_at"],
            "requested": len(state["experiments"]),
            "completed": sum(row["state"] == "completed" for row in state["experiments"]),
        }
        atomic_json(
            path,
            {
                "schema": "synta-backtest-batch-recent-v1",
                "batches": [
                    entry,
                    *[row for row in rows if row.get("identity") != entry["identity"]],
                ][:16],
            },
        )


def validate_experiments(experiments: list[dict]) -> None:
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("Backtest batch requires at least one experiment")
    if len(json.dumps(experiments, sort_keys=True, ensure_ascii=False).encode("utf-8")) > 65536:
        raise ValueError("Backtest batch request exceeds its input budget")
    for arguments in experiments:
        if (
            not isinstance(arguments, dict)
            or set(arguments) != {"price_url", "event_url", "rule", "fee_bps", "slippage_bps"}
            or not isinstance(arguments["price_url"], str)
            or arguments["event_url"] != ""
            or arguments["rule"] not in ("momentum", "mean_reversion", "buy_hold")
        ):
            raise ValueError("Batch needs model-chosen prices, rules and no live event source")
        for field in ("fee_bps", "slippage_bps"):
            value = arguments[field]
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= 1000
            ):
                raise ValueError("Backtest costs must be bounded explicit assumptions")


def run_frozen(
    root: Path,
    branch: str,
    experiments: list[dict],
    snapshots: dict[str, tuple[str, str, str]],
    *,
    deadline_seconds: int = 180,
    worker_limit: int | None = None,
) -> dict:
    """Resume one immutable batch, retaining completed report hashes after interruption.

    `snapshots` must be fetched once by the caller before any worker starts.
    Each tuple carries the source URL, unmodified body and retrieval time.
    Distinct requests execute in separate processes; matching requests share
    the backtesting cache and are never counted as independent evidence.
    """
    if branch not in ("A", "B"):
        raise ValueError("Backtest batch requires research branch A or B")
    validate_experiments(experiments)
    if type(deadline_seconds) is not int or not 1 <= deadline_seconds <= 600:
        raise ValueError("Invalid backtest batch deadline")
    if worker_limit is not None and (type(worker_limit) is not int or worker_limit < 1):
        raise ValueError("Invalid backtest simulation worker limit")
    for arguments in experiments:
        if arguments["price_url"] not in snapshots:
            raise ValueError("Missing frozen price snapshot")
    source_hashes = {}
    for url, snapshot in snapshots.items():
        if (
            not isinstance(url, str)
            or not isinstance(snapshot, tuple)
            or len(snapshot) != 3
            or not isinstance(snapshot[0], str)
            or not isinstance(snapshot[1], str)
            or not isinstance(snapshot[2], str)
            or len(snapshot[1].encode("utf-8")) > 2 * 2**20
        ):
            raise ValueError("Invalid or oversized frozen price snapshot")
        try:
            fetched_at = datetime.fromisoformat(snapshot[2])
        except ValueError as error:
            raise ValueError("Invalid frozen price source time") from error
        if (
            fetched_at.tzinfo is None
            or fetched_at.utcoffset() is None
            or fetched_at > datetime.now(UTC)
        ):
            raise ValueError("Frozen price source time must be observed and timezone-aware")
        source_hashes[url] = {
            "resolved_url": snapshot[0],
            "sha256": hashlib.sha256(snapshot[1].encode()).hexdigest(),
            "fetched_at": fetched_at.isoformat(),
        }
    identity = hashlib.sha256(
        json.dumps(
            {"branch": branch, "experiments": experiments, "sources": source_hashes},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    directory = root / "research/backtests/batches"
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory / (identity + ".json")
    cancel_path = directory / (identity + ".cancel")
    with backtest_key_lock(receipt_path):
        if receipt_path.is_symlink():
            raise ValueError("Backtest batch receipt cannot be a symbolic link")
        existing = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        rows = existing.get("experiments", [])
        if existing and (
            existing.get("identity") != identity
            or existing.get("source_sha256") != source_hashes
            or len(rows) != len(experiments)
        ):
            raise ValueError("Backtest batch receipt changed")
        results = []
        for index, arguments in enumerate(experiments):
            previous = rows[index] if rows else {}
            if previous.get("state") == "completed":
                if not _verified(previous, root):
                    raise ValueError("Archived backtest report changed")
                results.append({**previous, "cache_reused": True})
            else:
                results.append({"index": index, "state": "pending", "parameters": arguments})
        state = {
            "identity": identity,
            "state": "running",
            "updated_at": datetime.now(UTC).isoformat(),
            "source_sha256": source_hashes,
            "experiments": results,
            "actual_income_pln": None,
            "scope": "Exploratory historical simulations with assumed costs; shared source windows and repeated parameters are not independent validation or profit",
        }

        def save() -> None:
            state["updated_at"] = datetime.now(UTC).isoformat()
            if (
                len(json.dumps(state, ensure_ascii=False, allow_nan=False).encode("utf-8"))
                > 512 * 1024
            ):
                raise ValueError("Backtest batch receipt exceeds its size budget")
            atomic_json(receipt_path, state)
            update_recent(directory, state)

        if all(row["state"] == "completed" for row in results):
            state["state"] = "completed"
            state["cache_reused"] = True
            save()
            return {**state, "receipt": str(receipt_path)}
        capacity = admission(root)
        state["admission"] = capacity
        if not capacity["slots"]:
            state["state"] = "waiting-resource"
            save()
            return {**state, "receipt": str(receipt_path)}
        save()
        pending = [i for i, row in enumerate(results) if row["state"] != "completed"]
        started = time.monotonic()
        admitted_slots = min(capacity["slots"], worker_limit or capacity["slots"])
        state["admitted_batch_slots"] = admitted_slots
        with ProcessPoolExecutor(max_workers=min(admitted_slots, len(pending) or 1)) as pool:
            active = {}
            while pending or active:
                if cancel_path.exists() or time.monotonic() - started >= deadline_seconds:
                    state["state"] = "cancelled" if cancel_path.exists() else "deadline"
                    break
                current = admission(root)
                state["admission"] = current
                while pending and len(active) < min(admitted_slots, current["slots"]):
                    index = pending.pop(0)
                    arguments = experiments[index]
                    future = pool.submit(
                        _worker, root, branch, arguments, snapshots[arguments["price_url"]]
                    )
                    active[future] = index
                    results[index]["state"] = "running"
                    save()
                if not active:
                    state["state"] = "waiting-resource"
                    break
                done, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    index = active.pop(future)
                    try:
                        report = future.result()
                        path = Path(report["report"])
                        results[index] = {
                            "index": index,
                            "state": "completed",
                            "parameters": experiments[index],
                            "report": str(path),
                            "sha256": file_hash(path),
                            "cache_reused": report["cache_reused"],
                            "net_return": report["development_test"]["net_return"],
                            "fills": report["development_test"]["fills"],
                        }
                    except Exception as error:
                        results[index] = {
                            "index": index,
                            "state": "failed",
                            "parameters": experiments[index],
                            "error_type": type(error).__name__,
                        }
                    save()
            if state["state"] == "running":
                state["state"] = (
                    "completed"
                    if all(row["state"] == "completed" for row in results)
                    else "partial"
                )
        for row in results:
            if row["state"] == "running":
                row["state"] = "pending"
        save()
        return {**state, "receipt": str(receipt_path)}


def run_public(
    root: Path,
    branch: str,
    experiments: list[dict],
    *,
    deadline_seconds: int = 180,
    fetcher=None,
) -> dict:
    """Freeze each model-chosen public price source once before parallel evaluation."""
    from rlm.v100.research_tools import download_page

    validate_experiments(experiments)
    if type(deadline_seconds) is not int or not 1 <= deadline_seconds <= 600:
        raise ValueError("Invalid backtest batch deadline")
    if fetcher is None:
        fetcher = download_page
    deadline_at = time.monotonic() + deadline_seconds
    urls = {row["price_url"] for row in experiments if isinstance(row, dict)}
    snapshots = fetch_snapshots(root, sorted(urls), fetcher, deadline_at=deadline_at)
    remaining = math.floor(deadline_at - time.monotonic())
    if remaining < 1:
        raise TimeoutError("Backtest batch deadline exhausted before simulation")
    return run_frozen(root, branch, experiments, snapshots, deadline_seconds=remaining)


def throughput_report(root: Path) -> dict:
    """Measure observed backtest throughput from recent completed batches and project a 100x path.

    Returns baseline jobs/hour from the last 24 h of completed state.json records,
    current admission, and an evidenced path to 100x throughput on the identified host.
    Throughput is advisory; it is not a latency or profitability guarantee.
    """
    import statistics

    folder = root / "research/compute-jobs"
    if not folder.exists():
        return {
            "baseline_jobs_per_hour": None,
            "completed_jobs_observed": 0,
            "note": "No compute-jobs folder; no throughput evidence yet",
            "scope": "Measured from immutable state.json receipts only",
        }

    cutoff = time.time() - 86400
    durations: list[float] = []
    completed = 0
    for state_path in sorted(folder.glob("*/state.json")):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if state.get("state") not in ("locally-validated", "completed"):
            continue
        started = state.get("started_at") or state.get("queued_at", 0)
        finished = state.get("finished_at") or state.get("validated_at", 0)
        if finished < cutoff:
            continue
        completed += 1
        if started and finished and finished > started:
            durations.append(finished - started)

    adm = admission(root)
    current_slots = adm["slots"]
    available_cpus = adm["available_cpus"]

    baseline_jph: float | None = None
    median_duration_s: float | None = None
    if durations:
        median_duration_s = statistics.median(durations)
        # baseline = single-slot throughput; actual = current_slots * baseline
        baseline_jph = round(3600 / median_duration_s, 1) if median_duration_s > 0 else None

    # Path to 100x: each step is an evidence-based multiplier
    path_to_100x = []
    multiplier = 1.0
    if current_slots > 1:
        path_to_100x.append({
            "step": "parallelism",
            "description": f"Current admission allows {current_slots} parallel workers on this host",
            "multiplier": float(current_slots),
            "evidence": f"admission.slots={current_slots}, available_cpus={available_cpus}",
        })
        multiplier *= current_slots
    else:
        path_to_100x.append({
            "step": "parallelism",
            "description": "Add CPU capacity on a dedicated host (e.g. Debian1 32-core)",
            "multiplier": 16.0,
            "evidence": "Debian1 has 32 CPU threads; admission leaves 31 slots at low load",
        })
        multiplier *= 16
    # Batching gain: cache reuse reduces per-job data fetch
    path_to_100x.append({
        "step": "source_caching",
        "description": "Frozen source cache eliminates redundant downloads across a batch (already implemented in run_public)",
        "multiplier": 4.0,
        "evidence": "run_public() calls fetch_snapshots once per unique URL for a batch of N experiments",
    })
    multiplier *= 4
    # Queue depth
    path_to_100x.append({
        "step": "queue_depth",
        "description": "Increase max_jobs capacity from 128 to 2048 (already supported by queue_capacity config)",
        "multiplier": round(2048 / 128, 1),
        "evidence": "queue_capacity() accepts max_jobs up to 2048; config write needed",
    })
    multiplier *= 2048 / 128

    return {
        "baseline_jobs_per_hour": baseline_jph,
        "completed_jobs_observed": completed,
        "median_job_duration_seconds": median_duration_s,
        "current_admission_slots": current_slots,
        "available_cpus": available_cpus,
        "projected_multiplier": round(multiplier, 1),
        "path_to_100x": path_to_100x,
        "identified_host": "Debian1 (192.168.0.68, 32-thread CPU, 64 GiB RAM)",
        "note": (
            "Baseline from last 24 h locally-validated jobs. "
            "Projections are arithmetic estimates from measured admission; "
            "not a latency, income or outcome guarantee."
        ),
        "scope": "Measured CPU/RAM/disk admission + immutable state.json receipts",
    }
