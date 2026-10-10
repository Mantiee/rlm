"""Owned-computer worker: bounded CPU jobs over an authenticated shared folder.

This standalone file never imports the host controller, executes received code,
uses the RTX or turns a free managed Colab session into a distributed worker.
"""

import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path


def read_json(path: Path, limit: int = 2 * 2**20) -> dict:
    if path.is_symlink() or path.stat().st_size > limit:
        raise ValueError("Unsafe or oversized compute JSON")
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Compute JSON exceeds its limit")
    return json.loads(data)


def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "-" + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def claim(mailbox: Path, worker: str) -> tuple[Path, dict] | None:
    for path in sorted((mailbox / "jobs").glob("*.json"))[:128]:
        if not re.fullmatch(r"[a-f0-9]{24}", path.stem):
            continue
        if (mailbox / "closed" / (path.stem + ".json")).exists():
            continue
        directory = mailbox / "claims" / path.stem
        directory.parent.mkdir(parents=True, exist_ok=True)
        try:
            directory.mkdir()
        except FileExistsError:
            continue
        lease = {
            "job_id": path.stem,
            "worker": worker,
            "nonce": uuid.uuid4().hex,
            "started": time.time(),
            "heartbeat": time.time(),
        }
        atomic(directory / "lease.json", lease)
        return path, lease
    return None


def available() -> tuple[bool, str]:
    import psutil

    if psutil.virtual_memory().available < 6 * 2**30:
        return False, "Less than six GiB free host RAM"
    if psutil.cpu_percent(interval=0.1) > 65:
        return False, "Host CPU busy"
    if foreground_busy():
        return False, "Foreground heavy game; CPU experiment paused"
    if any(
        (p.info.get("name") or "").casefold() == "league of legends.exe"
        for p in psutil.process_iter(["name"])
    ):
        return False, "Game running; CPU experiment paused"
    return True, "ready"


def foreground_busy() -> bool:
    """Yield to heavy foreground games; browser presence is not load evidence."""
    if sys.platform != "win32":
        return False
    import psutil

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetForegroundWindow.restype = ctypes.c_void_p
    user32.GetWindowThreadProcessId.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    pid = ctypes.c_ulong()
    window = user32.GetForegroundWindow()
    if not window:
        return False
    user32.GetWindowThreadProcessId(window, ctypes.byref(pid))
    try:
        name = psutil.Process(pid.value).name().casefold()
    except psutil.Error:
        # A protected foreground process is a reason to yield, not crash/restart.
        return True
    return name in {
        "league of legends.exe",
        "valorant-win64-shipping.exe",
        "cs2.exe",
        "fortniteclient-win64-shipping.exe",
        "cyberpunk2077.exe",
        "overwatch.exe",
        "eldenring.exe",
        "gta5.exe",
        "rdr2.exe",
        "dota2.exe",
    }


def limit_child(pid: int) -> None:
    import psutil

    child = psutil.Process(pid)
    if sys.platform == "win32":
        child.nice(psutil.IDLE_PRIORITY_CLASS)
        child.cpu_affinity(child.cpu_affinity()[-2:])


def adaptive_child_budget(pid: int) -> dict:
    """Use spare headroom within the existing two-thread ceiling, never boost clocks."""
    import psutil

    cpu = psutil.cpu_percent(interval=0.1)
    ram = psutil.virtual_memory().available / 2**30
    slots = 1 if cpu >= 40 or ram < 8 else 2
    child = psutil.Process(pid)
    if sys.platform == "win32":
        allowed = psutil.Process(os.getpid()).cpu_affinity()
        child.cpu_affinity(allowed[-slots:])
    return {
        "cpu_affinity_slots": slots,
        "host_cpu_percent": cpu,
        "free_host_ram_gib": round(ram, 2),
    }


def recover_service(mailbox: Path, worker: str, kernel: Path, once: bool = False) -> None:
    """Recover transient SMB/telemetry failures without reinstalling or losing logs."""
    import psutil

    while True:
        try:
            service(mailbox, worker, kernel, once)
            return
        except (OSError, psutil.Error) as error:
            print(
                json.dumps(
                    {
                        "worker": worker,
                        "phase": "blocked",
                        "reason": str(error),
                        "retry_seconds": 30,
                    }
                ),
                flush=True,
            )
            if once:
                raise
            time.sleep(30)


def execute(mailbox: Path, path: Path, lease: dict, kernel: Path) -> None:
    import psutil

    job = read_json(path)
    if job.get("kernel_sha256") != hashlib.sha256(kernel.read_bytes()).hexdigest():
        raise ValueError("Update the owned worker: pinned kernel mismatch")
    if job.get("worker_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("Update the owned worker: pinned worker mismatch")
    # Child validates all received fields, uses only built-in models and safe tensors.
    work = kernel.parent / "work"
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v100-compute-", dir=work) as temporary:
        local = Path(temporary)
        (local / "job.json").write_bytes(path.read_bytes())
        environment = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="",
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
            NUMEXPR_NUM_THREADS="2",
        )
        result_path = mailbox / "results" / (path.stem + "-" + lease["nonce"])
        result_path.mkdir(parents=True, exist_ok=False)
        with (result_path / "worker.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", str(kernel), str(local / "job.json"), str(local / "output")],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                cwd=local,
            )
            started = time.monotonic()
            last_tick, active_seconds, suspended = started, 0.0, False
            try:
                limit_child(process.pid)
                while process.poll() is None:
                    lease_path = mailbox / "claims" / path.stem / "lease.json"
                    if read_json(lease_path)["nonce"] != lease["nonce"]:
                        raise RuntimeError("Compute lease revoked")
                    if (mailbox / "closed" / (path.stem + ".json")).exists():
                        raise RuntimeError("Compute job cancelled or already closed")
                    try:
                        owned = psutil.Process(process.pid)
                        rss = sum(
                            p.memory_info().rss for p in [owned] + owned.children(recursive=True)
                        )
                    except psutil.NoSuchProcess:
                        process.wait()
                        break
                    now = time.monotonic()
                    if not suspended:
                        active_seconds += now - last_tick
                    last_tick = now
                    if rss > 4 * 2**30 or active_seconds > 150 or now - started > 600:
                        raise RuntimeError(
                            "Compute child exceeded four GiB RAM, 150 active seconds or ten wall minutes"
                        )
                    ready, reason = available()
                    if psutil.virtual_memory().available < 2 * 2**30:
                        raise RuntimeError("Critical host RAM pressure; releasing owned child")
                    if not ready and not suspended:
                        for child in owned.children(recursive=True):
                            child.suspend()
                        owned.suspend()
                        suspended = True
                    elif ready and suspended:
                        owned.resume()
                        for child in owned.children(recursive=True):
                            child.resume()
                        suspended = False
                    lease["heartbeat"] = time.time()
                    atomic(lease_path, lease)
                    row = {
                        "job": path.stem,
                        "phase": "paused" if suspended else "training",
                        "reason": reason,
                        "active_seconds": round(active_seconds, 1),
                        "worker": lease["worker"],
                        "seconds": round(time.monotonic() - started, 1),
                        "rss_gib": round(rss / 2**30, 2),
                        "budget": adaptive_child_budget(process.pid),
                    }
                    atomic(
                        mailbox / "workers" / (lease["worker"] + ".json"),
                        row | {"updated": time.time()},
                    )
                    print(json.dumps(row), flush=True)
                    time.sleep(2)
                if process.returncode:
                    raise RuntimeError("Compute child failed; inspect worker.log")
            finally:
                if process.poll() is None:
                    try:
                        for child in psutil.Process(process.pid).children(recursive=True):
                            child.kill()
                    except psutil.NoSuchProcess:
                        pass
                    process.kill()
                process.wait()
        for name in ("report.json", "weights.safetensors"):
            source = local / "output" / name
            if source.is_symlink() or source.stat().st_size > 16 * 2**20:
                raise ValueError("Invalid compute result size")
            shutil.copyfile(source, result_path / name)
        atomic(
            result_path / "receipt.json",
            {
                **lease,
                "state": "complete",
                "job_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "files": {
                    name: hashlib.sha256((result_path / name).read_bytes()).hexdigest()
                    for name in ("report.json", "weights.safetensors")
                },
            },
        )


def service(mailbox: Path, worker: str, kernel: Path, once: bool = False) -> None:
    import psutil

    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,48}", worker):
        raise ValueError("Use a bounded alphanumeric worker name")
    for name in ("jobs", "claims", "results", "workers", "closed"):
        (mailbox / name).mkdir(parents=True, exist_ok=True)
    lock = mailbox / "workers" / (worker + ".lock")
    try:
        lock.mkdir()
    except FileExistsError:
        lease_path = lock / "owner.json"
        updated = (
            read_json(lease_path)["heartbeat"] if lease_path.exists() else lock.stat().st_mtime
        )
        if time.time() - updated < 300:
            raise ValueError("This worker name is already active; use another name") from None
        lock.rename(lock.with_name(lock.name + "-retired-" + uuid.uuid4().hex))
        lock.mkdir()
    owner = {"nonce": uuid.uuid4().hex, "heartbeat": time.time()}
    atomic(lock / "owner.json", owner)
    try:
        while True:
            owner["heartbeat"] = time.time()
            atomic(lock / "owner.json", owner)
            ready, reason = available()
            atomic(
                mailbox / "workers" / (worker + ".json"),
                {
                    "updated": time.time(),
                    "phase": "idle" if ready else "paused",
                    "reason": reason,
                    "device": "cpu",
                    "threads": 2,
                    "ram_limit_gib": 4,
                    "host_cpu_pause_percent": 40,
                    "min_free_host_ram_gib": 6,
                    "priority": "idle on Windows",
                    "gpu_enabled": False,
                },
            )
            task = claim(mailbox, worker) if ready else None
            if task:
                path, lease = task
                try:
                    execute(mailbox, path, lease, kernel)
                except (ValueError, RuntimeError, OSError, psutil.Error) as error:
                    destination = mailbox / "results" / (path.stem + "-" + lease["nonce"])
                    atomic(
                        destination / "receipt.json",
                        {**lease, "state": "failed", "detail": str(error)[:400]},
                    )
                    print(
                        json.dumps({"phase": "failed", "job": path.stem, "detail": str(error)}),
                        flush=True,
                    )
            elif not once:
                print(
                    json.dumps(
                        {"worker": worker, "phase": "idle" if ready else "paused", "reason": reason}
                    ),
                    flush=True,
                )
            if once:
                return
            time.sleep(10)
    finally:
        if read_json(lock / "owner.json")["nonce"] == owner["nonce"]:
            (lock / "owner.json").unlink()
            lock.rmdir()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mailbox", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_BACKEND_VERSION"):
        raise ValueError(
            "Distributed service is not supported in free managed Colab; use the interactive notebook"
        )
    recover_service(
        args.mailbox, args.name, Path(__file__).with_name("compute_kernel.py"), args.once
    )
