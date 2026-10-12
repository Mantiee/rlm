"""Independent availability recovery, without interrupting training or operator pauses."""

import argparse
import fcntl
import json
import shlex
import subprocess
import time
from pathlib import Path

from rlm.v100.common import atomic_json, load_profile
from rlm.v100.mission import status
from rlm.v100.supervisor import install

EXCLUSIVE = {"train", "mission-prepare", "optimize-mtp", "calibrate-training", "test-mtp"}


def exclusive_work(root: Path) -> bool:
    rows = subprocess.check_output(["ps", "-eo", "args="], text=True, timeout=10)
    for row in rows.splitlines():
        if "rlm.v100.cli" not in row:
            continue
        arguments = shlex.split(row)
        if "rlm.v100.cli" in arguments and arguments.count("--root") == 1:
            index = arguments.index("--root") + 1
            if index < len(arguments) and Path(arguments[index]).resolve() == root.resolve():
                if EXCLUSIVE.intersection(arguments):
                    return True
    return False


def preserved_profile(root: Path, record: dict) -> Path:
    candidates = []
    if record.get("learning", {}).get("live_profile"):
        candidates.append(Path(record["learning"]["live_profile"]))
    if record.get("run"):
        candidates.append(Path(record["run"]) / "input-profile.json")
    checkpoint = root / "research/availability-recovery-update/current.json"
    if checkpoint.is_file():
        candidates.append(Path(json.loads(checkpoint.read_text())["profile"]))
    errors = []
    for path in candidates:
        path = path.resolve()
        if path.is_relative_to(root / "research") and path.is_file():
            try:
                load_profile(path, root)
                return path
            except (ValueError, OSError, KeyError) as error:
                errors.append(str(error)[:200])
    raise ValueError("No valid preserved accepted profile: " + "; ".join(errors))


def recover(root: Path, resume: bool = False) -> dict:
    root = root.resolve()
    folder = root / "research/mission-watchdog"
    folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / "status.json"
    request = folder / "resume-request.json"
    if resume:
        atomic_json(request, {"requested": time.time()})
    resume = resume or request.exists()
    with (
        (folder / "lease.lock").open("a") as lease,
        (root / "research/mission-maintenance.lock").open("a") as maintenance,
    ):
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(maintenance, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"stage": "maintenance-in-progress", "updated": time.time()}
        try:
            record = status(root)
            pause = root / "research/supervisor/pause.json"
            if resume:
                pause.unlink(missing_ok=True)
            if pause.exists():
                result = {"stage": "operator-paused"}
            elif record.get("running"):
                request.unlink(missing_ok=True)
                result = {"stage": "mission-running", "run": record.get("run")}
            elif exclusive_work(root):
                result = {
                    "stage": "deferred-training",
                    "reason": "Owned exclusive job remains active",
                }
            else:
                previous = json.loads(receipt.read_text()) if receipt.is_file() else {}
                if not resume and time.time() < previous.get("next_retry", 0):
                    result = {**previous, "stage": "recovery-backoff"}
                else:
                    supervisor = root / "research/supervisor/status.json"
                    state = json.loads(supervisor.read_text()) if supervisor.is_file() else {}
                    if state.get("running") and 0 <= time.time() - state.get("updated", 0) < 15:
                        result = {"stage": "supervisor-recovering", "supervisor": state}
                    else:
                        profile = preserved_profile(root, record)
                        # Recheck after profile validation; another owned process may have started.
                        if status(root).get("running") or exclusive_work(root):
                            result = {"stage": "work-started-during-check"}
                        else:
                            result = {
                                "stage": "recovery-submitted",
                                "profile": str(profile),
                                "supervisor": install(root, profile),
                                "next_retry": time.time() + 300,
                            }
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
            result = {
                "stage": "recovery-blocked",
                "error": str(error)[:1000],
                "next_retry": time.time() + 300,
            }
        result["updated"] = time.time()
        atomic_json(receipt, result)
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(recover(args.root, args.resume)), flush=True)


if __name__ == "__main__":
    main()
