"""Single owned restart supervisor; checkpoints and explicit operator stops win."""

import fcntl
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from rlm.v100.common import atomic_json, load_profile


def loop(root: Path, profile: Path) -> None:
    from rlm.v100.mission import start, status, stop

    folder = root / "research/supervisor"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "lease.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        shutdown = False

        def interrupted(signum, frame):
            nonlocal shutdown
            shutdown = True

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        failures, next_retry, last_run = 0, 0.0, None
        while not shutdown and not (folder / "pause.json").exists():
            record = status(root)
            if record["running"]:
                if record.get("state", {}).get("phase") == "research-and-learning-loop":
                    failures = 0
            elif time.monotonic() >= next_retry:
                if (
                    record.get("run") == last_run
                    and record.get("state", {}).get("phase") == "failed"
                ):
                    failures += 1
                candidate = record.get("learning", {}).get("live_profile")
                prior_input = (
                    Path(record["run"]) / "input-profile.json" if record.get("run") else profile
                )
                # The installer already carried accepted weights into this profile.
                # Do not discard its new setup in favor of an older failed run.
                resume = profile
                if last_run is not None:
                    resume = (
                        Path(candidate)
                        if candidate and Path(candidate).exists()
                        else (prior_input if prior_input.exists() else profile)
                    )
                try:
                    if failures >= 2:
                        from rlm.v100.architecture_promotion import rollback

                        restored, receipt = rollback(
                            root,
                            load_profile(resume, root),
                            "Repeated owned mission serving failures",
                        )
                        if receipt["restored"]:
                            resume = folder / f"architecture-rollback-{time.time_ns()}.json"
                            atomic_json(resume, restored)
                            atomic_json(folder / "last-architecture-rollback.json", receipt)
                    launched = start(root, resume, max_context=131072, flash_attention="on")
                    last_run = launched["run"]
                    # Even a startup failure cannot create a tight GPU reload loop.
                    next_retry = time.monotonic() + min(900, 60 * 2 ** min(failures, 4))
                except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
                    failures += 1
                    next_retry = time.monotonic() + min(900, 30 * 2 ** min(failures, 5))
                    atomic_json(
                        folder / "last-error.json",
                        {
                            "error": type(error).__name__,
                            "detail": str(error)[:600],
                            "time": time.time(),
                        },
                    )
            atomic_json(
                folder / "status.json",
                {
                    "running": True,
                    "mission_running": record["running"],
                    "restart_failures": failures,
                    "next_retry_seconds": max(0, next_retry - time.monotonic()),
                    "last_run": last_run,
                    "updated": time.time(),
                },
            )
            time.sleep(2)
        stop(root)
        atomic_json(
            folder / "status.json",
            {
                "running": False,
                "state": "operator pause or supervisor stop",
                "updated": time.time(),
            },
        )


def install(root: Path, profile: Path) -> dict:
    """Use a user service if available, otherwise a detached owned supervisor."""
    from rlm.v100.serving import process_identity

    root, profile = root.resolve(), profile.resolve()
    folder = root / "research/supervisor"
    folder.mkdir(parents=True, exist_ok=True)
    unit = folder / "v100-mission.service"
    if any(
        character in str(root) + str(profile) + sys.executable
        for character in ("\n", '"', "%", "\\")
    ):
        raise ValueError("Unsupported service path characters")
    launcher = folder / "run-supervisor.sh"
    launcher.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nsource "
        + shlex.quote(str(root / "env.sh"))
        + "\n"
        + 'export PATH="$HOME/.local/bin:$PATH"\n'
        + "export HF_HUB_DOWNLOAD_TIMEOUT=600 HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DISABLE_XET=1\n"
        + "exec "
        + " ".join(
            shlex.quote(value)
            for value in (
                sys.executable,
                "-u",
                "-m",
                "rlm.v100.supervisor",
                str(root),
                str(profile),
            )
        )
        + "\n"
    )
    launcher.chmod(0o700)
    unit.write_text(
        "[Unit]\nDescription=Owned V100 research and learning supervisor\nAfter=network.target\n"
        "[Service]\nType=simple\n"
        f'ExecStart=/bin/bash "{launcher}"\n'
        f"WorkingDirectory={root}\n"
        "Environment=PYTHONNOUSERSITE=1\nRestart=on-failure\nRestartSec=30\nTimeoutStopSec=60\n"
        "[Install]\nWantedBy=default.target\n"
    )
    if shutil.which("systemd-analyze"):
        verified = subprocess.run(
            ["systemd-analyze", "verify", "--man=no", str(unit)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if verified.returncode:
            raise RuntimeError("Invalid owned service unit: " + verified.stderr[-1500:])
    check = (
        subprocess.run(["systemctl", "--user", "show-environment"], capture_output=True, timeout=10)
        if shutil.which("systemctl")
        else None
    )
    if check is not None and check.returncode == 0:
        subprocess.run(
            ["systemctl", "--user", "stop", "v100-mission.service"], capture_output=True, timeout=90
        )
        (folder / "pause.json").unlink(missing_ok=True)
        subprocess.run(["systemctl", "--user", "link", str(unit)], check=True)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(
            ["systemctl", "--user", "enable", "--now", "v100-mission.service"], check=True
        )
        linger = subprocess.run(
            ["loginctl", "show-user", str(os.getuid()), "-p", "Linger", "--value"],
            capture_output=True,
            text=True,
        )
        return {
            "mode": "user service",
            "survives_ssh_disconnect": linger.stdout.strip() == "yes",
            "boot_autostart_requires_linger": linger.stdout.strip() != "yes",
        }
    active = folder / "process.json"
    if active.exists():
        old = json.loads(active.read_text())
        try:
            if process_identity(old["pid"]) == old["start"]:
                arguments = Path(f"/proc/{old['pid']}/cmdline").read_bytes().split(b"\0")
                if b"rlm.v100.supervisor" not in arguments or str(root).encode() not in arguments:
                    raise ValueError("Detached supervisor identity differs; no process stopped")
                os.kill(old["pid"], signal.SIGTERM)
                deadline = time.monotonic() + 20
                while process_identity(old["pid"]) == old["start"]:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Previous owned supervisor still stopping; retry later")
                    time.sleep(0.1)
        except OSError:
            pass
    (folder / "pause.json").unlink(missing_ok=True)
    with (folder / "supervisor.log").open("ab") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", "rlm.v100.supervisor", str(root), str(profile)],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    value = {
        "pid": process.pid,
        "start": process_identity(process.pid),
        "mode": "detached supervisor",
        "boot_autostart": False,
    }
    atomic_json(active, value)
    return value


if __name__ == "__main__":
    loop(Path(sys.argv[1]), Path(sys.argv[2]))
