"""Owned SSH reverse tunnel. Only guest loopback reaches the public HTTP broker."""

import subprocess
from pathlib import Path


def command(root: Path) -> list[str]:
    from rlm.v100.desktop import PROXY_PORT, SSH_PORT

    folder = root / "research/desktop"
    return [
        "ssh",
        "-F",
        "/dev/null",
        "-N",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "UserKnownHostsFile=" + str(folder / "known_hosts"),
        "-i",
        str(folder / "client-key"),
        "-p",
        str(SSH_PORT),
        "-R",
        f"127.0.0.1:3128:127.0.0.1:{PROXY_PORT}",
        "root@127.0.0.1",
    ]


def close(process) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def configure_guest(root: Path) -> dict:
    from rlm.v100.desktop import run

    return run(
        root,
        """set -eu
for file in /etc/environment /etc/firefox/policies/policies.json; do
    if test -f "$file"; then sed -i 's|10.0.2.100:3128|127.0.0.1:3128|g' "$file"; fi
done
mkdir -p /etc/apt/apt.conf.d
printf '%s\\n' 'Acquire::http::Proxy "http://127.0.0.1:3128";' 'Acquire::https::Proxy "http://127.0.0.1:3128";' > /etc/apt/apt.conf.d/99v100-proxy
""",
        seconds=8,
        output_limit=500,
    )
