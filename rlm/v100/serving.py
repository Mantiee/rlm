"""Bind protected requests to a locally launched, versioned native server."""

import json
import os
import socket
from pathlib import Path
from urllib.parse import urlparse

from rlm.v100.common import atomic_json
from rlm.v100.protection import execution_hash, file_hash


def ensure_local_port_available(port: int) -> None:
    """Reject live listeners, while allowing a restarted server's TIME_WAIT sockets."""
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
            probe.listen(1)
        except OSError as error:
            raise OSError(
                error.errno, f"Local server port {port} unavailable: {error.strerror}"
            ) from error


def process_identity(pid: int) -> str:
    # Linux start time prevents a reused PID being accepted as the protected server.
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    if fields[0] in {"Z", "X", "x"}:
        raise ProcessLookupError(f"Process {pid} has exited")
    return fields[19]


def receipt_path(profile: dict, root: Path) -> Path:
    port = urlparse(profile["runtime"]["base_url"]).port or 8088
    return root / "research/state" / f"server-{port}.json"


def write_receipt(profile: dict, root: Path) -> None:
    files = {}
    for name in ("binary", "model", "draft_model"):
        if profile["server"][name]:
            path = Path(profile["server"][name])
            stat = path.stat()
            files[str(path)] = [stat.st_size, stat.st_mtime_ns, stat.st_ino]
    for path in Path(profile["server"]["binary"]).parent.glob("*.so*"):
        if path.is_file():
            stat = path.stat()
            files[str(path)] = [stat.st_size, stat.st_mtime_ns, stat.st_ino]
    atomic_json(
        receipt_path(profile, root),
        {
            "pid": os.getpid(),
            "process_start": process_identity(os.getpid()),
            "model_sha256": file_hash(Path(profile["server"]["model"])),
            "execution_sha256": execution_hash(profile),
            "files": files,
        },
    )


def assert_served_expert(client, profile: dict, root: Path, expert: dict | None = None) -> None:
    from rlm.v100.scratch_master import ScratchClient, is_scratch, verify

    if is_scratch(profile):
        verify(profile)
        if not isinstance(client, ScratchClient) or execution_hash(
            client.scratch_profile
        ) != execution_hash(profile):
            raise ValueError("Scratch client serves another frozen execution")
        if (
            expert
            and expert["profile"]["resources"]["scratch_hashes"]
            != profile["resources"]["scratch_hashes"]
        ):
            raise ValueError("Scratch expert identity differs")
        return
    receipt = json.loads(receipt_path(profile, root).read_text())
    if process_identity(receipt["pid"]) != receipt["process_start"]:
        raise ValueError("Protected server receipt belongs to a different process")
    expected_model = (
        next(value for name, value in expert["files"].items() if name.startswith("model-"))
        if expert
        else file_hash(Path(profile["server"]["model"]))
    )
    if receipt["model_sha256"] != expected_model or receipt["execution_sha256"] != execution_hash(
        profile
    ):
        raise ValueError("Protected expert is not the model launched at this endpoint")
    for raw, expected in receipt["files"].items():
        stat = Path(raw).stat()
        if [stat.st_size, stat.st_mtime_ns, stat.st_ino] != expected:
            raise ValueError("Protected server artifact changed since launch")
    if (
        Path(client.request("/props")["model_path"]).resolve()
        != Path(profile["server"]["model"]).resolve()
    ):
        raise ValueError("Native endpoint is serving a different expert")
