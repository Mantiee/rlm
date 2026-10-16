"""Bind protected requests to a locally launched, versioned native server."""

import hashlib
import ipaddress
import json
import os
import socket
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

from rlm.v100.common import atomic_json
from rlm.v100.protection import execution_hash, file_hash, file_identity


class ServingLeaseUnavailable(TimeoutError):
    """A coordinated candidate transition currently owns this local endpoint."""


def local_endpoint(profile: dict) -> tuple[str, int] | None:
    """Return a canonical loopback endpoint, or None for a remote backend."""
    try:
        origin = urlparse(profile["runtime"]["base_url"])
        host = (origin.hostname or "").casefold().rstrip(".")
        port = origin.port
    except (KeyError, TypeError, ValueError):
        return None
    if port is None:
        port = 443 if origin.scheme == "https" else 80
    if host == "localhost":
        return "loopback", port
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return None
    return ("loopback", port) if address.is_loopback or address.is_unspecified else None


@contextmanager
def endpoint_lease(
    root: Path,
    profile: dict,
    *,
    exclusive: bool,
    timeout: float | None = None,
):
    """Coordinate this checkout's local listener readers and candidate lifetimes.

    The lease is cooperative and root-scoped. It cannot prevent an unrelated
    program or another checkout from rebinding the same OS port.
    """
    endpoint = local_endpoint(profile)
    if endpoint is None:
        yield False
        return
    from rlm.v100.file_lock import LOCK_EX, LOCK_NB, LOCK_SH, LOCK_UN, flock

    encoded = json.dumps(
        [str(Path(root).resolve()), endpoint[0], endpoint[1]], separators=(",", ":")
    ).encode("utf-8")
    key = hashlib.sha256(encoded).hexdigest()
    folder = Path(root) / "research/state/serving-leases"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (key + ".lock")
    operation = LOCK_EX if exclusive else LOCK_SH
    deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
    with path.open("a+b") as handle:
        while True:
            try:
                flock(handle, operation | LOCK_NB)
                break
            except BlockingIOError as error:
                if deadline is not None and time.monotonic() >= deadline:
                    raise ServingLeaseUnavailable(
                        f"Local chat endpoint {endpoint[0]}:{endpoint[1]} is in a coordinated serving transition"
                    ) from error
                time.sleep(0.05)
        try:
            yield True
        finally:
            flock(handle, LOCK_UN)


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
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError as error:
        raise ProcessLookupError(f"Process {pid} no longer exists") from error
    fields = stat.rsplit(")", 1)[1].split()
    if fields[0] in {"Z", "X", "x"}:
        raise ProcessLookupError(f"Process {pid} has exited")
    return fields[19]


def receipt_path(profile: dict, root: Path) -> Path:
    port = urlparse(profile["runtime"]["base_url"]).port or 8088
    return root / "research/state" / f"server-{port}.json"


def write_receipt(
    profile: dict,
    root: Path,
    pid: int | None = None,
    *,
    check: Callable[[], None] | None = None,
    progress: Callable[[Path, int, int], None] | None = None,
) -> None:
    pid = os.getpid() if pid is None else pid
    identity = process_identity(pid)
    files = {}
    for name in ("binary", "model", "draft_model"):
        if profile["server"][name]:
            path = Path(profile["server"][name])
            files[str(path)] = list(file_identity(path))
    for path in Path(profile["server"]["binary"]).parent.glob("*.so*"):
        if path.is_file():
            files[str(path)] = list(file_identity(path))

    def hash_file(path: Path) -> str:
        return file_hash(path, check=check, progress=progress)

    model_hash = hash_file(Path(profile["server"]["model"]))
    native_hash = execution_hash(profile, hash_file=hash_file)
    if process_identity(pid) != identity:
        raise ProcessLookupError("Native process changed during artifact verification")
    for raw, expected in files.items():
        if list(file_identity(Path(raw))) != expected:
            raise ValueError("Server artifact changed during verification")
    if check is not None:
        check()
    atomic_json(
        receipt_path(profile, root),
        {
            "pid": pid,
            "process_start": identity,
            "model_sha256": model_hash,
            "execution_sha256": native_hash,
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
        if list(file_identity(Path(raw))) != expected:
            raise ValueError("Protected server artifact changed since launch")
    if (
        Path(client.request("/props")["model_path"]).resolve()
        != Path(profile["server"]["model"]).resolve()
    ):
        raise ValueError("Native endpoint is serving a different expert")
