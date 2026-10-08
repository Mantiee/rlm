"""Local restart probes must distinguish TIME_WAIT from an occupied listener."""

import errno
import socket
import sys

import pytest

from rlm.v100.serving import ensure_local_port_available


@pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0"])
def test_live_listener_still_blocks_reusable_probe(host):
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        with pytest.raises(OSError, match=f"port {port} unavailable") as failure:
            ensure_local_port_available(port)
        assert failure.value.errno == errno.EADDRINUSE


@pytest.mark.skipif(sys.platform != "linux", reason="Linux TIME_WAIT restart regression")
def test_restart_after_server_closes_connection_does_not_need_kernel_timeout():
    with socket.socket() as listener, socket.socket() as client:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        client.connect(("127.0.0.1", port))
        accepted, _ = listener.accept()
        # Server closes first, so its local endpoint enters TIME_WAIT.
        accepted.shutdown(socket.SHUT_WR)
        assert client.recv(1) == b""
        client.close()
        accepted.close()
    with socket.socket() as old_probe:
        with pytest.raises(OSError) as failure:
            old_probe.bind(("127.0.0.1", port))
        assert failure.value.errno == errno.EADDRINUSE
    ensure_local_port_available(port)
    # The probe itself leaves no resident listener behind.
    with socket.socket() as restarted:
        restarted.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        restarted.bind(("127.0.0.1", port))
        restarted.listen(1)
