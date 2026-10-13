import contextlib
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rlm.v100 import owned_maintenance, owned_node


def node_fixture(tmp_path):
    node = object.__new__(owned_node.Node)
    node.root = tmp_path / "work"
    node.maintenance = False
    node.drain_until = 0.0
    node.gpu_lock = threading.Lock()
    node.cpu_lock = threading.Lock()
    node.processes = {}
    node.generating = True
    return node


def test_drain_waits_for_gpu_without_cancelling_it_and_lease_expires(tmp_path):
    node = node_fixture(tmp_path)
    node.gpu_lock.acquire()
    with patch.object(owned_node.time, "monotonic", return_value=100):
        try:
            node.begin_maintenance()
        except BlockingIOError:
            pass
        else:
            raise AssertionError("Active GPU was not retained")
        assert node.draining() and not node.maintenance and node.generating
    node.gpu_lock.release()
    with patch.object(owned_node.time, "monotonic", return_value=166):
        assert not node.draining()


def test_drain_waits_for_cpu_then_acquires_maintenance(tmp_path):
    node = node_fixture(tmp_path)
    node.processes["owned-job"] = object()
    try:
        node.begin_maintenance()
    except BlockingIOError as error:
        assert "CPU" in str(error)
    else:
        raise AssertionError("Active CPU job was ignored")
    assert node.draining() and not node.maintenance
    assert "owned-job" in node.processes
    node.processes.clear()
    node.begin_maintenance()
    assert node.maintenance and not node.gpu_lock.locked()


def test_drain_keeps_gpu_telemetry_ready_for_running_generation(tmp_path, monkeypatch):
    import psutil

    node = node_fixture(tmp_path)
    node.source = tmp_path / "source"
    node.source.mkdir()
    node.cpu_slots = 8
    for name in ("compute_kernel.py", "compute_worker.py"):
        (node.source / name).write_text("fixture")
    node.drain_until = float("inf")
    monkeypatch.setattr(
        owned_node.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="NVIDIA GeForce RTX 3060 Ti, 35, 4096, 8192, 50, 90"
        ),
    )
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(available=16 * 2**30, total=32 * 2**30)
    )
    monkeypatch.setattr(psutil, "cpu_percent", lambda: 20)
    state = node.sample()
    assert state["ready"] is True
    assert state["cpu_ready"] is False
    assert state["maintenance_pending"] is True
    assert state["generating"] is True


def test_installer_retries_actual_busy_response_then_succeeds():
    replies = [(503, {"error": "Owned CPU jobs active"}), (200, {"state": "drained"})]
    output = io.StringIO()
    with (
        patch.object(owned_maintenance, "request_drain", side_effect=replies),
        patch.object(owned_maintenance.time, "sleep"),
        contextlib.redirect_stdout(output),
    ):
        assert owned_maintenance.drain(5)
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert records[0]["reason"] == "Owned CPU jobs active"
    assert records[-1]["stage"] == "owned-jobs-drained"


def test_installer_busy_budget_is_finite():
    clock = [0.0]
    with (
        patch.object(owned_maintenance, "request_drain", return_value=(503, {"error": "GPU busy"})),
        patch.object(owned_maintenance.time, "monotonic", side_effect=lambda: clock[0]),
        patch.object(
            owned_maintenance.time,
            "sleep",
            side_effect=lambda value: clock.__setitem__(0, clock[0] + value),
        ),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        assert owned_maintenance.drain(2) is False
    assert clock[0] == 2


def test_legacy_lan_listener_uses_actual_loopback_source():
    observed = []

    class LocalHandler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            observed.append(self.client_address[0])
            body = b'{"state":"drained"}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    connection_type = owned_maintenance.http.client.HTTPConnection
    with ThreadingHTTPServer(("127.0.0.2", 0), LocalHandler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        routes = []

        def connect(host, port, **kwargs):
            routes.append((host, kwargs["source_address"]))
            # An unused loopback port simulates the old missing listener.
            target = "127.0.0.1" if host == "127.0.0.1" else "127.0.0.2"
            return connection_type(target, server.server_port, **kwargs)

        try:
            with patch.object(owned_maintenance.http.client, "HTTPConnection", side_effect=connect):
                assert owned_maintenance.request_drain() == (200, {"state": "drained"})
        finally:
            server.shutdown()
            thread.join(timeout=2)
    assert routes == [("127.0.0.1", ("127.0.0.1", 0)), ("192.168.0.9", ("127.0.0.1", 0))]
    assert observed == ["127.0.0.1"]


def test_installer_does_not_confuse_loaded_models_with_running_jobs():
    source = (Path(__file__).parents[1] / "tools/bootstrap-synta-3060.sh").read_text()
    assert '"$SOURCE/owned_maintenance.py"' in source
    assert "if value.get('models'): raise SystemExit(75)" not in source
