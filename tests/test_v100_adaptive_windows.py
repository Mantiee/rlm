import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

from rlm.v100.windows_resource_proxy import Proxy, budget


def test_pressure_budget_yields_to_actual_load_and_does_not_consider_browser_presence():
    idle = budget(10, 12, 50, 12, 80, 0, False)
    video = budget(35, 12, 50, 12, 80, 45, False)
    busy = budget(55, 12, 50, 12, 80, 0, False)
    assert idle["ready"] and idle["duty_percent"] == 65 and idle["batch_tokens"] == 64
    assert video["ready"] and video["duty_percent"] == 35
    assert busy["ready"] and busy["duty_percent"] == 20 and busy["batch_tokens"] == 16
    for state in (
        budget(66, 12, 50, 12, 80, 0, False),
        budget(10, 5, 50, 12, 80, 0, False),
        budget(10, 12, 75, 12, 80, 0, False),
        budget(10, 12, 50, 2, 80, 0, False),
        budget(10, 12, 50, 12, 251, 0, False),
        budget(10, 12, 50, 12, 80, 0, True),
    ):
        assert not state["ready"] and state["reason"]


def test_proxy_denies_pressure_then_caps_and_forwards_real_inference(tmp_path):
    seen = []

    class Backend(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = b'{"done":true,"message":{"content":"OK"}}\n'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
    proxy.owner = "127.0.0.1"
    proxy.upstream_port = backend.server_port
    proxy.pressure = SimpleNamespace(
        state={"ready": False, "reason": "game"},
        not_before=0,
        lock=threading.Lock(),
        generating=False,
    )
    for server in (backend, proxy):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = http.client.HTTPConnection("127.0.0.1", proxy.server_port, timeout=3)
        client.request(
            "POST",
            "/api/chat",
            body=json.dumps({"model": "owned", "options": {"num_predict": -1, "num_batch": 999}}),
        )
        response = client.getresponse()
        assert response.status == 503 and not seen
        response.read()
        proxy.pressure.state = budget(55, 12, 50, 12, 80, 0, False)
        client.close()
        client = http.client.HTTPConnection("127.0.0.1", proxy.server_port, timeout=3)
        client.request(
            "POST",
            "/api/chat",
            body=json.dumps({"model": "owned", "options": {"num_predict": -1, "num_batch": 999}}),
        )
        response = client.getresponse()
        assert response.status == 200 and b"OK" in response.read()
        assert seen[0]["options"] == {"num_predict": 1, "num_batch": 16, "num_thread": 2}
        assert not proxy.pressure.generating and proxy.pressure.not_before > 0
        client.close()
    finally:
        for server in (proxy, backend):
            server.shutdown()
            server.server_close()


def test_windows_browser_is_not_classified_as_game(monkeypatch):
    from unittest.mock import Mock

    import psutil

    from rlm.v100 import compute_worker

    monkeypatch.setattr(compute_worker.sys, "platform", "win32")
    user = SimpleNamespace(
        GetForegroundWindow=Mock(return_value=1), GetWindowThreadProcessId=Mock()
    )
    monkeypatch.setattr(compute_worker.ctypes, "WinDLL", lambda *a, **k: user, raising=False)
    monkeypatch.setattr(psutil, "Process", lambda pid: SimpleNamespace(name=lambda: "chrome.exe"))
    assert not compute_worker.foreground_busy()
    monkeypatch.setattr(psutil, "Process", lambda pid: SimpleNamespace(name=lambda: "cs2.exe"))
    assert compute_worker.foreground_busy()
