import threading
import urllib.error
import urllib.request

from tools.synta_dashboard_redirect import RedirectServer


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def test_retired_dashboard_link_redirects_to_fixed_mission_control_origin():
    target = "http://192.168.0.68:8765/?v=108"
    server = RedirectServer(("127.0.0.1", 0), target)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/?attacker=ignored",
            headers={"Host": "attacker.example"},
        )
        opener = urllib.request.build_opener(NoRedirect)
        try:
            opener.open(request, timeout=2)
        except urllib.error.HTTPError as response:
            assert response.code == 302
            assert response.headers["Location"] == target
            assert response.headers["Cache-Control"] == "no-store"
        else:
            raise AssertionError("Expected a redirect response")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_redirect_service_has_health_probe_and_rejects_writes():
    server = RedirectServer(("127.0.0.1", 0), "http://192.168.0.68:8765/?v=108")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_port}/healthz", timeout=2
        ) as response:
            assert response.status == 200
            assert response.read() == b"ok"
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/", data=b"x", method="POST"
        )
        try:
            urllib.request.urlopen(request, timeout=2)
        except urllib.error.HTTPError as response:
            assert response.code == 405
        else:
            raise AssertionError("Expected the read-only route to reject writes")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
