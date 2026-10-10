import json
import threading
from contextlib import contextmanager
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from rlm.v100 import mission_chat, remote_access

CONFIG = {
    "owner_login": "operator@example.com",
    "origin": "https://debian1.test.ts.net",
    "dashboard": "http://192.168.0.68:8765",
}


@contextmanager
def gateway(root):
    with remote_access.GatewayServer(root, CONFIG, 0) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            yield server, f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join(3)


def request(base, path, owner=CONFIG["owner_login"], body=None, **headers):
    if owner:
        headers["Tailscale-User-Login"] = owner
    req = Request(base + path, data=body, headers=headers)
    with urlopen(req, timeout=3) as response:
        return response.status, response.headers, response.read()


def test_identity_required_for_every_read_and_write(tmp_path):
    with gateway(tmp_path) as (_, base):
        for owner in (None, "other@example.com"):
            for path in ("/", "/chat", "/api/chat", "/api/status"):
                with pytest.raises(HTTPError) as error:
                    request(base, path, owner)
                assert error.value.code == 403
            with pytest.raises(HTTPError) as error:
                request(base, "/api/chat", owner, body=b'{"message":"work"}')
            assert error.value.code == 403
    assert not (tmp_path / "research/state/user-chat.sqlite3").exists()


def test_owner_chat_csrf_exact_origin_queue_and_persistent_reply(tmp_path):
    with gateway(tmp_path) as (server, base):
        code, headers, body = request(base, "/chat")
        assert code == 200 and b"Awaiting saved result" in body
        assert "script-src 'sha256-" in headers["Content-Security-Policy"]
        assert server.server_address[0] == "127.0.0.1"
        _, _, body = request(base, "/api/chat")
        token = json.loads(body)["csrf"]
        valid = {
            "Origin": CONFIG["origin"],
            "X-Synta-CSRF": token,
            "Content-Type": "application/json",
        }
        for overrides in (
            {"Origin": "https://evil.example"},
            {"X-Synta-CSRF": "bad"},
            {"Sec-Fetch-Site": "cross-site"},
        ):
            with pytest.raises(HTTPError) as error:
                request(
                    base,
                    "/api/chat",
                    body=b'{"message":"Perform research"}',
                    **{**valid, **overrides},
                )
            assert error.value.code == 403
        code, _, body = request(base, "/api/chat", body=b'{"message":"Perform research"}', **valid)
        row = json.loads(body)
        assert code == 202 and row["state"] == "queued"
        identity = row["id"]
        reply = {"answer": "Actual saved response", "applied": [{"state": "failed"}]}
        with mission_chat.connect(tmp_path) as db:
            db.execute(
                "UPDATE requests SET state='completed',response=? WHERE id=?",
                (json.dumps(reply), identity),
            )
        _, _, body = request(base, "/api/chat/" + identity)
        assert json.loads(body)["response"] == reply
        _, _, body = request(base, "/api/chat")
        assert json.loads(body)["requests"][0]["response"] == reply


@pytest.mark.parametrize(
    "route",
    [
        "//evil.example/",
        "http://evil.example/",
        "/../env.sh",
        "/api/stop",
        "/paper/../../env.sh",
        "/api/chat/invalid",
    ],
)
def test_proxy_refuses_arbitrary_routes(route):
    assert not remote_access.proxy_allowed(route)


@pytest.mark.parametrize(
    "field,value",
    [
        ("owner_login", ""),
        ("origin", "http://debian1.ts.net"),
        ("origin", "https://evil.example"),
        ("dashboard", "http://0.0.0.0:8765"),
        ("dashboard", "http://8.8.8.8:8765"),
        ("dashboard", "http://192.168.0.68:22"),
    ],
)
def test_invalid_gateway_config_refused(field, value):
    with pytest.raises(ValueError):
        remote_access.validate_config({**CONFIG, field: value})


def test_remote_installer_keeps_other_services_and_windows_gpu_off():
    from pathlib import Path

    source = (Path(__file__).parents[1] / "tools/install-synta-remote.sh").read_text()
    assert "handlers != ours" in source and "AllowFunnel" in source
    assert "serve --bg --https=443 http://127.0.0.1:8786" in source
    assert "serve reset" not in source and "tailscale funnel" not in source
    assert "--advertise-routes" not in source and "nvidia-smi" not in source


def test_financial_remote_question_returns_without_gpu_queue(tmp_path, monkeypatch):
    from rlm.v100 import progress

    monkeypatch.setattr(
        progress, "snapshot", lambda root: {"phase": "research", "paper": {"executed_fills": 0}}
    )
    with gateway(tmp_path) as (_, base):
        _, _, body = request(base, "/api/chat")
        token = json.loads(body)["csrf"]
        code, _, body = request(
            base,
            "/api/chat",
            body=json.dumps({"message": "czy zarobiles w paper"}).encode(),
            **{
                "Origin": CONFIG["origin"],
                "X-Synta-CSRF": token,
                "Content-Type": "application/json",
            },
        )
        row = json.loads(body)
        assert code == 200 and row["state"] == "completed"
        assert row["response"]["responder"]["model"] == "controller-financial-evidence"
        assert "kapitału" in row["response"]["answer"]
