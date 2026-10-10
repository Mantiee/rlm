import json
import threading
from contextlib import contextmanager
from io import BytesIO
from urllib.error import HTTPError, URLError
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
        assert remote_access.NAVIGATION in body
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
    "opening",
    [
        b"<body>",
        b'<body class="dashboard">',
        b"<BODY class='dashboard'>",
        b'<body data-label="a > b">',
    ],
)
def test_dashboard_navigation_preserves_renderer_and_body_attributes(opening):
    source = b"<!doctype html><html>" + opening + b"<script>render()</script></body></html>"
    result = remote_access.dashboard_navigation(source)
    assert opening + remote_access.NAVIGATION in result
    assert result.replace(remote_access.NAVIGATION, b"", 1) == source
    assert remote_access.dashboard_navigation(result) == result


def test_dashboard_without_content_cannot_silently_omit_navigation():
    with pytest.raises(ValueError, match="no body content"):
        remote_access.dashboard_navigation(b"<html><head><title>Empty</title></head></html>")


def test_real_dashboard_renderer_with_implicit_body_retains_exact_script_and_policy(
    tmp_path, monkeypatch
):
    from rlm.v100.dashboard_layout import BASE_TEMPLATE, render_template, validate_template

    source = render_template(BASE_TEMPLATE, validate_template(BASE_TEMPLATE)).encode()
    assert b"<body" not in source
    policy = "default-src 'none'; script-src 'sha256-existing'; style-src 'unsafe-inline'"

    original_urlopen = remote_access.urlopen

    def upstream(req, timeout):
        if not req.full_url.startswith(CONFIG["dashboard"]):
            return original_urlopen(req, timeout=timeout)
        response = BytesIO(source)
        response.headers = {"Content-Type": "text/html", "Content-Security-Policy": policy}
        return response

    monkeypatch.setattr(remote_access, "urlopen", upstream)
    folder = tmp_path / "research/remote-access"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps(CONFIG))
    with gateway(tmp_path) as (server, base):
        code, headers, body = request(base, "/")
        verification = remote_access.verify_navigation(tmp_path, server.server_port)
    assert code == 200
    assert headers["Content-Security-Policy"] == policy
    assert body.replace(remote_access.NAVIGATION, b"", 1) == source
    assert body.index(remote_access.NAVIGATION) < body.index(b"<h1>")
    assert body.index(b"</style>") < body.index(remote_access.NAVIGATION)
    assert remote_access.dashboard_navigation(body) == body
    assert verification["navigation_verified"] is True
    assert verification["chat"] == CONFIG["origin"] + "/chat"


def test_targeted_update_verification_refuses_missing_links(tmp_path, monkeypatch):
    folder = tmp_path / "research/remote-access"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps(CONFIG))

    def missing_links(req, timeout):
        response = BytesIO(b"<html><h1>Dashboard</h1></html>")
        response.status = 200
        return response

    monkeypatch.setattr(remote_access, "urlopen", missing_links)
    with pytest.raises(RuntimeError, match="verification failed"):
        remote_access.verify_navigation(tmp_path)


def test_targeted_update_waits_for_gateway_listener_without_hiding_http_errors(
    tmp_path, monkeypatch
):
    folder = tmp_path / "research/remote-access"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps(CONFIG))
    calls = []

    def starting(req, timeout):
        calls.append(req.full_url)
        if len(calls) == 1:
            raise URLError(ConnectionRefusedError("Gateway starting"))
        response = BytesIO(remote_access.NAVIGATION)
        response.status = 200
        return response

    monkeypatch.setattr(remote_access, "urlopen", starting)
    monkeypatch.setattr(remote_access.time, "sleep", lambda seconds: None)
    assert remote_access.verify_navigation(tmp_path)["navigation_verified"] is True
    assert calls == [
        "http://127.0.0.1:8786/",
        "http://127.0.0.1:8786/",
        "http://127.0.0.1:8786/chat",
    ]

    def failed(req, timeout):
        raise HTTPError(req.full_url, 404, "Navigation failed", {}, None)

    monkeypatch.setattr(remote_access, "urlopen", failed)
    with pytest.raises(HTTPError):
        remote_access.verify_navigation(tmp_path)


@pytest.mark.parametrize(
    "head",
    [
        "<head><title>Łódź</title><!-- <body> --></head>",
        "<head>\r<title>Łódź\u2028test</title></head>",
        '<style>body:before {content:"<body>"}</style>',
        "<script>const fake = '<body><nav id=\"synta-remote-navigation\">';</script>",
    ],
)
def test_implicit_body_skips_metadata_comments_and_fake_tags(head):
    source = ("<html>" + head + "<h1>Dashboard</h1></html>").encode()
    result = remote_access.dashboard_navigation(source)
    assert (
        result
        == ("<html>" + head).encode() + remote_access.NAVIGATION + b"<h1>Dashboard</h1></html>"
    )


def test_existing_navigation_with_single_quoted_id_is_not_duplicated():
    source = b"<html><nav id='synta-remote-navigation'><a href='/chat'>Chat</a></nav></html>"
    assert remote_access.dashboard_navigation(source) == source


def test_gateway_dashboard_injects_mobile_links_and_preserves_upstream_policy(
    tmp_path, monkeypatch
):
    source = b'<html><body class="dashboard"><script>render()</script></body></html>'
    policy = "default-src 'none'; script-src 'sha256-existing'; style-src 'unsafe-inline'"
    calls = []

    def upstream(req, timeout):
        calls.append((req.full_url, timeout))
        response = BytesIO(source)
        response.headers = {"Content-Type": "text/html", "Content-Security-Policy": policy}
        return response

    monkeypatch.setattr(remote_access, "urlopen", upstream)
    with gateway(tmp_path) as (_, base):
        code, headers, body = request(base, "/")
    assert code == 200
    assert headers["Content-Security-Policy"] == policy
    assert body.replace(remote_access.NAVIGATION, b"", 1) == source
    assert b'href="/chat"' in body and b'href="/"' in body
    assert b"min-height:44px" in body
    assert calls == [(CONFIG["dashboard"] + "/", 5)]


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
