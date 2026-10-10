"""Owner-only Tailscale Serve gateway. Never bind the command API to the LAN."""

import argparse
import base64
import hashlib
import hmac
import ipaddress
import json
import re
import secrets
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

NAVIGATION = b"""<nav id="synta-remote-navigation" aria-label="Synta navigation" style="display:flex;flex-wrap:wrap;gap:12px;margin:0 0 20px;padding:8px;border:1px solid #536b88;border-radius:10px;background:#101725;font:16px system-ui"><a href="/" style="display:flex;align-items:center;min-height:44px;padding:0 16px;color:#9bcaff">Dashboard</a><a href="/chat" style="display:flex;align-items:center;min-height:44px;padding:0 16px;color:#9bcaff">Master chat</a></nav>"""


def dashboard_navigation(body: bytes) -> bytes:
    """Add gateway-owned navigation without modifying the renderer or its script."""
    if b'id="synta-remote-navigation"' in body:
        return body
    opening = re.search(rb"""<body\b(?:[^>"']|"[^"]*"|'[^']*')*>""", body, re.IGNORECASE)
    if opening is None:
        raise ValueError("Dashboard HTML has no body element for remote navigation")
    return body[: opening.end()] + NAVIGATION + body[opening.end() :]


CHAT_SCRIPT = r"""
const rows=new Map();let busy=false,csrf='';
async function api(path,options={}){const abort=new AbortController(),timer=setTimeout(()=>abort.abort(),8000);try{const r=await fetch(path,{cache:'no-store',signal:abort.signal,...options});if(!r.ok)throw Error('HTTP '+r.status);return await r.json()}finally{clearTimeout(timer)}}
function show(row){let card=rows.get(row.id);if(!card){card=document.createElement('article');card.dataset.key=row.id;const head=document.createElement('h3'),question=document.createElement('p'),status=document.createElement('p'),answer=document.createElement('pre'),details=document.createElement('details'),summary=document.createElement('summary'),receipts=document.createElement('pre');summary.textContent='Action receipts';details.append(summary,receipts);card.append(head,question,status,answer,details);document.getElementById('history').append(card);rows.set(row.id,card)}const [head,question,status,answer,details]=card.children;head.textContent=row.id;question.textContent='You: '+row.message;status.textContent=row.state+(row.processing?' | processing':'')+(row.error?' | '+row.error:'');answer.textContent=row.response?.answer||'Awaiting saved result. You can close and reopen this page.';details.lastChild.textContent=JSON.stringify(row.response?.applied||[],null,2)}
async function refresh(){if(busy)return;busy=true;try{const value=await api('/api/chat');csrf=value.csrf;for(const row of value.requests)show(row);document.getElementById('status').textContent='Connected | '+new Date().toLocaleTimeString()}catch(error){document.getElementById('status').textContent='Connection delayed: '+error.message}finally{busy=false}}
document.getElementById('send').onclick=async()=>{const input=document.getElementById('message'),button=document.getElementById('send'),message=input.value;if(!message.trim())return;button.disabled=true;try{if(!csrf)await refresh();const row=await api('/api/chat',{method:'POST',headers:{'Content-Type':'application/json','X-Synta-CSRF':csrf},body:JSON.stringify({message})});show(row);input.value='';await refresh()}catch(error){document.getElementById('status').textContent='Send failed: '+error.message+'; your message is retained.'}finally{button.disabled=false}};
refresh();setInterval(refresh,3000);
"""

CHAT_PAGE = (
    """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Synta remote chat</title><style>
body{background:#101725;color:#e1e9f6;font:16px system-ui;margin:0;padding:20px;max-width:960px;margin:auto}a{color:#9bcaff}nav{display:flex;gap:22px}article{background:#1a2538;border:1px solid #33475f;border-radius:12px;margin:16px 0;padding:16px}h3{font-size:12px;overflow-wrap:anywhere;color:#9cb0c9}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;line-height:1.6}textarea{box-sizing:border-box;width:100%;min-height:120px;background:#1a2538;color:#fff;padding:14px;font:inherit;border:1px solid #536b88;border-radius:8px}button{padding:12px 26px;background:#78b4ff;border:0;border-radius:8px;font:inherit;cursor:pointer}summary{cursor:pointer;color:#a6cfff}#composer{position:sticky;bottom:0;background:#101725;padding:12px 0}#status{color:#afc7e5}
</style>"""
    + NAVIGATION.decode()
    + """<h1>Synta master chat</h1><p>Authenticated owner commands use the same mission queue. Replies and execution receipts stay available after you leave.</p><p id="status">Connecting</p><main id="history"></main><section id="composer"><textarea id="message" maxlength="8000" placeholder="Message to Synta, /status, /goal ..."></textarea><button id="send">Send</button></section><script>"""
    + CHAT_SCRIPT
    + "</script></html>"
)


def validate_config(value: dict) -> dict:
    owner, origin = value.get("owner_login"), value.get("origin")
    if not isinstance(owner, str) or not owner.strip() or len(owner) > 254:
        raise ValueError("An explicit authenticated Tailscale owner is required")
    parsed = urlsplit(origin or "")
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or not parsed.hostname.endswith(".ts.net")
        or parsed.username
        or parsed.password
        or parsed.path
        or parsed.query
        or parsed.fragment
        or parsed.port not in (None, 443)
    ):
        raise ValueError("Use this device's private Tailscale HTTPS origin")
    upstream = urlsplit(value.get("dashboard", ""))
    address = ipaddress.ip_address(upstream.hostname or "")
    if (
        upstream.scheme != "http"
        or address.version != 4
        or not address.is_private
        or address.is_unspecified
        or address.is_multicast
        or upstream.port != 8765
        or upstream.username
        or upstream.password
        or upstream.path
        or upstream.query
        or upstream.fragment
    ):
        raise ValueError("Use the existing private IPv4 dashboard on port 8765")
    return dict(value)


def proxy_allowed(route: str) -> bool:
    parsed = urlsplit(route)
    if parsed.scheme or parsed.netloc or parsed.fragment:
        return False
    return parsed.path in (
        "/",
        "/api/status",
        "/api/trades",
        "/api/actions",
        "/api/live-inference",
        "/api/readiness",
    ) or bool(
        re.fullmatch(
            r"/paper/\d{8}-\d{6}-(?:all|daily|weekly)-[a-f0-9]{8}/(?:report\.html|report\.json|report\.md|trades\.csv)",
            parsed.path,
        )
    )


class GatewayServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, root: Path, config: dict, port: int = 8786):
        self.root = root.resolve()
        self.config = validate_config(config)
        self.csrf = secrets.token_urlsafe(32)
        self.slots = threading.BoundedSemaphore(8)
        super().__init__(("127.0.0.1", port), GatewayHandler)

    def process_request(self, request, client_address):
        request.settimeout(8)
        if not self.slots.acquire(blocking=False):
            try:
                request.sendall(b"HTTP/1.0 503 Busy\r\nContent-Length: 0\r\n\r\n")
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class GatewayHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def authorized(self) -> bool:
        values = self.headers.get_all("Tailscale-User-Login", [])
        return len(values) == 1 and hmac.compare_digest(
            values[0], self.server.config["owner_login"]
        )

    def reply(self, body: bytes, mime: str, code: int = 200, policy: str | None = None):
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            policy or "default-src 'none'; frame-ancestors 'none'; base-uri 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def json_reply(self, value: dict, code: int = 200):
        self.reply(
            json.dumps(value, ensure_ascii=False, allow_nan=False).encode(),
            "application/json",
            code,
        )

    def do_GET(self):  # noqa: N802
        if not self.authorized():
            self.json_reply({"error": "Authenticated Tailscale owner required"}, 403)
            return
        from rlm.v100.mission_chat import inspect, recent_requests

        route = urlsplit(self.path).path
        try:
            if route == "/chat":
                digest = base64.b64encode(hashlib.sha256(CHAT_SCRIPT.encode()).digest()).decode()
                self.reply(
                    CHAT_PAGE.encode(),
                    "text/html; charset=utf-8",
                    policy="default-src 'none'; script-src 'sha256-"
                    + digest
                    + "'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
                )
            elif route == "/api/chat":
                self.json_reply(
                    {"requests": recent_requests(self.server.root, 32), "csrf": self.server.csrf}
                )
            elif re.fullmatch(r"/api/chat/[a-f0-9]{32}", route):
                self.json_reply(inspect(self.server.root, route.rsplit("/", 1)[1]))
            elif proxy_allowed(self.path):
                upstream = Request(self.server.config["dashboard"] + self.path)
                with urlopen(upstream, timeout=5) as response:
                    body = response.read(8 * 2**20 + 1)
                    if len(body) > 8 * 2**20:
                        raise ValueError("Dashboard response exceeds 8 MiB")
                    if route == "/":
                        body = dashboard_navigation(body)
                    self.reply(
                        body,
                        response.headers.get("Content-Type", "application/octet-stream"),
                        policy=response.headers.get("Content-Security-Policy"),
                    )
            else:
                self.json_reply({"error": "Unknown route"}, 404)
        except HTTPError as error:
            self.json_reply({"error": "Dashboard unavailable", "upstream_status": error.code}, 503)
        except (URLError, TimeoutError, OSError, sqlite3.Error) as error:
            self.json_reply({"error": "Service delayed", "detail": str(error)[:200]}, 503)
        except (ValueError, KeyError):
            self.json_reply({"error": "Unavailable request or invalid data"}, 404)

    def do_POST(self):  # noqa: N802
        if not self.authorized():
            self.json_reply({"error": "Authenticated Tailscale owner required"}, 403)
            return
        if (
            self.path != "/api/chat"
            or self.headers.get("Origin") != self.server.config["origin"]
            or self.headers.get("Sec-Fetch-Site", "same-origin") != "same-origin"
            or not hmac.compare_digest(self.headers.get("X-Synta-CSRF", ""), self.server.csrf)
            or self.headers.get_content_type() != "application/json"
            or self.headers.get("Transfer-Encoding")
        ):
            self.json_reply({"error": "Same-origin authenticated chat request required"}, 403)
            return
        from rlm.v100.mission_chat import direct_facts, inspect, save_direct_reply, submit

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 1 <= length <= 32768:
                raise ValueError("Message body must fit 32 KiB")
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict) or set(value) != {"message"}:
                raise ValueError("Expected only a message")
            message = value["message"]
            if not isinstance(message, str) or not message.strip() or len(message) > 8000:
                raise ValueError("Chat needs 1-8000 characters")
            direct = direct_facts(self.server.root, message)
            if direct is not None:
                self.json_reply(save_direct_reply(self.server.root, message, direct))
                return
            identity = submit(self.server.root, value["message"])
            self.json_reply(inspect(self.server.root, identity), 202)
        except (ValueError, UnicodeError) as error:
            self.json_reply({"error": str(error)[:200]}, 400)
        except (OSError, sqlite3.Error) as error:
            self.json_reply({"error": "Chat queue delayed", "detail": str(error)[:200]}, 503)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    with GatewayServer(args.root, config) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
