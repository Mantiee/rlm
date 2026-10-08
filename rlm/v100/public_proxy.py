"""VM egress proxy: public HTTP(S) only, pinned DNS, no home-network access."""

import ipaddress
import select
import socket
import socketserver
import threading
import time
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlsplit


def public_address(host: str, port: int) -> tuple:
    if port not in (80, 443) or not host or len(host) > 253:
        raise ValueError("Only public HTTP/HTTPS ports are available")
    addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    if not addresses or any(
        not ipaddress.ip_address(row[4][0]).is_global
        or ipaddress.ip_address(row[4][0]).is_multicast
        or ipaddress.ip_address(row[4][0]).is_reserved
        for row in addresses
    ):
        raise ValueError("Loopback, private, link-local and special destinations are blocked")
    return addresses[0]


class PublicProxy(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = False
    daemon_threads = True

    def __init__(self, address):
        self.slots = threading.BoundedSemaphore(6)
        self.connections = threading.BoundedSemaphore(16)
        super().__init__(address, ProxyRequest)

    def process_request(self, request, client_address):
        if not self.connections.acquire(blocking=False):
            self.shutdown_request(request)
            return
        request.settimeout(10)
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.connections.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connections.release()


class ProxyRequest(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def do_CONNECT(self):  # noqa: N802 - HTTP handler API
        self.forward(True)

    def do_GET(self):  # noqa: N802
        self.forward(False)

    def do_HEAD(self):  # noqa: N802
        self.forward(False)

    def do_POST(self):  # noqa: N802
        self.forward(False)

    def forward(self, tunnel):
        if not self.server.slots.acquire(blocking=False):
            self.send_error(503, "VM connection budget exhausted")
            return
        peer = None
        try:
            parsed = urlsplit("//" + self.path if tunnel else self.path)
            if parsed.username or parsed.password or (not tunnel and parsed.scheme != "http"):
                raise ValueError("Invalid proxy destination")
            port = parsed.port or (443 if tunnel else 80)
            family, socktype, proto, _, address = public_address(parsed.hostname, port)
            peer = socket.socket(family, socktype, proto)
            peer.settimeout(10)
            peer.connect(address)  # Connect to the checked IP, never re-resolve DNS.
            if tunnel:
                self.send_response(200, "Connection established")
                self.end_headers()
            else:
                # No request smuggling/persistent connection to another host.
                if self.headers.get("Transfer-Encoding"):
                    raise ValueError("Chunked plaintext requests unsupported; use HTTPS")
                body_size = int(self.headers.get("Content-Length", "0"))
                if not 0 <= body_size <= 2**20:
                    raise ValueError("Plain HTTP upload too large")
                body = self.rfile.read(body_size) if body_size else b""
                path = parsed.path or "/"
                if parsed.query:
                    path += "?" + parsed.query
                headers = [
                    (k, v)
                    for k, v in self.headers.items()
                    if k.lower()
                    not in (
                        "host",
                        "connection",
                        "proxy-connection",
                        "proxy-authorization",
                        "keep-alive",
                        "upgrade",
                    )
                ]
                payload = f"{self.command} {path} HTTP/1.0\r\nHost: {parsed.netloc}\r\nConnection: close\r\n"
                payload += "".join(f"{k}: {v}\r\n" for k, v in headers) + "\r\n"
                peer.sendall(payload.encode("latin1") + body)
            # Six bounded connections, max 8 GiB each, ten minute lifetime.
            # Large downloads can resume inside the guest's fixed-size disk.
            deadline, count = time.monotonic() + 600, 0
            self.connection.settimeout(15)
            while time.monotonic() < deadline and count < 8 * 2**30:
                ready, _, _ = select.select([peer, self.connection], [], [], 15)
                if not ready:
                    break
                for source in ready:
                    data = source.recv(65536)
                    if not data:
                        return
                    destination = self.connection if source is peer else peer
                    destination.sendall(data)
                    count += len(data)
        except (ValueError, OSError):
            if peer is None:
                self.send_error(403, "Public destination unavailable or outside the network policy")
        finally:
            if peer is not None:
                peer.close()
            self.close_connection = True
            self.server.slots.release()
