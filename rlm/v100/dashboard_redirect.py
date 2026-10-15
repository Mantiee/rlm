"""Read-only redirect from the retired income dashboard to Mission Control."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class RedirectServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, target: str):
        self.target = target
        super().__init__(address, RedirectHandler)


class RedirectHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _redirect(self):
        self.send_response(302)
        self.send_header("Location", self.server.target)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path == "/healthz":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")
            return
        self._redirect()

    def do_HEAD(self):
        self._redirect()

    def do_POST(self):
        self.send_error(405, "Read-only redirect service")

    def log_message(self, format, *args):
        # Avoid logging user-provided URLs or query parameters.
        super().log_message("dashboard redirect: " + format, *args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args(argv)
    with RedirectServer((args.host, args.port), args.target) as server:
        server.serve_forever(poll_interval=0.5)


if __name__ == "__main__":
    main()
