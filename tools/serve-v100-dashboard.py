#!/usr/bin/env python3
"""Read-only LAN dashboard for an existing V100 installation; no model reload."""

import argparse
import base64
import hashlib
import ipaddress
import json
import math
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from rlm.v100.dashboard_layout import (
    APP_SCRIPT,
    BASE_TEMPLATE,
    GUEST_TEMPLATE,
    PAGE,
    REQUIRED_IDS,
    guest_layout,
    render_template,
    validate_template,
)

REPORT_ID = re.compile(r"\d{8}-\d{6}-(?:all|daily|weekly)-[0-9a-f]{8}")
ASSETS = {
    "report.html": "text/html; charset=utf-8",
    "report.json": "application/json",
    "report.md": "text/plain; charset=utf-8",
    "trades.csv": "text/csv; charset=utf-8",
}


__all__ = [
    "APP_SCRIPT",
    "BASE_TEMPLATE",
    "GUEST_TEMPLATE",
    "PAGE",
    "REQUIRED_IDS",
    "guest_layout",
    "render_template",
    "validate_template",
]


def validate_bind(address):
    ip = ipaddress.ip_address(address)
    if (
        ip.version != 4
        or not ip.is_private
        or ip.is_unspecified
        or ip.is_multicast
        or ip.is_reserved
    ):
        raise ValueError("Bind to an explicit private IPv4 address or 127.0.0.1")
    return address


def read_json(path):
    if path.stat().st_size > 2 * 2**20:
        raise ValueError("JSON snapshot exceeds 2 MiB")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected an object snapshot")
    return value


def paper_asset(root, route):
    parts = unquote(urlsplit(route).path).split("/")
    if (
        len(parts) != 4
        or parts[1] != "paper"
        or not REPORT_ID.fullmatch(parts[2])
        or parts[3] not in ASSETS
    ):
        raise ValueError("Unknown report route")
    base = (root / "research/paper/reports").resolve()
    path = base / parts[2] / parts[3]
    if (
        path.resolve().parent != base / parts[2]
        or not path.is_file()
        or path.stat().st_size > 8 * 2**20
    ):
        raise ValueError("Report is missing, oversized or outside its directory")
    return path, ASSETS[parts[3]]


class DashboardState:
    def __init__(self, root, mission_reader):
        self.root = root.resolve()
        self.mission_reader = mission_reader
        self.data = {"collected_at": time.time(), "errors": ["Pierwsze zbieranie danych trwa"]}
        self.stop = threading.Event()
        self.layout = {"guest_file": GUEST_TEMPLATE, "state": "waiting for guest template"}
        self.page = render_template(BASE_TEMPLATE, validate_template(BASE_TEMPLATE))
        saved = self.root / "research/dashboard/index.html"
        content, failure = BASE_TEMPLATE, None
        try:
            if saved.exists():
                if saved.stat().st_size > 256 * 1024:
                    raise ValueError("Saved layout exceeds 256 KiB")
                content = saved.read_text()
                validate_template(content)
        except (OSError, ValueError) as error:
            content, failure = BASE_TEMPLATE, str(error)[:300]
        self.publish_layout(content)
        if failure:
            self.layout["error"] = failure

    def publish_layout(self, content):
        identity = validate_template(content)
        folder = self.root / "research/dashboard"
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        history = folder / "layouts"
        history.mkdir(exist_ok=True, mode=0o700)
        version = history / (identity + ".html")
        if not version.exists():
            version.write_text(content)
        if self.layout.get("active_sha256") != identity:
            temporary = folder / "index.html.tmp"
            temporary.write_text(content)
            temporary.replace(folder / "index.html")
        self.page = render_template(content, identity)
        self.layout = {
            "guest_file": GUEST_TEMPLATE,
            "state": "validated layout active",
            "active_sha256": identity,
            "published_at": time.time(),
        }

    def sync_layout(self):
        try:
            content = guest_layout(self.root)
            try:
                validate_template(content)
            except ValueError:
                from rlm.v100.dashboard_editor import repair

                self.last_repair = repair(self.root)
                content = guest_layout(self.root)
            self.publish_layout(content)
            if hasattr(self, "last_repair"):
                self.layout["repair"] = self.last_repair
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            self.layout = {
                **self.layout,
                "state": "previous validated layout retained",
                "error": str(error)[:300],
            }
        folder = self.root / "research/dashboard"
        temporary = folder / "layout-status.json.tmp"
        temporary.write_text(json.dumps(self.layout, ensure_ascii=False))
        temporary.replace(folder / "layout-status.json")
        self.data = {**self.data, "layout": self.layout}

    def refresh(self):
        errors, report, gpu = [], {}, {}
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "rlm.v100.cli",
                    "--root",
                    str(self.root),
                    "mission-report",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=20,
            )
            if result.returncode:
                raise RuntimeError(result.stderr[-600:] or "mission-report failed")
            report = read_json(self.root / "research/mission/latest-report.json")
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            errors.append("Raport nie został odświeżony: " + str(error)[:600])
        mission = self.mission_reader(self.root)
        if report.get("mission_evidence", {}).get("run") != mission.get("run"):
            if report:
                errors.append(
                    "Raport dotyczy poprzedniego przebiegu; pokazano tylko bieżący status"
                )
            report = {}
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            values = result.stdout.strip().splitlines()[0].split(",")
            gpu = dict(
                zip(
                    ("utilization", "memory_used", "memory_total", "temperature", "power"),
                    (float(v.strip()) for v in values),
                    strict=True,
                )
            )
            if not all(math.isfinite(value) for value in gpu.values()):
                gpu = {}
                raise ValueError("GPU counters are nonfinite")
        except (OSError, ValueError, IndexError, subprocess.SubprocessError) as error:
            errors.append("Pomiar GPU niedostępny: " + str(error)[:200])
        log = ""
        if mission.get("run"):
            path = Path(mission["run"]) / "controller.log"
            if path.resolve().is_relative_to(self.root / "research/mission") and path.exists():
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 16000))
                    log = stream.read(16000).decode(errors="replace")
        reports = sorted(
            (self.root / "research/paper/reports").glob("*/report.html"),
            key=lambda p: p.stat().st_mtime_ns,
            reverse=True,
        )
        from rlm.v100.live_status import snapshot

        live = {}
        try:
            live = snapshot(self.root, mission, report, gpu)
        except (OSError, ValueError, RuntimeError) as error:
            errors.append("Podgląd pracy niedostępny: " + str(error)[:200])
        self.data = {
            "live": live,
            "collected_at": time.time(),
            "mission": mission,
            "report": report,
            "gpu": gpu,
            "controller_log": log,
            "paper_reports": [p.parent.name for p in reports if REPORT_ID.fullmatch(p.parent.name)][
                :12
            ],
            "errors": errors,
        }

    def loop(self):
        while not self.stop.is_set():
            try:
                self.refresh()
                self.sync_layout()
            except (OSError, ValueError, RuntimeError) as error:
                self.data = {"collected_at": time.time(), "errors": [str(error)[:600]]}
            self.stop.wait(20)


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, state):
        self.state = state
        self.slots = threading.BoundedSemaphore(8)
        super().__init__(address, DashboardHandler)

    def process_request(self, request, client_address):
        request.settimeout(10)
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


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):  # noqa: N802 - standard HTTP callback
        route = urlsplit(self.path).path
        try:
            if route == "/":
                body, mime = self.server.state.page.encode(), "text/html; charset=utf-8"
            elif route == "/api/status":
                body, mime = (
                    json.dumps(
                        self.server.state.data, ensure_ascii=False, allow_nan=False
                    ).encode(),
                    "application/json; charset=utf-8",
                )
            else:
                path, mime = paper_asset(self.server.state.root, self.path)
                body = path.read_bytes()
        except (ValueError, OSError):
            self.send_error(404, "Unavailable")
            return
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; script-src 'sha256-"
            + base64.b64encode(
                hashlib.sha256(body.split(b"<script>", 1)[1].split(b"</script>", 1)[0]).digest()
            ).decode()
            + "'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
            if route == "/"
            else "default-src 'none'; style-src 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802 - standard HTTP callback
        self.send_error(405, "Read-only dashboard")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bind", required=True, type=validate_bind)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Use an unprivileged port between 1024 and 65535")
    from rlm.v100.mission import status

    state = DashboardState(args.root, status)
    with DashboardServer((args.bind, args.port), state) as server:
        thread = threading.Thread(target=state.loop, daemon=True)
        thread.start()
        print(f"V100 dashboard: http://{args.bind}:{args.port} - read-only", flush=True)
        try:
            server.serve_forever()
        finally:
            state.stop.set()
            thread.join(timeout=1)


if __name__ == "__main__":
    main()
