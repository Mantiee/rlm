"""Owned Ollama admission/pacing proxy. Does not modify device power or clocks."""

import argparse
import http.client
import json
import math
import os
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

GAMES = {
    "league of legends.exe",
    "valorant-win64-shipping.exe",
    "cs2.exe",
    "fortniteclient-win64-shipping.exe",
    "cyberpunk2077.exe",
    "overwatch.exe",
    "eldenring.exe",
    "gta5.exe",
    "rdr2.exe",
    "dota2.exe",
}


def budget(
    cpu: float,
    ram: float,
    temperature: float,
    free_vram: float,
    watts: float,
    decoder: float,
    game: bool,
) -> dict:
    values = (cpu, ram, temperature, free_vram, watts, decoder)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Resource sensor returned non-finite values")
    reasons = []
    if game:
        reasons.append("heavy game running")
    if cpu > 65:
        reasons.append("host CPU above 65%")
    if ram < 6:
        reasons.append("free host RAM below 6 GiB")
    if temperature >= 75:
        reasons.append("GPU temperature at least 75 C")
    if free_vram < 3:
        reasons.append("free VRAM below 3 GiB")
    if watts > 250:
        reasons.append("board power above 250 W; stopping only owned inference")
    duty = 20 if cpu > 50 else 35 if cpu > 35 or decoder > 20 else 65
    return {
        "ready": not reasons,
        "reason": "; ".join(reasons) or "spare headroom",
        "duty_percent": duty,
        "batch_tokens": 16 if duty < 35 else 32 if duty < 65 else 64,
        "cpu_percent": cpu,
        "free_ram_gib": ram,
        "temperature_c": temperature,
        "free_vram_gib": free_vram,
        "board_watts": watts,
        "decoder_percent": decoder,
        "scope": "Measured admission and interruption after sampling; not a hard peak-power cap or crash guarantee. Browser presence does not pause work.",
    }


class Pressure:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.state = {"ready": False, "reason": "Waiting for resource sensors"}
        self.lock = threading.Lock()
        self.not_before = 0.0
        self.generating = False

    def stop_owned_runner(self) -> None:
        import psutil

        runtime = str(self.root / "runtime").casefold() + os.sep
        for process in psutil.process_iter(["name", "exe"]):
            executable = process.info.get("exe") or ""
            if process.info.get(
                "name", ""
            ).casefold() == "llama-server.exe" and executable.casefold().startswith(runtime):
                process.kill()

    def sample(self) -> dict:
        import psutil

        names = {str(p.info.get("name", "")).casefold() for p in psutil.process_iter(["name"])}
        executable = shutil.which("nvidia-smi")
        if not executable:
            raise RuntimeError("nvidia-smi sensor unavailable; GPU admission disabled")
        result = subprocess.run(
            [
                executable,
                "--query-gpu=temperature.gpu,memory.free,power.draw,utilization.decoder",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        temperature, vram, watts, decoder = map(
            float, result.stdout.strip().splitlines()[0].split(",")
        )
        return budget(
            psutil.cpu_percent(interval=0.2),
            psutil.virtual_memory().available / 2**30,
            temperature,
            vram / 1024,
            watts,
            decoder,
            bool(names & GAMES),
        )

    def monitor(self) -> None:
        while True:
            try:
                state = self.sample()
                if self.generating and not state["ready"]:
                    self.stop_owned_runner()
                    self.not_before = max(self.not_before, time.monotonic() + 60)
            except Exception as error:
                state = {"ready": False, "reason": str(error), "duty_percent": 20}
                if self.generating:
                    try:
                        self.stop_owned_runner()
                    except Exception as stop_error:
                        state["reason"] += "; owned runner stop failed: " + str(stop_error)
            self.state = {
                **state,
                "updated": time.time(),
                "contract": "synta-adaptive-windows-v1",
                "cooldown_seconds": round(max(0, self.not_before - time.monotonic()), 1),
                "generating": self.generating,
            }
            path = self.root / "logs/resource-status.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".new")
            temporary.write_text(json.dumps(self.state), encoding="utf-8")
            os.replace(temporary, path)
            time.sleep(2)


class Proxy(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        self.forward()

    def do_POST(self):  # noqa: N802
        self.forward()

    def forward(self) -> None:
        self.connection.settimeout(10)
        pressure = self.server.pressure
        if self.client_address[0] not in {
            self.server.owner,
            self.server.server_address[0],
            "127.0.0.1",
        }:
            self.send_error(403)
            return
        if self.path == "/api/synta-resources" and self.command == "GET":
            self.send_json(200, pressure.state)
            return
        if self.path not in {
            "/api/chat",
            "/api/generate",
            "/api/tags",
            "/api/ps",
            "/api/show",
            "/api/version",
            "/api/pull",
        }:
            self.send_error(404)
            return
        generating = self.command == "POST" and self.path in {"/api/chat", "/api/generate"}
        acquired = False
        connection = None
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= length <= 2**21:
                raise ValueError("Request exceeds 2 MiB")
            body = self.rfile.read(length)
            if generating:
                if not pressure.state["ready"] or time.monotonic() < pressure.not_before:
                    reason = (
                        pressure.state["reason"]
                        if not pressure.state["ready"]
                        else "Owned GPU cooldown: "
                        + str(round(pressure.not_before - time.monotonic(), 1))
                        + " s"
                    )
                    self.send_json(503, {"error": reason, "resources": pressure.state})
                    return
                acquired = pressure.lock.acquire(blocking=False)
                if not acquired:
                    self.send_json(503, {"error": "One owned GPU inference already active"})
                    return
                data = json.loads(body)
                options = data.setdefault("options", {})
                options["num_batch"] = max(
                    1, min(int(options.get("num_batch", 64)), pressure.state["batch_tokens"])
                )
                options["num_thread"] = 2
                options["num_predict"] = max(1, min(int(options.get("num_predict", 1024)), 2048))
                body = json.dumps(data).encode()
                pressure.generating = True
            started = time.monotonic()
            connection = http.client.HTTPConnection(
                "127.0.0.1", self.server.upstream_port, timeout=180
            )
            connection.request(
                self.command,
                self.path,
                body=body or None,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            self.send_response(response.status)
            self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            while True:
                chunk = response.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
            if generating:
                duty = pressure.state.get("duty_percent", 20)
                pressure.not_before = max(
                    pressure.not_before,
                    time.monotonic() + (time.monotonic() - started) * (100 - duty) / duty,
                )
        except (OSError, ValueError, http.client.HTTPException) as error:
            self.log_error("Owned transport failed: %s", error)
            self.close_connection = True
        finally:
            if connection:
                connection.close()
            if acquired:
                pressure.generating = False
                pressure.lock.release()

    def send_json(self, status: int, value: dict) -> None:
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        print(json.dumps({"time": time.time(), "transport": format % args}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bind", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--upstream-port", type=int, default=11436)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.bind, args.port), Proxy)
    server.daemon_threads = True
    server.owner, server.upstream_port = args.owner, args.upstream_port
    server.pressure = Pressure(args.root)
    threading.Thread(target=server.pressure.monitor, daemon=True).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
