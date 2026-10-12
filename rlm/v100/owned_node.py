"""Dedicated owned Linux inference and fixed-kernel CPU service; no remote shell."""

import argparse
import base64
import hashlib
import http.client
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODELS = {"qwen3.5:2b-q4_K_M", "qwen3.5:4b-q4_K_M"}
CONTRACT = "synta-dedicated-linux-v1"
CPU_RAM = 768 * 2**20


def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".new")
    temporary.write_text(json.dumps(value, allow_nan=False))
    os.replace(temporary, path)


def identity(value: str) -> str:
    if not re.fullmatch(r"[a-f0-9]{24}-[a-f0-9]{32}", value):
        raise ValueError("Invalid owned CPU identity")
    return value


def resource_budget(ram_available: float, temperature: float, gpu_free: float) -> dict:
    if not all(math.isfinite(v) and v >= 0 for v in (ram_available, temperature, gpu_free)):
        raise ValueError("Invalid resource telemetry")
    reasons = []
    if ram_available < 1:
        reasons.append("Host RAM reserve below 1 GiB")
    if temperature >= 80:
        reasons.append("GPU temperature reached 80 C")
    if gpu_free < 0.25:
        reasons.append("GPU free memory below 256 MiB")
    return {"ready": not reasons, "reason": "; ".join(reasons) or "dedicated owned capacity"}


class Node:
    def __init__(self, root: Path, source: Path):
        import psutil

        self.root, self.source = root, source
        self.state = {"ready": False, "reason": "Waiting for measured telemetry"}
        self.gpu_lock = threading.Lock()
        self.cpu_lock = threading.Lock()
        self.generating = False
        self.maintenance = False
        self.processes = {}
        self.cpu_slots = max(
            1,
            min(
                16,
                (os.cpu_count() or 2) // 2,
                int(psutil.virtual_memory().total * 0.4 / CPU_RAM),
            ),
        )
        self.cpu_pool = ThreadPoolExecutor(max_workers=self.cpu_slots)
        for receipt in (root / "cpu").glob("*/status.json"):
            value = json.loads(receipt.read_text())
            if value.get("state") in ("queued", "running"):
                atomic(
                    receipt,
                    {
                        **value,
                        "state": "failed",
                        "detail": "Owned service restarted; retry through controller",
                        "updated": time.time(),
                    },
                )

    def sample(self) -> dict:
        import psutil

        measured = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,temperature.gpu,memory.free,memory.total,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=3,
        )
        lines = measured.stdout.strip().splitlines()
        if len(lines) != 1:
            raise ValueError("One operator-owned RTX 3060 Ti required")
        name, temperature, free, total, utilization, power = [
            s.strip() for s in lines[0].split(",")
        ]
        if "RTX 3060 Ti" not in name:
            raise ValueError("GPU differs from the explicitly registered RTX 3060 Ti")
        ram = psutil.virtual_memory()
        values = resource_budget(ram.available / 2**30, float(temperature), float(free) / 1024)
        if self.maintaining():
            values.update(ready=False, reason="Owned update window")
        with self.cpu_lock:
            active = len(self.processes)
        return {
            **values,
            "cpu_ready": not self.maintaining() and ram.available > 2**30,
            "contract": CONTRACT,
            "updated": time.time(),
            "name": "debian-3060ti",
            "gpu_name": name,
            "temperature_c": float(temperature),
            "gpu_utilization_percent": float(utilization),
            "board_watts": float(power),
            "free_vram_gib": float(free) / 1024,
            "total_vram_gib": float(total) / 1024,
            "free_ram_gib": ram.available / 2**30,
            "cpu_percent": psutil.cpu_percent(),
            "cpu_threads": os.cpu_count(),
            "cpu_slots": self.cpu_slots,
            "cpu_active": active,
            "generating": self.generating,
            "duty_percent": 100,
            "revision": self.source.name,
            "kernel_sha256": hashlib.sha256(
                (self.source / "compute_kernel.py").read_bytes()
            ).hexdigest(),
            "worker_sha256": hashlib.sha256(
                (self.source / "compute_worker.py").read_bytes()
            ).hexdigest(),
            "scope": "Dedicated capacity for real queued work; no guaranteed utilization or hardware-failure immunity. Remote metrics remain untrusted.",
        }

    def maintaining(self) -> bool:
        return self.maintenance or (self.root.parent / "installing").exists()

    def monitor(self):
        while True:
            try:
                self.state = self.sample()
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                import psutil

                try:
                    ram_ready = psutil.virtual_memory().available > 2**30
                except (OSError, psutil.Error):
                    ram_ready = False
                self.state = {
                    "ready": False,
                    "cpu_ready": not self.maintaining() and ram_ready,
                    "cpu_slots": self.cpu_slots,
                    "cpu_active": len(self.processes),
                    "contract": CONTRACT,
                    "reason": str(error),
                    "updated": time.time(),
                    "generating": self.generating,
                    "kernel_sha256": hashlib.sha256(
                        (self.source / "compute_kernel.py").read_bytes()
                    ).hexdigest(),
                    "worker_sha256": hashlib.sha256(
                        (self.source / "compute_worker.py").read_bytes()
                    ).hexdigest(),
                }
            try:
                atomic(self.root / "resource-status.json", self.state)
            except OSError as error:
                print(
                    json.dumps({"telemetry_write_error": str(error), "updated": time.time()}),
                    flush=True,
                )
            time.sleep(2)

    def submit_cpu(self, value: dict) -> dict:
        key = identity(value["identity"])
        raw = base64.b64decode(value["job"], validate=True)
        if len(raw) > 2 * 2**20 or hashlib.sha256(raw).hexdigest() != value["sha256"]:
            raise ValueError("CPU request input hash/size differs")
        job = json.loads(raw)
        if not isinstance(job, dict):
            raise ValueError("Fixed CPU job must be a JSON object")
        if __package__:
            from rlm.v100.compute_kernel import validate
        else:
            from compute_kernel import validate

        validate(job)
        if (
            job["id"] != key[:24]
            or job.get("kernel_sha256") != self.state.get("kernel_sha256")
            or job.get("worker_sha256") != self.state.get("worker_sha256")
        ):
            raise ValueError("Pinned CPU input or local kernel identity differs")
        folder = self.root / "cpu" / key
        with self.cpu_lock:
            if folder.exists():
                if (folder / "job.json").read_bytes() != raw:
                    raise ValueError("CPU identity reused with different input")
                return json.loads((folder / "status.json").read_text())
            if (
                len(self.processes) >= self.cpu_slots
                or self.maintaining()
                or not self.state.get("cpu_ready")
            ):
                raise BlockingIOError("Owned CPU capacity busy or telemetry unavailable")
            folder.mkdir(parents=True)
            (folder / "job.json").write_bytes(raw)
            status = {"state": "queued", "updated": time.time(), "job_sha256": value["sha256"]}
            atomic(folder / "status.json", status)
            self.processes[key] = None
            self.cpu_pool.submit(self.run_cpu, key, folder)
        return status

    def run_cpu(self, key: str, folder: Path):
        import psutil

        status = json.loads((folder / "status.json").read_text())
        try:
            environment = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2",
                "PYTHONNOUSERSITE": "1",
            }
            with (folder / "worker.log").open("w") as output:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        str(self.source / "compute_kernel.py"),
                        str(folder / "job.json"),
                        str(folder / "output"),
                    ],
                    cwd=folder,
                    env=environment,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
                with self.cpu_lock:
                    self.processes[key] = process
                started = time.monotonic()
                try:
                    while process.poll() is None:
                        if (folder / "cancel").exists():
                            raise RuntimeError("Cancelled by owning controller")
                        if (
                            time.monotonic() - started > 150
                            or psutil.Process(process.pid).memory_info().rss > CPU_RAM
                            or psutil.virtual_memory().available < 2**30
                        ):
                            raise RuntimeError("Owned CPU time/RAM reserve exceeded")
                        atomic(
                            folder / "status.json",
                            {**status, "state": "running", "updated": time.time()},
                        )
                        time.sleep(1)
                    if process.returncode:
                        raise RuntimeError("Fixed CPU kernel failed; inspect worker.log")
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.wait()
            files = {}
            for name in ("report.json", "weights.safetensors"):
                path = folder / "output" / name
                if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 2**20:
                    raise ValueError("Invalid CPU result size/type")
                files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            status.update(
                state="complete",
                files=files,
                worker="debian-3060ti-cpu",
                worker_implementation_sha256=hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            )
        except (OSError, ValueError, RuntimeError, psutil.Error) as error:
            status.update(state="failed", detail=str(error)[:400])
        finally:
            atomic(folder / "status.json", {**status, "updated": time.time()})
            with self.cpu_lock:
                self.processes.pop(key, None)

    def cpu_result(self, key: str) -> dict:
        folder = self.root / "cpu" / identity(key)
        status = json.loads((folder / "status.json").read_text())
        if status["state"] == "complete":
            status["artifacts"] = {
                name: base64.b64encode((folder / "output" / name).read_bytes()).decode()
                for name in ("report.json", "weights.safetensors")
            }
        return status


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        self.forward()

    def do_POST(self):  # noqa: N802
        self.forward()

    def do_DELETE(self):  # noqa: N802
        self.forward()

    def send_json(self, code: int, value: dict):
        body = json.dumps(value, allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def forward(self):
        self.connection.settimeout(10)
        if self.client_address[0] not in (
            self.server.owner,
            "127.0.0.1",
            self.server.server_address[0],
        ):
            self.send_error(403)
            return
        node = self.server.node
        acquired, upstream = False, None
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= length <= 3 * 2**20:
                raise ValueError("Node input exceeds 3 MiB")
            body = self.rfile.read(length)
            if (
                self.command == "POST"
                and self.path == "/api/synta-maintenance"
                and self.client_address[0] == "127.0.0.1"
            ):
                acquired = node.gpu_lock.acquire(blocking=False)
                if not acquired:
                    raise BlockingIOError("Owned inference active")
                with node.cpu_lock:
                    if node.processes:
                        raise BlockingIOError("Owned CPU jobs active")
                    node.maintenance = True
                self.send_json(200, {"state": "drained"})
                return
            if self.command == "GET" and self.path == "/api/synta-resources":
                self.send_json(200, node.state)
                return
            if self.path == "/api/synta-cpu" and self.command == "POST":
                self.send_json(202, node.submit_cpu(json.loads(body)))
                return
            if self.path.startswith("/api/synta-cpu/"):
                key = identity(self.path.removeprefix("/api/synta-cpu/"))
                if self.command == "DELETE":
                    (node.root / "cpu" / key / "cancel").touch()
                    self.send_json(200, {"state": "cancel requested"})
                elif self.command == "GET":
                    self.send_json(200, node.cpu_result(key))
                else:
                    self.send_error(405)
                return
            allowed = {
                "GET": {"/api/version", "/api/tags", "/api/ps"},
                "POST": {"/api/show", "/api/chat"},
            }
            if self.path not in allowed.get(self.command, set()):
                self.send_error(404)
                return
            if self.command == "POST":
                data = json.loads(body)
                if (
                    not isinstance(data, dict)
                    or not isinstance(data.get("model"), str)
                    or data.get("model") not in MODELS
                ):
                    raise ValueError("Model outside pinned dedicated-node allowlist")
                if self.path == "/api/chat":
                    if node.maintaining() or not node.state.get("ready"):
                        raise BlockingIOError(node.state.get("reason", "Node not ready"))
                    acquired = node.gpu_lock.acquire(blocking=False)
                    if not acquired:
                        raise BlockingIOError("Owned GPU inference busy")
                    if node.maintaining():
                        raise BlockingIOError("Owned update window")
                    options = data.setdefault("options", {})
                    options["num_ctx"] = 32768
                    options["num_predict"] = min(
                        4096, max(1, int(options.get("num_predict", 1024)))
                    )
                    options["num_batch"] = min(256, max(1, int(options.get("num_batch", 64))))
                    options["num_thread"] = os.cpu_count() or 2
                    data["keep_alive"] = -1
                    node.generating = True
                body = json.dumps(data).encode()
            upstream = http.client.HTTPConnection(
                "127.0.0.1", self.server.upstream_port, timeout=180
            )
            upstream.request(
                self.command,
                self.path,
                body=body or None,
                headers={"Content-Type": "application/json"},
            )
            response = upstream.getresponse()
            self.send_response(response.status)
            self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            while chunk := response.read1(65536):
                if acquired and not node.state.get("ready"):
                    raise RuntimeError("Owned generation cancelled on measured pressure")
                self.wfile.write(chunk)
                self.wfile.flush()
        except BlockingIOError as error:
            self.send_json(503, {"error": str(error)})
        except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
            self.send_json(400, {"error": str(error)[:300]})
        except (OSError, RuntimeError, http.client.HTTPException) as error:
            self.log_error("Owned transport ended: %s", error)
            self.close_connection = True
        finally:
            if upstream:
                upstream.close()
            if acquired:
                node.generating = False
                node.gpu_lock.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bind", default="192.168.0.9")
    parser.add_argument("--owner", default="192.168.0.68")
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--upstream-port", type=int, default=11436)
    args = parser.parse_args()
    node = Node(args.root.resolve(), args.source.resolve())
    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    server.owner, server.node, server.upstream_port = args.owner, node, args.upstream_port
    threading.Thread(target=node.monitor, daemon=True).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
