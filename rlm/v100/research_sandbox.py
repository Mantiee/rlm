"""CPU experiment workspace with a public GET broker and no host/network mounts."""

import json
import os
import signal
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

from rlm.v100.common import atomic_json
from rlm.v100.protection import file_hash

BOOTSTRAP = """import json, socket, resource, os
resource.setrlimit(resource.RLIMIT_AS, (3*2**30, 3*2**30))
resource.setrlimit(resource.RLIMIT_FSIZE, (8*2**20, 8*2**20))
resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
def fetch(url):
    with socket.socket(socket.AF_UNIX) as s:
        s.connect('/broker/http.sock')
        s.sendall((json.dumps({'url':url})+'\\n').encode())
        f=s.makefile('rb')
        response=json.loads(f.readline(600000))
        if 'error' in response: raise RuntimeError(response['error'])
        return response
os.chdir('/work')
exec(compile(open('/experiment.py').read(), '/experiment.py', 'exec'), {'fetch':fetch, '__name__':'__main__'})
"""


def runtime(root: Path) -> str:
    pin = json.loads((root / "research/sandbox-runtime.json").read_text())
    if file_hash(Path(pin["binary"])) != pin["sha256"]:
        raise ValueError("Sandbox binary changed")
    return pin["binary"]


def command(root: Path, source: Path, work: Path, broker: Path) -> list[str]:
    args = [
        runtime(root),
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
        "--clearenv",
    ]
    for path in dict.fromkeys(
        [Path("/usr"), Path("/lib"), Path("/lib64"), Path(sys.prefix), Path(sys.base_prefix)]
    ):
        if path.exists():
            args += ["--ro-bind", str(path), str(path)]
    return args + [
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--size",
        "33554432",
        "--tmpfs",
        "/tmp",
        "--ro-bind",
        str(source),
        "/experiment.py",
        "--bind",
        str(work),
        "/work",
        "--ro-bind",
        str(broker),
        "/broker",
        "--chdir",
        "/work",
        "--setenv",
        "HOME",
        "/work",
        "--setenv",
        "OMP_NUM_THREADS",
        "2",
        "--setenv",
        "OPENBLAS_NUM_THREADS",
        "2",
        "--setenv",
        "MKL_NUM_THREADS",
        "2",
        sys.executable,
        "-I",
        "-u",
        "-c",
        BOOTSTRAP,
    ]


def run(root: Path, branch: str, code: str, seconds: int = 60) -> dict:
    if branch not in ("A", "B") or not isinstance(code, str) or not 1 <= len(code) <= 12000:
        raise ValueError("CPU experiment needs A/B and 1-12000 code characters")
    if type(seconds) is not int or not 1 <= seconds <= 60:
        raise ValueError("CPU experiment timeout must be 1-60 seconds")
    compile(code, "experiment.py", "exec")
    # Verify the namespace runtime BEFORE executing any model-generated code.
    runtime(root)
    import shutil

    import psutil

    if shutil.disk_usage(root).free < 10 * 2**30 or psutil.virtual_memory().available < 4 * 2**30:
        raise RuntimeError("CPU experiment deferred: need 4 GiB available RAM and 10 GiB disk")
    folder = root / "research/python-experiments" / ("run-" + uuid.uuid4().hex[:12])
    folder.mkdir(parents=True)
    work, source = folder / "work", folder / "experiment.py"
    work.mkdir()
    source.write_text(code)
    observations = []

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            from rlm.v100.research_tools import download_page

            self.connection.settimeout(35)
            try:
                request = json.loads(self.rfile.readline(4097))
                if set(request) != {"url"} or len(observations) >= 8:
                    raise ValueError("Public GET budget exhausted or invalid request")
                observations.append({"requested_url": request["url"], "started": time.time()})
                url, body = download_page(request["url"], max_bytes=128 * 1024)
                import hashlib

                result = {
                    "url": url,
                    "body": body,
                    "sha256": hashlib.sha256(body.encode()).hexdigest(),
                    "fetched_at": time.time(),
                }
                observations[-1].update({k: v for k, v in result.items() if k != "body"})
                (folder / f"source-{len(observations)}.txt").write_text(body)
            except Exception as error:
                result = {"error": str(error)[:300]}
            self.wfile.write((json.dumps(result) + "\n").encode())

    started, timed_out = time.monotonic(), False
    with tempfile.TemporaryDirectory(prefix="v100-web-") as tmp:
        with socketserver.UnixStreamServer(str(Path(tmp) / "http.sock"), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with (folder / "output.log").open("wb") as log:
                    process = subprocess.Popen(
                        command(root, source, work, Path(tmp)),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    try:
                        os.sched_setaffinity(process.pid, sorted(os.sched_getaffinity(0))[:2])
                        deadline = time.monotonic() + seconds
                        while process.poll() is None:
                            used = sum(
                                p.lstat().st_size for p in work.rglob("*") if not p.is_symlink()
                            )
                            if used > 32 * 2**20 or len(list(work.rglob("*"))) > 2048:
                                raise subprocess.TimeoutExpired("workspace quota", seconds)
                            if time.monotonic() >= deadline:
                                raise subprocess.TimeoutExpired("experiment", seconds)
                            try:
                                process.wait(timeout=0.5)
                            except subprocess.TimeoutExpired:
                                pass
                    except subprocess.TimeoutExpired:
                        timed_out = True
                    finally:
                        # Kill descendants as well; no background work escapes its budget.
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
            finally:
                server.shutdown()
                thread.join(timeout=40)
    result = {
        "branch": branch,
        "directory": str(folder),
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "seconds": time.monotonic() - started,
        "output": (folder / "output.log").read_text(errors="replace")[-12000:],
        "sources": observations,
        "status": "experiment, not independently verified truth",
    }
    atomic_json(folder / "result.json", result)
    return result
