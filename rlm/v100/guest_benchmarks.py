"""Official coding graders inside the RAM-bounded private guest, never the host."""

import base64
import hashlib
import json
import shlex
import zlib
from pathlib import Path

from rlm.v100.common import atomic_json
from rlm.v100.protection import file_hash


def grade(root: Path, snapshot: Path, question: dict, answer: dict, folder: Path) -> dict:
    from rlm.v100.desktop import run

    pin = json.loads((snapshot / "manifest.json").read_text())
    worker = Path(__file__).with_name("benchmark_worker.py")
    if file_hash(worker) != pin["worker_sha256"]:
        raise ValueError("Official grading adapter changed; prepare a new snapshot")
    # Agentic repository repair uses a privileged Docker harness and large images.
    # Never silently execute it on the Debian host or use a substitute score.
    if question.get("task") == "agentic_coding":
        raise ValueError(
            "Official agentic-coding Docker harness needs a separately provisioned grading VM; full report remains incomplete"
        )
    folder.mkdir(parents=True, exist_ok=True)
    setup = run(
        root,
        """set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
if ! command -v bwrap >/dev/null || ! command -v python3 >/dev/null; then
  apt-get update -qq
  apt-get install -y -qq bubblewrap python3-venv git
fi
mkdir -p /workspace/official-benchmarks
if [ ! -x /workspace/official-benchmarks/venv/bin/python ]; then
  python3 -m venv /workspace/official-benchmarks/venv
fi
""",
        seconds=120,
    )
    if setup["exit_code"]:
        raise RuntimeError("Guest grader prerequisites unavailable: " + json.dumps(setup)[:500])
    revision = pin["scorer_revision"]
    import re

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Invalid official scorer pin")
    install = run(
        root,
        """set -euo pipefail
cd /workspace/official-benchmarks
if [ ! -f installed-REVISION ]; then
  venv/bin/pip install --disable-pip-version-check 'livebench @ git+https://github.com/LiveBench/LiveBench.git@REVISION' >/workspace/official-benchmarks/install.log 2>&1
  venv/bin/python -c 'from livebench.gen_ground_truth_judgment import play_a_match_gt'
  touch installed-REVISION
fi
venv/bin/pip freeze
""".replace("REVISION", revision),
        seconds=120,
    )
    if install["exit_code"]:
        raise RuntimeError(
            "Guest official grader installation deferred; inspect guest /workspace/official-benchmarks/install.log"
        )
    freeze = install["stdout"]
    digest = hashlib.sha256(freeze.encode()).hexdigest()
    runtime_pin = snapshot / "guest-grader-runtime.json"
    if (
        runtime_pin.exists()
        and json.loads(runtime_pin.read_text())["requirements_sha256"] != digest
    ):
        raise ValueError("Guest grader dependencies changed; reports cannot be compared")
    if not runtime_pin.exists():
        atomic_json(
            runtime_pin,
            {"requirements_sha256": digest, "requirements": freeze, "scorer_revision": revision},
        )
    request = {"question": question, "model": "local-candidate", "answer": answer}
    atomic_json(folder / "request.json", request)
    encoded = base64.b64encode(zlib.compress(json.dumps(request).encode())).decode()
    if len(encoded) > 2**20:
        raise ValueError("Coding question exceeds transfer budget")
    # Fresh directory per case; input is readonly in a nested namespace. Candidate
    # execution cannot modify the trusted installed scorer or later question files.
    identity = hashlib.sha256((str(folder) + digest).encode()).hexdigest()[:24]
    path = "/workspace/official-benchmarks/case-" + identity
    result = run(
        root, "mkdir -p " + shlex.quote(path) + "\n: > " + shlex.quote(path + "/input.b64"), 10
    )
    if result["exit_code"]:
        raise RuntimeError("Guest coding transfer directory unavailable")
    for offset in range(0, len(encoded), 8000):
        result = run(
            root,
            "printf '%s' "
            + shlex.quote(encoded[offset : offset + 8000])
            + " >> "
            + shlex.quote(path + "/input.b64"),
            10,
        )
        if result["exit_code"]:
            raise RuntimeError("Guest coding input transfer failed")
    script = """import base64,json,resource,zlib
from pathlib import Path
from livebench.common import MatchSingle
from livebench.gen_ground_truth_judgment import play_a_match_gt
resource.setrlimit(resource.RLIMIT_CPU,(90,90))
resource.setrlimit(resource.RLIMIT_FSIZE,(8*2**20,8*2**20))
data=json.loads(zlib.decompress(base64.b64decode(Path('/input.b64').read_text())))
result=play_a_match_gt(MatchSingle(data['question'],data['model'],data['answer']))
print('V100_OFFICIAL_RESULT:'+json.dumps(result,allow_nan=False),flush=True)
"""
    invocation = """set -euo pipefail
bwrap --unshare-all --die-with-parent --new-session --cap-drop ALL --clearenv \\
 --ro-bind /usr /usr --ro-bind /lib /lib --ro-bind /lib64 /lib64 \\
 --ro-bind /workspace/official-benchmarks/venv /workspace/official-benchmarks/venv \\
 --ro-bind CASE/input.b64 /input.b64 --proc /proc --dev /dev --tmpfs /tmp \\
 --setenv HOME /tmp --setenv OMP_NUM_THREADS 1 --setenv OPENBLAS_NUM_THREADS 1 \\
 --chdir /tmp /workspace/official-benchmarks/venv/bin/python -I -c SCRIPT
""".replace("CASE", path).replace("SCRIPT", shlex.quote(script))
    result = run(root, invocation, seconds=110, output_limit=12000)
    (folder / "grader.log").write_text(json.dumps(result, indent=2))
    if result["exit_code"]:
        raise RuntimeError("Official coding grader failed inside guest; no fabricated score")
    lines = result["stdout"].splitlines()
    if not lines or not lines[-1].startswith("V100_OFFICIAL_RESULT:"):
        raise ValueError("Guest grader returned no complete official result")
    grading = json.loads(lines[-1].split(":", 1)[1])
    if grading.get("eval_status") == "eval_error" or grading.get("error_msg"):
        raise ValueError("Official coding scorer error: " + str(grading)[:300])
    import math

    score = grading.get("score")
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid official coding score")
    atomic_json(folder / "result.json", grading)
    return grading
