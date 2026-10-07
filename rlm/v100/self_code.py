"""Pinned own source; model edits remain isolated candidates, never live host code."""

import json
import re
import shutil
import subprocess
import sys
import uuid
from importlib.metadata import distribution
from pathlib import Path

from rlm.v100.common import atomic_json
from rlm.v100.protection import file_hash


def prepare(root: Path) -> dict:
    from rlm.v100.mission import status

    if status(root)["running"] or sys.prefix == sys.base_prefix:
        raise ValueError("Stop the mission and use the isolated continual venv")
    bwrap = shutil.which("bwrap")
    if not bwrap:
        folder = root / "tools/bubblewrap" / uuid.uuid4().hex[:12]
        folder.mkdir(parents=True)
        subprocess.run(["apt", "download", "bubblewrap"], cwd=folder, check=True, timeout=120)
        packages = list(folder.glob("bubblewrap_*.deb"))
        if len(packages) != 1:
            raise ValueError("Expected one APT-verified sandbox package")
        subprocess.run(["dpkg-deb", "-x", str(packages[0]), str(folder / "extracted")], check=True)
        bwrap = str(folder / "extracted/usr/bin/bwrap")
    subprocess.run(
        [
            bwrap,
            "--unshare-all",
            "--ro-bind",
            "/",
            "/",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "/usr/bin/true",
        ],
        check=True,
        timeout=15,
    )
    atomic_json(
        root / "research/sandbox-runtime.json", {"binary": bwrap, "sha256": file_hash(Path(bwrap))}
    )
    origin = json.loads(distribution("rlms").read_text("direct_url.json") or "{}")
    revision = origin.get("vcs_info", {}).get("commit_id", "")
    if origin.get("url") != "https://github.com/Mantiee/rlm.git" or not re.fullmatch(
        r"[0-9a-f]{40}", revision
    ):
        raise ValueError("Self-code requires the explicitly pinned fork installation")
    source = root / "src/self-code" / revision
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", origin["url"], str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", revision], check=True)
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if actual != revision or dirty:
        raise ValueError("Experimental source revision or tracked files changed")
    subprocess.run(
        [
            "uv",
            "--no-config",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--index-url",
            "https://pypi.org/simple",
            "pytest==8.4.2",
        ],
        check=True,
    )
    value = {
        "source": str(source),
        "revision": revision,
        "scope": "Isolated algorithm candidates; host controller and system prompts remain unchanged",
    }
    atomic_json(root / "research/self-code-source.json", value)
    return value


def execute(root: Path, branch: str, name: str, arguments: dict) -> dict:
    from rlm.v100.code_lab import PROMPT_FILES, algorithm_source, create_code, source_files

    source = Path(json.loads((root / "research/self-code-source.json").read_text())["source"])
    if name == "list_algorithm_files" and not arguments:
        return {
            "files": [
                p
                for p in source_files(source)
                if p.startswith("rlm/") and p.endswith(".py") and p not in PROMPT_FILES
            ]
        }
    if name == "read_algorithm_file" and set(arguments) == {"filename"}:
        return {
            "filename": arguments["filename"],
            "source": algorithm_source(source, arguments["filename"]),
        }
    if name == "create_code_candidate" and set(arguments) == {
        "filename",
        "find",
        "replace",
        "hypothesis",
    }:
        output = root / "research/code-candidates" / ("candidate-" + uuid.uuid4().hex[:12])
        report = create_code(
            source,
            arguments["filename"],
            {k: arguments[k] for k in ("find", "replace", "hypothesis")},
            output,
        )
        atomic_json(output / "proposal.json", {"branch": branch})
        return {
            "directory": str(output),
            "status": report["status"],
            "next": "Run check_code_candidate; unit checks alone do not justify promotion",
        }
    raise ValueError("Invalid own-code tool request")


def admitted(root: Path) -> dict:
    from rlm.v100.code_lab import verify_code

    result = {}
    for folder in sorted((root / "research/code-candidates").glob("candidate-*")):
        if (
            not (folder / "proposal.json").exists()
            or not (folder / "check-result.json").exists()
            or (folder / "attempted.json").exists()
        ):
            continue
        branch = json.loads((folder / "proposal.json").read_text())["branch"]
        checked = json.loads((folder / "check-result.json").read_text())
        source = verify_code(folder)
        if (
            branch in ("A", "B")
            and branch not in result
            and checked.get("passed")
            and checked["source_files"] == source["source_files"]
        ):
            result[branch] = folder
            atomic_json(
                folder / "attempted.json",
                {
                    "status": "admitted for independent training and task-quality testing; no live code replacement"
                },
            )
    return result
