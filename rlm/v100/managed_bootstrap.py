"""Operator-authorized GitHub updates and owned-host preparation, with receipts."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path

if __package__:
    from rlm.v100.inference_benchmark import save
else:
    from inference_benchmark import save

REPOSITORY = "Mantiee/rlm"
BRANCH = "v100-rtx-reconnect"


def fetch(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "SyntaManagedUpdater/1.0 (+https://github.com/Mantiee/rlm)"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(4 * 1024**2 + 1)
    if len(data) > 4 * 1024**2:
        raise ValueError("Release metadata/script exceeds its byte budget")
    return data


def release(revision: str | None = None) -> tuple[str, dict]:
    if revision is None:
        revision = json.loads(
            fetch(f"https://api.github.com/repos/{REPOSITORY}/git/ref/heads/{BRANCH}")
        )["object"]["sha"]
    if not re.fullmatch("[0-9a-f]{40}", revision):
        raise ValueError("Release must resolve to an immutable commit")
    value = json.loads(
        fetch(
            f"https://raw.githubusercontent.com/{REPOSITORY}/{revision}/tools/managed-release.json"
        )
    )
    if value.get("schema") != "synta-managed-release-v1" or type(value.get("version")) is not int:
        raise ValueError("Unknown managed release contract")
    if not 83 <= value["version"] <= 9999 or not isinstance(value.get("files"), dict):
        raise ValueError("Invalid managed release version or artifact inventory")
    for key, suffix in (("linux_installer", ".sh"), ("windows_prepare", ".ps1")):
        name = value.get(key, "")
        if name not in value["files"] or not name.startswith("tools/") or not name.endswith(suffix):
            raise ValueError("Entrypoint is not a declared release artifact")
    return revision, value


def materialize(folder: Path, revision: str, manifest: dict) -> None:
    for name, expected in manifest["files"].items():
        if not re.fullmatch(r"(?:tools|rlm/v100)/[A-Za-z0-9_.-]+", name) or not re.fullmatch(
            "[0-9a-f]{64}", expected
        ):
            raise ValueError("Release artifact path/hash invalid")
        data = fetch(f"https://raw.githubusercontent.com/{REPOSITORY}/{revision}/{name}")
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Release artifact checksum mismatch: " + name)
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".new")
        temporary.write_bytes(data)
        os.replace(temporary, path)
    save(folder / "release.json", {**manifest, "revision": revision})


def linux(root: Path, folder: Path, revision: str, manifest: dict) -> dict:
    python = root / "venvs/v100-continual/bin/python"
    watchdog = folder / "rlm/v100/mission_watchdog.py"
    if watchdog.is_file():
        subprocess.run([str(python), str(watchdog), "--root", str(root)], check=True, timeout=120)
    processes = subprocess.check_output(["ps", "-eo", "args="], text=True)
    exclusive = any(
        "rlm.v100.cli" in line
        and re.search(
            r" (train|mission-prepare|optimize-mtp|calibrate-training|test-mtp)( |$)", line
        )
        for line in processes.splitlines()
    )
    installed = subprocess.check_output(
        [str(python), "-c", 'from importlib.metadata import version; print(version("rlms"))'],
        text=True,
    ).strip()
    target = f"0.1.3+v100.{manifest['version']}"
    match = re.fullmatch(r"0\.1\.3\+v100\.(\d+)", installed)
    if not match or int(match[1]) > manifest["version"]:
        raise ValueError("Managed updater refuses an unknown package or a downgrade")
    if installed != target:
        if exclusive:
            return {
                "stage": "deferred-training",
                "installed": installed,
                "target": target,
                "revision": revision,
            }
        subprocess.run(
            ["/bin/bash", str(folder / manifest["linux_installer"]), revision], check=True
        )
    knowledge_status = root / "research/knowledge/import-status.json"
    importing = any(
        "wikimedia_exports.py" in line or "rlm.v100.wiki_ingest" in line
        for line in processes.splitlines()
    )
    wiki_error = None
    try:
        complete = (
            knowledge_status.exists()
            and json.loads(knowledge_status.read_text()).get("stage") == "complete"
        )
    except (OSError, ValueError) as error:
        complete = False
        wiki_error = str(error)
    wiki_receipt = {
        "stage": "already-running" if importing else "complete" if complete else "queued"
    }
    if wiki_error:
        wiki_receipt["previous_status_error"] = wiki_error
    if not importing and not complete:
        wiki = folder / "rlm/v100/wikimedia_exports.py"
        try:
            subprocess.run(
                [
                    "systemd-run",
                    "--user",
                    "--unit=synta-managed-wikipedia",
                    "--collect",
                    "--property=Type=exec",
                    "--property=Nice=10",
                    str(python),
                    "-u",
                    str(wiki),
                    "--root",
                    str(root),
                    "--language",
                    "en",
                ],
                check=True,
            )
        except (OSError, subprocess.SubprocessError) as error:
            wiki_receipt = {"stage": "blocked", "error": str(error)}
    source_receipt = {"stage": "not prepared"}
    try:
        subprocess.run(
            [
                str(python),
                str(folder / "rlm/v100/source_sync.py"),
                "--root",
                str(root),
                "--revision",
                revision,
            ],
            check=True,
        )
        source_receipt = {"stage": "prepared", "revision": revision}
    except (OSError, subprocess.SubprocessError) as error:
        source_receipt = {"stage": "blocked", "error": str(error)}
    # Prepare assets and remote roles without pausing the active master for native maintenance.
    subprocess.run([str(python), "-m", "rlm.v100.managed_prepare", "--root", str(root)], check=True)
    preparation = root / "research/managed-update/preparation.json"
    observed = json.loads(preparation.read_text()) if preparation.exists() else {"stage": "unknown"}
    return {
        "stage": "package-ready-preparation-recorded"
        if observed.get("stage") == "prepared" and wiki_receipt["stage"] != "blocked"
        else "package-ready-preparation-partial",
        "wiki": wiki_receipt,
        "source": source_receipt,
        "preparation_stage": observed.get("stage"),
        "version": target,
        "revision": revision,
        "scope": "Roles require successful host qualification; accepted learning weights retained",
    }


def windows(root: Path, folder: Path, revision: str, manifest: dict) -> dict:
    script = folder / manifest["windows_prepare"]
    subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-Revision",
            revision,
            "-Bundle",
            str(folder),
        ],
        check=True,
    )
    receipt = root / "logs/managed-models.json"
    models = json.loads(receipt.read_text(encoding="utf-8-sig")) if receipt.exists() else {}
    return {
        "stage": "windows-prepared" if models.get("stage") == "prepared" else "windows-partial",
        "preparation_stage": models.get("stage", "unknown"),
        "revision": revision,
        "scope": "Prepared owned models and scoped connection; Debian independently validates role quality",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--revision")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    state = (
        args.root / "research/managed-update/status.json"
        if os.name != "nt"
        else args.root / "logs/managed-update.json"
    )
    try:
        revision, manifest = release(args.revision)
        folder = args.root / "tools/managed-releases" / revision
        if state.exists():
            previous = json.loads(state.read_text())
            if (
                previous.get("revision") == revision
                and previous.get("stage")
                in ("package-ready-preparation-recorded", "windows-prepared")
                and time.time() - previous.get("updated", 0) < 21600
            ):
                print(json.dumps({"stage": "already-checked", "revision": revision}), flush=True)
                return
        materialize(folder, revision, manifest)
        bootstrap = args.root / "tools/managed-bootstrap"
        bootstrap.mkdir(parents=True, exist_ok=True)
        for module in ("managed_bootstrap", "inference_benchmark"):
            source = folder / f"rlm/v100/{module}.py"
            if source.exists():
                destination = bootstrap / (module + ".py")
                temporary = destination.with_suffix(".new")
                temporary.write_bytes(source.read_bytes())
                os.replace(temporary, destination)
        save(state, {"stage": "release-verified", "revision": revision, "updated": time.time()})
        result = (
            windows(args.root, folder, revision, manifest)
            if os.name == "nt"
            else linux(args.root, folder, revision, manifest)
        )
        save(state, {**result, "updated": time.time()})
        print(json.dumps(result), flush=True)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        save(
            state,
            {
                "stage": "failed",
                "error": str(error),
                "updated": time.time(),
                "scope": "Inspect receipt; no update success or model improvement claimed",
            },
        )
        raise


if __name__ == "__main__":
    main()
