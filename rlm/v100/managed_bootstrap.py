"""Operator-authorized GitHub updates and owned-host preparation, with receipts."""

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

if __package__:
    from rlm.v100.inference_benchmark import save
else:
    from inference_benchmark import save

REPOSITORY = "Mantiee/rlm"
BRANCH = "v100-rtx-reconnect"


def active_learning_cycle(root: Path) -> bool:
    from rlm.v100.mission import status

    current = status(root)
    learning = current.get("learning", {})
    return (
        current.get("running") is True
        and learning.get("stage")
        in (
            "candidate-training-and-independent-tests",
            "scratch-architecture-training-and-tests",
            "alternative-architecture-training-and-tests",
        )
        and (learning.get("cycle_in_progress") or 0) > (learning.get("completed_cycles") or 0)
    )


def dashboard_update(root: Path, folder: Path, revision: str) -> dict:
    """Upgrade the passive UI in a separate code directory, never the trainer."""
    python = root / "venvs/v100-continual/bin/python"
    uv = root.parent / ".local/bin/uv"
    if not uv.is_file():
        uv = Path(shutil.which("uv") or "/unavailable")
    if not python.is_file() or not uv.is_file():
        return {"stage": "blocked", "reason": "Existing Python/uv unavailable"}
    path = root / "research/dashboard-deployment.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("revision") == revision and previous.get("stage") == "ready":
            return previous
    target = root / "tools/dashboard-runtime" / revision
    override = Path.home() / ".config/systemd/user/v100-dashboard.service.d/20-code.conf"
    old_override = override.read_bytes() if override.exists() else None
    entrypoint = root / "bin/serve-v100-dashboard.py"
    old_entrypoint = entrypoint.read_bytes() if entrypoint.exists() else None
    changed = False
    try:
        subprocess.run(
            [
                str(uv),
                "--no-config",
                "pip",
                "install",
                "--python",
                str(python),
                "--target",
                str(target),
                "--no-deps",
                f"git+https://github.com/{REPOSITORY}.git@{revision}",
            ],
            check=True,
            timeout=180,
        )
        entrypoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = entrypoint.with_suffix(".new")
        temporary.write_bytes((folder / "tools/serve-v100-dashboard.py").read_bytes())
        os.replace(temporary, entrypoint)
        changed = True
        override.parent.mkdir(parents=True, exist_ok=True)
        override.write_text(f'[Service]\nEnvironment="PYTHONPATH={target}"\n')
        changed = True
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True, timeout=15)
        subprocess.run(
            ["systemctl", "--user", "restart", "v100-dashboard.service"], check=True, timeout=30
        )
        subprocess.run(
            [
                str(python),
                "-c",
                "import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from rlm.v100.dashboard_health import wait;wait(root=Path(sys.argv[2]))",
                str(target),
                str(root),
            ],
            check=True,
            timeout=90,
        )
        result = {
            "stage": "ready",
            "revision": revision,
            "code_path": str(target),
            "scope": "Passive renderer upgraded separately; accepted trainer package and active cycle unchanged",
        }
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        rollback_error = None
        if changed:
            try:
                if old_override is None:
                    override.unlink(missing_ok=True)
                else:
                    override.write_bytes(old_override)
                if old_entrypoint is not None:
                    entrypoint.write_bytes(old_entrypoint)
                subprocess.run(["systemctl", "--user", "daemon-reload"], check=False, timeout=15)
                subprocess.run(
                    ["systemctl", "--user", "restart", "v100-dashboard.service"],
                    check=False,
                    timeout=30,
                )
            except (OSError, subprocess.SubprocessError) as rollback:
                rollback_error = str(rollback)[:400]
        result = {"stage": "blocked", "error": str(error)[:400], "rollback_error": rollback_error}
    save(path, {**result, "updated": time.time()})
    return result


def remote_chat_update(root: Path, dashboard: dict, revision: str) -> dict:
    """Refresh the authenticated read-only gateway without restarting the learner."""
    if dashboard.get("stage") != "ready" or dashboard.get("revision") != revision:
        return {"stage": "blocked", "reason": "Verified separate runtime unavailable"}
    target = Path(dashboard["code_path"])
    override = Path.home() / ".config/systemd/user/synta-remote.service.d/20-code.conf"
    path = root / "research/remote-chat-deployment.json"
    if path.is_file():
        previous = json.loads(path.read_text())
        if previous.get("stage") == "configured" and previous.get("revision") == revision:
            return previous
    old = override.read_bytes() if override.is_file() else None
    try:
        subprocess.run(
            [
                str(root / "venvs/v100-continual/bin/python"),
                "-c",
                "import sys;sys.path.insert(0,sys.argv[1]);from rlm.v100.remote_access import GatewayServer;from rlm.v100.chat_progress import requested;assert requested('Możesz już skutecznie testami zarabiać? trenujesz już przez 4 dni…')",
                str(target),
            ],
            check=True,
            timeout=30,
        )
        override.parent.mkdir(parents=True, exist_ok=True)
        override.write_text(f'[Service]\nEnvironment="PYTHONPATH={target}"\n')
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True, timeout=15)
        subprocess.run(
            ["systemctl", "--user", "try-restart", "synta-remote.service"], check=True, timeout=30
        )
        subprocess.run(
            [
                str(root / "venvs/v100-continual/bin/python"),
                "-c",
                "import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from rlm.v100.mission_chat import resolve_failed_status_questions;print(resolve_failed_status_questions(Path(sys.argv[2])))",
                str(target),
                str(root),
            ],
            check=True,
            timeout=60,
        )
        result = {
            "stage": "configured",
            "revision": revision,
            "scope": "Authenticated gateway code refreshed; actual browser response requires a receipt. Accepted trainer and queued action worker unchanged",
        }
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        if old is None:
            override.unlink(missing_ok=True)
        else:
            override.write_bytes(old)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=False, timeout=15)
        result = {"stage": "blocked", "error": str(error)[:400]}
    save(path, {**result, "updated": time.time()})
    return result


def fetch(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "SyntaManagedUpdater/1.0 (+https://github.com/Mantiee/rlm)"}
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read(4 * 1024**2 + 1)
            if len(data) > 4 * 1024**2:
                raise ValueError("Release metadata/script exceeds its byte budget")
            return data
        except urllib.error.HTTPError as error:
            if error.code != 429 and not 500 <= error.code < 600:
                raise
            if attempt == 2:
                raise
        except (OSError, urllib.error.URLError):
            if attempt == 2:
                raise
        time.sleep(attempt + 1)
    raise RuntimeError("Metadata fetch retry budget exhausted")


def queue_chat_probe(root: Path, dashboard: dict, revision: str) -> dict:
    if (
        dashboard.get("stage") != "ready"
        or not (root / "research/remote-access/config.json").is_file()
    ):
        return {"stage": "not-configured"}
    receipt = root / "research/chat-probe.json"
    if receipt.is_file():
        previous = json.loads(receipt.read_text())
        if previous.get("revision") == revision and previous.get("stage") == "verified":
            return previous
    try:
        active = subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", "synta-chat-probe.service"],
            timeout=15,
            check=False,
        )
        if active.returncode == 0:
            return {"stage": "already-running"}
        target = Path(dashboard["code_path"]) / "rlm/v100/chat_probe.py"
        unit = Path.home() / ".config/systemd/user/synta-chat-probe.service"
        unit.parent.mkdir(parents=True, exist_ok=True)
        unit.write_text(
            f"[Unit]\nDescription=Synta fixed read-only chat verification\n[Service]\nType=oneshot\nExecStart={root}/venvs/v100-continual/bin/python -u {target} --root {root} --revision {revision}\nNice=10\nTimeoutStartSec=4min\n"
        )
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True, timeout=15)
        subprocess.run(
            ["systemctl", "--user", "start", "--no-block", "synta-chat-probe.service"],
            check=True,
            timeout=15,
        )
        return {"stage": "queued", "revision": revision}
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {"stage": "blocked", "error": str(error)[:400]}


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


def refresh_private_sync(root: Path, folder: Path) -> None:
    """Refresh an already opted-in sync from the checksum-verified release bundle."""
    destination = root / "tools/private-sync"
    if not (destination / "private_sync.py").is_file():
        return
    for name in (
        "rlm/v100/private_sync.py",
        "rlm/v100/release_selector.py",
        "tools/audit-synta-hosts.py",
    ):
        source = folder / name
        temporary = destination / (source.name + ".new")
        temporary.write_bytes(source.read_bytes())
        os.replace(temporary, destination / source.name)


def refresh_bootstrap(root: Path, folder: Path) -> None:
    """Install verified dependencies before replacing the standalone entrypoint."""
    modules = ("inference_benchmark", "release_selector", "managed_bootstrap")
    sources = [folder / f"rlm/v100/{module}.py" for module in modules]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(
                "Bootstrap dependency missing from verified bundle: " + source.name
            )
        compile(source.read_bytes(), str(source), "exec")
    bootstrap = root / "tools/managed-bootstrap"
    bootstrap.mkdir(parents=True, exist_ok=True)
    for source in sources:
        destination = bootstrap / source.name
        temporary = destination.with_suffix(".new")
        temporary.write_bytes(source.read_bytes())
        os.replace(temporary, destination)


def bootstrap_selection(root: Path, peer: bool = False) -> str | None:
    name = "rlm.v100.release_selector" if __package__ else "release_selector"
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as error:
        if __package__ or error.name != name:
            raise
        revision, manifest = release()
        artifact = "rlm/v100/release_selector.py"
        expected = manifest["files"][artifact]
        body = fetch(f"https://raw.githubusercontent.com/{REPOSITORY}/{revision}/{artifact}")
        if hashlib.sha256(body).hexdigest() != expected:
            raise ValueError("Missing bootstrap dependency checksum differs") from error
        destination = Path(__file__).with_name("release_selector.py")
        compile(body, str(destination), "exec")
        temporary = destination.with_suffix(".new")
        temporary.write_bytes(body)
        os.replace(temporary, destination)
        spec = importlib.util.spec_from_file_location(
            "synta_recovered_release_selector", destination
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Recovered bootstrap dependency cannot be loaded") from error
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(
            json.dumps(
                {
                    "stage": "bootstrap-dependency-restored",
                    "revision": revision,
                    "artifact": artifact,
                }
            ),
            flush=True,
        )
    return module.selection(root, peer=peer)


def remote_preparation(root: Path, folder: Path, revision: str) -> dict:
    """Use the pinned source as a separate process while the learner package waits."""
    python = root / "venvs/v100-continual/bin/python"
    source = folder / "rlm/v100/source_sync.py"
    if not source.is_file():
        return {"stage": "not available in bundle"}
    try:
        active = subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", "synta-role-qualification.service"],
            check=False,
            timeout=15,
        )
        if active.returncode == 0:
            return {"stage": "already-running"}
        subprocess.run(
            [str(python), str(source), "--root", str(root), "--revision", revision],
            check=True,
            timeout=180,
        )
        snapshot = root / "research/managed-source" / revision
        ledger = json.loads((snapshot / "MANAGED_SOURCE.json").read_text())
        manifest = json.loads((folder / "release.json").read_text())
        for name in (
            "rlm/v100/managed_prepare.py",
            "rlm/v100/model_roles.py",
            "rlm/v100/remote_helper.py",
        ):
            if ledger["files"].get(name) != manifest["files"].get(name):
                raise ValueError("Remote preparation source differs from verified bundle: " + name)
        unit = Path.home() / ".config/systemd/user/synta-role-qualification.service"
        unit.parent.mkdir(parents=True, exist_ok=True)
        unit.write_text(
            f"[Unit]\nDescription=Synta independent remote model qualification\n[Service]\nType=oneshot\nWorkingDirectory={snapshot}\nExecStart={python} -u -m rlm.v100.managed_prepare --root {root} --remote-only\nNice=10\nTimeoutStartSec=15min\n"
        )
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True, timeout=15)
        subprocess.run(
            ["systemctl", "--user", "start", "--no-block", "synta-role-qualification.service"],
            check=True,
            timeout=15,
        )
        return {
            "stage": "queued",
            "revision": revision,
            "scope": "Remote roles only; active accepted learner retained",
        }
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        return {"stage": "blocked", "error": str(error)[:400]}


def linux(root: Path, folder: Path, revision: str, manifest: dict) -> dict:
    python = root / "venvs/v100-continual/bin/python"
    dashboard = dashboard_update(root, folder, revision) if manifest["version"] >= 91 else {}
    remote_chat = remote_chat_update(root, dashboard, revision) if manifest["version"] >= 98 else {}
    if manifest["version"] >= 101:
        remote_chat = {**remote_chat, "probe": queue_chat_probe(root, dashboard, revision)}
    watchdog = folder / "rlm/v100/mission_watchdog.py"
    recovery = {"stage": "not-configured"}
    if watchdog.is_file():
        try:
            subprocess.run(
                [str(python), str(watchdog), "--root", str(root)], check=True, timeout=120
            )
            recovery = {"stage": "completed"}
        except (OSError, subprocess.SubprocessError) as error:
            recovery = {"stage": "blocked", "error": str(error)[:400]}
        recovery.update(revision=revision, updated=time.time())
        save(root / "research/managed-mission-recovery.json", recovery)
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
    remote_roles = remote_preparation(root, folder, revision) if manifest["version"] >= 96 else {}
    if installed != target:
        if manifest["version"] >= 91 and active_learning_cycle(root):
            return {
                "stage": "deferred-active-learning-cycle",
                "installed": installed,
                "target": target,
                "revision": revision,
                "dashboard": dashboard,
                "remote_chat": remote_chat,
                "remote_qualification": remote_roles,
                "mission_recovery": recovery,
                "scope": "Training and independent evaluation complete before a package switch",
            }
        if exclusive:
            return {
                "stage": "deferred-training",
                "installed": installed,
                "target": target,
                "revision": revision,
                "remote_qualification": remote_roles,
                "mission_recovery": recovery,
            }
        if recovery["stage"] == "blocked":
            return {
                "stage": "deferred-mission-recovery",
                "installed": installed,
                "target": target,
                "revision": revision,
                "dashboard": dashboard,
                "remote_chat": remote_chat,
                "remote_qualification": remote_roles,
                "mission_recovery": recovery,
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
    if manifest["version"] >= 91:
        unit = Path.home() / ".config/systemd/user/synta-model-preparation.service"
        unit.parent.mkdir(parents=True, exist_ok=True)
        unit.write_text(
            f"[Unit]\nDescription=Synta optional model preparation\n[Service]\nType=oneshot\nExecStart={python} -u -m rlm.v100.managed_prepare --root {root}\nNice=10\nTimeoutStartSec=30min\n"
        )
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True, timeout=15)
        subprocess.run(
            ["systemctl", "--user", "start", "--no-block", "synta-model-preparation.service"],
            check=True,
            timeout=15,
        )
    else:
        subprocess.run(
            [str(python), "-m", "rlm.v100.managed_prepare", "--root", str(root)], check=True
        )
    preparation = root / "research/managed-update/preparation.json"
    observed = json.loads(preparation.read_text()) if preparation.exists() else {"stage": "unknown"}
    return {
        "stage": "package-ready-preparation-recorded"
        if observed.get("stage") == "prepared" and wiki_receipt["stage"] != "blocked"
        else "package-ready-preparation-partial",
        "wiki": wiki_receipt,
        "source": source_receipt,
        "preparation_stage": observed.get("stage"),
        "dashboard": dashboard,
        "remote_qualification": remote_roles,
        "mission_recovery": recovery,
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
            "-WindowStyle",
            "Hidden",
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
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
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
        selected = args.revision or bootstrap_selection(args.root, peer=os.name == "nt")
        revision, manifest = release(selected)
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
        if os.name != "nt":
            refresh_private_sync(args.root, folder)
        refresh_bootstrap(args.root, folder)
        save(state, {"stage": "release-verified", "revision": revision, "updated": time.time()})
        result = (
            windows(args.root, folder, revision, manifest)
            if os.name == "nt"
            else linux(args.root, folder, revision, manifest)
        )
        save(state, {**result, "updated": time.time()})
        print(json.dumps(result), flush=True)
    except (
        OSError,
        ValueError,
        RuntimeError,
        ImportError,
        KeyError,
        subprocess.SubprocessError,
    ) as error:
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
