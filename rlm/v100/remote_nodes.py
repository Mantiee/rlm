"""Reconnect the explicitly owned Linux node; preserve local validation of CPU results."""

import argparse
import base64
import copy
import hashlib
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

import requests

from rlm.v100.common import atomic_json, load_profile
from rlm.v100.model_policy import NODE_MODELS, retired
from rlm.v100.model_roles import evaluate, json_hash
from rlm.v100.remote_helper import BACKEND, ROLE_MODELS, STABLE_METADATA, metadata_sha

ENDPOINT = "http://192.168.0.9:11435"
CONTRACT = "synta-dedicated-linux-v1"


def request(method: str, path: str, value: dict | None = None) -> dict:
    if path not in ("/api/synta-resources", "/api/tags", "/api/synta-cpu") and not re.fullmatch(
        r"/api/synta-cpu/[a-f0-9]{24}-[a-f0-9]{32}", path
    ):
        raise ValueError("Unknown owned-node API path")
    with requests.Session() as session:
        session.trust_env = False
        with session.request(
            method, ENDPOINT + path, json=value, timeout=(2, 10), allow_redirects=False, stream=True
        ) as response:
            if 300 <= response.status_code < 400:
                raise ValueError("Owned node redirected its response")
            response.raise_for_status()
            body = bytearray()
            for part in response.iter_content(65536):
                body.extend(part)
                if len(body) > 48 * 2**20:
                    raise ValueError("Owned-node output exceeds 48 MiB")
            result = json.loads(body)
    if not isinstance(result, dict):
        raise ValueError("Owned-node JSON object required")
    return result


def selected_node(root: Path) -> Path | None:
    folder = root / "research/owned-node"
    if not (root / "research/owned-node.json").exists() or not (folder / "status.json").exists():
        return None
    status = json.loads((folder / "status.json").read_text())
    if (
        not status.get("admitted")
        or not status.get("resources", {}).get("ready")
        or time.time() - status.get("updated", 0) > 90
    ):
        return None
    path = Path(status["profile"]).resolve()
    if not path.is_relative_to(folder.resolve()):
        raise ValueError("Owned node profile escaped its recorded folder")
    value = json.loads(path.read_text())
    if json_hash(value) != status["profile_sha256"] or value["runtime"]["base_url"] != ENDPOINT:
        raise ValueError("Owned node profile identity changed")
    if retired(value["runtime"]["model_name"]):
        return None
    return path


def qualify(root: Path, resources: dict) -> dict:
    from rlm.v100.competition import helper_client
    from rlm.v100.mission import status

    mission = status(root)
    if not mission.get("run"):
        raise ValueError("Need the existing owned mission profile")
    original = load_profile(Path(mission["run"]) / "input-profile.json", root)
    rows = request("GET", "/api/tags")["models"]
    reports = []
    winners = []
    folder = root / "research/owned-node"
    folder.mkdir(parents=True, exist_ok=True)
    lock_path = folder / "selection-lock.json"
    locked = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    if locked:
        if locked.get("model") not in NODE_MODELS:
            raise ValueError("Unknown locked owned-node model")
        if not any(
            row.get("name") == locked["model"] and row.get("digest") == locked["digest"]
            for row in rows
        ):
            return {
                "admitted": False,
                "phase": "locked-model-unavailable",
                "model": locked["model"],
                "reason": "Pinned model digest changed or disappeared; CPU jobs remain independent",
            }
    for model in (locked["model"],) if locked else NODE_MODELS:
        matching = [
            r
            for r in rows
            if r.get("name") == model and not r.get("remote_host") and not r.get("remote_model")
        ]
        if len(matching) != 1 or not re.fullmatch(r"[a-f0-9]{64}", matching[0].get("digest", "")):
            continue
        digest = matching[0]["digest"]
        key = hashlib.sha256(
            json.dumps(
                {
                    "digest": digest,
                    "revision": resources.get("revision"),
                    "context": 32768,
                    "panel": "arithmetic-accounting-evidence-v1",
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        receipt = folder / (key + ".qualification.json")
        profile_path = folder / (key + ".profile.json")
        cached = json.loads(receipt.read_text()) if receipt.exists() else {}
        if cached.get("error") and time.time() - cached.get("updated", 0) < 300:
            reports.append(cached)
            continue
        if receipt.exists() and profile_path.exists():
            report = cached
        else:
            profile = copy.deepcopy(original)
            profile["runtime"].update(
                backend=BACKEND,
                base_url=ENDPOINT,
                model_name=model,
                model_version="owned-3060-" + digest[:12],
                context_window=32768,
                tool_protocol="json",
                max_output_tokens=256,
                enable_thinking=False,
                max_timeout=120,
            )
            profile["server"].update(
                binary="",
                model="",
                draft_model="",
                adapter="",
                gpu_layers=0,
                slots=1,
                context_per_slot=32768,
            )
            profile["resources"] = {
                "device": "remote",
                "model_digest": digest,
                "metadata_sha256": "",
                "metadata_hash_scheme": STABLE_METADATA,
                "max_vram_gib": ROLE_MODELS[model][1],
                "helper_batch_tokens": 64,
                "helper_duty_percent": 100,
                "dedicated_host": True,
                "on_demand": True,
                "allow_cpu_offload": False,
                "require_adaptive_windows_guard": True,
                "worker_name": "debian-3060ti",
            }
            try:
                client = helper_client(profile, root)
                client.timeout = 5
                profile["resources"]["metadata_sha256"] = metadata_sha(
                    client.identity(), STABLE_METADATA
                )
                client = helper_client(profile, root)
                client.timeout = 15
                client.request_deadline = time.monotonic() + 60
                measurement = evaluate(client)
            except (OSError, ValueError, RuntimeError, requests.RequestException) as error:
                report = {
                    "admitted": False,
                    "model": model,
                    "error": str(error)[:400],
                    "updated": time.time(),
                }
                atomic_json(receipt, report)
                reports.append(report)
                continue
            report = {
                "admitted": measurement["passed"] == measurement["total"]
                and measurement["critical_passed"],
                "measurement": measurement,
                "model": model,
                "digest": digest,
                "scope": "Finite advisory helper qualification. No master weight promotion, broad intelligence certification or verified income.",
            }
            profile["runtime"]["max_output_tokens"] = 2048
            maximum = max(1, (32768 - 2048) // profile["memory"]["chunk_tokens"])
            profile["memory"]["fanout"] = min(profile["memory"]["fanout"], maximum)
            profile["memory"]["retrieve_count"] = min(profile["memory"]["retrieve_count"], maximum)
            atomic_json(profile_path, profile)
            report["profile_sha256"] = json_hash(profile)
            atomic_json(receipt, report)
        reports.append(report)
        if report.get("admitted"):
            profile = json.loads(profile_path.read_text())
            if report.get("profile_sha256") != json_hash(profile):
                raise ValueError("Owned qualification profile changed")
            winners.append((report["measurement"]["seconds"], profile_path, profile))
    atomic_json(folder / "qualifications.json", {"reports": reports})
    if not winners:
        return {
            "admitted": False,
            "phase": "reachable-unqualified",
            "reason": "No downloaded candidate passed every panel case",
        }
    _, path, profile = min(winners, key=lambda row: row[0])
    if not locked:
        atomic_json(
            lock_path,
            {
                "model": profile["runtime"]["model_name"],
                "digest": profile["resources"]["model_digest"],
                "metadata_sha256": profile["resources"]["metadata_sha256"],
                "scope": "Fixed foundation identity; code updates do not reselect this role",
            },
        )
    return {
        "admitted": True,
        "phase": "qualified",
        "profile": str(path),
        "profile_sha256": json_hash(profile),
        "model": profile["runtime"]["model_name"],
    }


def cpu_bridge(root: Path, resources: dict) -> None:
    from rlm.v100.distributed_compute import mailbox_path

    if not (root / "research/compute-config.json").exists():
        return
    mailbox = mailbox_path(root)
    worker = "debian-3060ti-cpu"
    phase = (
        "idle"
        if resources.get("cpu_ready")
        and resources.get("cpu_active", 0) < resources.get("cpu_slots", 1)
        else "busy"
    )
    atomic_json(
        mailbox / "workers" / (worker + ".json"),
        {
            "updated": time.time(),
            "phase": phase,
            "device": "cpu",
            "threads": resources.get("cpu_threads"),
            "slots": resources.get("cpu_slots"),
            "name": worker,
            "gpu_enabled": False,
            "reason": "Dedicated fixed-kernel CPU pool; per-job threads=2, multiple independent jobs in parallel",
        },
    )
    for path in sorted((mailbox / "jobs").glob("*.json"))[:16]:
        if not re.fullmatch(r"[a-f0-9]{24}", path.stem):
            continue
        original = root / "research/compute-jobs" / path.stem / "job.json"
        if not original.is_file():
            continue
        if (
            path.is_symlink()
            or path.stat().st_size > 2 * 2**20
            or path.read_bytes() != original.read_bytes()
        ):
            raise ValueError("Owned CPU mailbox input differs from the frozen host job")
        claim = mailbox / "claims" / path.stem
        if not claim.resolve().is_relative_to(mailbox.resolve()):
            raise ValueError("Owned CPU claim escaped its mailbox")
        lease_path = claim / "lease.json"
        closed = (mailbox / "closed" / (path.stem + ".json")).exists()
        if closed and not lease_path.exists():
            continue
        if not claim.exists():
            if closed or phase != "idle":
                continue
            job = json.loads(path.read_text())
            if job.get("kernel_sha256") != resources.get("kernel_sha256") or job.get(
                "worker_sha256"
            ) != resources.get("worker_sha256"):
                continue
            claim.parent.mkdir(parents=True, exist_ok=True)
            try:
                claim.mkdir()
            except FileExistsError:
                continue
            atomic_json(
                lease_path,
                {
                    "job_id": path.stem,
                    "worker": worker,
                    "nonce": uuid.uuid4().hex,
                    "started": time.time(),
                    "heartbeat": time.time(),
                },
            )
        if not lease_path.exists():
            continue
        lease = json.loads(lease_path.read_text())
        if lease.get("worker") != worker:
            continue
        nonce = lease.get("nonce", "")
        if not re.fullmatch(r"[a-f0-9]{32}", nonce):
            raise ValueError("Owned CPU lease nonce changed")
        key = path.stem + "-" + nonce
        if closed:
            request("DELETE", "/api/synta-cpu/" + key)
            continue
        destination = mailbox / "results" / key
        if not destination.resolve().is_relative_to(mailbox.resolve()):
            raise ValueError("Owned CPU result escaped its mailbox")
        if (destination / "receipt.json").exists():
            continue
        try:
            result = request("GET", "/api/synta-cpu/" + key)
        except requests.HTTPError as error:
            if error.response.status_code != 400:
                raise
            raw = path.read_bytes()
            try:
                result = request(
                    "POST",
                    "/api/synta-cpu",
                    {
                        "identity": key,
                        "job": base64.b64encode(raw).decode(),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    },
                )
            except requests.HTTPError as submission:
                if submission.response.status_code == 503:
                    lease_path.unlink()
                    claim.rmdir()
                    phase = "busy"
                    continue
                raise
        if result.get("state") not in ("queued", "running", "complete", "failed"):
            raise ValueError("Unknown owned CPU lifecycle")
        atomic_json(lease_path, {**lease, "heartbeat": time.time()})
        if result["state"] == "complete":
            if result.get("job_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError("Owned CPU result input changed")
            files = result.get("artifacts", {})
            if set(files) != {"report.json", "weights.safetensors"}:
                raise ValueError("Owned CPU artifact set changed")
            destination.mkdir(parents=True, exist_ok=True)
            for name, encoded in files.items():
                data = base64.b64decode(encoded, validate=True)
                if len(data) > 16 * 2**20 or hashlib.sha256(data).hexdigest() != result[
                    "files"
                ].get(name):
                    raise ValueError("Owned CPU artifact hash/size differs")
                target = destination / name
                temporary = target.with_name(name + "." + uuid.uuid4().hex)
                with temporary.open("xb") as stream:
                    stream.write(data)
                os.replace(temporary, target)
            atomic_json(
                destination / "receipt.json",
                {
                    **lease,
                    "state": "complete",
                    "job_sha256": result["job_sha256"],
                    "files": result["files"],
                    "worker_implementation_sha256": result.get("worker_implementation_sha256"),
                    "scope": "Remote metrics untrusted; normal host safe-tensor replay is required",
                },
            )
        elif result["state"] == "failed":
            atomic_json(
                destination / "receipt.json",
                {**lease, "state": "failed", "detail": result.get("detail", "Owned kernel failed")},
            )


def tick(root: Path) -> dict:
    status_path = root / "research/owned-node/status.json"
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    try:
        resources = request("GET", "/api/synta-resources")
        if (
            resources.get("contract") != CONTRACT
            or abs(time.time() - resources.get("updated", 0)) > 15
        ):
            raise ValueError("Dedicated node telemetry contract/time differs")
        # CPU results and observations do not wait behind optional GPU qualification.
        from rlm.v100.distributed_compute import tick as compute_tick
        from rlm.v100.goal_compute import tick as goal_tick

        cpu_errors = {}
        for name, operation in (
            ("transport", lambda: cpu_bridge(root, resources)),
            ("validation", lambda: compute_tick(root)),
            ("goal_dispatch", lambda: goal_tick(root)),
        ):
            try:
                operation()
            except (
                OSError,
                ValueError,
                RuntimeError,
                KeyError,
                requests.RequestException,
            ) as error:
                cpu_errors[name] = str(error)[:400]
        qualification = root / "research/owned-node/qualification-state.json"
        qualified = (
            json.loads(qualification.read_text())
            if qualification.exists()
            else {"admitted": False, "phase": "qualification pending"}
        )
        if qualified.get("resources_revision") != resources.get("revision") or not resources.get(
            "ready"
        ):
            qualified = {
                "admitted": False,
                "phase": "GPU unavailable or new revision; CPU independent",
                "reason": resources.get("reason"),
            }
        result = {
            **previous,
            **qualified,
            "resources": resources,
            "endpoint": ENDPOINT,
            "updated": time.time(),
            "cpu_errors": cpu_errors,
            "reason": qualified.get("reason", ""),
        }
    except (OSError, ValueError, KeyError, RuntimeError, requests.RequestException) as error:
        result = {
            **previous,
            "admitted": False,
            "phase": "unavailable",
            "reason": str(error)[:400],
            "endpoint": ENDPOINT,
            "updated": time.time(),
        }
    atomic_json(status_path, result)
    return result


def qualification_loop(root: Path) -> None:
    while True:
        try:
            resources = request("GET", "/api/synta-resources")
            if (
                resources.get("contract") != CONTRACT
                or not resources.get("ready")
                or abs(time.time() - resources.get("updated", 0)) > 15
            ):
                raise ValueError("Owned GPU is not ready for qualification")
            result = {
                **qualify(root, resources),
                "resources_revision": resources.get("revision"),
                "qualification_updated": time.time(),
            }
        except (OSError, ValueError, RuntimeError, KeyError, requests.RequestException) as error:
            result = {
                "admitted": False,
                "phase": "qualification deferred",
                "reason": str(error)[:400],
            }
        atomic_json(root / "research/owned-node/qualification-state.json", result)
        time.sleep(60)


def main():
    from importlib.metadata import version

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    if (
        not re.fullmatch(r"0\.1\.3\+v100\.(\d+)", version("rlms"))
        or int(version("rlms").rsplit(".", 1)[1]) < 87
    ):
        raise SystemExit("Waiting for deferred Synta v87 package update; no train interrupted")
    threading.Thread(target=qualification_loop, args=(args.root.resolve(),), daemon=True).start()
    while True:
        print(json.dumps(tick(args.root.resolve())), flush=True)
        time.sleep(20)


if __name__ == "__main__":
    main()
