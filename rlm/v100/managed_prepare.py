"""Prepare native assets and discover the explicitly owned Windows peer."""

import argparse
import json
import signal
import subprocess
import time
from pathlib import Path

import requests
from rlm.v100.common import atomic_json
from rlm.v100.model_candidates import prepare


def helper_endpoint(root: Path) -> str | None:
    choices = ["http://192.168.0.61:11435"]
    try:
        peers = json.loads(
            subprocess.check_output(["tailscale", "status", "--json"], text=True, timeout=10)
        )
        if not isinstance(peers, dict) or not isinstance(peers.get("Peer", {}), dict):
            raise ValueError("Tailnet peer inventory malformed")
        hosts = [
            item
            for item in peers.get("Peer", {}).values()
            if "100.78.125.51" in item.get("TailscaleIPs", [])
        ]
        if len(hosts) == 1 and "100.78.125.51" in hosts[0].get("TailscaleIPs", []):
            choices.append("http://100.78.125.51:11437")
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    with requests.Session() as session:
        session.trust_env = False
        for origin in choices:
            try:
                response = session.get(origin + "/api/tags", timeout=(3, 5), allow_redirects=False)
                response.raise_for_status()
                value = response.json()
                if not isinstance(value, dict) or not isinstance(value.get("models"), list):
                    raise ValueError("Helper identity inventory malformed")
                return origin
            except (ValueError, requests.RequestException):
                continue
    return None


def memory_pilot(root: Path) -> dict:
    from rlm.v100.architectures import (
        DEFAULT_BUDGET,
        candidate_path,
        prepared_inputs,
        run_candidate,
    )
    from rlm.v100.goals import load_goal
    from rlm.v100.ngram_memory import propose

    goal = load_goal(root)
    if not goal:
        return {"state": "blocked", "reason": "No frozen goal"}
    pool, suite = prepared_inputs(root)
    identity = "managed-ngram-v83-" + goal["id"][:12]
    folder = candidate_path(root, identity)
    evidence = folder / "trial/quality.json"
    if evidence.exists():
        return json.loads(evidence.read_text())
    if folder.exists():
        # Preserve the failed/interrupted archive and retry in one new versioned trial.
        identity = "managed-ngram-v89-" + goal["id"][:12]
        folder = candidate_path(root, identity)
        evidence = folder / "trial/quality.json"
        if evidence.exists():
            return json.loads(evidence.read_text())
        if (folder / "trial").exists():
            return {
                "state": "interrupted-pilot",
                "path": str(folder),
                "scope": "New attempt retained; inspect its failure before another retry",
            }
    if not folder.exists():
        propose(
            root,
            "A",
            identity,
            32,
            1,
            1024,
            3,
            "Compare causal n-gram memory enabled and disabled on the same held-out goal workload. Reject absent improvement; this is a small independent CPU model, not pretrained Gemma/Qwen extension.",
        )
    records = [json.loads(line) for line in pool.read_text().splitlines() if line.strip()]
    required = max(
        (
            len(
                "".join(
                    str(m["role"]) + ": " + str(m["content"]) + "\n" for m in row["messages"]
                ).encode()
            )
            + 32
            for row in records
        ),
        default=256,
    )
    if required > 8192:
        return {
            "state": "blocked",
            "reason": "Full verified examples exceed the 8192-byte CPU pilot context; no truncation performed",
            "required_bytes": required,
        }
    budget = dict(DEFAULT_BUDGET) | {"context_window": max(256, required), "threads": 4}
    return run_candidate(root, identity, pool, suite, budget)


def run(root: Path) -> dict:
    record = {
        "updated": time.time(),
        "stage": "preparing-assets",
        "assets": [],
        "roles": "not qualified",
        "scope": "Asset preparation only; no accepted weights replaced",
    }
    path = root / "research/managed-update/preparation.json"
    # Restore helper communication and qualify already downloaded roles before bulk native assets.
    record["stage"] = "qualifying-roles"
    atomic_json(path, record)
    endpoint = helper_endpoint(root)
    record["helper_endpoint"] = endpoint
    if endpoint:
        profile = root / "research/researcher-rtx3090.json"
        if profile.exists():
            value = json.loads(profile.read_text())
            value["runtime"]["base_url"] = endpoint
            value["resources"]["require_adaptive_windows_guard"] = True
            # This changes only transport to the explicitly owned peer; pinned model/template remains.
            atomic_json(profile, value)
        try:
            from rlm.v100.model_roles import qualify

            record["roles"] = qualify(root)
        except (
            OSError,
            ValueError,
            RuntimeError,
            subprocess.SubprocessError,
            requests.RequestException,
        ) as error:
            record["roles"] = {"state": "blocked", "error": str(error)}
    record["stage"] = "preparing-assets"
    atomic_json(path, record)
    native = root / "src/llama-v0.6.0/build-v100/bin/llama-server"
    for role in ("fast", "planner"):
        try:
            profile, artifact = prepare(root, native, role, 32768)
            atomic_json(root / f"research/model-roles/candidates/{role}.json", profile)
            record["assets"].append({"role": role, "state": "downloaded-hash-verified", **artifact})
        except (
            OSError,
            ValueError,
            RuntimeError,
            subprocess.SubprocessError,
            requests.RequestException,
        ) as error:
            record["assets"].append({"role": role, "state": "blocked", "error": str(error)})
        atomic_json(path, record)
    record["debian_foundation"] = {
        "state": "accepted-learner-retained",
        "reason": "No old Qwen foundation retried; new serving candidates are qualified separately from learned weights",
    }
    try:
        record["memory_pilot"] = memory_pilot(root)
    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.SubprocessError,
        requests.RequestException,
    ) as error:
        record["memory_pilot"] = {"state": "blocked", "error": str(error)}
    try:
        from rlm.v100.managed_native import run as native_qualification

        record["native_optimization"] = native_qualification(root)
    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.SubprocessError,
        requests.RequestException,
    ) as error:
        record["native_optimization"] = {"state": "blocked", "error": str(error)}
    roles = record["roles"]
    record["stage"] = (
        "prepared"
        if isinstance(roles, dict)
        and roles.get("roles")
        and all(item["state"] == "downloaded-hash-verified" for item in record["assets"])
        and record.get("native_optimization", {}).get("state")
        in ("measured", "deferred-live-priority")
        and record.get("memory_pilot", {}).get("state") not in ("blocked", "interrupted-pilot")
        and record.get("debian_foundation", {}).get("state") != "blocked"
        else "partially-blocked"
    )
    record["updated"] = time.time()
    atomic_json(path, record)
    return record


def remote_only(root: Path) -> dict:
    """Qualify advisory Windows roles without running any V100 training/native pilot."""
    from rlm.v100.model_roles import qualify

    target = root / "research/model-roles/remote-preparation.json"
    record = {
        "stage": "qualifying-remote-roles",
        "updated": time.time(),
        "scope": "Independent advisory roles; accepted learner, LoRA and V100 endpoint unchanged",
    }
    atomic_json(target, record)
    endpoint = helper_endpoint(root)
    if endpoint is None:
        record.update(stage="blocked", error="Owned Windows helper unavailable")
    else:
        profile = root / "research/researcher-rtx3090.json"
        if profile.exists():
            value = json.loads(profile.read_text())
            value["runtime"]["base_url"] = endpoint
            value["resources"]["require_adaptive_windows_guard"] = True
            atomic_json(profile, value)
        try:
            record["roles"] = qualify(root)
            record["stage"] = "qualified" if record["roles"].get("roles") else "no-role-admitted"
        except (
            OSError,
            ValueError,
            RuntimeError,
            subprocess.SubprocessError,
            requests.RequestException,
        ) as error:
            record.update(stage="blocked", error=str(error)[:400])
    record["updated"] = time.time()
    atomic_json(target, record)
    return record


def main() -> None:
    def interrupted(signum, frame):
        raise InterruptedError("Managed preparation interrupted; resume preserved mission profile")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--remote-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(remote_only(args.root) if args.remote_only else run(args.root), indent=2))


if __name__ == "__main__":
    main()
