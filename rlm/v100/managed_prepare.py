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
                if not isinstance(value.get("models"), list):
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
        return {
            "state": "interrupted-pilot",
            "path": str(folder),
            "scope": "Inspect existing trial; never overwrite its frozen evidence",
        }
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
    return run_candidate(root, identity, pool, suite, dict(DEFAULT_BUDGET))


def run(root: Path) -> dict:
    record = {
        "updated": time.time(),
        "stage": "preparing-assets",
        "assets": [],
        "roles": "not qualified",
        "scope": "Asset preparation only; no accepted weights replaced",
    }
    path = root / "research/managed-update/preparation.json"
    native = root / "src/llama-v0.6.0/build-v100/bin/llama-server"
    for role in ("fast", "planner"):
        try:
            profile, artifact = prepare(root, native, role, 32768)
            atomic_json(root / f"research/model-roles/candidates/{role}.json", profile)
            record["assets"].append({"role": role, "state": "downloaded-hash-verified", **artifact})
        except (OSError, ValueError, RuntimeError) as error:
            record["assets"].append({"role": role, "state": "blocked", "error": str(error)})
        atomic_json(path, record)
    try:
        from huggingface_hub import HfApi

        from rlm.v100.foundation import propose

        identity = HfApi().model_info("Qwen/Qwen3.5-4B").sha
        record["debian_foundation"] = propose(
            root,
            "A",
            "Qwen/Qwen3.5-4B",
            identity,
            "Compare a separate Qwen3.5 foundation against the accepted learner using retained quality gates; preserve predecessor and reject regressions. Faster small-model serving is a hypothesis requiring measurement.",
        )
    except (OSError, ValueError, RuntimeError) as error:
        record["debian_foundation"] = {"state": "blocked", "error": str(error)}
    try:
        record["memory_pilot"] = memory_pilot(root)
    except (OSError, ValueError, RuntimeError) as error:
        record["memory_pilot"] = {"state": "blocked", "error": str(error)}
    try:
        from rlm.v100.managed_native import run as native_qualification

        record["native_optimization"] = native_qualification(root)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        record["native_optimization"] = {"state": "blocked", "error": str(error)}
    record["stage"] = "qualifying-roles"
    atomic_json(path, record)
    endpoint = helper_endpoint(root)
    record["helper_endpoint"] = endpoint
    if endpoint:
        profile = root / "research/researcher-rtx3090.json"
        if profile.exists():
            value = json.loads(profile.read_text())
            value["runtime"]["base_url"] = endpoint
            # This changes only transport to the explicitly owned peer; pinned model/template remains.
            atomic_json(profile, value)
        try:
            from rlm.v100.model_roles import qualify

            record["roles"] = qualify(root)
        except (OSError, ValueError, RuntimeError) as error:
            record["roles"] = {"state": "blocked", "error": str(error)}
    roles = record["roles"]
    record["stage"] = (
        "prepared"
        if isinstance(roles, dict)
        and roles.get("roles")
        and all(item["state"] == "downloaded-hash-verified" for item in record["assets"])
        else "partially-blocked"
    )
    record["updated"] = time.time()
    atomic_json(path, record)
    return record


def main() -> None:
    def interrupted(signum, frame):
        raise InterruptedError("Managed preparation interrupted; resume preserved mission profile")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.root), indent=2))


if __name__ == "__main__":
    main()
