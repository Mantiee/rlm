"""One bounded owner-authorized read-only chat round trip through the actual gateway."""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

from rlm.v100.common import atomic_json
from rlm.v100.remote_access import validate_config


def run(root: Path, revision: str, timeout: int = 180, port: int = 8786) -> dict:
    if not re.fullmatch(r"[a-f0-9]{40}", revision) or not 1 <= timeout <= 180:
        raise ValueError("Probe requires an immutable revision and a 1-180 second budget")
    config = validate_config(json.loads((root / "research/remote-access/config.json").read_text()))
    receipt = root / "research/chat-probe.json"
    previous = json.loads(receipt.read_text()) if receipt.is_file() else {}
    if previous.get("revision") == revision and previous.get("stage") == "verified":
        return previous
    result = {
        "stage": "starting",
        "revision": revision,
        "updated": time.time(),
        "scope": "Local authenticated gateway and generated reply; not proof of browser/phone routing, tool execution, learning or income",
    }
    if previous.get("revision") == revision and previous.get("previous_attempt"):
        result["previous_attempt"] = previous["previous_attempt"]
    if previous.get("revision") == revision and previous.get("transport_failures"):
        result["transport_failures"] = previous["transport_failures"][-2:]
    opener = build_opener(ProxyHandler({}))
    headers = {"Tailscale-User-Login": config["owner_login"]}

    def request(path: str, data: dict | None = None) -> dict:
        req = Request(
            f"http://127.0.0.1:{port}" + path,
            headers=headers,
            data=json.dumps(data).encode() if data else None,
        )
        with opener.open(req, timeout=5) as response:
            body = response.read(8 * 2**20 + 1)
        if len(body) > 8 * 2**20:
            raise ValueError("Probe response exceeds 8 MiB")
        return json.loads(body)

    atomic_json(receipt, result)
    try:
        from rlm.v100.model_roles import selected_profile

        planner = selected_profile(root, "planner")
        admission = hashlib.sha256(planner.read_bytes()).hexdigest() if planner else None
        result["planner_admission"] = admission
        session = request("/api/chat")
        headers.update(
            {
                "Origin": config["origin"],
                "Content-Type": "application/json",
                "X-Synta-CSRF": session["csrf"],
            }
        )
        identity = previous.get("request_id") if previous.get("revision") == revision else None
        attempts = previous.get("transport_attempts", 1) if identity else 1
        if identity is not None and admission and previous.get("planner_admission") != admission:
            old = request("/api/chat/" + identity)
            if old.get("state") == "failed":
                # Only this immutable, action-free canary can be retried after
                # a new verified admission. Never replay an operator command.
                result["previous_attempt"] = {
                    "request_id": identity,
                    "error": old.get("error"),
                    "planner_admission": previous.get("planner_admission"),
                }
                identity = None
                attempts = 1
                result.pop("transport_failures", None)

        def submit_probe() -> str:
            return request(
                "/api/chat",
                {
                    "message": f"What is 19 times 23? Reply with only 437. Do not use tools or change anything. Read-only verification ID: {revision}"
                },
            )["id"]

        if identity is None:
            identity = submit_probe()
        if not re.fullmatch(r"[a-f0-9]{32}", identity):
            raise ValueError("Gateway returned an invalid chat request ID")
        result.update(stage="waiting", request_id=identity, transport_attempts=attempts)
        atomic_json(receipt, result)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            row = request("/api/chat/" + identity)
            response = row.get("response") or {}
            if row.get("state") == "completed":
                answer = response.get("answer", "").strip()
                responder = response.get("responder", {})
                if (
                    answer != "437"
                    or response.get("actions")
                    or response.get("applied")
                    or response.get("tool_receipts")
                    or not responder.get("model")
                    or responder["model"].startswith("controller-")
                ):
                    raise ValueError(
                        "Chat probe did not return the exact generated read-only answer"
                    )
                result.update(stage="verified", answer=answer, responder=responder)
                break
            if row.get("state") in ("failed", "cancelled"):
                if (
                    row.get("state") == "failed"
                    and str(row.get("error", "")).startswith(
                        "503 Server Error: Service Unavailable"
                    )
                    and attempts < 3
                    and time.monotonic() + 3 < deadline
                ):
                    result.setdefault("transport_failures", []).append(
                        {"request_id": identity, "error": row["error"]}
                    )
                    time.sleep(2)
                    identity = submit_probe()
                    if not re.fullmatch(r"[a-f0-9]{32}", identity):
                        raise ValueError("Gateway returned an invalid chat request ID")
                    attempts += 1
                    result.update(request_id=identity, transport_attempts=attempts)
                    atomic_json(receipt, result)
                    continue
                raise RuntimeError("Chat probe ended: " + str(row.get("error") or row.get("state")))
            time.sleep(1)
        else:
            result.update(
                stage="waiting-timeout",
                error="Saved read-only request has not completed within the probe budget",
            )
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        result.update(stage="blocked", error=str(error)[:500])
    result["updated"] = time.time()
    atomic_json(receipt, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.revision)))


if __name__ == "__main__":
    main()
