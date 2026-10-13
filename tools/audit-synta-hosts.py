"""Read-only local audit of the three explicitly owned Synta hosts."""

import json
import re
import sqlite3
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path


def receipt(path):
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 2 * 2**20:
        return {"state": "unavailable", "path": str(path)}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        return {"error": str(error)}


def probe(item):
    name, url = item
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=5) as response:
            body = response.read(8 * 2**20 + 1)
        if len(body) > 8 * 2**20:
            raise ValueError("Response exceeds audit byte budget")
        value = json.loads(body)
        if name == "dashboard":
            value = {
                "collected_at": value.get("collected_at"),
                "mission": value.get("mission"),
                "errors": value.get("errors"),
                "current_activity": value.get("report", {}).get("current_activity"),
            }
        return name, {"reachable": True, "observed_at": time.time(), "data": value}
    except (OSError, ValueError) as error:
        return name, {"reachable": False, "error": str(error)}


def latest_role_report(root):
    reports = list((root / "research/model-roles").glob("*/qualification.json"))
    if not reports:
        return {}
    return receipt(max(reports, key=lambda path: path.stat().st_mtime))


def chat_probe_queue(root, probe_receipt):
    identity, revision = probe_receipt.get("request_id", ""), probe_receipt.get("revision", "")
    if not re.fullmatch("[a-f0-9]{32}", identity) or not re.fullmatch("[a-f0-9]{40}", revision):
        return {"state": "no-saved-probe"}
    path = root / "research/state/user-chat.sqlite3"
    if not path.is_file() or path.is_symlink():
        return {"state": "unavailable"}
    try:
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.5) as db:
            db.row_factory = sqlite3.Row
            row = db.execute(
                "SELECT id,message,state,attempts,error,updated,CASE WHEN length(response)<=65536 THEN response END AS response FROM requests WHERE id=?",
                (identity,),
            ).fetchone()
        if row is None:
            return {"state": "missing"}
        expected = f"What is 19 times 23? Reply with only 437. Do not use tools or change anything. Read-only verification ID: {revision}"
        if row["message"] != expected:
            return {"state": "rejected", "error": "Saved ID is not the fixed authorized probe"}
        response = json.loads(row["response"]) if row["response"] else {}
        return {
            "id": identity,
            "revision": revision,
            "state": row["state"],
            "attempts": row["attempts"],
            "error": row["error"],
            "updated": row["updated"],
            "response": {
                key: response.get(key)
                for key in ("answer", "actions", "applied", "tool_receipts", "responder")
            },
            "scope": "Read-only observation of the one fixed probe; no other operator messages exported",
        }
    except (OSError, ValueError, sqlite3.Error) as error:
        return {"state": "unavailable", "error": str(error)[:300]}


def main():
    from rlm.v100.mission import status
    from rlm.v100.mission_view import snapshot

    root = Path.home() / "ai-v100"
    report = {
        "collected_at": time.time(),
        "installed_package": version("rlms"),
        "mission": status(root),
        "receipts": {},
    }
    for name, path in {
        "updater": "managed-update/status.json",
        "preparation": "managed-update/preparation.json",
        "roles": "model-roles/active.json",
        "node": "owned-node/status.json",
        "wiki_import": "knowledge/import-status.json",
        "wiki_index": "knowledge/index-progress.json",
        "wiki_goal_use": "knowledge/goal-use.json",
        "private_sync": "private-sync/status.json",
        "remote_preparation": "model-roles/remote-preparation.json",
        "remote_chat_deployment": "remote-chat-deployment.json",
        "chat_probe": "chat-probe.json",
        "mission_recovery": "mission-watchdog/status.json",
        "supervisor": "supervisor/status.json",
        "supervisor_pause": "supervisor/pause.json",
    }.items():
        report["receipts"][name] = receipt(root / "research" / path)
    report["receipts"]["chat_probe_queue"] = chat_probe_queue(
        root, report["receipts"]["chat_probe"]
    )
    report["receipts"]["current_activity"] = snapshot(root, report["mission"], {})
    reports = list((root / "research/model-roles").glob("*/qualification.json")) + list(
        (root / "research/owned-node").glob("*.qualification.json")
    )
    report["qualification_details"] = [
        {"path": str(path), "data": receipt(path)}
        for path in sorted(reports, key=lambda p: p.stat().st_mtime, reverse=True)[:5]
    ]
    report["role_metadata"] = []
    for role, entry in report["receipts"]["roles"].get("roles", {}).items():
        profile_path = Path(entry["profile"]).resolve()
        if not profile_path.is_relative_to((root / "research/model-roles").resolve()):
            continue
        profile = receipt(profile_path)
        item = {"role": role, "profile": profile}
        snapshot_path = profile.get("resources", {}).get("metadata_snapshot")
        if snapshot_path:
            snapshot_path = (root / snapshot_path).resolve()
            if snapshot_path.is_relative_to((root / "research").resolve()):
                item["pinned_snapshot"] = receipt(snapshot_path)
        report["role_metadata"].append(item)
    for entry in [latest_role_report(root)]:
        for candidate in entry.get("reports", []):
            snapshot_path = candidate.get("metadata_snapshot")
            if snapshot_path and candidate.get("role") == "planner":
                snapshot_path = Path(snapshot_path).resolve()
                if snapshot_path.is_relative_to((root / "research/model-roles").resolve()):
                    report["role_metadata"].append(
                        {"role": "planner-candidate", "current_snapshot": receipt(snapshot_path)}
                    )
    urls = [("dashboard", "http://192.168.0.68:8765/api/status")]
    for host, origin in (
        ("windows", "http://192.168.0.61:11435"),
        ("node3060", "http://192.168.0.9:11435"),
    ):
        urls += [
            (host + "_" + route.split("/")[-1], origin + route)
            for route in ("/api/synta-resources", "/api/tags", "/api/ps")
        ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        report["hosts"] = dict(pool.map(probe, urls))
    command = [
        "systemctl",
        "--user",
        "list-units",
        "--all",
        "--no-pager",
        "v100-mission.service",
        "synta-managed-update.service",
        "synta-model-preparation.service",
        "synta-role-qualification.service",
        "synta-chat-probe.service",
        "*wikipedia*",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        report["services"] = result.stdout[-12000:] + result.stderr[-2000:]
    except (OSError, subprocess.SubprocessError) as error:
        report["services"] = str(error)
    report["receipts"]["updater_service"] = {}
    for field, command in (
        (
            "state",
            [
                "systemctl",
                "--user",
                "show",
                "synta-managed-update.service",
                "--property=ActiveState,SubState,Result,ExecMainStatus,ExecMainStartTimestamp,ExecMainExitTimestamp",
            ],
        ),
        (
            "recent_log",
            [
                "journalctl",
                "--user",
                "-u",
                "synta-managed-update.service",
                "-n",
                "12",
                "--no-pager",
                "-o",
                "cat",
            ],
        ),
    ):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            report["receipts"]["updater_service"][field] = (result.stdout + result.stderr)[-4000:]
        except (OSError, subprocess.SubprocessError) as error:
            report["receipts"]["updater_service"][field] = str(error)[:400]
    output = root / "research/host-audit.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("Saved:", output)


if __name__ == "__main__":
    main()
