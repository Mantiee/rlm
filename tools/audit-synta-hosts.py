"""Read-only local audit of the three explicitly owned Synta hosts."""

import json
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
    }.items():
        report["receipts"][name] = receipt(root / "research" / path)
    report["receipts"]["current_activity"] = snapshot(root, report["mission"], {})
    reports = list((root / "research/model-roles").glob("*/qualification.json")) + list(
        (root / "research/owned-node").glob("*.qualification.json")
    )
    report["qualification_details"] = [
        {"path": str(path), "data": receipt(path)}
        for path in sorted(reports, key=lambda p: p.stat().st_mtime, reverse=True)[:5]
    ]
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
        "*wikipedia*",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        report["services"] = result.stdout[-12000:] + result.stderr[-2000:]
    except (OSError, subprocess.SubprocessError) as error:
        report["services"] = str(error)
    output = root / "research/host-audit.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("Saved:", output)


if __name__ == "__main__":
    main()
