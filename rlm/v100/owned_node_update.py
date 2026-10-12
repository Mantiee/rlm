"""Root-owned updater for the explicitly installed, dedicated Linux worker only."""

import argparse
import hashlib
import json
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path


def read(url: str, maximum: int = 2 * 2**20) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "SyntaOwnedNodeUpdater/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError("Release metadata exceeds its byte budget")
    return data


def run(base: Path, state: Path) -> dict:
    if __package__:
        from rlm.v100.owned_node import atomic
    else:
        from owned_node import atomic

    target = state / "managed-update.json"
    try:
        ref = json.loads(
            read("https://api.github.com/repos/Mantiee/rlm/git/ref/heads/v100-rtx-reconnect")
        )
        revision = ref["object"]["sha"]
        if not re.fullmatch("[a-f0-9]{40}", revision):
            raise ValueError("GitHub channel did not return an immutable revision")
        prefix = f"https://raw.githubusercontent.com/Mantiee/rlm/{revision}/"
        manifest = json.loads(read(prefix + "tools/managed-release.json"))
        path = manifest.get("node3060_installer")
        if path != "tools/bootstrap-synta-3060.sh" or path not in manifest["files"]:
            raise ValueError("Release does not declare the owned 3060 installer")
        current = (base / "current").resolve().name
        receipt = state / "installation.json"
        installed = json.loads(receipt.read_text()) if receipt.exists() else {}
        if (
            current == revision
            and installed.get("stage") == "prepared"
            and installed.get("revision") == revision
        ):
            result = {"stage": "current", "revision": revision}
        else:
            body = read(prefix + path)
            if hashlib.sha256(body).hexdigest() != manifest["files"][path]:
                raise ValueError("Owned installer checksum differs")
            script = base / "updates" / (revision + ".sh")
            script.parent.mkdir(parents=True, exist_ok=True)
            script.write_bytes(body)
            process = subprocess.run(["/bin/bash", str(script), revision, "--update"], check=False)
            if process.returncode not in (0, 75):
                raise RuntimeError(
                    "Owned node update failed; previous revision retained where possible"
                )
            result = {
                "stage": "deferred-busy" if process.returncode == 75 else "installed",
                "revision": revision,
            }
    except (OSError, ValueError, KeyError, RuntimeError, urllib.error.URLError) as error:
        result = {"stage": "blocked", "error": str(error)[:400]}
    import time

    atomic(target, {**result, "updated": time.time()})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("/opt/synta3060"))
    parser.add_argument("--state", type=Path, default=Path("/var/lib/synta3060"))
    args = parser.parse_args()
    print(json.dumps(run(args.base, args.state)), flush=True)


if __name__ == "__main__":
    main()
