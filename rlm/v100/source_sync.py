"""Fetch the pinned public source snapshot for isolated inspection and proposals."""

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def sync(root: Path, revision: str) -> dict:
    if not re.fullmatch("[0-9a-f]{40}", revision):
        raise ValueError("Exact source revision required")
    destination = root / "research/managed-source" / revision
    manifest = destination / "MANAGED_SOURCE.json"
    if manifest.exists():
        value = json.loads(manifest.read_text())
        if value["revision"] != revision:
            raise ValueError("Source snapshot revision changed")
        for relative, expected in value["files"].items():
            target = (destination / relative).resolve()
            if (
                not target.is_relative_to(destination.resolve())
                or target.is_symlink()
                or not target.is_file()
            ):
                raise ValueError("Cached source snapshot path changed")
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError(
                    "Cached source snapshot bytes changed; isolated edits need a separate copy"
                )
        return value
    if destination.exists():
        raise ValueError("Incomplete source directory retained; inspect before retry")
    url = f"https://codeload.github.com/Mantiee/rlm/zip/{revision}"
    request = urllib.request.Request(
        url, headers={"User-Agent": "SyntaSourceSync/1.0 (+https://github.com/Mantiee/rlm)"}
    )
    with urllib.request.urlopen(request, timeout=60) as reply:
        data = reply.read(64 * 2**20 + 1)
    if len(data) > 64 * 2**20:
        raise ValueError("Source archive exceeds 64 MiB download budget")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        folder = Path(temporary) / "source"
        folder.mkdir()
        files = {}
        expanded = 0
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                parts = Path(member.filename).parts
                if (
                    len(parts) < 2
                    or any(part in ("..", ".") for part in parts)
                    or Path(member.filename).is_absolute()
                ):
                    raise ValueError("Source archive path escapes its snapshot")
                relative = Path(*parts[1:])
                if relative.suffix not in (
                    ".py",
                    ".md",
                    ".txt",
                    ".toml",
                    ".sh",
                    ".ps1",
                    ".json",
                    ".js",
                    ".html",
                    ".css",
                ):
                    continue
                expanded += member.file_size
                if expanded > 256 * 2**20 or member.file_size > 8 * 2**20:
                    raise ValueError("Source archive exceeds expanded text budget")
                content = archive.read(member)
                target = folder / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                files[relative.as_posix()] = hashlib.sha256(content).hexdigest()
        if "rlm/v100/mission_chat.py" not in files:
            raise ValueError("Source snapshot lacks the master controller")
        value = {
            "source": str(destination),
            "revision": revision,
            "files": files,
            "archive_sha256": hashlib.sha256(data).hexdigest(),
            "scope": "Pinned public source for inspection and isolated edits. No model-authored patch installed on either host.",
        }
        (folder / "MANAGED_SOURCE.json").write_text(json.dumps(value, indent=2))
        shutil.move(str(folder), str(destination))
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    value = sync(args.root, args.revision)
    if os.name != "nt":
        target = args.root / "research/self-code-source.json"
        temporary = target.with_suffix(".new")
        temporary.write_text(json.dumps(value, indent=2))
        os.replace(temporary, target)
    print(
        json.dumps(
            {
                "source": value["source"],
                "revision": value["revision"],
                "file_count": len(value["files"]),
            }
        )
    )


if __name__ == "__main__":
    main()
