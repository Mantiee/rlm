"""Snapshot and restore only the owned rlms package during a guarded cutover."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path


def tree_hashes(folder: Path) -> dict[str, str]:
    return {
        str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def snapshot(root: Path, destination: Path) -> dict:
    root, destination = root.resolve(), destination.resolve()
    if not destination.is_relative_to(root / "research") or destination.exists():
        raise ValueError("Use a new backup directory inside owned research")
    distribution = importlib.metadata.distribution("rlms")
    package = Path(distribution.locate_file("rlm")).resolve()
    site = package.parent
    if not package.is_relative_to(root / "venvs") or not package.is_dir():
        raise ValueError("Refuse to snapshot a foreign package installation")
    metadata = [
        site / path.parts[0]
        for path in distribution.files or []
        if path.name == "METADATA" and path.parts[0].startswith("rlms-")
    ]
    if len(metadata) != 1 or not metadata[0].is_dir():
        raise ValueError("Cannot identify exactly one rlms distribution")
    destination.mkdir(parents=True, mode=0o700)
    for path in [package, metadata[0]]:
        shutil.copytree(path, destination / path.name, symlinks=False)
    receipt = {
        "version": distribution.version,
        "site": str(site),
        "package": package.name,
        "metadata": metadata[0].name,
        "hashes": tree_hashes(destination),
    }
    (destination / "snapshot.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def restore(root: Path, destination: Path) -> dict:
    root, destination = root.resolve(), destination.resolve()
    if not destination.is_relative_to(root / "research"):
        raise ValueError("Backup is outside owned research")
    receipt = json.loads((destination / "snapshot.json").read_text())
    site = Path(receipt["site"]).resolve()
    if not site.is_relative_to(root / "venvs"):
        raise ValueError("Refuse to restore into a foreign environment")
    if receipt["package"] != "rlm" or not receipt["metadata"].startswith("rlms-"):
        raise ValueError("Invalid package recovery inventory")
    if Path(receipt["metadata"]).name != receipt["metadata"]:
        raise ValueError("Invalid metadata recovery path")
    for name, expected in receipt["hashes"].items():
        path = (destination / name).resolve()
        if (
            not path.is_relative_to(destination)
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("Package recovery snapshot failed integrity verification")
    displaced = destination / "displaced-runtime"
    displaced.mkdir(mode=0o700)
    # Keep failed files for diagnosis; never recursively delete a package tree.
    for path in [site / "rlm", *site.glob("rlms-*.dist-info")]:
        if path.exists():
            path.rename(displaced / path.name)
    for name in [receipt["package"], receipt["metadata"]]:
        shutil.copytree(destination / name, site / name)
    return {"restored_version": receipt["version"], "displaced": str(displaced)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["snapshot", "restore"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    result = (
        snapshot(args.root, args.destination)
        if args.action == "snapshot"
        else restore(args.root, args.destination)
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
