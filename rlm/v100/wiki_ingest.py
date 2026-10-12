"""Acquire a dated official Wikipedia export and index all current articles."""

import argparse
import hashlib
import json
import os
import re
import shutil
import urllib.request
from pathlib import Path

from rlm.v100.inference_benchmark import save
from rlm.v100.knowledge_rag import import_wiki


def fetch(root: Path, language: str) -> Path:
    if language not in ("pl", "en"):
        raise ValueError("Supported official export languages: pl, en")
    wiki = language + "wiki"
    with urllib.request.urlopen(
        f"https://dumps.wikimedia.org/{wiki}/latest/dumpstatus.json", timeout=30
    ) as response:
        status = json.load(response)
    job = status["jobs"]["articlesmultistreamdump"]
    if job["status"] != "done":
        raise RuntimeError("Latest official Wikipedia export is not complete")
    matches = [
        (name, item)
        for name, item in job["files"].items()
        if re.fullmatch(wiki + r"-[0-9]{8}-pages-articles-multistream\.xml\.bz2", name)
    ]
    if len(matches) != 1:
        raise ValueError(
            "No unambiguous complete single-file Wikipedia export; inspect publisher export format"
        )
    name, metadata = matches[0]
    size, checksum = metadata["size"], metadata["sha1"]
    if type(size) is not int or size <= 0 or not re.fullmatch("[0-9a-f]{40}", checksum):
        raise ValueError("Invalid official export size/checksum")
    folder = root / "research/knowledge/exports"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    if not path.exists():
        if shutil.disk_usage(folder).free < size * 5 + 8 * 1024**3:
            raise RuntimeError(
                "Full Wikipedia acquisition needs five times compressed size plus 8 GiB free disk for export and index"
            )
        date = name.split("-")[1]
        url = f"https://dumps.wikimedia.org/{wiki}/{date}/{name}"
        temporary = path.with_suffix(".partial")
        received = 0
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            while block := response.read(4 * 1024**2):
                received += len(block)
                if received > size:
                    raise ValueError("Export exceeds official size")
                output.write(block)
                if received % (128 * 1024**2) < len(block):
                    print(
                        json.dumps(
                            {"stage": "downloading-wikipedia", "bytes": received, "total": size}
                        ),
                        flush=True,
                    )
        if received != size:
            raise ValueError("Incomplete Wikipedia export")
        with temporary.open("rb") as stream:
            if hashlib.file_digest(stream, "sha1").hexdigest() != checksum:
                raise ValueError("Official Wikipedia checksum mismatch")
        os.replace(temporary, path)
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha1").hexdigest() != checksum:
            raise ValueError("Cached export differs from official checksum")
    save(
        path.with_suffix(".receipt.json"),
        {
            "publisher": "Wikimedia",
            "filename": name,
            "bytes": size,
            "publisher_sha1": checksum,
            "language": language,
            "scope": "Public wikitext; preserve source/revision attribution and applicable licensing",
        },
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--language", choices=("pl", "en"), default="pl")
    args = parser.parse_args()
    if os.name != "nt":
        os.nice(10)
    path = fetch(args.root, args.language)
    receipt = import_wiki(args.root, path, args.language, 1000000000)
    save(args.root / "research/knowledge/import.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
