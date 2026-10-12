"""Official current XML exports with identifiable requests and SHA-256 verification."""

import argparse
import hashlib
import json
import os
import re
import shutil
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

if __package__:
    from rlm.v100.knowledge_rag import import_wiki
else:
    from knowledge_rag import import_wiki

USER_AGENT = "SyntaKnowledgeIndexer/1.0 (+https://github.com/Mantiee/rlm; public Wikipedia RAG)"
BASE = "https://dumps.wikimedia.org/other/mediawiki_content_current"


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".new")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def open_export(url: str, timeout: int = 60, headers: dict | None = None):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        if error.code in (403, 429):
            detail = error.read(1024).decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Wikimedia HTTP {error.code} for {url}; identifiable User-Agent sent. No retry/address rotation. Provider detail: {detail}"
            ) from error
        raise


def manifest_rows(text: str, wiki: str, date: str) -> list[tuple[str, str]]:
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})\s+\*?(?:\./)?([^\s]+)", line)
        if not match or not re.fullmatch(
            re.escape(wiki + "-" + date) + r"-[a-z0-9]+\.xml\.bz2", match[2]
        ):
            raise ValueError("Unexpected official export checksum/filename; no file downloaded")
        rows.append((match[2], match[1]))
    if not rows or len(rows) > 100000 or len({name for name, _ in rows}) != len(rows):
        raise ValueError("Empty, duplicate or oversized Wikipedia manifest")
    return rows


def select_export(language: str, today: datetime | None = None) -> tuple[str, str, str, list]:
    if language not in ("en", "pl"):
        raise ValueError("Supported Wikipedia languages: en, pl")
    wiki = language + "wiki"
    now = today or datetime.now(UTC)
    month = now.year * 12 + now.month - 1
    for offset in range(6):
        year, index = divmod(month - offset, 12)
        date = f"{year:04d}-{index + 1:02d}-01"
        origin = f"{BASE}/{wiki}/{date}/xml/bzip2"
        try:
            with open_export(origin + "/SHA256SUMS", timeout=30) as response:
                raw = response.read(16 * 1024**2 + 1)
            if len(raw) > 16 * 1024**2:
                raise ValueError("Official checksum manifest exceeds read budget")
            text = raw.decode("utf-8")
            return date, origin, text, manifest_rows(text, wiki, date)
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
    raise RuntimeError(
        "No completed current Wikipedia export found in the latest six monthly manifests"
    )


def acquire(
    folder: Path, origin: str, name: str, expected: str, cancel: threading.Event | None = None
) -> Path:
    path = folder / name
    if path.exists():
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("Cached Wikipedia part checksum mismatch")
        return path
    partial = path.with_suffix(".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    digest = hashlib.sha256()
    if offset:
        print(
            json.dumps({"stage": "verifying-resume-prefix", "part": name, "bytes": offset}),
            flush=True,
        )
        with partial.open("rb") as stream:
            while block := stream.read(4 * 1024**2):
                digest.update(block)
        if digest.hexdigest() == expected:
            os.replace(partial, path)
            return path
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with open_export(origin + "/" + name, headers=headers) as response:
        if offset:
            value = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)", value)
            if (
                getattr(response, "status", 200) != 206
                or not match
                or int(match[1]) != offset
                or int(match[2]) != int(match[3]) - 1
            ):
                raise ValueError(
                    "Server did not confirm the requested byte range; existing prefix preserved"
                )
            size = int(match[3])
        else:
            size = response.headers.get("Content-Length")
            size = int(size) if size else None
        if size is not None and shutil.disk_usage(folder).free < size * 5 + 8 * 1024**3:
            raise RuntimeError("Insufficient free disk for this Wikipedia part and its index")
        received = offset
        started = reported = time.monotonic()
        with partial.open("ab" if offset else "wb") as output:
            while block := response.read(4 * 1024**2):
                if cancel is not None and cancel.is_set():
                    raise RuntimeError(
                        "Other Wikipedia transfer failed; prefix retained for next run"
                    )
                received += len(block)
                if size is not None and received > size:
                    raise ValueError("Wikipedia part exceeds published HTTP size")
                if shutil.disk_usage(folder).free < 4 * 1024**3:
                    raise RuntimeError("Wikipedia download paused: preserve 4 GiB free disk")
                digest.update(block)
                output.write(block)
                now = time.monotonic()
                if now - reported >= 5:
                    rate = (received - offset) / max(now - started, 0.001)
                    print(
                        json.dumps(
                            {
                                "stage": "download-progress",
                                "part": name,
                                "bytes": received,
                                "total_bytes": size,
                                "mib_per_second": round(rate / 1024**2, 2),
                                "eta_seconds": round((size - received) / rate)
                                if size is not None and rate
                                else None,
                            }
                        ),
                        flush=True,
                    )
                    reported = now
        if (size is not None and received != size) or digest.hexdigest() != expected:
            raise ValueError("Incomplete or incorrect Wikipedia part; not admitted to index")
    os.replace(partial, path)
    return path


def run(root: Path, language: str, connections: int = 2) -> dict:
    if connections not in (1, 2):
        raise ValueError("Wikipedia transfers use at most two connections")
    status_path = root / "research/knowledge/import-status.json"
    date, origin, manifest, rows = select_export(language)
    folder = root / "research/knowledge/exports" / (language + "wiki") / date
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SHA256SUMS").write_text(manifest, encoding="utf-8")
    record = {
        "stage": "export-selected",
        "language": language,
        "date": date,
        "origin": origin,
        "parts_total": len(rows),
        "parts_indexed": 0,
        "pages_imported": 0,
        "scope": "Archived RAG knowledge; no weight updates or verified income",
    }
    save(status_path, record)
    print(json.dumps(record), flush=True)
    cancel = threading.Event()
    with ThreadPoolExecutor(max_workers=connections) as executor:
        pending = {
            index: executor.submit(acquire, folder, origin, name, digest, cancel)
            for index, (name, digest) in enumerate(rows[:connections])
        }
        try:
            for number, (name, digest) in enumerate(rows, 1):
                record.update(stage="downloading-part", part=name, part_number=number)
                save(status_path, record)
                print(json.dumps(record), flush=True)
                path = pending.pop(number - 1).result()
                following = number - 1 + connections
                if following < len(rows):
                    next_name, next_digest = rows[following]
                    pending[following] = executor.submit(
                        acquire, folder, origin, next_name, next_digest, cancel
                    )
                record.update(stage="indexing-part")
                save(status_path, record)
                print(json.dumps(record), flush=True)
                result = import_wiki(root, path, language, 1000000000)
                record.update(
                    parts_indexed=number,
                    pages_imported=record["pages_imported"] + result["pages_imported"],
                )
                save(
                    path.with_suffix(".receipt.json"),
                    {**result, "publisher_sha256": digest, "source": origin + "/" + name},
                )
                save(status_path, record)
        except BaseException:
            cancel.set()
            for future in pending.values():
                future.cancel()
            raise
    record.update(stage="complete")
    save(status_path, record)
    print(json.dumps(record), flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--language", choices=("en", "pl"), default="en")
    parser.add_argument("--connections", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if os.name != "nt":
        os.nice(10)
    folder = args.root / "research/knowledge"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "import.lock").open("a") as lock:
        if os.name != "nt":
            import fcntl

            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit(
                    "Another Wikipedia importer owns the index; no download started"
                ) from None
        try:
            run(args.root, args.language, args.connections)
        except (OSError, ValueError, RuntimeError) as error:
            status_path = args.root / "research/knowledge/import-status.json"
            previous = json.loads(status_path.read_text()) if status_path.exists() else {}
            save(
                status_path,
                {
                    **previous,
                    "stage": "failed",
                    "language": args.language,
                    "error": str(error),
                    "scope": "No complete corpus claimed; existing indexed data retained",
                },
            )
            raise


if __name__ == "__main__":
    main()
