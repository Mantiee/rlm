"""Prepare hash-verified native candidates without replacing accepted model weights."""

import argparse
import hashlib
import json
import os
import re
import shutil
import time
import urllib.request
from pathlib import Path

if __package__:
    from rlm.v100.inference_benchmark import run, save
else:
    from inference_benchmark import run, save

CATALOG = {
    "fast": ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf"),
    "planner": ("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
}


def json_url(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as reply:
        return json.load(reply)


def artifact(role: str) -> dict:
    repo, filename = CATALOG[role]
    metadata = json_url(f"https://huggingface.co/api/models/{repo}?blobs=true")
    revision = metadata["sha"]
    if not re.fullmatch("[0-9a-f]{40}", revision):
        raise ValueError("Model repository did not return an immutable revision")
    rows = [row for row in metadata["siblings"] if row["rfilename"] == filename]
    if len(rows) != 1:
        raise ValueError("Expected one named single-file GGUF artifact")
    lfs = rows[0]["lfs"]
    digest, size = lfs["sha256"], lfs["size"]
    if not re.fullmatch("[0-9a-f]{64}", digest) or not 0 < size < 32 * 1024**3:
        raise ValueError("Invalid model hash or download size")
    return dict(
        role=role, repo=repo, revision=revision, filename=filename, sha256=digest, bytes=size
    )


def download(root: Path, record: dict) -> Path:
    folder = root / "models/candidates" / record["role"] / record["revision"]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / record["filename"]
    if path.exists():
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != record["sha256"]:
                raise ValueError("Existing candidate checksum mismatch")
        return path
    if shutil.disk_usage(folder).free < record["bytes"] + 8 * 1024**3:
        raise RuntimeError("Candidate download requires model size plus 8 GiB free disk")
    save(folder / "artifact.json", record)
    temporary = path.with_suffix(".partial")
    digest = hashlib.sha256()
    received = 0
    url = (
        f"https://huggingface.co/{record['repo']}/resolve/{record['revision']}/{record['filename']}"
    )
    if temporary.exists():
        received = temporary.stat().st_size
        if received > record["bytes"]:
            raise ValueError("Partial candidate exceeds the pinned artifact size")
        with temporary.open("rb") as stream:
            while block := stream.read(4 * 1024**2):
                digest.update(block)
    if received < record["bytes"]:
        headers = {"User-Agent": "SyntaModelAcquirer/1.0 (+https://github.com/Mantiee/rlm)"}
        if received:
            headers["Range"] = f"bytes={received}-"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=60) as reply:
            if received:
                content_range = reply.headers.get("Content-Range", "")
                expected_range = rf"bytes {received}-[0-9]+/{record['bytes']}"
                if reply.status != 206 or not re.fullmatch(expected_range, content_range):
                    raise ValueError(
                        "Model server did not honor exact pinned resume range; partial retained"
                    )
            last_update = time.monotonic()
            with temporary.open("ab" if received else "wb") as output:
                while block := reply.read(4 * 1024**2):
                    received += len(block)
                    if received > record["bytes"]:
                        raise ValueError("Candidate exceeds declared artifact size")
                    digest.update(block)
                    output.write(block)
                    if time.monotonic() - last_update >= 10:
                        print(
                            json.dumps(
                                {
                                    "stage": "model-download-progress",
                                    "role": record["role"],
                                    "bytes": received,
                                    "total_bytes": record["bytes"],
                                }
                            ),
                            flush=True,
                        )
                        last_update = time.monotonic()
    if received != record["bytes"] or digest.hexdigest() != record["sha256"]:
        raise ValueError("Downloaded candidate checksum/size mismatch; not activated")
    os.replace(temporary, path)
    return path


def prepare(root: Path, binary: Path, role: str, context: int = 32768) -> tuple[dict, dict]:
    if role not in CATALOG or context not in (32768, 131072, 262144, 524288):
        raise ValueError("Unsupported candidate role or context")
    record = artifact(role)
    model = download(root, record)
    server = dict(
        tune_kernels=True,
        binary=str(binary.resolve()),
        model=str(model.resolve()),
        model_sha256=record["sha256"],
        context_per_slot=context,
        slots=1,
        threads=4,
        batch_size=128,
        ubatch_size=64,
        flash_attention="on",
        cache_type="q8_0",
        jinja=True,
        gpu_layers=999,
        draft_model="",
        spec_type="none",
    )
    if context > 262144:
        server.update(rope_scaling="yarn", rope_scale=2, yarn_orig_ctx=262144)
    return {"server": server}, record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--role", choices=CATALOG, required=True)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8097)
    args = parser.parse_args()
    if not args.binary.is_file() or args.output.exists():
        raise ValueError("Require an existing native runtime and new output directory")
    profile, record = prepare(args.root, args.binary, args.role, args.context)
    save(args.output.with_suffix(".candidate.json"), {"profile": profile, "artifact": record})
    result = run(profile, args.output, args.port)
    result["candidate"] = record
    result["production_activated"] = False
    result["scope"] = (
        "Candidate throughput only; independent role-quality and long-context tests required before production promotion"
    )
    save(args.output / "results.json", result)
    print(json.dumps(result["selection"]))


if __name__ == "__main__":
    main()
