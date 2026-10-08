"""Pinned independent A/B parents and finite regression evidence across restarts."""

import json
from pathlib import Path

from rlm.v100.common import atomic_json
from rlm.v100.protection import file_hash


def read(profile: dict) -> tuple[dict, list[dict]]:
    parents, gates = {}, []
    for branch, entry in profile.get("resources", {}).get("branch_lineages", {}).items():
        if branch not in ("A", "B"):
            raise ValueError("Unknown branch lineage")
        for key in ("profile", "quality"):
            if file_hash(Path(entry[key])) != entry[key + "_sha256"]:
                raise ValueError("Pinned branch parent or quality evidence changed")
        parent = json.loads(Path(entry["profile"]).read_text())
        if parent["training"]["base_model"] != profile["training"]["base_model"]:
            raise ValueError("Branch parent targets a different pretrained base")
        parents[branch] = parent
        gates.append(json.loads(Path(entry["quality"]).read_text()))
    return parents, gates


def record(profile: dict, directory: Path, verdict: dict) -> dict:
    entries = dict(profile.get("resources", {}).get("branch_lineages", {}))
    for branch in ("A", "B"):
        if not verdict["branches"][branch]["eligible"]:
            continue
        parent = json.loads((directory / branch / "serving.json").read_text())
        adapter = directory / branch / "training/candidate"
        parent["training"].update(init_adapter=str(adapter), teacher_adapter=str(adapter))
        public = directory / branch / "public-quality.json"
        if parent.get("resources", {}).get("public_benchmarks"):
            parent["resources"].update(
                public_baseline=str(public), public_baseline_sha256=file_hash(public)
            )
        parent.setdefault("resources", {}).pop("branch_lineages", None)
        path = directory / branch / "lineage.json"
        atomic_json(path, parent)
        quality = directory / branch / "development-quality.json"
        entries[branch] = {
            "profile": str(path),
            "profile_sha256": file_hash(path),
            "quality": str(quality),
            "quality_sha256": file_hash(quality),
        }
    return entries
