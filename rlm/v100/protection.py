"""Immutable inference snapshots and fixed, source-connected training splits."""

import copy
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
from pathlib import Path

from rlm.v100.common import atomic_json


def file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 2**20), b""):
            value.update(block)
    return value.hexdigest()


def assert_candidate_output(output: Path, base: Path, adapter: Path | None, root: Path) -> None:
    destination = output.resolve()
    protected = [base.resolve(), (root / "research/experts").resolve()]
    if adapter is not None:
        protected.append(adapter.resolve())
    for path in protected:
        if (
            destination == path
            or destination.is_relative_to(path)
            or path.is_relative_to(destination)
        ):
            raise ValueError(f"Candidate output overlaps protected data: {path}")


def fixed_split(records: list[dict], ledger: Path) -> tuple[list[dict], list[dict]]:
    """Persist source roles; reject new records connecting train and validation sources."""
    groups: list[set[str]] = []
    for record in records:
        sources = set(record.get("document_ids") or [record["group"]])
        if not all(isinstance(source, str) and source for source in sources):
            raise ValueError("Source identifiers must be nonempty strings")
        for group in [group for group in groups if group & sources]:
            sources |= group
            groups.remove(group)
        groups.append(sources)
    if len(groups) < 2:
        raise ValueError("Need at least two independent source groups for held-out validation")
    groups.sort(key=lambda group: hashlib.sha256(json.dumps(sorted(group)).encode()).hexdigest())
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(ledger) as db:
        db.execute("CREATE TABLE IF NOT EXISTS roles(source TEXT PRIMARY KEY, role TEXT NOT NULL)")
        db.execute("BEGIN IMMEDIATE")
        existing = dict(db.execute("SELECT source, role FROM roles"))
        initial_eval = max(1, len(groups) // 5)
        assignments = dict(existing)
        for index, group in enumerate(groups):
            old_roles = {existing[source] for source in group if source in existing}
            if len(old_roles) > 1:
                raise ValueError(
                    "New record connects train and held-out sources; split cannot change"
                )
            if old_roles:
                role = old_roles.pop()
            elif not existing:
                role = "validation" if index < initial_eval else "train"
            else:
                key = hashlib.sha256(json.dumps(sorted(group)).encode()).hexdigest()
                role = "validation" if int(key, 16) % 5 == 0 else "train"
            if role == "audit":
                raise ValueError("Audit sources cannot be used for training or development")
            for source in group:
                assignments[source] = role
                db.execute("INSERT OR IGNORE INTO roles VALUES(?,?)", (source, role))
        train, validation = [], []
        for record in records:
            sources = record.get("document_ids") or [record["group"]]
            role = assignments[sources[0]]
            (train if role == "train" else validation).append(record)
        if not train or not validation:
            raise ValueError("Dataset must include both fixed train and validation sources")
        db.commit()
    return train, validation


def reserve_audit_sources(ledger: Path, sources: list[str]) -> None:
    if not sources or not all(isinstance(source, str) and source for source in sources):
        raise ValueError("Audit requires explicit source identifiers")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(ledger) as db:
        db.execute("CREATE TABLE IF NOT EXISTS roles(source TEXT PRIMARY KEY, role TEXT NOT NULL)")
        db.execute("BEGIN IMMEDIATE")
        for source in sources:
            old = db.execute("SELECT role FROM roles WHERE source=?", (source,)).fetchone()
            if old and old[0] != "audit":
                raise ValueError("Previously used training/development source cannot become audit")
            db.execute("INSERT OR IGNORE INTO roles VALUES(?, 'audit')", (source,))
        db.commit()


def compare_reports(baseline: dict, candidate: dict) -> dict:
    if baseline.get("schema") != "v100-quality-v1" or candidate.get("schema") != "v100-quality-v1":
        raise ValueError("Requires quality reports, not speed benchmarks")
    if baseline["suite_sha256"] != candidate["suite_sha256"]:
        raise ValueError("Quality suites differ")
    if baseline["generation"] != candidate["generation"]:
        raise ValueError("Generation conditions differ")
    if baseline["memory_mode"] != candidate["memory_mode"]:
        raise ValueError("Memory conditions differ")
    old = {row["id"]: row for row in baseline["cases"]}
    new = {row["id"]: row for row in candidate["cases"]}
    if (
        not old
        or old.keys() != new.keys()
        or len(old) != len(baseline["cases"])
        or len(new) != len(candidate["cases"])
    ):
        raise ValueError("Missing or duplicate quality cases")
    for rows in (old, new):
        if not all(type(row["passed"]) is bool for row in rows.values()):
            raise ValueError("Quality verdicts must be boolean")
    regressions = [name for name in old if old[name]["passed"] and not new[name]["passed"]]
    improvements = [name for name in old if not old[name]["passed"] and new[name]["passed"]]
    return {
        "passed": not regressions,
        "regressions": regressions,
        "improvements": improvements,
        "scope": "Only the supplied finite suite; no universal no-forgetting guarantee",
    }


class ExpertRegistry:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def manifest_path(self, expert_id: str) -> Path:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}", expert_id):
            raise ValueError("Invalid expert identifier")
        return self.directory / expert_id / "expert.json"

    def register(
        self,
        expert_id: str,
        profile: dict,
        description: str,
        baseline_report: dict | list[dict] | None = None,
        candidate_report: dict | None = None,
    ) -> dict:
        path = self.manifest_path(expert_id)
        if path.parent.exists():
            raise FileExistsError(f"Expert is immutable and already exists: {expert_id}")
        if not description.strip():
            raise ValueError("Expert requires a task description")
        if baseline_report is None and candidate_report is None and self.list():
            raise ValueError("Only the initial baseline can be registered without quality reports")
        gate = None
        if baseline_report is not None or candidate_report is not None:
            if baseline_report is None or candidate_report is None:
                raise ValueError("Both quality reports are required")
            baselines = baseline_report if isinstance(baseline_report, list) else [baseline_report]
            if not baselines:
                raise ValueError("At least one parent quality report is required")
            gates = [compare_reports(parent, candidate_report) for parent in baselines]
            gate = {"passed": all(item["passed"] for item in gates), "parents": gates}
            if not gate["passed"]:
                raise ValueError(f"Protected task regressions: {gates}")
            provenance = Path(profile["server"]["model"]).with_suffix(".provenance.json")
            if provenance.exists():
                data = json.loads(provenance.read_text())
                if data["model_sha256"] != candidate_report["model_sha256"]:
                    raise ValueError("Export provenance evaluated a different model")
                if len(data["adapter"].get("parents", [])) > 1:
                    expected = set(data["parent_model_sha256"])
                    evaluated = {item["model_sha256"] for item in baselines}
                    if len(expected) < 2 or not expected <= evaluated:
                        raise ValueError(
                            "Bred candidates require quality reports from both parents"
                        )
            if candidate_report["model_sha256"] != file_hash(Path(profile["server"]["model"])):
                raise ValueError("Candidate report evaluated a different model")
            if candidate_report["execution_sha256"] != execution_hash(profile):
                raise ValueError("Candidate report used different execution settings")
        staging = Path(tempfile.mkdtemp(prefix=".register-", dir=self.directory))
        preserved = copy.deepcopy(profile)
        files = {}
        try:
            for key in ("model", "binary", "draft_model"):
                raw = profile["server"][key]
                if not raw:
                    continue
                source = Path(raw)
                name = f"{key}-{source.name}"
                target = staging / name
                # Independent copies: modifying the original cannot mutate the snapshot.
                shutil.copyfile(source, target)
                source_hash, target_hash = file_hash(source), file_hash(target)
                if source_hash != target_hash:
                    raise ValueError("Artifact changed while taking its snapshot")
                target.chmod(0o555 if key == "binary" else 0o444)
                files[name] = target_hash
                preserved["server"][key] = str(path.parent / name)
            # llama.cpp places shared libraries beside its executables. Preserve them
            # too: relocating only llama-server would break its $ORIGIN RUNPATH.
            for source in sorted(Path(profile["server"]["binary"]).parent.glob("*.so*")):
                if source.is_file():
                    target = staging / source.name
                    shutil.copyfile(source, target)
                    files[source.name] = file_hash(target)
                    if files[source.name] != file_hash(source):
                        raise ValueError("Native library changed while taking its snapshot")
                    target.chmod(0o444)
            preserved["server"]["library_path"] = str(path.parent)
            if candidate_report is not None:
                if (
                    files[Path(preserved["server"]["model"]).name]
                    != candidate_report["model_sha256"]
                ):
                    raise ValueError("Model changed after evaluation and before snapshot commit")
                staged_profile = copy.deepcopy(preserved)
                for key in ("model", "binary", "draft_model"):
                    if staged_profile["server"][key]:
                        staged_profile["server"][key] = str(
                            staging / Path(staged_profile["server"][key]).name
                        )
                if execution_hash(staged_profile) != candidate_report["execution_sha256"]:
                    raise ValueError("Execution artifacts changed before snapshot commit")
            report = {
                "schema": "v100-expert-v1",
                "id": expert_id,
                "description": description,
                "profile": preserved,
                "files": files,
                "gate": gate,
                "quality_reports": {"baseline": baseline_report, "candidate": candidate_report},
            }
            atomic_json(staging / "expert.json", report)
            (staging / "expert.json").chmod(0o444)
            # mkdir is an exclusive claim; concurrent registrations never replace an expert.
            path.parent.mkdir()
            for item in staging.iterdir():
                os.rename(item, path.parent / item.name)
            staging.rmdir()
            return report
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def get(self, expert_id: str, verify: bool = True) -> dict:
        path = self.manifest_path(expert_id)
        expert = json.loads(path.read_text())
        if expert["id"] != expert_id or expert["schema"] != "v100-expert-v1":
            raise ValueError("Invalid expert manifest")
        if verify:
            for name, expected in expert["files"].items():
                if Path(name).name != name or file_hash(path.parent / name) != expected:
                    raise ValueError(f"Protected expert artifact changed: {name}")
        return expert

    def list(self) -> list[dict]:
        return [
            self.get(path.parent.name, verify=False)
            for path in sorted(self.directory.glob("*/expert.json"))
        ]


def execution_hash(profile: dict) -> str:
    server = {
        key: value
        for key, value in profile["server"].items()
        if key not in ("model", "binary", "draft_model", "library_path")
    }
    payload = {
        "runtime": profile["runtime"],
        "server": server,
        "binary_sha256": file_hash(Path(profile["server"]["binary"])),
        "draft_sha256": file_hash(Path(profile["server"]["draft_model"]))
        if profile["server"]["draft_model"]
        else None,
        "native_libraries": {
            path.name: file_hash(path)
            for path in sorted(Path(profile["server"]["binary"]).parent.glob("*.so*"))
            if path.is_file()
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
