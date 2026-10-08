"""One-use post-freeze randomized checks; detailed cases never enter model feedback."""

import json
import secrets
from pathlib import Path

from rlm.v100.common import atomic_json
from rlm.v100.insights import verified_record
from rlm.v100.protection import compare_reports, file_hash, reserve_audit_sources


def create(root: Path, parent: dict, candidate: Path, folder: Path, count: int = 12) -> Path:
    if type(count) is not int or not 8 <= count <= 32:
        raise ValueError("Fresh audit needs 8-32 independently generated cases")
    # Freeze identities BEFORE drawing tasks. No seeds supplied by the optimizer.
    identities = {
        "parent_sha256": file_hash(Path(parent["server"]["model"])),
        "candidate_sha256": file_hash(candidate),
    }
    folder.mkdir(parents=True, exist_ok=False)
    rng, rows, sources = secrets.SystemRandom(), [], []
    while len(rows) < count:
        a, b, c = (rng.randint(2, 999) for _ in range(3))
        kind = len(rows) % 3
        task = (
            {"kind": "arithmetic", "expression": f"({a}-{b})*{c}"}
            if kind == 0
            else {"kind": "linear_equation", "expression": f"{a}*x+{b}={c}"}
            if kind == 1
            else {"kind": "decimal_calculation", "expression": f"{a}.{b:03d}*{c}/10000"}
        )
        record = verified_record(task)
        if record["group"] in sources:
            continue
        sources.extend(record["document_ids"])
        rows.append(
            {
                "id": record["group"],
                "skill": task["kind"],
                "match": "exact",
                "messages": record["messages"][:-1],
                "expected": record["messages"][-1]["content"],
            }
        )
    reserve_audit_sources(
        Path(parent["training"].get("split_ledger", root / "research/state/splits.sqlite3")),
        sources,
    )
    reserve_audit_sources(root / "research/state/architecture-splits.sqlite3", sources)
    suite = folder / "suite.jsonl"
    suite.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    atomic_json(
        folder / "manifest.json",
        {
            "schema": "v100-fresh-audit-v1",
            "state": "sealed",
            **identities,
            "suite_sha256": file_hash(suite),
            "count": count,
            "scope": "Fresh finite arithmetic/equation/decimal checks, not proof of any user goal or universal retention",
        },
    )
    return folder


def parent_report(root: Path, parent: dict, folder: Path) -> None:
    from rlm.v100.competition import helper_client, managed_server
    from rlm.v100.evaluation import evaluate_suite

    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["state"] != "sealed" or manifest["parent_sha256"] != file_hash(
        Path(parent["server"]["model"])
    ):
        raise ValueError("Fresh audit already exposed or parent changed")
    # Any interrupted/failed attempt is consumed; never rerun revealed cases.
    atomic_json(folder / "manifest.json", {**manifest, "state": "exposed"})
    path = folder / "parent-profile.json"
    atomic_json(path, parent)
    with managed_server(path, root, folder / "parent-server.log"):
        evaluate_suite(
            helper_client(parent), parent, folder / "suite.jsonl", folder / "parent.json"
        )


def candidate_report(client, candidate: dict, folder: Path) -> dict:
    from rlm.v100.evaluation import evaluate_suite

    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["state"] != "exposed" or (folder / "candidate.json").exists():
        raise ValueError("Fresh audit is single-use")
    if manifest["candidate_sha256"] != file_hash(Path(candidate["server"]["model"])) or manifest[
        "suite_sha256"
    ] != file_hash(folder / "suite.jsonl"):
        raise ValueError("Frozen candidate or audit suite changed")
    # Burn before inference, including failures and transport errors.
    atomic_json(folder / "manifest.json", {**manifest, "state": "consumed"})
    report = evaluate_suite(client, candidate, folder / "suite.jsonl", folder / "candidate.json")
    baseline = json.loads((folder / "parent.json").read_text())
    gate = compare_reports(baseline, report)
    complete = all(
        not row.get("error") and row.get("finish_reason") != "length"
        for r in (baseline, report)
        for row in r["cases"]
    )
    gate["passed"] = gate["passed"] and complete
    summary = {
        "passed": gate["passed"],
        "complete": complete,
        "count": manifest["count"],
        "scope": manifest["scope"],
    }
    atomic_json(
        folder / "gate.json",
        {
            **summary,
            "parent_report_sha256": file_hash(folder / "parent.json"),
            "candidate_report_sha256": file_hash(folder / "candidate.json"),
        },
    )
    return summary


def verified_gate(folder: Path, candidate: Path) -> dict:
    manifest = json.loads((folder / "manifest.json").read_text())
    gate = json.loads((folder / "gate.json").read_text())
    if (
        manifest["state"] != "consumed"
        or manifest["candidate_sha256"] != file_hash(candidate)
        or manifest["suite_sha256"] != file_hash(folder / "suite.jsonl")
    ):
        raise ValueError("Fresh audit proof differs from frozen candidate")
    for role in ("parent", "candidate"):
        report_path = folder / f"{role}.json"
        report = json.loads(report_path.read_text())
        if (
            gate[f"{role}_report_sha256"] != file_hash(report_path)
            or report["model_sha256"] != manifest[f"{role}_sha256"]
            or report["suite_sha256"] != manifest["suite_sha256"]
            or len(report["cases"]) != manifest["count"]
        ):
            raise ValueError("Fresh audit report changed or is incomplete")
    old, new = [
        json.loads((folder / f"{role}.json").read_text()) for role in ("parent", "candidate")
    ]
    passed = compare_reports(old, new)["passed"] and all(
        not r.get("error") and r.get("finish_reason") != "length"
        for report in (old, new)
        for r in report["cases"]
    )
    if passed != gate["passed"]:
        raise ValueError("Fresh audit verdict changed")
    return {"passed": passed, "count": manifest["count"], "scope": manifest["scope"]}
