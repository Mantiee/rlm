"""Activate a different native or isolated architecture from frozen host evidence."""

import copy
import json
import shutil
from pathlib import Path

from rlm.v100.common import atomic_json
from rlm.v100.protection import ExpertRegistry, compare_reports, execution_hash, file_hash


def complete(report: dict) -> bool:
    return bool(report.get("cases")) and all(
        not row.get("error") and row.get("finish_reason") != "length" for row in report["cases"]
    )


def probe(root: Path, profile: dict, folder: Path) -> None:
    from rlm.v100.competition import helper_client, managed_server
    from rlm.v100.serving import assert_served_expert

    path = folder / "activation-probe.json"
    atomic_json(path, profile)
    with managed_server(path, root, folder / "activation-probe.log"):
        client = helper_client(profile, root)
        assert_served_expert(client, profile, root)
        client.sampling_args["max_tokens"] = min(64, profile["runtime"]["max_output_tokens"])
        client.enable_thinking = False
        answer = client.completion("Return only the integer result of 2+2.")
        if not answer.strip() or client.get_response_info()["finish_reason"] == "length":
            raise ValueError("Architecture did not finish its activation inference probe")


def activate(root: Path, parent: dict, result: dict, gates: list[dict]) -> tuple[dict, dict]:
    """Return a new profile; caller atomically commits live.json after success."""
    from rlm.v100.fresh_audit import verified_gate
    from rlm.v100.lineages import read

    if result.get("eligible") is not True:
        return parent, {"activated": False, "reason": "Candidate failed its quality gates"}
    if not result.get("expert_id") or not result.get("fresh_audit"):
        raise ValueError("Architecture has no frozen activation proof")
    if result["expert_id"] in parent.get("resources", {}).get("quarantined_foundations", []):
        return parent, {
            "activated": False,
            "reason": "Architecture quarantined after serving failure",
        }
    registry = ExpertRegistry(root / "research/experts")
    expert = registry.get(result["expert_id"])
    candidate = copy.deepcopy(expert["profile"])
    quality = expert["quality_reports"]["candidate"]
    if (
        not complete(quality)
        or quality["model_sha256"] != file_hash(Path(candidate["server"]["model"]))
        or quality["execution_sha256"] != execution_hash(candidate)
    ):
        raise ValueError("Architecture activation evidence or execution changed")
    if not gates or not all(compare_reports(g, quality)["passed"] for g in gates):
        raise ValueError("Architecture failed retained ancestor gates")
    parent_sha = file_hash(Path(parent["server"]["model"]))
    current_report = next((g for g in reversed(gates) if g["model_sha256"] == parent_sha), None)
    if current_report is None or not complete(current_report):
        raise ValueError("Architecture requires a complete evaluation of the live predecessor")
    improvement = compare_reports(current_report, quality)["improvements"]
    if not improvement:
        return parent, {
            "activated": False,
            "reason": "No measured task improvement; expert retained",
        }
    audit = Path(result["fresh_audit"])
    manifest = json.loads((audit / "manifest.json").read_text())
    if (
        manifest["parent_sha256"] != parent_sha
        or not verified_gate(audit, Path(candidate["server"]["model"]))["passed"]
    ):
        raise ValueError("Architecture needs a passed fresh audit against the live predecessor")
    resources = candidate.setdefault("resources", {})
    if parent.get("resources", {}).get("public_benchmarks"):
        from rlm.v100.public_benchmarks import compare

        baseline = Path(parent["resources"]["public_baseline"])
        public = Path(result["public_quality"])
        public_report = json.loads(public.read_text())
        if public_report.get("model_sha256") != quality["model_sha256"]:
            raise ValueError("Architecture public report evaluated another model")
        if (
            file_hash(baseline) != parent["resources"]["public_baseline_sha256"]
            or file_hash(public) != result["public_quality_sha256"]
            or not compare(json.loads(baseline.read_text()), public_report)["passed"]
        ):
            raise ValueError("Architecture public benchmark proof changed or failed")
        resources.update(public_baseline=str(public), public_baseline_sha256=file_hash(public))
    # No LoRA from one foundation is attached to a different foundation.
    _, branch_gates = read(parent)
    predecessor_id = "predecessor-" + parent_sha[:24] + "-" + execution_hash(parent)[:12]
    if not registry.manifest_path(predecessor_id).exists():
        registry.register(
            predecessor_id,
            parent,
            "Automatic architecture rollback",
            current_report,
            current_report,
        )
    registry.get(predecessor_id)
    folder = registry.manifest_path(result["expert_id"]).parent
    adapter_path = candidate["training"].get("init_adapter")
    if adapter_path:
        from rlm.v100.breeding import base_signature, verified_adapter

        metadata, _ = verified_adapter(Path(adapter_path))
        if metadata["base"] != base_signature(Path(candidate["training"]["base_model"])):
            raise ValueError("Architecture continuation adapter targets a changed base")
        frozen = folder / "continuation-adapter"
        frozen.mkdir(exist_ok=True)
        for name in (
            "adapter_config.json",
            "adapter_model.safetensors",
            "v100-adapter.json",
            "v100-export.json",
        ):
            source = Path(adapter_path) / name
            if not source.exists() and name == "v100-export.json":
                continue
            destination = frozen / name
            if destination.exists() and file_hash(destination) != file_hash(source):
                raise ValueError("Architecture continuation snapshot already differs")
            if not destination.exists():
                shutil.copyfile(source, destination)
                destination.chmod(0o444)
            if file_hash(destination) != file_hash(source):
                raise ValueError("Architecture continuation changed during snapshot")
        verified_adapter(frozen)
        candidate["training"].update(init_adapter=str(frozen), teacher_adapter=str(frozen))
    protection = folder / "retained-gates.json"
    evidence = {"schema": "v100-architecture-gates-v1", "reports": gates + branch_gates + [quality]}
    if protection.exists() and json.loads(protection.read_text()) != evidence:
        raise ValueError("Architecture ancestor evidence already differs")
    atomic_json(protection, evidence)
    protection.chmod(0o444)
    resources["protected_architecture_gates"] = {
        "path": str(protection),
        "sha256": file_hash(protection),
    }
    history = list(parent.get("resources", {}).get("architecture_history", []))
    history.append(
        {
            "predecessor_expert": predecessor_id,
            "previous_branch_lineages": parent.get("resources", {}).get("branch_lineages", {}),
            "selected_expert": result["expert_id"],
            "fresh_audit": str(audit),
            "improved_cases": len(improvement),
        }
    )
    resources.update(architecture_history=history, active_foundation_expert=result["expert_id"])
    resources.pop("branch_lineages", None)
    probe(root, candidate, folder)
    receipt = {
        "activated": True,
        "expert_id": result["expert_id"],
        "predecessor_expert": predecessor_id,
        "improved_cases": len(improvement),
        "scope": "Frozen finite quality, public and fresh gates; universal retention is not guaranteed",
    }
    atomic_json(folder / "activation.json", receipt)
    return candidate, receipt


def rollback(root: Path, current: dict, reason: str) -> tuple[dict, dict]:
    """Recover an immutable predecessor after repeated serving failures."""
    history = current.get("resources", {}).get("architecture_history", [])
    if not history:
        return current, {"restored": False, "reason": "No architecture predecessor"}
    entry = history[-1]
    registry = ExpertRegistry(root / "research/experts")
    predecessor = registry.get(entry["predecessor_expert"])
    restored = copy.deepcopy(predecessor["profile"])
    resources = restored.setdefault("resources", {})
    rejected = set(current.get("resources", {}).get("quarantined_foundations", []))
    rejected.add(entry["selected_expert"])
    resources["quarantined_foundations"] = sorted(rejected)
    receipt = {
        "restored": True,
        "predecessor_expert": entry["predecessor_expert"],
        "quarantined_expert": entry["selected_expert"],
        "reason": reason[:400],
    }
    resources["architecture_rollback"] = receipt
    return restored, receipt
