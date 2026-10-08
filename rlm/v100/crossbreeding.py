"""One bounded child of accepted A/B parents, evaluated independently."""

import copy
import json
import shutil
import subprocess
from pathlib import Path

from rlm.v100.common import atomic_json


def try_child(
    root: Path, profile: dict, suite: Path, gates: list[dict], output: Path
) -> tuple[dict, dict | None]:
    from rlm.v100.breeding import breed_adapters
    from rlm.v100.competition import command, helper_client, managed_server, require_idle_gpu
    from rlm.v100.evaluation import evaluate_suite
    from rlm.v100.lineages import read
    from rlm.v100.protection import compare_reports

    parents, quality = read(profile)
    if set(parents) != {"A", "B"}:
        return profile, None
    adapters = [Path(parents[b]["training"]["init_adapter"]) for b in ("A", "B")]
    rank = sum(json.loads((p / "adapter_config.json").read_text())["r"] for p in adapters)
    if rank > 64 or shutil.disk_usage(root).free < 80 * 2**30:
        return profile, {"status": "deferred", "reason": "child exceeds rank/disk budget"}
    require_idle_gpu()
    output.mkdir()
    training = output / "training"
    training.mkdir()
    breed_adapters(
        *adapters, training / "candidate", 0.5, Path(profile["training"]["base_model"]), root
    )
    chosen = copy.deepcopy(profile)
    chosen["training"]["output"] = str(training)
    config = output / "export-profile.json"
    atomic_json(config, chosen)
    subprocess.run(command(root, config, "export-model"), check=True, timeout=3600)
    chosen["server"].update(model=str(training / "export-Q6_K.gguf"), draft_model="")
    chosen["runtime"]["model_version"] = output.name
    chosen.setdefault("resources", {}).pop("mtp_validation", None)
    chosen["training"].update(
        init_adapter=str(training / "candidate"), teacher_adapter=str(training / "candidate")
    )
    serving = output / "serving.json"
    atomic_json(serving, chosen)
    audit = None
    if chosen.get("resources", {}).get("fresh_audit_required"):
        from rlm.v100 import fresh_audit

        audit = fresh_audit.create(
            root, profile, Path(chosen["server"]["model"]), output / "fresh-audit"
        )
        fresh_audit.parent_report(root, profile, audit)
    with managed_server(serving, root, output / "server.log"):
        report = evaluate_suite(
            helper_client(chosen, root), chosen, suite, output / "development-quality.json"
        )
        public_passed = True
        if chosen.get("resources", {}).get("public_benchmarks"):
            from rlm.v100.protection import file_hash
            from rlm.v100.public_benchmarks import compare, evaluate

            public_path = output / "public-quality.json"
            candidate = evaluate(root, chosen, public_path)
            for parent in parents.values():
                reference = Path(parent["resources"]["public_baseline"])
                if file_hash(reference) != parent["resources"]["public_baseline_sha256"]:
                    raise ValueError("Official ancestor report changed")
                public_passed &= compare(json.loads(reference.read_text()), candidate)["passed"]
            chosen["resources"].update(
                public_baseline=str(public_path), public_baseline_sha256=file_hash(public_path)
            )
            atomic_json(serving, chosen)
        if audit is not None:
            fresh_audit.candidate_report(helper_client(chosen), chosen, audit)
    fresh_passed = (
        audit is None or fresh_audit.verified_gate(audit, Path(chosen["server"]["model"]))["passed"]
    )
    score = sum(c["passed"] for c in report["cases"])
    passed = (
        public_passed
        and fresh_passed
        and not any(c.get("error") for c in report["cases"])
        and all(compare_reports(parent, report)["passed"] for parent in [*gates, *quality])
        and score > max(sum(c["passed"] for c in p["cases"]) for p in quality)
    )
    verdict = {
        "status": "accepted child" if passed else "rejected child; accepted parents retained",
        "passed": passed,
        "score": score,
        "quality_report": str(output / "development-quality.json"),
    }
    atomic_json(output / "verdict.json", verdict)
    return (chosen if passed else profile), verdict
