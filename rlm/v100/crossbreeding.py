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
    with managed_server(serving, root, output / "server.log"):
        report = evaluate_suite(
            helper_client(chosen, root), chosen, suite, output / "development-quality.json"
        )
    score = sum(c["passed"] for c in report["cases"])
    passed = (
        not any(c.get("error") for c in report["cases"])
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
