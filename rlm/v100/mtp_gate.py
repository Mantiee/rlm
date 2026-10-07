"""Enable measured MTP only after speed and independent fixed-suite checks."""

import copy
import json
from pathlib import Path

from rlm.v100.common import atomic_json, load_profile
from rlm.v100.protection import compare_reports, execution_hash, file_hash
from rlm.v100.speculative import DEFAULT_DRAFT_TOKENS, prepare_mtp, recommend_mtp, test_mtp


def identity(profile: dict) -> str:
    target = copy.deepcopy(profile)
    target["server"].update(draft_model="", draft_tokens=2, spec_type="draft-mtp")
    return execution_hash(target)


def valid(profile: dict) -> bool:
    proof = profile.get("resources", {}).get("mtp_validation")
    if not isinstance(proof, dict) or not profile["server"].get("draft_model"):
        return False
    try:
        if (
            proof["execution"] != identity(profile)
            or proof["draft_hash"] != file_hash(Path(profile["server"]["draft_model"]))
            or proof["draft_tokens"] != profile["server"]["draft_tokens"]
        ):
            return False
        paths = proof["reports"]
        if len(paths) != 3 or any(
            file_hash(Path(path)) != digest for path, digest in paths.items()
        ):
            return False
        values = {Path(path).name: json.loads(Path(path).read_text()) for path in paths}
        before, after = values["quality-baseline.json"], values["quality-candidate.json"]
        return (
            compare_reports(before, after)["passed"]
            and before["model_sha256"]
            == after["model_sha256"]
            == file_hash(Path(profile["server"]["model"]))
            and not any(case.get("error") for case in after["cases"])
            and recommend_mtp(values["comparison.json"], False)["recommended_speed_profile"]
            == f"mtp{proof['draft_tokens']}"
        )
    except (OSError, ValueError, TypeError, KeyError):
        return False


def optimize(root: Path, path: Path, suite: Path, repeats: int = 3) -> Path:
    from rlm.v100.competition import helper_client, managed_server, require_idle_gpu
    from rlm.v100.evaluation import evaluate_suite
    from rlm.v100.mission import setup_profile, status

    if status(root)["running"]:
        raise ValueError("Stop the mission before exclusive MTP experiments")
    require_idle_gpu()
    source = load_profile(path, root)
    source = setup_profile(source, source["runtime"]["context_window"])
    source["server"]["draft_model"] = ""
    source.setdefault("resources", {}).pop("mtp_validation", None)
    prepared = root / "research/mtp-input.json"
    atomic_json(prepared, source)
    # Preserve previous experiments before creating a new target-bound sweep.
    import time

    for name in ("baseline", *(f"mtp{n}" for n in DEFAULT_DRAFT_TOKENS)):
        experiment = root / f"research/v100-{name}.json"
        if experiment.exists():
            experiment.rename(experiment.with_name(experiment.name + f".backup-{time.time_ns()}"))
    prepare_mtp(prepared, root)
    for name in ("baseline", *(f"mtp{n}" for n in DEFAULT_DRAFT_TOKENS)):
        generated = load_profile(root / f"research/v100-{name}.json", root)
        if identity(generated) != identity(source):
            raise ValueError("MTP sweep changed the target execution configuration")
    folder = test_mtp(root, repeats)
    speed = json.loads((folder / "comparison.json").read_text())
    selected = recommend_mtp(speed, False)["recommended_speed_profile"]
    baseline = root / "research/v100-baseline.json"
    if selected == "baseline":
        print("No measured MTP speed gain; retain baseline", flush=True)
        return baseline
    candidate = root / f"research/v100-{selected}.json"
    reports = {}
    for name, profile_path in (("baseline", baseline), ("candidate", candidate)):
        with managed_server(profile_path, root, folder / f"quality-{name}.server.log") as profile:
            reports[name] = evaluate_suite(
                helper_client(profile, root), profile, suite, folder / f"quality-{name}.json"
            )
    verdict = compare_reports(reports["baseline"], reports["candidate"])
    atomic_json(folder / "quality-verdict.json", verdict)
    if not verdict["passed"] or any(case.get("error") for case in reports["candidate"]["cases"]):
        print("MTP failed finite quality gate; retain baseline", flush=True)
        return baseline
    chosen = load_profile(candidate, root)
    chosen.setdefault("resources", {})["mtp_validation"] = {
        "execution": identity(chosen),
        "draft_hash": file_hash(Path(chosen["server"]["draft_model"])),
        "draft_tokens": chosen["server"]["draft_tokens"],
        "reports": {
            str(p): file_hash(p)
            for p in (
                folder / "comparison.json",
                folder / "quality-baseline.json",
                folder / "quality-candidate.json",
            )
        },
    }
    destination = folder / "validated-profile.json"
    atomic_json(destination, chosen)
    return destination
