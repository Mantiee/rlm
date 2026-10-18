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


def quality_verdict(baseline: dict, candidate: dict, control: dict | None = None) -> dict:
    """Compare quality runs with optional noise calibration via a control run.

    Without *control*: strict rule — any regression fails (original behaviour).

    With *control* (a second baseline run after the candidate, same settings):
    - Counts baseline↔control flips (cases that changed between the two baseline
      runs purely due to stochastic sampling).
    - Passes only when:
        (a) candidate regressions ≤ baseline↔control flips  (noise-calibrated), and
        (b) candidate passes at least as many cases as the weaker baseline run.
    - A deterministic suite gives 0 flips, so the result equals the strict rule.
    - The control must share model_sha256, suite_sha256, generation settings and
      memory_mode with both baseline and candidate (compare_reports validates this).
    """
    # Validate that all reports are structurally compatible via compare_reports.
    strict = compare_reports(baseline, candidate)

    if control is None:
        return strict

    # Validate control is comparable to baseline.
    compare_reports(baseline, control)

    base_cases = {c["id"]: c["passed"] for c in baseline["cases"]}
    ctrl_cases = {c["id"]: c["passed"] for c in control["cases"]}
    cand_cases = {c["id"]: c["passed"] for c in candidate["cases"]}

    flips = sum(1 for cid in base_cases if base_cases[cid] != ctrl_cases.get(cid))
    regressions = sum(
        1 for cid in base_cases if base_cases[cid] and not cand_cases.get(cid)
    )
    base_pass = sum(1 for v in base_cases.values() if v)
    ctrl_pass = sum(1 for v in ctrl_cases.values() if v)
    cand_pass = sum(1 for v in cand_cases.values() if v)
    weaker_baseline = min(base_pass, ctrl_pass)

    noise_ok = regressions <= flips and cand_pass >= weaker_baseline
    return {
        **strict,
        "passed": noise_ok,
        "noise_calibrated": True,
        "flips": flips,
        "regressions": regressions,
        "candidate_passed": cand_pass,
        "weaker_baseline_passed": weaker_baseline,
    }


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
        n_reports = len(paths)
        if n_reports not in (3, 4) or any(
            file_hash(Path(path)) != digest for path, digest in paths.items()
        ):
            return False
        values = {Path(path).name: json.loads(Path(path).read_text()) for path in paths}
        before = values["quality-baseline.json"]
        after = values["quality-candidate.json"]
        control = values.get("quality-control.json")  # present in 4-report proof only
        verdict = quality_verdict(before, after, control)
        return (
            verdict["passed"]
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
    # Run a noise-calibration control: baseline once more, after the candidate.
    # This lets quality_verdict distinguish real regressions from sampling variance.
    with managed_server(baseline, root, folder / "quality-control.server.log") as profile:
        reports["control"] = evaluate_suite(
            helper_client(profile, root),
            profile,
            suite,
            folder / "quality-control.json",
        )
    verdict = quality_verdict(reports["baseline"], reports["candidate"], reports["control"])
    atomic_json(folder / "quality-verdict.json", verdict)
    if not verdict["passed"] or any(case.get("error") for case in reports["candidate"]["cases"]):
        print("MTP failed noise-calibrated quality gate; retain baseline", flush=True)
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
                folder / "quality-control.json",
            )
        },
    }
    destination = folder / "validated-profile.json"
    atomic_json(destination, chosen)
    return destination
