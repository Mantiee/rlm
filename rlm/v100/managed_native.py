"""One owned maintenance window; always resume the frozen accepted mission profile."""

import copy
import fcntl
import json
import subprocess
import time
from pathlib import Path

from rlm.v100.common import atomic_json, load_profile


def exclusive(processes: str) -> bool:
    return any(
        "rlm.v100.cli" in line
        and any(
            " " + mode + " " in line + " "
            for mode in (
                "train",
                "mission-prepare",
                "optimize-mtp",
                "calibrate-training",
                "test-mtp",
            )
        )
        for line in processes.splitlines()
    )


def run(root: Path, *, allow_maintenance: bool = False) -> dict:
    if not allow_maintenance:
        return {
            "state": "deferred-live-priority",
            "scope": "Master/chat keep the accepted server. Native maintenance requires a separate explicit window.",
        }
    lock = root / "research/state/architecture-cuda.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"state": "deferred-training", "scope": "Owned CUDA budget is already leased"}
        try:
            return maintenance(root)
        finally:
            fcntl.flock(lease, fcntl.LOCK_UN)


def qualify_settings(
    root: Path, profile: dict, selection: dict, folder: Path
) -> tuple[Path | None, dict]:
    from rlm.v100.architecture_promotion import complete
    from rlm.v100.competition import helper_client, managed_server
    from rlm.v100.evaluation import evaluate_suite
    from rlm.v100.goals import load_goal
    from rlm.v100.lineages import read
    from rlm.v100.protection import compare_reports

    if selection.get("state") != "measured" or selection.get("winner") == "baseline":
        return None, {"activated": False, "reason": "No measured faster configuration"}
    candidate = copy.deepcopy(profile)
    candidate["server"].update(selection["server"])
    candidate["server"].pop("tune_kernels", None)
    suite = root / "research/income-challenge-v1/development.jsonl"
    load_goal(root, suite)
    reports = []
    for name, settings in (("before", profile), ("after", candidate)):
        path = folder / (name + ".profile.json")
        atomic_json(path, settings)
        with managed_server(path, root, folder / (name + ".quality.log")):
            reports.append(
                evaluate_suite(
                    helper_client(settings, root),
                    settings,
                    suite,
                    folder / (name + ".quality.json"),
                )
            )
    _, retained = read(profile)
    decisions = [compare_reports(previous, reports[1]) for previous in [reports[0], *retained]]
    passed = complete(reports[1]) and all(row["passed"] for row in decisions)
    evidence = {
        "activated": passed,
        "gates": decisions,
        "speedup": selection["speedup"],
        "scope": "Same model, sampling, context and finite retained task suite; measured serving optimization, not learned profit",
    }
    atomic_json(folder / "settings-admission.json", evidence)
    return (folder / "after.profile.json" if passed else None), evidence


def maintenance(root: Path) -> dict:
    from rlm.v100.inference_benchmark import run as benchmark
    from rlm.v100.mission import status, stop
    from rlm.v100.supervisor import install

    receipt = root / "research/managed-update/native-qualification.json"
    processes = subprocess.check_output(["ps", "-eo", "args="], text=True)
    if exclusive(processes):
        return {"state": "deferred-training", "scope": "No mission or training stopped"}
    current = status(root)
    if not current.get("running"):
        return {"state": "deferred-mission-not-running"}
    source = current.get("learning", {}).get("live_profile")
    if not source:
        source = str(Path(current["run"]) / "input-profile.json")
    profile = load_profile(Path(source), root)
    if profile["runtime"].get("backend"):
        return {"state": "deferred-non-native-master"}
    from rlm.v100.model_roles import json_hash
    from rlm.v100.protection import file_hash

    identity = {
        "profile_sha256": json_hash(profile),
        "model_sha256": file_hash(Path(profile["server"]["model"])),
    }
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous.get("state") == "measured" and all(
            previous.get(k) == v for k, v in identity.items()
        ):
            return previous
    folder = root / "research/performance" / ("managed-" + str(time.time_ns()))
    folder.mkdir(parents=True)
    preserved = folder / "accepted-profile.json"
    atomic_json(preserved, profile)
    result = {
        "state": "maintenance-starting",
        "profile": str(preserved),
        "updated": time.time(),
        **identity,
    }
    atomic_json(receipt, result)
    resume_profile = preserved
    try:
        stop(root)
        subprocess.run(
            ["systemctl", "--user", "stop", "v100-mission.service"], check=True, timeout=90
        )
        processes = subprocess.check_output(["ps", "-eo", "args="], text=True)
        if exclusive(processes) or status(root).get("running"):
            raise RuntimeError("Owned GPU did not become exclusive; benchmark not started")
        settings = {"server": {**profile["server"], "tune_kernels": True}}
        measurement = benchmark(settings, folder / "accepted-master", 8097)
        result.update(state=measurement["selection"]["state"], selection=measurement["selection"])
        result["scope"] = (
            "Native n-gram/cache/offload/thread/Flash Attention comparison on the accepted GGUF; "
            "measurement proposal only. Independent protected quality gates precede changed serving settings."
        )
        try:
            promoted, admission = qualify_settings(root, profile, measurement["selection"], folder)
            result["settings_admission"] = admission
            if promoted is not None:
                resume_profile = promoted
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            result["settings_admission"] = {
                "activated": False,
                "state": "blocked",
                "error": str(error),
            }
        candidates = {}
        for role in ("fast", "planner"):
            path = root / f"research/model-roles/candidates/{role}.json"
            if not path.exists():
                continue
            try:
                candidates[role] = benchmark(json.loads(path.read_text()), folder / role, 8097)[
                    "selection"
                ]
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                candidates[role] = {"state": "blocked", "error": str(error)}
        # Test a real long prefill on the small Qwen candidate; allocate only if resource gates admit it.
        path = root / "research/model-roles/candidates/fast.json"
        if path.exists():
            candidate = json.loads(path.read_text())
            candidate["server"].update(
                context_per_slot=524288,
                rope_scaling="yarn",
                rope_scale=2,
                yarn_orig_ctx=262144,
                tune_kernels=False,
            )
            try:
                candidates["long_context"] = benchmark(candidate, folder / "context-524288", 8097)[
                    "selection"
                ]
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                candidates["long_context"] = {"state": "blocked", "error": str(error)}
        result["candidates"] = candidates
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        result.update(state="blocked", error=str(error))
    finally:
        try:
            result["resume"] = install(root, resume_profile)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            result.update(
                state="resume-failed", resume_error=str(error), recovery_profile=str(preserved)
            )
            if resume_profile != preserved:
                try:
                    result["rollback_resume"] = install(root, preserved)
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as recovery:
                    result["rollback_error"] = str(recovery)
        if resume_profile != preserved:
            result["profile_sha256"] = json_hash(json.loads(resume_profile.read_text()))
        result["updated"] = time.time()
        atomic_json(receipt, result)
    return result
