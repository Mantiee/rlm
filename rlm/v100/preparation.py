"""Reviewable stopped-mission setup with persistent stage logs and no hidden promotion."""

import json
import time
from pathlib import Path

from rlm.v100.common import atomic_json, load_profile


def prepare(root: Path, optimize: bool = False, calibrate: bool = False) -> Path:
    from rlm.v100.competition import helper_client
    from rlm.v100.income_skills import prepare as prepare_skills
    from rlm.v100.mission import setup_profile, status
    from rlm.v100.remote_helper import prepare_remote

    previous = status(root)
    if previous["running"]:
        raise ValueError("Stop the mission first; existing checkpoints will be retained")
    run = Path(previous["run"])
    path = run / "learning/live.json"
    if not path.exists():
        path = run / "input-profile.json"
    profile = load_profile(path, root)
    profile["server"]["flash_attention"] = "on"
    profile = setup_profile(profile, 131072)
    directory = root / "research/preparation" / ("run-" + str(time.time_ns()))
    directory.mkdir(parents=True)
    prepared = directory / "prepared-profile.json"
    atomic_json(prepared, profile)
    stages = {}

    def stage(name, action, required=False):
        print(json.dumps({"preparation": name, "status": "starting"}), flush=True)
        try:
            value = action()
            stages[name] = {"status": "prepared", "result": str(value)}
        except Exception as error:
            stages[name] = {
                "status": "failed" if required else "deferred",
                "error": type(error).__name__,
                "detail": str(error)[:500],
            }
            atomic_json(directory / "stages.json", stages)
            print(json.dumps({name: stages[name]}), flush=True)
            if required:
                raise
        atomic_json(directory / "stages.json", stages)
        print(json.dumps({name: stages[name]}), flush=True)

    def helper():
        target = root / "research/researcher-rtx3090.json"
        old = load_profile(target, root)
        # Validate pinned identity before altering context/budget, not a blind re-pin.
        helper_client(old, root).identity()
        if old["runtime"]["context_window"] == 32768:
            chosen = old
        else:
            backup = target.with_name(f"researcher-rtx3090.backup-{time.time_ns()}.json")
            target.rename(backup)
            try:
                target = prepare_remote(
                    profile,
                    root,
                    old["runtime"]["base_url"],
                    32768,
                    old["resources"]["model_digest"],
                )
                chosen = load_profile(target, root)
            except Exception:
                target.write_bytes(backup.read_bytes())
                raise
        chosen["resources"].update(helper_batch_tokens=16, helper_duty_percent=30)
        chosen["runtime"]["max_timeout"] = 600
        helper_client(chosen, root).loaded()
        atomic_json(target, chosen)
        return target

    stage("rtx-helper", helper, required=True)
    stage("verified-cost-skills", lambda: prepare_skills(root, profile), required=True)
    from rlm.v100.spot_bootstrap import prepare as prepare_spot

    stage("paper-spot-primary-sources", lambda: prepare_spot(root), required=True)
    from rlm.v100.mission_semantic import prepare as prepare_memory
    from rlm.v100.self_code import prepare as prepare_code

    stage("cpu-hybrid-memory", lambda: prepare_memory(root))
    stage("isolated-self-code", lambda: prepare_code(root))
    if calibrate:
        from rlm.v100.training_calibration import calibrate as measure

        candidates = []

        def measure_training():
            result = measure(root, prepared, root / "research/income-challenge-v1/pool.jsonl")
            candidates.append(result)
            return result

        stage("training-calibration", measure_training)
        if candidates:
            atomic_json(prepared, load_profile(candidates[-1], root))
    if optimize:
        from rlm.v100.mtp_gate import optimize as measure_mtp

        candidates = []

        def measure_draft():
            result = measure_mtp(
                root, prepared, root / "research/income-challenge-v1/development.jsonl"
            )
            candidates.append(result)
            return result

        stage("measured-mtp", measure_draft)
        if candidates:
            atomic_json(prepared, load_profile(candidates[-1], root))
    atomic_json(
        root / "research/prepared-mission.json",
        {
            "profile": str(prepared),
            "stages": str(directory / "stages.json"),
            "mission_started": False,
        },
    )
    print("PREPARED PROFILE:", prepared, flush=True)
    return prepared
