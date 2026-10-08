"""Assemble already measured components without repeating long calibration sweeps."""

import copy
import json
import time
from pathlib import Path

from rlm.v100.common import atomic_json, load_profile


def measured_speed(root: Path, profile: dict) -> tuple[dict, Path | None]:
    """Adopt a validated native speed profile only for identical target weights."""
    from rlm.v100.mtp_gate import valid
    from rlm.v100.protection import file_hash

    paths = sorted(
        (root / "research/logs").glob("mtp-ab-*/validated-profile.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    current_hash = None
    for path in paths:
        candidate = load_profile(path, root)
        if not valid(candidate):
            continue
        if current_hash is None:
            current_hash = file_hash(Path(profile["server"]["model"]))
        if file_hash(Path(candidate["server"]["model"])) != current_hash:
            continue
        chosen = copy.deepcopy(profile)
        chosen["server"] = candidate["server"]
        chosen["runtime"] = candidate["runtime"]
        chosen.setdefault("resources", {})["mtp_validation"] = candidate["resources"][
            "mtp_validation"
        ]
        # Validation binds runtime/native settings, while training ancestry and
        # both accepted lineages continue to come from the serving checkpoint.
        if not valid(chosen):
            raise ValueError("Merged measured configuration failed its bound speed/quality proof")
        return chosen, path
    return profile, None


def choose_profile(root: Path) -> Path:
    from rlm.v100.mission import status
    from rlm.v100.mtp_gate import valid

    candidates = []
    previous = status(root)
    live = previous.get("learning", {}).get("live_profile")
    if live and Path(live).exists():
        # Serving checkpoints take precedence over newly generated speed-test profiles.
        return Path(live)
    for path in (root / "research/logs").glob("mtp-ab-*/validated-profile.json"):
        if valid(load_profile(path, root)):
            candidates.append(path)
    prepared = root / "research/prepared-mission.json"
    if prepared.exists():
        candidates.append(Path(json.loads(prepared.read_text())["profile"]))
    candidates.extend((root / "research/preparation").glob("run-*/profile-output8192.json"))
    baseline = root / "research/v100-baseline.json"
    if baseline.exists():
        candidates.append(baseline)
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        raise ValueError("No previous measured/prepared profile. Pass --profile explicitly")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def prepare(
    root: Path,
    path: Path | None = None,
    desktop: bool = True,
    benchmarks: bool = True,
    paper: bool | None = None,
) -> Path:
    from rlm.v100.mission import status
    from rlm.v100.public_benchmarks import current
    from rlm.v100.public_benchmarks import prepare as prepare_benchmarks
    from rlm.v100.remote_helper import canonicalize_remote
    from rlm.v100.self_code import prepare as prepare_code

    if status(root)["running"]:
        raise ValueError("Stop the owned mission before changing its setup")
    source = path or choose_profile(root)
    profile = load_profile(source, root)
    profile, speed_path = measured_speed(root, profile)
    folder = root / "research/campaign" / f"run-{time.time_ns()}"
    folder.mkdir(parents=True)
    stages = {}

    def stage(name, function):
        print(json.dumps({"campaign": name, "state": "starting"}), flush=True)
        try:
            value = function()
            stages[name] = {"state": "ready", "result": str(value)}
        except Exception as error:
            stages[name] = {
                "state": "deferred",
                "error": type(error).__name__,
                "detail": str(error)[:600],
            }
        atomic_json(folder / "stages.json", stages)
        print(json.dumps({"campaign": name, **stages[name]}), flush=True)

    stage("helper-identity", lambda: canonicalize_remote(root))
    stage("own-source-and-cpu-sandbox", lambda: prepare_code(root))
    if benchmarks:
        stage("official-public-benchmarks", lambda: current(root) or prepare_benchmarks(root, 20))
    if desktop:
        from rlm.v100.desktop import prepare as prepare_desktop

        stage("private-desktop", lambda: prepare_desktop(root))
    resources = profile.setdefault("resources", {})
    resources.update(
        fresh_audit_required=True,
        paper_research_enabled=resources.get("paper_research_enabled", False)
        if paper is None
        else paper,
        resident_drones=True,
        independent_branch_serving=True,
        interactive_lab=True,
        public_benchmarks=bool(current(root)),
        mission_max_context=131072,
    )
    destination = folder / "profile.json"
    atomic_json(destination, profile)
    atomic_json(
        root / "research/campaign/current.json",
        {
            "profile": str(destination),
            "source": str(source),
            "speed_profile": str(speed_path) if speed_path else None,
            "stages": stages,
        },
    )
    return destination
