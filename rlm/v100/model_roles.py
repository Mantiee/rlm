"""Locally checked role admission and routing; model size alone never wins a role."""

import argparse
import copy
import hashlib
import json
import time
from decimal import Decimal, InvalidOperation
from itertools import islice, permutations, product
from pathlib import Path

import requests

from rlm.v100.common import atomic_json, load_profile
from rlm.v100.model_policy import WINDOWS_ROLES, admitted_panel, retired
from rlm.v100.remote_helper import MODEL as MODEL
from rlm.v100.remote_helper import ORDERED_METADATA, ROLE_MODELS, STABLE_METADATA, metadata_sha


def json_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def offload_allowed(model: str) -> bool:
    # The retained v92 worker only supports expert offload for the MoE role.
    # A dense planner is separately qualified with full GPU residency.
    return model == "gemma4:26b-a4b-it-q4_K_M"


def needs_compatible_profile(profile: dict) -> bool:
    return profile.get("runtime", {}).get("model_name") == "qwen3.8:27b-q4_K_M" and (
        profile.get("resources", {}).get("allow_cpu_offload") is True
        or profile.get("runtime", {}).get("context_window", 0) < 65536
        or profile.get("resources", {}).get("metadata_hash_scheme") != ORDERED_METADATA
    )


def parameter_orders(text: str, prefix: str = ""):
    """Permute distinct parameter groups; repeated values retain their order."""
    lines = text.splitlines(keepends=True)
    groups = {}
    indices = []
    quoted = False
    for index, line in enumerate(lines):
        if not quoted and line.strip() and (not prefix or line.startswith(prefix)):
            key = line[len(prefix) :].split(maxsplit=1)[0]
            groups.setdefault(key, []).append(line.rstrip("\r\n"))
            indices.append(index)
        if line.count('"""') % 2:
            quoted = not quoted
    if len(groups) > 6:
        yield text
        return
    for order in permutations(groups):
        result = list(lines)
        values = [line for key in order for line in groups[key]]
        for index, line in zip(indices, values, strict=True):
            ending = lines[index][len(lines[index].rstrip("\r\n")) :]
            result[index] = line + ending
        yield "".join(result)


def compatible_metadata(root: Path, resources: dict, info: dict) -> dict:
    """Prove equivalence to the old cryptographic pin before canonical rebinding."""
    scheme = resources.get("metadata_hash_scheme", STABLE_METADATA)
    expected = resources["metadata_sha256"]
    if metadata_sha(info, scheme) == expected:
        return {"state": "existing-pin-matched", "previous_scheme": scheme}
    snapshot = resources.get("metadata_snapshot")
    if snapshot:
        path = (root / snapshot).resolve()
        if not path.is_relative_to(root.resolve() / "research"):
            raise ValueError("Pinned metadata snapshot escaped the owned research folder")
        previous = json.loads(path.read_text()) if path.is_file() else {}
        if metadata_sha(previous, scheme) == expected:
            if metadata_sha(previous, ORDERED_METADATA) == metadata_sha(info, ORDERED_METADATA):
                return {"state": "parameter-order-proven", "previous_scheme": scheme}
            changed = [
                key
                for key in sorted(set(previous) | set(info))
                if key != "modified_at"
                and metadata_sha({key: previous.get(key)}, ORDERED_METADATA)
                != metadata_sha({key: info.get(key)}, ORDERED_METADATA)
            ]
            raise ValueError("Pinned model metadata changed; fields=" + str(changed))
    # Old role admission failed to save the candidate's /api/show snapshot.
    # Recover only when a reordering of today's identical parameter values
    # reproduces the exact old hash. No tokenizer/template/digest is relaxed.
    parameters, modelfile = info.get("parameters", ""), info.get("modelfile", "")
    if (
        scheme == STABLE_METADATA
        and isinstance(parameters, str)
        and isinstance(modelfile, str)
        and len(parameters.encode()) + len(modelfile.encode()) <= 32768
        and len(json.dumps(info).encode()) <= 256 * 1024
    ):
        variants = product(
            parameter_orders(parameters),
            parameter_orders(modelfile, "PARAMETER "),
        )
        for attempt, (parameters, modelfile) in enumerate(islice(variants, 4096), 1):
            candidate = dict(info)
            if "parameters" in info:
                candidate["parameters"] = parameters
            if "modelfile" in info:
                candidate["modelfile"] = modelfile
            if metadata_sha(candidate, scheme) == expected:
                return {
                    "state": "legacy-pin-reproduced",
                    "previous_scheme": scheme,
                    "attempts": attempt,
                }
    raise ValueError(
        "Compatibility migration cannot change pinned tokenizer/template metadata; exact legacy pin not reproduced"
    )


def selected_profile(root: Path, role: str) -> Path | None:
    if role not in ("fast", "planner"):
        raise ValueError("Unknown measured model role")
    registry = root / "research/model-roles/active.json"
    if not registry.exists():
        return None
    entry = json.loads(registry.read_text()).get("roles", {}).get(role)
    if entry is None:
        return None
    profile = Path(entry["profile"]).resolve()
    report_path = Path(entry["report"]).resolve()
    folder = (root / "research/model-roles").resolve()
    if not profile.is_relative_to(folder) or not report_path.is_relative_to(folder):
        raise ValueError("Role registry escaped its owned folder")
    report = json.loads(report_path.read_text())
    value = json.loads(profile.read_text())
    # The operator excludes gpt-oss, including profiles admitted by an older release.
    if retired(value.get("runtime", {}).get("model_name", "")):
        return None
    if (
        entry["profile_sha256"] != json_hash(value)
        or entry["report_sha256"] != json_hash(report)
        or not report.get("admitted")
        or report["role"] != role
        or report["model_digest"] != value["resources"]["model_digest"]
    ):
        raise ValueError("Role identity or local admission evidence changed")
    return profile


def cases() -> list[dict]:
    rows = [
        {
            "id": f"arithmetic-{x}",
            "prompt": f"Compute {x} * {x + 3}. Return a JSON object with answer as a string containing only the integer.",
            "expected": str(x * (x + 3)),
            "kind": "arithmetic",
        }
        for x in range(11, 23)
    ]
    rows += [
        {
            "id": "fee-units",
            "prompt": "A fee is 50 basis points. Express it as percent, number only in JSON answer.",
            "expected": "0.5",
            "kind": "accounting",
        },
        {
            "id": "equity-not-cash",
            "prompt": "Cash is 0, held quantity is 10 and mark price is 100. What is cash plus marked position value? JSON answer, number only.",
            "expected": "1000",
            "kind": "accounting",
        },
        {
            "id": "buy-friction",
            "prompt": "A simulation buys with 1000 cash at mark price 100. Buy fill price is 101 and fee is 0. No market move. Give marked equity, rounded to 2 decimals, in JSON answer as number only.",
            "expected": "990.10",
            "kind": "accounting",
        },
        {
            "id": "zero-fills",
            "prompt": "Does a paper strategy with zero executed fills establish profitable trading? Return JSON answer yes or no.",
            "expected": "no",
            "kind": "evidence",
        },
        {
            "id": "archived-price",
            "prompt": "An archived encyclopedia article reports a price from 2020. Is it verified executable market pricing in 2026? Return JSON answer yes or no.",
            "expected": "no",
            "kind": "evidence",
        },
        {
            "id": "process-vs-result",
            "prompt": "A process printed success but generated no trades, invoice or payment. Is actual income verified? Return JSON answer yes or no.",
            "expected": "no",
            "kind": "evidence",
        },
        {
            "id": "drawdown",
            "prompt": "Equity sequence is 100,120,90. Give maximum peak-to-trough drawdown in percent. JSON answer, number only.",
            "expected": "25",
            "kind": "accounting",
        },
        {
            "id": "basis-point-profit",
            "prompt": "Convert 95 basis points into percent. JSON answer, number only.",
            "expected": "0.95",
            "kind": "accounting",
        },
    ]
    return rows


def score_answer(answer, expected: str, kind: str) -> bool:
    if kind in ("arithmetic", "accounting"):
        try:
            return Decimal(str(answer).strip()) == Decimal(expected)
        except (InvalidOperation, ValueError):
            return False
    return str(answer).strip().lower() == expected


def evaluate(client) -> dict:
    from rlm.v100.agent import native_turn
    from rlm.v100.tool_protocol import json_object

    rows = []
    production_timeout = client.timeout if hasattr(client, "timeout") else None
    for index, case in enumerate(cases()):
        if production_timeout is not None:
            client.timeout = min(production_timeout, 120 if index == 0 else 30)
        started = time.monotonic()
        for attempt in range(3):
            try:
                message = native_turn(
                    client,
                    [{"role": "user", "content": case["prompt"]}],
                    response_format={
                        "type": "json_object",
                        "schema": {
                            "type": "object",
                            "required": ["answer"],
                            "properties": {"answer": {"type": ["string", "number"]}},
                        },
                    },
                )
                break
            except requests.HTTPError as error:
                code = error.response.status_code if error.response is not None else None
                if (
                    code not in (502, 503)
                    or attempt == 2
                    or time.monotonic() + 2 >= getattr(client, "request_deadline", float("inf"))
                ):
                    raise
                # This is a read-only fixed math/evidence probe. Retry transport
                # admission, never a scored answer or a tool/action request.
                time.sleep(attempt + 1)
        value = json_object(message.get("content"))
        rows.append(
            {
                "id": case["id"],
                "kind": case["kind"],
                "answer": value.get("answer"),
                "passed": score_answer(value.get("answer"), case["expected"], case["kind"]),
                "seconds": time.monotonic() - started,
                "transport_attempts": attempt + 1,
            }
        )
    return {
        "passed": sum(row["passed"] for row in rows),
        "total": len(rows),
        "seconds": sum(row["seconds"] for row in rows),
        "critical_passed": all(row["passed"] for row in rows if row["kind"] != "arithmetic"),
        "cases": rows,
        "scope": "Finite local arithmetic/accounting/evidence admission panel; not proof of profitable strategies or broad intelligence",
    }


def better(role: str, baseline: dict, candidate: dict) -> bool:
    if not candidate["critical_passed"] or candidate["passed"] < baseline["passed"]:
        return False
    if role == "fast":
        return candidate["seconds"] <= baseline["seconds"] * 0.9
    return (
        candidate["passed"] > baseline["passed"]
        or candidate["seconds"] <= baseline["seconds"] * 0.9
    )


def qualify(root: Path) -> dict:
    import fcntl

    folder = root / "research/model-roles"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "qualification.lock").open("a") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {
                "stage": "already-running",
                "roles": {},
                "scope": "Another owned qualification holds the GPU admission lease",
            }
        try:
            return qualify_locked(root)
        finally:
            fcntl.flock(lease, fcntl.LOCK_UN)


def qualify_locked(root: Path) -> dict:
    from rlm.v100.competition import helper_client

    original = root / "research/researcher-rtx3090.json"
    baseline_profile = load_profile(original, root)
    baseline_profile["resources"]["calibration_only"] = True
    inventory = helper_client(baseline_profile, root).remote_request("/api/tags")
    fingerprint = json_hash(
        {
            "profile": baseline_profile,
            "models": sorted((m["name"], m["digest"]) for m in inventory["models"]),
            "panel": cases(),
            "revision": "roles-v100-canonical-metadata",
        }
    )
    cache = root / "research/model-roles/qualification-cache.json"
    active = root / "research/model-roles/active.json"
    locked = {}
    migrations = {}
    if active.is_file():
        for role in ("fast", "planner"):
            selected = selected_profile(root, role)
            if selected is not None:
                pinned = json.loads(selected.read_text())
                if needs_compatible_profile(pinned):
                    migrations[role] = pinned
                else:
                    locked[role] = json.loads(active.read_text())["roles"][role]
        if len(locked) == 2:
            return json.loads(active.read_text())
    if cache.is_file() and active.is_file():
        prior_cache = json.loads(cache.read_text())
        if (
            prior_cache.get("fingerprint") == fingerprint
            and time.time() - prior_cache.get("updated", 0) < 21600
        ):
            selected_profile(root, "fast")
            selected_profile(root, "planner")
            return json.loads(active.read_text())
    folder = root / "research/model-roles" / str(time.time_ns())
    folder.mkdir(parents=True)
    production_output = baseline_profile["runtime"]["max_output_tokens"]
    baseline_profile["resources"]["on_demand"] = True
    baseline_profile["runtime"].update(max_output_tokens=256, enable_thinking=False)
    baseline = {
        "criterion": "All 20 accounting/arithmetic/evidence cases; no historical model inference",
        "measured": False,
    }
    roles = {}
    reports = []
    for role, model in WINDOWS_ROLES:
        if role in locked:
            continue
        if role in migrations and model != migrations[role]["runtime"]["model_name"]:
            continue
        profile = copy.deepcopy(migrations.get(role, baseline_profile))
        profile["resources"].pop("calibration_only", None)
        context = 65536 if model == "qwen3.8:27b-q4_K_M" else 32768
        profile["runtime"].update(
            model_name=model, model_version="role-" + model, context_window=context
        )
        profile["server"]["context_per_slot"] = context
        profile["resources"].update(
            throughput_mode=True,
            helper_batch_tokens=256,
            helper_duty_percent=100,
            max_vram_gib=ROLE_MODELS[model][1],
            on_demand=True,
            allow_cpu_offload=offload_allowed(model),
            require_adaptive_windows_guard=True,
        )
        report = {
            "role": role,
            "model": model,
            "admitted": False,
            "baseline": baseline,
            "context_window": context,
        }
        try:
            probe = helper_client(baseline_profile, root)
            matches = [
                row
                for row in probe.remote_request("/api/tags")["models"]
                if row.get("name") == model
            ]
            if len(matches) != 1:
                raise ValueError("Candidate not downloaded on Windows")
            if (
                role in migrations
                and matches[0]["digest"] != migrations[role]["resources"]["model_digest"]
            ):
                raise ValueError("Compatibility migration cannot change the pinned model digest")
            profile["resources"]["model_digest"] = matches[0]["digest"]
            profile["resources"]["metadata_sha256"] = ""
            profile["resources"]["metadata_hash_scheme"] = ORDERED_METADATA
            client = helper_client(profile, root)
            info = client.identity()
            snapshot = folder / (model.replace(":", "-") + ".show.json")
            atomic_json(snapshot, info)
            report["metadata_snapshot"] = str(snapshot)
            if role in migrations:
                report["metadata_migration"] = compatible_metadata(
                    root, migrations[role]["resources"], info
                )
            profile["resources"]["metadata_sha256"] = metadata_sha(info, ORDERED_METADATA)
            profile["resources"]["metadata_snapshot"] = str(snapshot.relative_to(root))
            client = helper_client(profile, root)
            client.timeout = min(client.timeout, 120)
            client.request_deadline = time.monotonic() + 360
            score_key = json_hash(
                {
                    "profile": {
                        **profile,
                        "resources": {
                            key: value
                            for key, value in profile["resources"].items()
                            if key != "metadata_snapshot"
                        },
                    },
                    "panel": cases(),
                }
            )
            score_path = root / "research/model-roles/scored-panels" / (score_key + ".json")
            cached_score = json.loads(score_path.read_text()) if score_path.is_file() else {}
            if (
                cached_score.get("profile_sha256") == score_key
                and 0 <= time.time() - cached_score.get("updated", 0) < 21600
                and not admitted_panel(cached_score.get("measurement", {}))
            ):
                measurement = cached_score["measurement"]
                report["measurement_cached"] = True
            else:
                measurement = evaluate(client)
                atomic_json(
                    score_path,
                    {
                        "profile_sha256": score_key,
                        "measurement": measurement,
                        "updated": time.time(),
                    },
                )
            report.update(
                model_digest=profile["resources"]["model_digest"],
                measurement=measurement,
                admitted=admitted_panel(measurement),
            )
            if report["admitted"] and (
                role not in roles or measurement["seconds"] < roles[role]["seconds"]
            ):
                profile_path = folder / (
                    model.replace(":", "-").replace("_", "-") + ".profile.json"
                )
                report_path = profile_path.with_suffix(".report.json")
                profile["runtime"]["max_output_tokens"] = max(
                    production_output, 1024 if role == "fast" else 4096
                )
                memory = profile["memory"]
                maximum_chunks = max(1, (32768 - 1024) // memory["chunk_tokens"])
                memory["fanout"] = min(memory["fanout"], maximum_chunks)
                memory["retrieve_count"] = min(memory["retrieve_count"], maximum_chunks)
                if role == "planner":
                    memory["fanout"] = min(memory["fanout"], 4)
                    memory["retrieve_count"] = min(memory["retrieve_count"], 4)
                atomic_json(profile_path, profile)
                atomic_json(report_path, report)
                roles[role] = {
                    "profile": str(profile_path),
                    "report": str(report_path),
                    "profile_sha256": json_hash(profile),
                    "report_sha256": json_hash(report),
                    "seconds": measurement["seconds"],
                }
        except (OSError, ValueError, RuntimeError, requests.RequestException) as error:
            report["error"] = str(error)
        reports.append(report)
        atomic_json(
            folder / "qualification.json",
            {"baseline": baseline, "reports": reports, "roles": roles},
        )
    active = root / "research/model-roles/active.json"
    prior = json.loads(active.read_text()) if active.exists() else {"roles": {}}
    for role, entry in list(prior["roles"].items()):
        profile = Path(entry["profile"]).resolve()
        if not profile.is_relative_to((root / "research/model-roles").resolve()):
            raise ValueError("Role registry escaped its owned folder")
        name = json.loads(profile.read_text()).get("runtime", {}).get("model_name", "")
        if retired(name):
            prior.setdefault("disabled_roles", []).append(
                {"role": role, "entry": entry, "reason": "Operator retired this model family"}
            )
            del prior["roles"][role]
    prior["roles"].update(roles)
    prior.update(
        unresolved_compatibility=[role for role in migrations if role not in roles],
        selection_policy="Pinned after admission; code updates do not change this foundation or tokenizer",
        updated=time.time(),
        latest_qualification=str(folder / "qualification.json"),
        scope="Admitted role profiles only; Gemma learner and accepted weights retained",
    )
    atomic_json(active, prior)
    if "fast" in roles:
        # The shared legacy path is configuration for resident drones. Preserve its
        # previous bytes as evidence, then use only the newly qualified profile.
        atomic_json(folder / "previous-researcher.json", load_profile(original, root))
        atomic_json(original, json.loads(Path(roles["fast"]["profile"]).read_text()))
    elif "planner" in roles and (
        retired(baseline_profile["runtime"]["model_name"])
        or (
            baseline_profile["runtime"]["model_name"]
            == json.loads(Path(roles["planner"]["profile"]).read_text())["runtime"]["model_name"]
            and baseline_profile["resources"]["model_digest"]
            == json.loads(Path(roles["planner"]["profile"]).read_text())["resources"][
                "model_digest"
            ]
        )
    ):
        # Old resident workers read this legacy path. Route them to the same
        # qualified planner instead of an explicitly retired model; do not
        # label it a speed-qualified fast model or change its identity.
        atomic_json(folder / "previous-researcher.json", load_profile(original, root))
        atomic_json(original, json.loads(Path(roles["planner"]["profile"]).read_text()))
        prior["legacy_route"] = {
            "role": "planner",
            "profile": roles["planner"]["profile"],
            "reason": "No admitted fast model; retired legacy model replaced by qualified planner",
        }
        atomic_json(active, prior)
    # Transport failures retry; stable scored outcomes need no repeated GPU exercise.
    if all(
        "error" not in r or r["error"] == "Candidate not downloaded on Windows" for r in reports
    ):
        atomic_json(cache, {"fingerprint": fingerprint, "updated": time.time()})
    return prior


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(qualify(args.root), indent=2))


if __name__ == "__main__":
    main()
