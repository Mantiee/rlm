"""Locally checked role admission and routing; model size alone never wins a role."""

import argparse
import copy
import hashlib
import json
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

import requests

from rlm.v100.common import atomic_json, load_profile
from rlm.v100.model_policy import WINDOWS_ROLES, admitted_panel, retired
from rlm.v100.remote_helper import MODEL as MODEL
from rlm.v100.remote_helper import ROLE_MODELS, STABLE_METADATA, metadata_sha


def json_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


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
        value = json_object(message.get("content"))
        rows.append(
            {
                "id": case["id"],
                "kind": case["kind"],
                "answer": value.get("answer"),
                "passed": score_answer(value.get("answer"), case["expected"], case["kind"]),
                "seconds": time.monotonic() - started,
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
            "revision": "roles-v96",
        }
    )
    cache = root / "research/model-roles/qualification-cache.json"
    active = root / "research/model-roles/active.json"
    locked = {}
    if active.is_file():
        for role in ("fast", "planner"):
            if selected_profile(root, role) is not None:
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
        profile = copy.deepcopy(baseline_profile)
        profile["resources"].pop("calibration_only", None)
        profile["runtime"].update(
            model_name=model, model_version="role-" + model, context_window=32768
        )
        profile["server"]["context_per_slot"] = 32768
        profile["resources"].update(
            throughput_mode=True,
            helper_batch_tokens=256,
            helper_duty_percent=100,
            max_vram_gib=ROLE_MODELS[model][1],
            on_demand=True,
            allow_cpu_offload=role == "planner",
            require_adaptive_windows_guard=True,
        )
        report = {"role": role, "model": model, "admitted": False, "baseline": baseline}
        try:
            probe = helper_client(baseline_profile, root)
            matches = [
                row
                for row in probe.remote_request("/api/tags")["models"]
                if row.get("name") == model
            ]
            if len(matches) != 1:
                raise ValueError("Candidate not downloaded on Windows")
            profile["resources"]["model_digest"] = matches[0]["digest"]
            profile["resources"]["metadata_sha256"] = ""
            profile["resources"]["metadata_hash_scheme"] = STABLE_METADATA
            client = helper_client(profile, root)
            info = client.identity()
            profile["resources"]["metadata_sha256"] = metadata_sha(info, STABLE_METADATA)
            client = helper_client(profile, root)
            client.timeout = min(client.timeout, 120)
            client.request_deadline = time.monotonic() + 360
            measurement = evaluate(client)
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
