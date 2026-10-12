"""Locally checked role admission and routing; model size alone never wins a role."""

import argparse
import copy
import hashlib
import json
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

from rlm.v100.common import atomic_json, load_profile
from rlm.v100.remote_helper import MODEL, ROLE_MODELS, STABLE_METADATA, metadata_sha


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
    for case in cases():
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
    if baseline_profile["runtime"]["model_name"] != MODEL:
        raise ValueError("Role qualification requires the preserved pinned 9B baseline")
    folder = root / "research/model-roles" / str(time.time_ns())
    folder.mkdir(parents=True)
    production_output = baseline_profile["runtime"]["max_output_tokens"]
    baseline_profile["resources"]["on_demand"] = True
    baseline_profile["runtime"].update(max_output_tokens=256, enable_thinking=False)
    baseline = evaluate(helper_client(baseline_profile, root))
    roles = {}
    reports = []
    for role, model in [
        ("fast", "qwen3.5:2b-q4_K_M"),
        ("fast", "qwen3.5:4b-q4_K_M"),
        ("planner", "qwen3.5:35b-a3b-q4_K_M"),
        ("planner", "gpt-oss:20b"),
        ("planner", "gpt-oss:120b"),
    ]:
        profile = copy.deepcopy(baseline_profile)
        profile["runtime"].update(
            model_name=model, model_version="role-" + model, context_window=32768
        )
        profile["server"]["context_per_slot"] = 32768
        profile["resources"].update(
            max_vram_gib=ROLE_MODELS[model][1],
            on_demand=True,
            allow_cpu_offload=role == "planner",
            require_adaptive_windows_guard=True,
        )
        if model.startswith("gpt-oss:"):
            profile["runtime"].update(enable_thinking=True, max_output_tokens=1024)
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
            measurement = evaluate(client)
            report.update(
                model_digest=profile["resources"]["model_digest"],
                measurement=measurement,
                admitted=better(role, baseline, measurement)
                and (
                    model != "gpt-oss:120b"
                    or (
                        measurement["seconds"] <= baseline["seconds"] * 1.25
                        and all(row["seconds"] <= 15 for row in measurement["cases"])
                    )
                ),
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
        except (OSError, ValueError, RuntimeError) as error:
            report["error"] = str(error)
        reports.append(report)
        atomic_json(
            folder / "qualification.json",
            {"baseline": baseline, "reports": reports, "roles": roles},
        )
    active = root / "research/model-roles/active.json"
    prior = json.loads(active.read_text()) if active.exists() else {"roles": {}}
    prior["roles"].update(roles)
    prior.update(
        updated=time.time(),
        latest_qualification=str(folder / "qualification.json"),
        scope="Admitted role profiles only; Gemma learner and accepted weights retained",
    )
    atomic_json(active, prior)
    return prior


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(qualify(args.root), indent=2))


if __name__ == "__main__":
    main()
