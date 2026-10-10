"""Goal-linked, precommitted forecasts over host-observed public numeric outcomes.

Sources and timestamps are host-collected; forecasts never certify their own labels.
This is predictive association, not causal inference or a profit evaluator.
"""

import hashlib
import json
import math
import shutil
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

from rlm.v100.common import atomic_json


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def source_key(url: str) -> str:
    parsed = urlsplit(url)
    return digest([parsed.hostname, parsed.path])


def visible(row: dict, excerpt_limit: int) -> dict:
    return {
        "source": row["url"][:160],
        "acquired": row["acquired"],
        "value": row["value"],
        "excerpt": row["excerpt"][:excerpt_limit],
    }


@contextmanager
def database(root: Path):
    folder = root / "research/goal-learning"
    folder.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(folder / "ledger.sqlite3", timeout=30)
    db.row_factory = sqlite3.Row
    try:
        db.execute(
            "CREATE TABLE IF NOT EXISTS observations (id TEXT PRIMARY KEY, time REAL, data TEXT)"
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS forecasts (id TEXT PRIMARY KEY, due REAL, data TEXT, outcome TEXT)"
        )
        with db:
            yield db
    finally:
        db.close()


def observe(root: Path, url: str, field: str = "") -> dict:
    """Archive publicly fetched evidence, optionally selecting a JSON numeric metric."""
    from rlm.v100.research_tools import TextOnly, download_page

    if not isinstance(field, str) or len(field) > 200:
        raise ValueError("JSON field path exceeds budget")
    with database(root) as db:
        if db.execute("SELECT count(*) FROM observations").fetchone()[0] >= 8192:
            raise ValueError(
                "Goal evidence archive budget exhausted; retain snapshots and review capacity"
            )
    if shutil.disk_usage(root).free < 2 * 2**30:
        raise ValueError("Goal evidence requires 2 GiB of free disk headroom")
    resolved, body = download_page(url, max_bytes=256 * 1024)
    acquired = time.time()
    value = None
    if field:
        selected = json.loads(body)
        for key in field.split("."):
            selected = selected[int(key)] if isinstance(selected, list) else selected[key]
        if (
            type(selected) not in (int, float, str)
            or isinstance(selected, str)
            and len(selected) > 64
        ):
            raise ValueError("Outcome must be a finite JSON number")
        value = float(selected)
        if not math.isfinite(value):
            raise ValueError("Outcome must be a finite JSON number")
        excerpt = body[:4000]
    else:
        parser = TextOnly()
        parser.feed(body)
        excerpt = " ".join(parser.parts)[:4000] or body[:4000]
    row = {
        "url": url,
        "resolved_url": resolved,
        "field": field,
        "value": value,
        "acquired": acquired,
        "body_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "excerpt": excerpt,
    }
    identity = digest(row)
    archive = root / "research/goal-learning/sources" / (identity + ".json")
    atomic_json(archive, {"observation": row, "body": body})
    with database(root) as db:
        db.execute(
            "INSERT OR IGNORE INTO observations VALUES (?, ?, ?)",
            (identity, acquired, json.dumps(row)),
        )
    return {
        "id": identity,
        **row,
        "scope": "Host retrieval time, not proven original publication time",
    }


def observation(root: Path, identity: str) -> dict:
    with database(root) as db:
        row = db.execute("SELECT data FROM observations WHERE id=?", (identity,)).fetchone()
    if row is None:
        raise ValueError("Unknown host observation")
    value = json.loads(row[0])
    archive = json.loads(
        (root / "research/goal-learning/sources" / (identity + ".json")).read_text()
    )
    if (
        digest(value) != identity
        or archive["observation"] != value
        or hashlib.sha256(archive["body"].encode()).hexdigest() != value["body_sha256"]
    ):
        raise ValueError("Observation archive changed")
    return value


def predict(root: Path, branch: str, specification: dict) -> dict:
    from rlm.v100.planning import read

    fields = {
        "question",
        "rationale",
        "evidence",
        "target",
        "horizon_seconds",
        "threshold",
        "probabilities",
    }
    if (
        not isinstance(specification, dict)
        or set(specification) != fields
        or branch not in ("A", "B")
    ):
        raise ValueError("Incomplete forecast specification")
    for key in ("question", "rationale"):
        if not isinstance(specification[key], str) or not 1 <= len(specification[key]) <= 1200:
            raise ValueError("Question and rationale need 1-1200 characters")
    evidence = specification["evidence"]
    horizon = specification["horizon_seconds"]
    threshold = specification["threshold"]
    probabilities = specification["probabilities"]
    if (
        not isinstance(evidence, list)
        or not 1 <= len(evidence) <= 8
        or any(not isinstance(identity, str) for identity in evidence)
        or len(set(evidence)) != len(evidence)
    ):
        raise ValueError("Choose 1-8 distinct host evidence IDs")
    if type(horizon) is not int or not 60 <= horizon <= 604800:
        raise ValueError("Horizon must be 60 seconds to 7 days")
    if type(threshold) not in (float, int) or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Precommit a positive absolute change threshold in target units")
    if (
        not isinstance(probabilities, list)
        or len(probabilities) != 3
        or any(
            type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
            for p in probabilities
        )
        or abs(sum(probabilities) - 1) > 1e-6
    ):
        raise ValueError("Probabilities [down, flat, up] must sum to one")
    if not isinstance(specification["target"], str):
        raise ValueError("Target must be a host observation ID")
    target = observation(root, specification["target"])
    features = [observation(root, identity) for identity in evidence]
    now = time.time()
    if target["value"] is None or not 0 <= now - target["acquired"] <= 60:
        raise ValueError("Target must be a numeric observation fetched within 60 seconds")
    if any(item["acquired"] > now for item in features):
        raise ValueError("Future evidence is forbidden")
    plan = read(root)
    if not plan.get("long"):
        raise ValueError("Operator long-term goal must exist before goal training")
    row = {
        "branch": branch,
        "created": now,
        "due": now + horizon,
        "plan": plan,
        "specification": specification,
    }
    identity = digest(row)
    with database(root) as db:
        db.execute("BEGIN IMMEDIATE")
        active = db.execute(
            "SELECT count(*) FROM forecasts WHERE outcome IS NULL AND due>=?", (now - 300,)
        ).fetchone()[0]
        if active >= 32:
            raise ValueError("At most 32 pending forecasts")
        db.execute(
            "INSERT INTO forecasts VALUES (?, ?, ?, NULL)", (identity, row["due"], json.dumps(row))
        )
    return {"id": identity, **row, "state": "precommitted", "weights_changed": False}


def settle(root: Path, identity: str, observed_id: str) -> dict:
    with database(root) as db:
        db.execute("BEGIN IMMEDIATE")
        item = db.execute("SELECT data, outcome FROM forecasts WHERE id=?", (identity,)).fetchone()
        if item is None:
            raise ValueError("Unknown forecast")
        prediction = json.loads(item[0])
        if digest(prediction) != identity:
            raise ValueError("Forecast commitment changed")
        if item[1] is not None:
            return json.loads(item[1])
        spec = prediction["specification"]
        before, after = observation(root, spec["target"]), observation(root, observed_id)
        if (after["url"], after["field"]) != (before["url"], before["field"]):
            raise ValueError("Outcome series differs from committed target")
        if not prediction["due"] <= after["acquired"] <= prediction["due"] + 300:
            raise ValueError("Outcome outside horizon + 5 minute tolerance; no fabricated label")
        delta = after["value"] - before["value"]
        if not math.isfinite(delta):
            raise ValueError("Observed change is outside the finite scoring range")
        actual = 2 if delta > spec["threshold"] else 0 if delta < -spec["threshold"] else 1
        probabilities = spec["probabilities"]
        brier = sum((p - (index == actual)) ** 2 for index, p in enumerate(probabilities))
        result = {
            "observation": observed_id,
            "actual": actual,
            "delta": delta,
            "brier": brier,
            "flat_baseline_brier": 0 if actual == 1 else 2,
            "uniform_baseline_brier": 2 / 3,
            "correct": probabilities.index(max(probabilities)) == actual,
            "scope": "Observed predictive association; no causal or profit certification",
        }
        db.execute("UPDATE forecasts SET outcome=? WHERE id=?", (json.dumps(result), identity))
    return result


def tick(root: Path) -> None:
    from rlm.v100.activity import ActivityLog

    with database(root) as db:
        pending = list(
            db.execute(
                "SELECT id, data FROM forecasts WHERE outcome IS NULL AND due<=? AND due>=? LIMIT 4",
                (time.time(), time.time() - 300),
            )
        )
    for item in pending:
        prediction = json.loads(item["data"])
        target = observation(root, prediction["specification"]["target"])
        try:
            acquired = observe(root, target["url"], target["field"])
            result = settle(root, item["id"], acquired["id"])
            ActivityLog(root, prediction["branch"], "goal-learning").write(
                "metrics", "forecast-resolved", {"id": item["id"], **result}
            )
        except Exception as error:
            ActivityLog(root, prediction["branch"], "goal-learning").write(
                "errors", "forecast-unresolved", {"id": item["id"], "error": str(error)[:400]}
            )


def records(root: Path, active_only: bool = True) -> list[dict]:
    with database(root) as db:
        items = list(
            db.execute(
                "SELECT id, data, outcome FROM forecasts WHERE outcome IS NOT NULL ORDER BY due"
            )
        )
    from rlm.v100.goals import load_goal

    current_goal = load_goal(root)
    result = []
    for item in items:
        forecast = json.loads(item["data"])
        if active_only and (
            not current_goal or forecast["plan"]["long"]["id"] != current_goal["id"]
        ):
            continue
        if digest(forecast) != item["id"]:
            raise ValueError("Forecast commitment changed")
        saved = json.loads(item["outcome"])
        # Recompute without trusting a model-generated or edited success field.
        before = observation(root, forecast["specification"]["target"])
        after = observation(root, saved["observation"])
        spec = forecast["specification"]
        if (before["url"], before["field"]) != (after["url"], after["field"]) or not forecast[
            "due"
        ] <= after["acquired"] <= forecast["due"] + 300:
            raise ValueError("Invalid outcome provenance")
        delta = after["value"] - before["value"]
        actual = 2 if delta > spec["threshold"] else 0 if delta < -spec["threshold"] else 1
        features = [observation(root, key) for key in spec["evidence"]]
        if (
            any(row["acquired"] > forecast["created"] for row in features)
            or before["acquired"] > forecast["created"]
        ):
            raise ValueError("Training evidence contains future observations")
        # All examples sharing an outcome series stay together in the host split ledger.
        series = digest([source_key(before["url"]), before["field"]])
        result.append(
            {
                "id": "goal-" + item["id"],
                "group": "goal-series-" + series,
                "document_ids": [
                    "goal-source-" + source_key(before["url"]),
                    "goal-series-" + series,
                    *["goal-source-" + source_key(row["url"]) for row in features],
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "task": "Forecast down/flat/up over the stated horizon. Return exactly one word: down, flat or up. This is a labeled observational example, not a causal or income claim.",
                                "goal": forecast["plan"]["long"]["text"][:400],
                                "question": spec["question"][:300],
                                "horizon_seconds": spec["horizon_seconds"],
                                "threshold": spec["threshold"],
                                "target_at_prediction": visible(before, 0),
                                "evidence_at_prediction": [
                                    visible(row, 800 // len(features)) for row in features
                                ],
                            }
                        ),
                    },
                    {"role": "assistant", "content": ["down", "flat", "up"][actual]},
                ],
                "verification": {
                    "kind": "goal_observation",
                    "accepted": True,
                    "forecast_id": item["id"],
                    "goal_id": forecast["plan"]["long"]["id"],
                    "domain": "goal-pattern-" + series,
                },
            }
        )
    return result


def admit(root: Path, candidates: list[dict], profile: dict) -> list[dict]:
    """Token-check canonical examples on CPU before they enter an automatic training pool."""
    if not candidates:
        return []
    from transformers import AutoTokenizer

    from rlm.v100.goals import load_goal
    from rlm.v100.training import encode_record

    goal_id = (load_goal(root) or {}).get("id")
    settings = profile["training"]
    admitted, deferred = [], []
    try:
        tokenizer = AutoTokenizer.from_pretrained(settings["base_model"], local_files_only=True)
    except (OSError, ValueError) as error:
        atomic_json(
            root / "research/goal-learning/admission.json",
            {
                "goal_id": goal_id,
                "admitted": 0,
                "deferred": len(candidates),
                "reason": str(error)[:400],
            },
        )
        return []
    for row in candidates:
        try:
            encode_record(row, tokenizer, settings["max_length"])
        except ValueError as error:
            deferred.append({"id": row["id"], "reason": str(error)[:300]})
        else:
            admitted.append(row)
    atomic_json(
        root / "research/goal-learning/admission.json",
        {
            "goal_id": goal_id,
            "admitted": len(admitted),
            "deferred_count": len(deferred),
            "deferred": deferred[:16],
            "max_length": settings["max_length"],
            "scope": "Examples remain archived; no unmeasured sequence-length increase",
        },
    )
    return admitted


def status(root: Path) -> dict:
    admission = root / "research/goal-learning/admission.json"
    path = root / "research/goal-learning/ledger.sqlite3"
    items, all_items = [], []
    if path.exists():
        db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1)
        db.row_factory = sqlite3.Row
        try:
            db.execute("BEGIN")
            items = list(
                db.execute(
                    "SELECT id, due, data, outcome FROM forecasts ORDER BY due DESC LIMIT 16"
                )
            )
            all_items = list(db.execute("SELECT data, due, outcome FROM forecasts"))
        finally:
            db.close()
    from rlm.v100.goals import load_goal

    goal = load_goal(root)
    active_items = [
        item
        for item in all_items
        if goal and json.loads(item["data"])["plan"]["long"]["id"] == goal["id"]
    ]
    current = [
        item
        for item in items
        if goal and json.loads(item["data"])["plan"]["long"]["id"] == goal["id"]
    ]
    resolved = [json.loads(item["outcome"]) for item in current if item["outcome"]]
    admission_record = json.loads(admission.read_text()) if admission.exists() else None
    if admission_record and admission_record.get("goal_id") != (goal or {}).get("id"):
        admission_record = {
            "state": "historical or unattributed",
            "goal_id": admission_record.get("goal_id"),
            "scope": "Not admission evidence for the active goal",
        }
    return {
        "goal_id": goal["id"] if goal else None,
        "resolved_total": sum(item["outcome"] is not None for item in active_items),
        "pending_total": sum(
            item["outcome"] is None and item["due"] + 300 >= time.time() for item in active_items
        ),
        "expired_total": sum(
            item["outcome"] is None and item["due"] + 300 < time.time() for item in active_items
        ),
        "learning_task": {
            "inputs": "Archived evidence available before prediction",
            "targets": "Independently observed future down/flat/up outcomes",
            "metric": "Brier score and source-disjoint goal development accuracy",
            "training": "Supervised adapter candidates using verified outcomes, including failed predictions",
            "acceptance": "Goal development improvement plus retention and independent audit gates",
            "scope": "Generic numeric forecast task across sources; arbitrary goals need independently verifiable task labels",
        },
        "training_admission": admission_record,
        "forecasts": [
            {
                "id": item["id"],
                "due": item["due"],
                "branch": json.loads(item["data"])["branch"],
                "question": json.loads(item["data"])["specification"]["question"][:200],
                "probabilities": json.loads(item["data"])["specification"]["probabilities"],
                "outcome": json.loads(item["outcome"]) if item["outcome"] else None,
                "state": "resolved"
                if item["outcome"]
                else "expired"
                if item["due"] + 300 < time.time()
                else "pending",
            }
            for item in current
        ],
        "recent_resolved": len(resolved),
        "recent_brier": sum(row["brier"] for row in resolved) / len(resolved) if resolved else None,
        "scope": "Recent observational score only; generalization and causal/profit claims require separate tests",
    }


def development_suite(root: Path, dataset: Path, ledger: Path, destination: Path) -> Path | None:
    """Goal-linked development gate, never advertised as an untouched final audit."""
    from rlm.v100.goals import load_goal
    from rlm.v100.training import load_records

    goal = load_goal(root)
    if not goal or not (root / "research/goal-learning/ledger.sqlite3").exists():
        return None
    _, validation = load_records(dataset, ledger)
    cases = [
        row
        for row in validation
        if row.get("verification", {}).get("kind") == "goal_observation"
        and row["verification"]["goal_id"] == goal["id"]
    ]
    if len(cases) < 4 or len({row["group"] for row in cases}) < 2:
        return None
    cases = sorted(cases, key=lambda row: row["id"])[-16:]
    with destination.open("x") as handle:
        for row in cases:
            handle.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "skill": "goal-pattern",
                        "match": "exact",
                        "messages": row["messages"][:-1],
                        "expected": row["messages"][-1]["content"],
                    }
                )
                + "\n"
            )
    return destination


def verified_gate(folder: Path, parent: Path, candidate: Path) -> dict:
    from rlm.v100.protection import compare_reports, file_hash

    previous = json.loads((folder / "goal-parent.json").read_text())
    current = json.loads((folder / "goal-candidate.json").read_text())
    if previous["model_sha256"] != file_hash(parent) or current["model_sha256"] != file_hash(
        candidate
    ):
        raise ValueError("Goal development reports belong to different weights")
    suite_hash = file_hash(folder / "goal-development.jsonl")
    if any(report["suite_sha256"] != suite_hash for report in (previous, current)):
        raise ValueError("Goal development suite changed")
    gate = compare_reports(previous, current)
    complete = all(
        not row.get("error") and row.get("finish_reason") != "length"
        for report in (previous, current)
        for row in report["cases"]
    )
    gate["passed"] = gate["passed"] and complete
    gate["complete"] = complete
    gate["scope"] = (
        "Goal-linked source-held-out development only; reused validation is not future profit or causal proof"
    )
    gate["goal_passed_cases"] = sum(row["passed"] for row in current["cases"])
    expected = [
        json.loads(line)["expected"]
        for line in (folder / "goal-development.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if not expected or any(value not in ("down", "flat", "up") for value in expected):
        raise ValueError("Invalid goal development labels")
    gate["best_constant_baseline_cases"] = max(
        expected.count(value) for value in ("down", "flat", "up")
    )
    gate["beats_best_constant_baseline"] = (
        gate["goal_passed_cases"] > gate["best_constant_baseline_cases"]
    )
    gate["passed"] = gate["passed"] and gate["beats_best_constant_baseline"]
    return gate
