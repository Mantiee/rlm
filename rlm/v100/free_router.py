"""Explicit zero-price external consultations; no paid or automatic model route."""

import hashlib
import json
import os
import sqlite3
import time
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

import requests

from rlm.v100.common import atomic_json

ORIGIN = "https://openrouter.ai/api/v1"
KEY_VARIABLE = "V100_FREE_ROUTER_KEY"


def api(method: str, path: str, key: str = "", payload: dict | None = None) -> dict:
    if (method, path) not in (("GET", "/models"), ("GET", "/key"), ("POST", "/chat/completions")):
        raise ValueError("Unsupported free-router capability")
    headers = {"Authorization": "Bearer " + key} if key else {}
    # No caller-controlled endpoint, redirect, proxy or .netrc credentials.
    with requests.Session() as session:
        session.trust_env = False
        with session.request(
            method,
            ORIGIN + path,
            headers=headers,
            json=payload,
            timeout=(10, 90),
            allow_redirects=False,
            stream=True,
        ) as response:
            if response.status_code != 200:
                raise RuntimeError(
                    f"Free consultation unavailable (HTTP {response.status_code}); no paid fallback"
                )
            content = bytearray()
            for chunk in response.iter_content(65536):
                content.extend(chunk)
                if len(content) > 8 * 2**20:
                    raise ValueError("Free-router response exceeds its text budget")
            return json.loads(content)


def free_model(row: dict) -> bool:
    model_id, pricing = row.get("id"), row.get("pricing")
    if (
        not isinstance(model_id, str)
        or not model_id.endswith(":free")
        or not isinstance(pricing, dict)
        or not {"prompt", "completion"} <= pricing.keys()
    ):
        return False
    if "text" not in row.get("architecture", {}).get("output_modalities", []):
        return False
    try:
        return all(
            Decimal(str(price)).is_finite() and Decimal(str(price)) == 0
            for price in pricing.values()
        )
    except (InvalidOperation, ValueError):
        return False


def catalog(query: str = "") -> list[dict]:
    if not isinstance(query, str) or len(query) > 120:
        raise ValueError("Free-model search exceeds its budget")
    rows = api("GET", "/models")["data"]
    return [
        row
        for row in rows
        if free_model(row)
        and query.casefold() in (row["id"] + " " + row.get("name", "")).casefold()
    ]


def public_catalog(query: str = "") -> dict:
    rows = sorted(catalog(query), key=lambda row: row["id"])
    return {
        "models": [
            {
                "id": row["id"],
                "name": row.get("name", row["id"]),
                "context_length": row.get("context_length"),
                "description": row.get("description", "")[:180],
            }
            for row in rows[:12]
        ],
        "matching_models": len(rows),
        "key_configured": bool(os.environ.get(KEY_VARIABLE)),
        "scope": "Live zero-price text-model catalog, not an intelligence ranking or guarantee of available quota; narrow query if truncated",
    }


def reserve_request(root: Path, key: str) -> None:
    path = root / "research/state/free-consultations.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(key.encode()).hexdigest()
    day = datetime.now(UTC).date().isoformat()
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS quotas(day TEXT, identity TEXT, attempts INTEGER NOT NULL, PRIMARY KEY(day,identity))"
        )
        db.execute("BEGIN IMMEDIATE")
        previous = db.execute(
            "SELECT attempts FROM quotas WHERE day=? AND identity=?", (day, identity)
        ).fetchone()
        if previous and previous[0] >= 40:
            raise RuntimeError("Shared free consultation budget exhausted for today")
        db.execute(
            "INSERT INTO quotas VALUES(?,?,1) ON CONFLICT(day,identity) DO UPDATE SET attempts=attempts+1",
            (day, identity),
        )
        # One shared account budget for A/B/testers; no parallel bursts or retries.
        db.execute(
            "CREATE TABLE IF NOT EXISTS cadence(identity TEXT PRIMARY KEY, last_attempt REAL NOT NULL)"
        )
        prior = db.execute(
            "SELECT last_attempt FROM cadence WHERE identity=?", (identity,)
        ).fetchone()
        if prior and time.time() - prior[0] < 5:
            raise RuntimeError("Free consultation cooldown active; retry in a later research turn")
        db.execute("INSERT OR REPLACE INTO cadence VALUES(?,?)", (identity, time.time()))


def consult(root: Path, branch: str, model_id: str, question: str) -> dict:
    blocker = root / "research/state/free-router-blocked.json"
    if blocker.exists():
        raise RuntimeError(
            "Free connector disabled after a cost-guard failure; needs operator review"
        )
    if (
        branch not in ("A", "B")
        or not isinstance(model_id, str)
        or not model_id.endswith(":free")
        or len(model_id) > 200
    ):
        raise ValueError("Only explicit :free model selections are permitted")
    if not isinstance(question, str) or not 1 <= len(question) <= 3000:
        raise ValueError("Free consultation question must contain 1-3000 characters")
    key = os.environ.get(KEY_VARIABLE, "")
    if not key:
        raise RuntimeError(
            "Free connector needs a free-tier OpenRouter account key configured locally"
        )
    if api("GET", "/key", key).get("data", {}).get("is_free_tier") is not True:
        raise ValueError(
            "Free consultation requires an account reported as free tier; funded/unknown account blocked"
        )
    matches = [row for row in catalog(model_id) if row["id"] == model_id]
    if len(matches) != 1:
        raise ValueError("Selected model is absent or no longer has strictly zero pricing")
    reserve_request(root, key)
    start = time.monotonic()
    response = api(
        "POST",
        "/chat/completions",
        key,
        {
            "model": model_id,
            "messages": [{"role": "user", "content": question}],
            "max_tokens": 768,
            "stream": False,
            "temperature": 0,
            "plugins": [],
            "provider": {
                "max_price": {"prompt": 0, "completion": 0, "request": 0},
                "allow_fallbacks": False,
                "require_parameters": True,
                "enforce_distillable_text": True,
            },
        },
    )
    if response.get("model") not in (model_id, model_id.removesuffix(":free")):
        raise ValueError("Free consultation returned a different model; reject its output")
    cost = response.get("usage", {}).get("cost")
    if cost is not None:
        try:
            valid_cost = Decimal(str(cost)).is_finite() and Decimal(str(cost)) == 0
        except (InvalidOperation, ValueError):
            valid_cost = False
        if not valid_cost:
            atomic_json(
                blocker,
                {
                    "reason": "Provider reported nonzero or invalid cost despite zero-price routing",
                    "model": model_id,
                    "time": time.time(),
                },
            )
            raise RuntimeError(
                "Provider cost guard failed; connector disabled pending operator review"
            )
    answer = response["choices"][0]["message"].get("content")
    if not isinstance(answer, str) or not answer or len(answer) > 12000:
        raise ValueError("Free consultation returned no bounded text answer")
    result = {
        "model": model_id,
        "answer": answer[:2000],
        "wall_seconds": time.monotonic() - start,
        "status": "external advisory answer, not verified truth or training approval",
    }
    from rlm.v100.experiments import SharedLab

    shared = SharedLab(root / "research/state/competition.sqlite3")
    try:
        shared.append(
            branch,
            "free-consultation",
            {
                "model": model_id,
                "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                "wall_seconds": result["wall_seconds"],
                "status": result["status"],
            },
        )
    finally:
        shared.close()
    return result
