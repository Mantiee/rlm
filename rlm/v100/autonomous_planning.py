"""Asynchronous model-owned replanning; the operator's long-term goal remains frozen."""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rlm.v100.common import atomic_json, load_profile
from rlm.v100.planning import read, update

POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="synta-planning")
PENDING = {}


def context(root: Path) -> dict:
    value = {"plans": read(root), "evidence": {}}
    for name in ("income-work", "market-research", "goal-learning", "goal-compute"):
        path = root / "research" / name / "status.json"
        if path.is_file() and path.stat().st_size < 2 * 2**20:
            observed = json.loads(path.read_text())
            # Bounded receipts, not a copy of source documents or model reasoning.
            value["evidence"][name] = {
                key: observed[key]
                for key in (
                    "state",
                    "error",
                    "reason",
                    "goal_id",
                    "result_path",
                    "updated",
                    "next_evidence",
                )
                if key in observed
            }
    return value


def run(root: Path) -> dict:
    from rlm.v100.agent import native_turn
    from rlm.v100.competition import helper_client
    from rlm.v100.income_work import accepted_master
    from rlm.v100.mission_chat import backend_preflight
    from rlm.v100.remote_helper import selected_helper
    from rlm.v100.tool_protocol import json_object

    inputs = context(root)
    goal = inputs["plans"].get("long") or {}
    if not goal.get("id"):
        return {"stage": "waiting-for-long-term-goal"}
    candidates = []
    errors = []
    for role in ("planner", "fast"):
        try:
            path = selected_helper(root, role)
            if path not in candidates:
                candidates.append(path)
        except (OSError, ValueError) as error:
            errors.append(str(error)[:200])
    client = None
    for path in candidates:
        try:
            candidate = helper_client(load_profile(path, root), root, "A")
            candidate.timeout = 5
            candidate.request_deadline = time.monotonic() + 60
            backend_preflight(candidate)
            client = candidate
            break
        except (OSError, ValueError, RuntimeError) as error:
            errors.append(str(error)[:200])
    if client is None:
        client = accepted_master(root, "A")
    client.request_deadline = time.monotonic() + 60
    client.timeout = 60
    client.enable_thinking = False
    client.activity_actor = "autonomous-planner"
    client.sampling_args = dict(client.sampling_args) | {"max_tokens": 1024}
    message = native_turn(
        client,
        [
            {
                "role": "system",
                "content": "Replan the next short-term experiment and medium-term milestones to fulfill the exact operator long-term goal. Return English JSON strings short and mid, each under 2000 characters. Each plan needs a measurable success criterion and a concrete next action. Revise repeated failures and losing experiments instead of repeating them. An opportunity ID is not a host observation ID: inspect observed targets before predictions. Separate retrieval, verified learning, simulated P&L and actual income. Existing legal, capital and authorization constraints remain binding. No invented receipts, income or performance. Do not change the long-term goal or claim that planned work ran.",
            },
            {"role": "user", "content": json.dumps(inputs, ensure_ascii=False)},
        ],
        response_format={
            "type": "json_object",
            "schema": {
                "type": "object",
                "required": ["short", "mid"],
                "additionalProperties": False,
                "properties": {
                    key: {"type": "string", "minLength": 1, "maxLength": 2000}
                    for key in ("short", "mid")
                },
            },
        },
    )
    proposed = json_object(message.get("content"))
    if set(proposed) != {"short", "mid"} or any(
        not isinstance(t, str) or not 1 <= len(t.strip()) <= 2000 for t in proposed.values()
    ):
        raise ValueError("Autonomous plan is incomplete or exceeds its budget")
    receipts = [
        update(root, horizon, proposed[horizon], "A", expected_goal_id=goal["id"])
        for horizon in ("short", "mid")
    ]
    return {
        "stage": "replanned",
        "goal_id": goal["id"],
        "receipts": receipts,
        "model": client.model_name,
        "scope": "Model-authored plans, not completed work or verified income",
    }


def tick(root: Path) -> None:
    path = root / "research/plans/autonomous-status.json"
    key = str(root.resolve())
    future = PENDING.get(key)
    if future is not None:
        if not future.done():
            return
        try:
            result = future.result()
        except Exception as error:
            result = {"stage": "planning-blocked", "error": str(error)[:600]}
        previous = json.loads(path.read_text()) if path.is_file() else {}
        atomic_json(
            path, {**previous, **result, "updated": time.time(), "next_poll": time.time() + 300}
        )
        del PENDING[key]
        return
    previous = json.loads(path.read_text()) if path.is_file() else {}
    if time.time() < previous.get("next_poll", 0):
        return
    inputs = context(root)
    # Timestamps alone do not constitute new evidence.
    evidence = {
        name: {k: v for k, v in row.items() if k != "updated"}
        for name, row in inputs["evidence"].items()
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            {"goal": inputs["plans"].get("long"), "evidence": evidence}, sort_keys=True
        ).encode()
    ).hexdigest()
    if (
        previous.get("stage") == "replanned"
        and previous.get("fingerprint") == fingerprint
        and time.time() - previous.get("last_replan", 0) < 1800
    ):
        return
    atomic_json(
        path,
        {
            "stage": "planning",
            "updated": time.time(),
            "fingerprint": fingerprint,
            "last_replan": time.time(),
        },
    )
    PENDING[key] = POOL.submit(run, root)
