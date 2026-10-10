"""Model-selected R&D budgets; fixed quality evaluation never reads this policy."""

import copy
import fcntl
import json
from pathlib import Path

from rlm.v100.common import atomic_json

DEFAULTS = {
    "master": {"thinking": True, "max_tokens": 4096, "batch_tokens": 512},
    "helper": {"thinking": False, "max_tokens": 1024, "batch_tokens": 16},
}


def settings(root: Path) -> dict:
    path = root / "research/research-policy.json"
    return json.loads(path.read_text()) if path.exists() else copy.deepcopy(DEFAULTS)


def choose(root: Path, target: str, thinking: bool, max_tokens: int, batch_tokens: int) -> dict:
    if target not in DEFAULTS or type(thinking) is not bool:
        raise ValueError("Choose master/helper and an explicit thinking boolean")
    ceiling = 8192 if target == "master" else 4096
    if type(max_tokens) is not int or not 256 <= max_tokens <= ceiling:
        raise ValueError(f"Output tokens must be between 256 and {ceiling}")
    batches = (128, 256, 512) if target == "master" else (16,)
    if target == "helper":
        guard_path = root / "research/researcher-rtx3090.json"
        if guard_path.exists():
            guard = json.loads(guard_path.read_text()).get("resources", {})
            if guard.get("require_adaptive_windows_guard") is True:
                batches = (16, 32, 64)
    if type(batch_tokens) is not int or batch_tokens not in batches:
        raise ValueError(
            "Batch outside host budget; Windows adaptive proxy can lower the requested batch"
        )
    lock = root / "research/research-policy.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        policy = settings(root)
        policy[target] = {
            "thinking": thinking,
            "max_tokens": max_tokens,
            "batch_tokens": batch_tokens,
        }
        atomic_json(root / "research/research-policy.json", policy)
    return {
        "target": target,
        **policy[target],
        "scope": "R&D requests only; fixed evaluations and training gates remain unchanged",
        "batch_effect": "next helper request"
        if target == "helper"
        else "proposal only; native V100 batch requires stopped-server benchmark",
    }


def apply(client, root: Path):
    result = copy.copy(client)
    if getattr(client, "activity_actor", "").startswith("chat"):
        # Interactive latency limits must not inherit long research thinking/output budgets.
        return result
    if getattr(client, "research_device", None) == "cpu":
        return result
    target = "helper" if hasattr(result, "helper_batch_tokens") else "master"
    policy = settings(root)[target]
    result.enable_thinking = policy["thinking"]
    result.sampling_args = dict(client.sampling_args)
    result.sampling_args["max_tokens"] = min(policy["max_tokens"], result.context_window // 4)
    ceiling = getattr(client, "research_token_ceiling", None)
    if ceiling is not None:
        result.sampling_args["max_tokens"] = min(result.sampling_args["max_tokens"], ceiling)
    if getattr(client, "research_thinking_allowed", True) is False:
        result.enable_thinking = False
    if target == "helper":
        result.helper_batch_tokens = min(policy["batch_tokens"], client.helper_batch_tokens)
        # Pacing is operator-owned, not a model-selected research budget.
        result.helper_duty_percent = client.helper_duty_percent
    return result
