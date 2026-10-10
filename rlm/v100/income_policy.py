"""Operator-controlled hypothetical research policy. No spending authorization."""

import json
import re
import time
import unicodedata
from pathlib import Path

from rlm.v100.common import atomic_json


def normalized(message: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", message.casefold()) if not unicodedata.combining(c)
    )


def read(root: Path) -> dict:
    path = root / "research/income-research-policy.json"
    return (
        json.loads(path.read_text())
        if path.exists()
        else {
            "capital_research": False,
            "real_orders": False,
            "spending": False,
            "scope": "Hypothetical research only; no spending, accounts, outreach or real orders",
        }
    )


def setting(message: str) -> bool | None:
    value = normalized(message)
    if "?" in value or re.search(
        r"\b(?:nie zmien|nie ustaw|nie wlacz|do not|dont|przyklad|example|quote|cytat)\b", value
    ):
        return None
    if not re.search(
        r"\b(?:zmien|ustaw|przelacz|wlacz|wylacz|zaloz|zakladaj|masz zalozyc|switch|enable|disable|change|assume)\b", value
    ):
        return None
    if not re.search(r"kapital|capital|deposit|wplat|funded|zero.upfront", value):
        return None
    if re.search(
        r"(?:wylacz|disable).*(?:kapital|capital|funded)|(?:wlacz|enable|switch|przelacz).*(?:zero.upfront|bez wplat)",
        value,
    ):
        return False
    return True


def update(root: Path, message: str) -> dict:
    enabled = setting(message)
    if enabled is None:
        raise ValueError("Current operator message must explicitly set the research policy")
    previous = read(root)
    atomic_json(
        root / "research/income-policy-history" / f"{time.time_ns()}.json",
        {
            "previous": previous,
            "operator_message": message,
        },
    )
    value = {
        **previous,
        "capital_research": enabled,
        "updated": time.time(),
        "real_orders": False,
        "spending": False,
    }
    atomic_json(root / "research/income-research-policy.json", value)
    from rlm.v100.market_research import commission

    value["dispatch"] = commission(root)
    return value
