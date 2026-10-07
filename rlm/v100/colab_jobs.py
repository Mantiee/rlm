"""Interactive Colab proposals; never remote workers or quota bypass."""

import uuid
from pathlib import Path

from rlm.v100.common import atomic_json


def propose(root: Path, branch: str, steps: int, purpose: str) -> dict:
    if (
        type(steps) is not int
        or not 10 <= steps <= 100
        or not isinstance(purpose, str)
        or not 1 <= len(purpose) <= 600
    ):
        raise ValueError("Colab pilot requires 10-100 steps and a short purpose")
    path = root / "research/colab-proposals" / (uuid.uuid4().hex[:12] + ".json")
    value = {
        "branch": branch,
        "steps": steps,
        "purpose": purpose,
        "max_seconds": 120,
        "status": "awaiting interactive user session",
        "notebook": "https://colab.research.google.com/github/Mantiee/rlm/blob/v100-rtx-reconnect/tools/colab-income-pilot.ipynb",
        "scope": "Optional tiny separate-model pilot. No automatic account/session, distributed worker, cookie rotation, paid plan or live-model promotion.",
    }
    atomic_json(path, value)
    return {**value, "proposal": str(path)}
