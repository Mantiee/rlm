"""Explicit controller reads of recorded plans, independent of inference availability."""

import json
import re
from pathlib import Path

from rlm.v100.chat_goals import normalized


def requested(message: str) -> bool:
    text = normalized(message).strip()
    return text in ("/plan", "/plans", "/cel") or bool(
        re.fullmatch(
            r"(?:jaki (?:jest )?(?:twoj |zapisany |aktualny )?(?:cel|plan)|"
            r"pokaz (?:zapisany |aktualny )?(?:cel|plan|plany)|"
            r"what is (?:your |the recorded |the current )?(?:goal|plan)|"
            r"show (?:the recorded |the current )?(?:goal|plan|plans))[?.!]*",
            text,
        )
    )


def respond(root: Path) -> dict:
    from rlm.v100.planning import read

    plans = read(root)
    return {
        "answer": "Recorded plans / Zapisane plany:\n"
        + json.dumps(plans, ensure_ascii=False, indent=2),
        "actions": [],
        "applied": [],
        "tool_receipts": [{"tool": "get_plan", "result": plans, "executor": "controller-readonly"}],
        "responder": {
            "model": "controller-plan-read",
            "inference_used": False,
            "scope": "Read-only controller command; no model generation or model tool use",
        },
    }
