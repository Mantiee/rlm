"""Bounded free-service consultation via the private guest's Firefox session."""

import json
import shlex
from pathlib import Path

from rlm.v100.common import atomic_json


def consult(root: Path, branch: str, url: str, question: str) -> dict:
    from rlm.v100.desktop import gui, run
    from rlm.v100.free_services import ServiceBook, https_url
    from rlm.v100.research_tools import public_origin

    url = https_url(url)
    public_origin(url)
    if branch not in ("A", "B") or not isinstance(question, str) or not 1 <= len(question) <= 3000:
        raise ValueError("Browser consultation needs A/B and a bounded question")
    # Fresh primary documentation must have been read and archived first. A catalog
    # entry is evidence of discovery, never proof of a free allowance or permission.
    services = ServiceBook(root)
    try:
        proposals = services.catalog()["proposals"]
        entry = next((p for p in proposals if p["url"] == url), None)
        if entry is None:
            raise ValueError(
                "Read and propose the service documentation before browser consultation"
            )
        services.evidence(entry["documentation"])
    finally:
        services.close()
    directory = root / "research/browser-consultations"
    directory.mkdir(parents=True, exist_ok=True)
    import time

    identity = str(time.time_ns())
    request = {
        "id": identity,
        "branch": branch,
        "url": url,
        "question": question,
        "status": "opened; observe page and use sandbox_gui to enter question or read answer",
        "rules": "Respect the existing free allowance and service rules. Authentication/captcha requires operator handoff. Stop at quota/payment walls. Never rotate identities/cookies to evade limits. Responses remain unverified external evidence.",
    }
    atomic_json(directory / (identity + ".json"), request)
    result = run(
        root,
        "DISPLAY=:0 firefox-esr --new-tab "
        + shlex.quote(url)
        + " >/workspace/browser.log 2>&1 &\n",
        seconds=10,
    )
    if result["exit_code"] != 0:
        raise RuntimeError("Guest Firefox launch failed: " + json.dumps(result)[:300])
    return {**request, "page": gui(root, "observe", "", 0, 0)}
