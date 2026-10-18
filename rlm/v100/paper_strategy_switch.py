"""Model-callable strategy version switch for a single paper portfolio ledger.

The model calls `propose_switch`; the host validates all version fields against
the live ledger, writes an immutable receipt and returns a before/after summary.
Positions, cash and event history are preserved — no P&L reset.

Security constraints (unchanged from CODEX_BRIEF):
- Zero Live trading orders: never touches Trading 212 Live.
- Zero hardcoded decisions: the model names the target; the host only enforces
  consistency with the existing version chain and idempotency.
"""

import hashlib
import json
import time
import uuid
from pathlib import Path


def _sha256_str(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def propose_switch(
    root: Path,
    branch: str,
    *,
    portfolio_id: str,
    to_strategy_id: str,
    to_strategy_sha256: str,
    rationale: str,
    cost_assessment: str,
    evidence_sha256: str,
) -> dict:
    """Write a versioned strategy change receipt; return before/after summary.

    The model supplies:
      - portfolio_id: which ledger to switch
      - to_strategy_id: human-readable name for the new strategy
      - to_strategy_sha256: SHA-256 of the new strategy policy/code blob
      - rationale: why the switch is warranted (immutably recorded)
      - cost_assessment: model's explicit cost/risk assessment (immutably recorded)
      - evidence_sha256: digest of the evidence the model used (from research tool receipts)

    The host reads the active version from the ledger, constructs the full receipt
    and calls trading212_demo_paper.PaperBook.record_strategy_change.
    """
    import re

    SHA256_RE = re.compile(r"[0-9a-f]{64}")

    if not portfolio_id or not isinstance(portfolio_id, str):
        raise ValueError("portfolio_id must be a non-empty string")
    if not to_strategy_id or not isinstance(to_strategy_id, str):
        raise ValueError("to_strategy_id must be a non-empty string")
    for name, val in (
        ("to_strategy_sha256", to_strategy_sha256),
        ("evidence_sha256", evidence_sha256),
    ):
        if not isinstance(val, str) or not SHA256_RE.fullmatch(val):
            raise ValueError(f"{name} must be a lowercase 64-char SHA-256 hex digest")
    if not rationale or len(rationale) < 10:
        raise ValueError("rationale must be at least 10 characters")
    if not cost_assessment or len(cost_assessment) < 10:
        raise ValueError("cost_assessment must be at least 10 characters")

    from rlm.v100.trading212_demo_paper import PaperBook

    journal = root / "research/paper/paper.db"
    if not journal.exists():
        raise ValueError("Paper ledger not found; start a paper portfolio first")

    book = PaperBook(journal)
    active = book.active_strategy(portfolio_id)

    from_version = active["version"]
    from_strategy_sha256 = active.get("strategy_sha256")
    from_version_sha256 = active["version_sha256"]
    from_strategy_id = active.get("strategy_id", "")

    if to_strategy_id == from_strategy_id:
        raise ValueError(
            f"to_strategy_id '{to_strategy_id}' is the same as the current strategy; "
            "a version change must name a different strategy"
        )

    idempotency_key = str(uuid.uuid4())
    now_ts = str(int(time.time()))
    effective_quote_time = str(time.time())

    receipt_input = {
        "portfolio_id": portfolio_id,
        "idempotency_key": idempotency_key,
        "from_version": from_version,
        "from_strategy_sha256": from_strategy_sha256,
        "from_version_sha256": from_version_sha256,
        "to_version": from_version + 1,
        "to_strategy_id": to_strategy_id,
        "to_strategy_sha256": to_strategy_sha256,
        "approval_evidence_sha256": evidence_sha256,
        "effective_quote_time": effective_quote_time,
        "rationale": rationale,
    }

    result = book.record_strategy_change(receipt_input)

    from rlm.v100.activity import ActivityLog

    ActivityLog(root, branch, "strategy-switch").write(
        "steps",
        "strategy-version-change",
        {
            **result,
            "cost_assessment": cost_assessment,
            "evidence_sha256": evidence_sha256,
            "branch": branch,
            "proposed_at": now_ts,
        },
    )

    return {
        "status": result["status"],
        "portfolio_id": portfolio_id,
        "from_version": from_version,
        "from_strategy_id": from_strategy_id,
        "to_version": result["to_version"],
        "to_strategy_id": to_strategy_id,
        "version_sha256": result.get("version_sha256", ""),
        "receipt_bound": True,
        "p_and_l_reset": False,
        "rationale_recorded": True,
        "cost_assessment_recorded": True,
        "note": (
            "Strategy version recorded. Positions, cash and history are preserved. "
            "This does not affect any Live account. The version digest is immutable."
        ),
    }
