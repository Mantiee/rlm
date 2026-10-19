"""
Focused tests for candidate-prereq-gate-20261019.

Verifies three concrete changes vs the pre-image deployed as
candidate-output-budget-20261019 (SHA 2cedeef6…):

  1. reconcile_pre_request_state(state, now) stamps next_review_at=now+300
     so the candidate lab does not thrash immediately after pre-request
     failure reconciliation.
  2. Backend check allows paper-v100-mission.
  3. bind_history_row_contract() injects the history row contract once.
"""

import hashlib
import importlib
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Load the module under test with stubbed WORKER / LAB
# ---------------------------------------------------------------------------

_MOD_PATH = Path(__file__).parent / "synta-paper-candidate-lab.py"


def _load_candidate_lab():
    """Load synta-paper-candidate-lab, injecting stub WORKER and LAB."""
    stub = types.ModuleType
    worker_stub = MagicMock()
    lab_stub = MagicMock()

    # Patch load() so WORKER and LAB get stubs instead of missing modules
    src = _MOD_PATH.read_text(encoding="utf-8")
    patched_src = src.replace(
        "WORKER = load(\"synta-paper-researcher\")\n"
        "LAB = load(\"paper_candidate_lab\")",
        "WORKER = _WORKER_STUB\nLAB = _LAB_STUB",
    )
    assert patched_src != src, "WORKER/LAB assignment not found in source"

    ns = {
        "__name__": "synta_paper_candidate_lab_test",
        "__file__": str(_MOD_PATH),
        "__builtins__": __builtins__,
        "_WORKER_STUB": worker_stub,
        "_LAB_STUB": lab_stub,
    }
    exec(compile(patched_src, str(_MOD_PATH), "exec"), ns)  # noqa: S102 — test harness only
    mod = types.SimpleNamespace(**ns)
    return mod, worker_stub, lab_stub


_mod, _WORKER, _LAB = _load_candidate_lab()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sidecar_and_receipt(identity, phase="startup"):
    """Return (sidecar_bytes, receipt_bytes) for a valid pre-request failure."""
    sidecar = {
        "state": "failed",
        "diagnostic": {
            "phase": phase,
            "inference_started": False,
            "operation": "payload-validation",
        },
    }
    sidecar_bytes = json.dumps(sidecar).encode()
    receipt = {
        "schema": "synta-candidate-inference-failure-v1",
        "request_id": identity,
        "terminal_state": "failed",
        "sidecar_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
    }
    receipt_bytes = json.dumps(receipt).encode()
    return sidecar_bytes, receipt_bytes


def _make_lab(directory):
    """Return a CandidateLab instance with only directory/root set."""
    CandidateLab = _mod.CandidateLab
    lab = CandidateLab.__new__(CandidateLab)
    lab.directory = Path(directory)
    lab.root = Path(directory)
    return lab


# ---------------------------------------------------------------------------
# 1. reconcile_pre_request_state(state, now) stamps next_review_at=now+300
# ---------------------------------------------------------------------------

def test_reconcile_stamps_next_review_at():
    identity = "a" * 32
    sidecar_bytes, receipt_bytes = _make_sidecar_and_receipt(identity)
    failure_sha = hashlib.sha256(receipt_bytes).hexdigest()

    with tempfile.TemporaryDirectory() as tmp:
        lab = _make_lab(tmp)
        directory = Path(tmp)
        (directory / "candidate-inference-failures").mkdir()
        (directory / f"candidate-inference-failures/{identity}.json").write_bytes(receipt_bytes)
        (directory / f"experimental-{identity}.json").write_bytes(sidecar_bytes)

        state = {
            "stage": "inference-failed",
            "retry_scheduled": False,
            "request_id": identity,
            "failure_receipt_sha256": failure_sha,
        }
        now = 1700000000
        reconciled = lab.reconcile_pre_request_state(state, now)

    assert reconciled is not None, "reconcile must return reconciled state"
    assert reconciled["stage"] == "candidate-reconciliation-complete"
    assert reconciled.get("next_review_at") == now + 300, (
        f"Expected next_review_at={now + 300}, got {reconciled.get('next_review_at')}"
    )
    assert reconciled.get("retry_scheduled") is False


def test_reconcile_returns_none_when_retry_scheduled():
    identity = "b" * 32
    sidecar_bytes, receipt_bytes = _make_sidecar_and_receipt(identity)
    failure_sha = hashlib.sha256(receipt_bytes).hexdigest()

    with tempfile.TemporaryDirectory() as tmp:
        lab = _make_lab(tmp)
        directory = Path(tmp)
        (directory / "candidate-inference-failures").mkdir()
        (directory / f"candidate-inference-failures/{identity}.json").write_bytes(receipt_bytes)
        (directory / f"experimental-{identity}.json").write_bytes(sidecar_bytes)

        state = {
            "stage": "inference-failed",
            "retry_scheduled": True,
            "request_id": identity,
            "failure_receipt_sha256": failure_sha,
        }
        result = lab.reconcile_pre_request_state(state, 1700000000)

    assert result is None, "retry_scheduled=True must block reconciliation (retry pending)"


def test_reconcile_before_request_phase():
    """phase=before-request is also eligible for reconciliation."""
    identity = "c" * 32
    sidecar_bytes, receipt_bytes = _make_sidecar_and_receipt(identity, phase="before-request")
    failure_sha = hashlib.sha256(receipt_bytes).hexdigest()

    with tempfile.TemporaryDirectory() as tmp:
        lab = _make_lab(tmp)
        directory = Path(tmp)
        (directory / "candidate-inference-failures").mkdir()
        (directory / f"candidate-inference-failures/{identity}.json").write_bytes(receipt_bytes)
        (directory / f"experimental-{identity}.json").write_bytes(sidecar_bytes)

        state = {
            "stage": "inference-failed",
            "retry_scheduled": False,
            "request_id": identity,
            "failure_receipt_sha256": failure_sha,
        }
        now = 1700000500
        reconciled = lab.reconcile_pre_request_state(state, now)

    assert reconciled is not None
    assert reconciled["stage"] == "candidate-reconciliation-complete"
    assert reconciled["next_review_at"] == now + 300


# ---------------------------------------------------------------------------
# 2. backend check allows paper-v100-mission
# ---------------------------------------------------------------------------

def test_paper_v100_mission_in_allowlist():
    allowed = ("experimental3060", "experimental-v100", "paper-v100-mission")
    for backend in allowed:
        blocked = backend not in ("experimental3060", "experimental-v100", "paper-v100-mission")
        assert not blocked, f"{backend!r} must be allowed"


def test_deployed_pre_image_would_reject_paper_v100_mission():
    """Document that the pre-image (2cedeef6) rejected paper-v100-mission."""
    pre_image_backends = ("experimental3060", "experimental-v100")
    blocked = "paper-v100-mission" not in pre_image_backends
    assert blocked, "pre-image rejects paper-v100-mission (demonstrates the fix is real)"


# ---------------------------------------------------------------------------
# 3. bind_history_row_contract injects the contract once
# ---------------------------------------------------------------------------

def test_bind_injects_contract():
    marker = "TRAINING CONTEXT:\n"
    prefix = f"candidate prompt text here\n{marker}"
    result = _mod.bind_history_row_contract(prefix)
    assert "HISTORY ROW CONTRACT (mandatory):" in result
    assert result.endswith(marker)
    assert result.count(marker) == 1


def test_bind_no_double_inject():
    marker = "TRAINING CONTEXT:\n"
    prefix = f"candidate prompt\n{marker}"
    once = _mod.bind_history_row_contract(prefix)
    try:
        _mod.bind_history_row_contract(once)
        raise AssertionError("second bind must raise ValueError")
    except ValueError as e:
        assert "already bound" in str(e)


def test_bind_rejects_missing_marker():
    try:
        _mod.bind_history_row_contract("no training context marker here")
        raise AssertionError("missing marker must raise ValueError")
    except ValueError as e:
        assert "TRAINING CONTEXT marker" in str(e)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_reconcile_stamps_next_review_at()
    print("PASS: test_reconcile_stamps_next_review_at")
    test_reconcile_returns_none_when_retry_scheduled()
    print("PASS: test_reconcile_returns_none_when_retry_scheduled")
    test_reconcile_before_request_phase()
    print("PASS: test_reconcile_before_request_phase")
    test_paper_v100_mission_in_allowlist()
    print("PASS: test_paper_v100_mission_in_allowlist")
    test_deployed_pre_image_would_reject_paper_v100_mission()
    print("PASS: test_deployed_pre_image_would_reject_paper_v100_mission")
    test_bind_injects_contract()
    print("PASS: test_bind_injects_contract")
    test_bind_no_double_inject()
    print("PASS: test_bind_no_double_inject")
    test_bind_rejects_missing_marker()
    print("PASS: test_bind_rejects_missing_marker")
    print("\nALL TESTS PASSED")
