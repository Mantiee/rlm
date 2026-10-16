"""Income work reports host-classified execution, not a count of successful reads."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlm.v100 import income_work, research_contract, researchers
from tests.test_v100_recovery import financial_world


def classified(phase, **changes):
    return {
        "id": "quality-identity",
        "state": phase,
        "work_receipts": [],
        "process_receipts": [],
        "forecast_commitments": [],
        **changes,
    }


def execute(root, monkeypatch, result, previous=None):
    goal, _ = financial_world(root)
    monkeypatch.setattr(
        income_work, "accepted_master", lambda *a: SimpleNamespace(sampling_args={})
    )
    calls = []

    def research(client, branch, job, context, path):
        calls.append(context)
        return copy.deepcopy(result)

    monkeypatch.setattr(researchers, "research_task", research)
    monkeypatch.setattr(
        research_contract,
        "assessment",
        lambda *a, **k: pytest.fail("Do not increment assessment/dedup twice"),
    )
    monkeypatch.setattr(
        income_work, "native_turn", lambda *a, **k: pytest.fail("No additional model generation")
    )
    receipt = income_work.execute_goal_work(
        root, {"id": "job", "branch": "A"}, goal, previous or {}
    )
    assert len(calls) == 1
    return receipt, calls


@pytest.mark.parametrize(
    "phase",
    [
        "advisory only - no executed test",
        "proposal registered - no executed test",
        "repeated advisory",
        "rejected test",
    ],
)
def test_reads_and_proposals_never_become_experiment_completion(tmp_path, monkeypatch, phase):
    result = {
        "research_quality": classified(phase),
        "research_trace": [
            {"tool": "read_public_page", "result": {"text": "terms"}},
            {"tool": "calculate", "result": {"value": 123}},
            {"tool": "register_income_opportunity", "result": {"id": "proposal"}},
        ],
    }
    receipt, _ = execute(tmp_path, monkeypatch, result)
    assert receipt["state"] == phase
    assert receipt["successful_tool_receipts"] == 3
    assert receipt["executed_work_receipts"] == 0
    assert receipt["forecast_commitments"] == []
    assert receipt["actual_income_pln"] is None
    assert "activity only" in receipt["successful_tool_receipts_scope"]
    assert json.loads(Path(receipt["result_path"]).read_text())["result"] == result


def test_archived_process_remains_executed_but_unvalidated(tmp_path, monkeypatch):
    quality = classified(
        "process executed - task validation missing",
        process_receipts=[
            {
                "kind": "executed-unvalidated",
                "directory": "/archived/work",
                "receipt_sha256": "a" * 64,
            }
        ],
    )
    result = {
        "research_quality": quality,
        "research_trace": [
            {"tool": "run_research_python", "result": {"exit_code": 0, "timed_out": False}}
        ],
    }
    receipt, _ = execute(tmp_path, monkeypatch, result)
    assert receipt["state"] == quality["state"]
    assert receipt["executed_work_receipts"] == 1
    assert receipt["research_quality"] == quality
    assert receipt["execution_checkpoint"]["quality_id"] == quality["id"]
    assert "not task validation or income" in receipt["execution_checkpoint"]["scope"]
    assert receipt["actual_income_pln"] is None


def test_forecast_checkpoint_keeps_commitment_pending(tmp_path, monkeypatch):
    quality = classified(
        "forecast committed - outcome pending", forecast_commitments=["future-commitment"]
    )
    receipt, _ = execute(tmp_path, monkeypatch, {"research_quality": quality, "research_trace": []})
    assert receipt["state"] == quality["state"]
    assert receipt["forecast_commitments"] == ["future-commitment"]
    assert receipt["executed_work_receipts"] == 0
    assert receipt["execution_checkpoint"]["result_path"] == receipt["result_path"]


@pytest.mark.parametrize(
    "quality",
    [
        None,
        {},
        {"state": "profitable"},
        classified("executed research receipt", work_receipts="not a list"),
    ],
)
def test_missing_or_malformed_classification_cannot_promote_tool_activity(
    tmp_path, monkeypatch, quality
):
    receipt, _ = execute(
        tmp_path,
        monkeypatch,
        {
            "research_quality": quality,
            "research_trace": [{"tool": "sandbox_run", "result": {"exit_code": 0}}],
        },
    )
    assert receipt["state"] == "advisory only; execution classification unavailable"
    assert receipt["executed_work_receipts"] == 0 and receipt["actual_income_pln"] is None


def test_next_work_receives_prior_checkpoint_and_original_classification(tmp_path, monkeypatch):
    first, _ = execute(
        tmp_path,
        monkeypatch,
        {
            "research_quality": classified("proposal registered - no executed test"),
            "research_trace": [],
        },
    )
    second, calls = execute(
        tmp_path,
        monkeypatch,
        {"research_quality": classified("advisory only - no executed test"), "research_trace": []},
        previous=first,
    )
    context = calls[0][0]
    assert context["previous_work"]["execution_checkpoint"] == first["execution_checkpoint"]
    assert context["previous_experiment_evidence"]["research_quality"] == first["research_quality"]
    assert Path(first["result_path"]).exists() and Path(second["result_path"]).exists()
