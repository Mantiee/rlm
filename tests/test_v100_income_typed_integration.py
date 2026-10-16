"""Typed local work precedes broad research and never dead-ends unchanged inputs."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlm.v100 import (
    goal_learning,
    income_experiment,
    income_work,
    research_sandbox,
    research_tools,
    researchers,
)
from rlm.v100.common import atomic_json
from tests.test_v100_recovery import financial_world


@pytest.fixture
def context(tmp_path, monkeypatch):
    goal, _ = financial_world(tmp_path)
    monkeypatch.setattr(
        research_tools,
        "download_page",
        lambda url, max_bytes: (url, "Public free service terms and permitted local preparation."),
    )
    source = goal_learning.observe(tmp_path, "https://example.org/official-terms")
    atomic_json(
        tmp_path / "research/income-work/dossiers" / (source["id"] + ".json"),
        {"goal_id": goal["id"], "source": source, "analysis": None},
    )
    monkeypatch.setattr(
        income_work, "accepted_master", lambda *a: SimpleNamespace(sampling_args={})
    )
    proposal = {
        "hypothesis": "Produce a reproducible terms checklist",
        "code": "from pathlib import Path\nPath('checklist.txt').write_text('Source reviewed; eligibility and payment unverified')",
        "measurement": "Presence of checklist and missing dependencies",
        "failure_condition": "Missing checklist",
        "artifacts": ["checklist.txt"],
        "seconds": 10,
    }
    turns, runs, broad = [], [], []

    def model(*args, **kwargs):
        turns.append(1)
        return {"content": json.dumps(proposal)}

    def child(root, branch, code, seconds):
        runs.append(1)
        folder = root / "research/python-experiments/run-integration-fixture"
        work = folder / "work"
        work.mkdir(parents=True)
        result = subprocess.run(
            [sys.executable, "-I", "-c", code],
            cwd=work,
            capture_output=True,
            text=True,
            timeout=seconds,
        )
        receipt = {
            "directory": str(folder),
            "exit_code": result.returncode,
            "timed_out": False,
            "output": result.stdout,
        }
        atomic_json(folder / "result.json", receipt)
        return receipt

    def research(client, branch, job, observations, root):
        broad.append(observations)
        return {
            "research_trace": [],
            "research_quality": {
                "state": "advisory only - no executed test",
                "work_receipts": [],
                "process_receipts": [],
                "forecast_commitments": [],
            },
        }

    monkeypatch.setattr(income_experiment, "native_turn", model)
    monkeypatch.setattr(research_sandbox, "run", child)
    monkeypatch.setattr(researchers, "research_task", research)
    return tmp_path, goal, source, proposal, turns, runs, broad


def execute(context, previous=None):
    root, goal, *_ = context
    return income_work.execute_goal_work(root, {"id": "job", "branch": "A"}, goal, previous or {})


def test_income_dossier_commissions_real_local_artifact_before_broad_loop(context):
    root, goal, source, _, turns, runs, broad = context
    result = execute(context)
    assert result["mode"] == "model-selected local experiment"
    assert result["state"] == "executed-unvalidated"
    assert result["executed_work_receipts"] == 1 and result["actual_income_pln"] is None
    assert len(turns) == len(runs) == 1 and not broad
    experiment = result["typed_local_experiment"]["experiment"]
    assert Path(experiment["artifacts"][0]["path"]).read_text().startswith("Source reviewed")
    assert experiment["inputs"][0]["id"] == source["id"]
    assert Path(result["result_path"]).exists()


def test_completed_unchanged_commission_continues_broad_research_without_rerun(context):
    _, _, _, _, turns, runs, broad = context
    first = execute(context)
    second = execute(context, first)
    assert second["mode"] == "model-selected experiments"
    assert len(turns) == len(runs) == len(broad) == 1
    prior = broad[0][0]["previous_experiment_evidence"]["typed_local_checkpoint"]
    assert prior["state"] == "executed-unvalidated" and prior["artifacts"]
    assert "unchanged inputs" in prior["scope"]
    assert second["actual_income_pln"] is None


def test_invalid_model_proposal_recorded_once_then_broad_path_seeks_new_work(context):
    _, _, _, proposal, turns, runs, broad = context
    proposal.pop("code")
    first = execute(context)
    assert first["state"] == "typed local experiment blocked"
    assert "exact typed specification" in first["error"]
    second = execute(context, first)
    assert len(turns) == 1 and not runs and len(broad) == 1
    assert second["typed_local_experiment"]["reused"]


def test_no_income_dossier_preserves_original_broad_tool_path(context):
    root, _, source, _, turns, runs, broad = context
    (root / "research/income-work/dossiers" / (source["id"] + ".json")).unlink()
    result = execute(context)
    assert result["mode"] == "model-selected experiments"
    assert not turns and not runs and len(broad) == 1


def test_numeric_market_observation_is_not_income_terms_evidence(context, monkeypatch):
    root, goal, source, _, turns, runs, broad = context
    path = root / "research/income-work/dossiers" / (source["id"] + ".json")
    monkeypatch.setattr(
        research_tools, "download_page", lambda url, max_bytes: (url, '{"price":100}')
    )
    price = goal_learning.observe(root, "https://example.org/price", "price")
    atomic_json(path, {"goal_id": goal["id"], "source": price})
    execute(context)
    assert not turns and not runs and len(broad) == 1


def test_unresolved_planning_attempt_is_not_replayed_but_other_research_continues(context):
    root, _, _, _, turns, runs, broad = context
    first = execute(context)
    path = Path(first["result_path"])
    record = json.loads(path.read_text())
    record.pop("experiment")
    record["state"] = "planning"
    atomic_json(path, record)
    second = execute(context, first)
    assert len(turns) == len(runs) == len(broad) == 1
    assert second["typed_local_experiment"]["state"] == "planning"
