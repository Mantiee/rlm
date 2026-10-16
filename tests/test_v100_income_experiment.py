"""Typed local execution delegates to the existing sandbox and keeps real receipts."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rlm.v100 import goal_learning, research_sandbox, research_tools
from rlm.v100 import income_experiment as lab
from rlm.v100.common import atomic_json
from tests.test_v100_recovery import financial_world


@pytest.fixture
def world(tmp_path, monkeypatch):
    goal, _ = financial_world(tmp_path)
    monkeypatch.setattr(
        research_tools, "download_page", lambda url, max_bytes: (url, '{"value":100}')
    )
    evidence = goal_learning.observe(tmp_path, "https://example.org/metric", "value")
    proposal = {
        "hypothesis": "A local summary can preserve source value",
        "code": "from pathlib import Path\nPath('result.json').write_text('{\"value\":100}')\n",
        "measurement": "Record output field value and input source hash",
        "failure_condition": "Missing or incorrect value",
        "artifacts": ["result.json"],
        "seconds": 10,
    }
    calls = []

    def child(root, branch, code, seconds):
        # Known safe fixture code runs a real local child; production sandbox is delegated,
        # not reproduced or claimed as verified by this fixture.
        calls.append(code)
        folder = root / "research/python-experiments" / f"run-fixture-{len(calls)}"
        work = folder / "work"
        work.mkdir(parents=True)
        process = subprocess.run(
            [sys.executable, "-I", "-c", code],
            cwd=work,
            capture_output=True,
            text=True,
            timeout=seconds,
        )
        result = {
            "branch": branch,
            "directory": str(folder),
            "exit_code": process.returncode,
            "timed_out": False,
            "output": process.stdout,
        }
        atomic_json(folder / "result.json", result)
        return result

    monkeypatch.setattr(research_sandbox, "run", child)
    return tmp_path, goal, evidence, proposal, calls


def test_model_proposes_then_real_child_artifact_is_hashed_once(world, monkeypatch):
    root, goal, evidence, proposal, calls = world
    turns = []

    def turn(client, messages, **kwargs):
        turns.append(messages)
        assert kwargs["response_format"]["schema"] == lab.specification_schema()
        return {"content": json.dumps(proposal)}

    monkeypatch.setattr(lab, "native_turn", turn)
    receipt = lab.propose_and_run(root, "A", object(), [evidence["id"]])
    assert len(turns) == len(calls) == 1
    assert receipt["state"] == "executed-unvalidated" and receipt["actual_income_pln"] is None
    artifact = receipt["artifacts"][0]
    assert json.loads(Path(artifact["path"]).read_text()) == {"value": 100}
    assert len(artifact["sha256"]) == 64 and artifact["bytes"] > 0
    assert receipt["process_receipt"]["kind"] == "executed-unvalidated"
    repeated = lab.execute_proposal(
        root, "A", goal["id"], {**proposal, "hypothesis": "A paraphrased title"}, [evidence["id"]]
    )
    assert repeated["reused"] and repeated["id"] == receipt["id"] and len(calls) == 1


def test_fresh_retrieval_same_content_does_not_rerun(world):
    root, goal, evidence, proposal, calls = world
    first = lab.execute_proposal(root, "A", goal["id"], proposal, [evidence["id"]])
    new = goal_learning.observe(root, "https://example.org/metric", "value")
    second = lab.execute_proposal(root, "A", goal["id"], proposal, [new["id"]])
    assert second["reused"] and second["id"] == first["id"] and len(calls) == 1


def test_genuinely_changed_source_content_commissions_new_run(world, monkeypatch):
    root, goal, evidence, proposal, calls = world
    first = lab.execute_proposal(root, "A", goal["id"], proposal, [evidence["id"]])
    monkeypatch.setattr(
        research_tools, "download_page", lambda url, max_bytes: (url, '{"value":101}')
    )
    new = goal_learning.observe(root, "https://example.org/metric", "value")
    second = lab.execute_proposal(root, "A", goal["id"], proposal, [new["id"]])
    assert second["id"] != first["id"] and len(calls) == 2


@pytest.mark.parametrize(
    "changes",
    [
        {"artifacts": ["../escape"]},
        {"artifacts": ["/tmp/escape"]},
        {"artifacts": ["C:\\escape"]},
        {"artifacts": ["."]},
        {"artifacts": []},
        {"seconds": True},
        {"seconds": 61},
        {"code": "invalid python ("},
        {"measurement": ""},
        {"extra": "unsupported"},
    ],
)
def test_invalid_typed_plan_never_executes(world, changes):
    root, goal, evidence, proposal, calls = world
    with pytest.raises((ValueError, SyntaxError)):
        lab.execute_proposal(root, "A", goal["id"], {**proposal, **changes}, [evidence["id"]])
    assert calls == []


def test_missing_declared_artifact_is_failure_not_success(world):
    root, goal, evidence, proposal, calls = world
    proposal["code"] = "print('banner only')"
    receipt = lab.execute_proposal(root, "A", goal["id"], proposal, [evidence["id"]])
    assert receipt["state"] == "blocked or failed" and "artifact missing" in receipt["error"]
    assert receipt["actual_income_pln"] is None


def test_changed_goal_blocks_model_plan_before_child(world, monkeypatch):
    root, _, evidence, proposal, calls = world

    def turn(*args, **kwargs):
        monkeypatch.setattr(lab, "load_goal", lambda root: {"id": "new goal"})
        return {"content": json.dumps(proposal)}

    monkeypatch.setattr(lab, "native_turn", turn)
    with pytest.raises(ValueError, match="unchanged active goal"):
        lab.propose_and_run(root, "A", object(), [evidence["id"]])
    assert not calls


def test_unresolved_running_checkpoint_is_never_automatically_replayed(world):
    root, goal, evidence, proposal, calls = world
    receipt = lab.execute_proposal(root, "A", goal["id"], proposal, [evidence["id"]])
    path = root / "research/income-work/local-experiments" / receipt["id"] / "checkpoint.json"
    saved = json.loads(path.read_text())
    saved["state"] = "running"
    atomic_json(path, saved)
    again = lab.execute_proposal(root, "A", goal["id"], proposal, [evidence["id"]])
    assert again["reused"] and again["state"] == "running" and len(calls) == 1


def test_missing_sandbox_runtime_does_not_run_unsandboxed(world, monkeypatch):
    root, goal, evidence, proposal, calls = world
    monkeypatch.setattr(
        research_sandbox,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("isolated runtime unavailable")),
    )
    receipt = lab.execute_proposal(root, "A", goal["id"], proposal, [evidence["id"]])
    assert receipt["state"] == "blocked or failed" and "runtime unavailable" in receipt["error"]
    assert not calls


@pytest.mark.parametrize("exit_code,timed_out", [(1, False), (-9, True)])
def test_unsuccessful_process_is_not_an_executed_artifact(world, monkeypatch, exit_code, timed_out):
    root, goal, evidence, proposal, calls = world

    def failure(root, branch, code, seconds):
        folder = root / "research/python-experiments/run-failed"
        folder.mkdir(parents=True)
        result = {
            "directory": str(folder),
            "exit_code": exit_code,
            "timed_out": timed_out,
            "output": "failed",
        }
        atomic_json(folder / "result.json", result)
        return result

    monkeypatch.setattr(research_sandbox, "run", failure)
    receipt = lab.execute_proposal(root, "A", goal["id"], proposal, [evidence["id"]])
    assert receipt["state"] == "blocked or failed"
    assert receipt["process_receipt"] is None and receipt["actual_income_pln"] is None


def test_tampered_source_archive_cannot_commission_code(world):
    root, goal, evidence, proposal, calls = world
    path = root / "research/goal-learning/sources" / (evidence["id"] + ".json")
    value = json.loads(path.read_text())
    value["body"] = "changed"
    atomic_json(path, value)
    with pytest.raises(ValueError, match="archive changed"):
        lab.execute_proposal(root, "A", goal["id"], proposal, [evidence["id"]])
    assert not calls
