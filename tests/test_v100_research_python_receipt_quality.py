import json

import pytest

from rlm.v100.research_contract import assessment


def receipt(
    tmp_path, *, exit_code=0, timed_out=False, output="Measured 12 samples", artifact=False
):
    folder = tmp_path / "research/python-experiments/run-fixture"
    (folder / "work").mkdir(parents=True)
    if artifact:
        (folder / "work/result.csv").write_text("sample,value\n1,12\n", encoding="utf-8")
    value = {
        "directory": str(folder),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "output": output,
        "status": "experiment, not independently verified truth",
    }
    (folder / "result.json").write_text(json.dumps(value), encoding="utf-8")
    return value


def assess(tmp_path, value):
    return assessment(
        tmp_path,
        {
            "hypothesis": "Public data transformation may be useful",
            "suggested_test": "Produce and check a reproducible data artifact",
            "research_trace": [{"tool": "run_research_python", "result": value}],
        },
    )


def test_archived_python_process_is_executed_but_unvalidated(tmp_path):
    value = assess(tmp_path, receipt(tmp_path, artifact=True))
    assert value["state"] == "process executed - task validation missing"
    process = value["process_receipts"][0]
    assert process["kind"] == "executed-unvalidated"
    assert process["artifacts"][0]["bytes"] > 0
    assert len(process["artifacts"][0]["sha256"]) == 64
    assert not value["work_receipts"]
    assert "income unverified" in process["scope"]


@pytest.mark.parametrize(
    "kwargs", [{"exit_code": 1}, {"timed_out": True}, {"output": ""}, {"output": "   "}]
)
def test_failed_timed_out_or_empty_process_does_not_count(tmp_path, kwargs):
    value = assess(tmp_path, receipt(tmp_path, **kwargs))
    assert value["process_receipts"] == []
    assert value["state"] == "advisory only - no executed test"


def test_nonempty_artifact_without_stdout_is_unvalidated_process(tmp_path):
    value = assess(tmp_path, receipt(tmp_path, artifact=True, output=""))
    assert len(value["process_receipts"]) == 1
    assert value["state"] == "process executed - task validation missing"


def test_tampered_receipt_or_outside_archive_is_not_execution_proof(tmp_path):
    original = receipt(tmp_path)
    changed = {**original, "output": "Invented successful result"}
    assert assess(tmp_path, changed)["process_receipts"] == []
    original["directory"] = str(tmp_path / "outside")
    assert assess(tmp_path, original)["process_receipts"] == []
