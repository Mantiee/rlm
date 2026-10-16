"""Interrupted development baselines retain exact cases, failures and identity."""

import json
from types import SimpleNamespace

import pytest

from rlm.v100 import evaluation
from rlm.v100.common import atomic_json
from tests.test_v100_challenge import bootstrap


def inputs(tmp_path):
    profile = bootstrap(tmp_path)
    suite = tmp_path / "suite.jsonl"
    suite.write_text(
        "\n".join(
            json.dumps(
                dict(
                    id=str(i),
                    skill="logic",
                    messages=[dict(role="user", content=str(i))],
                    expected="4",
                    match="exact",
                )
            )
            for i in range(3)
        ),
        encoding="utf-8",
    )
    output = tmp_path / "research/mission/run-original/baseline-32768.json"
    output.parent.mkdir(parents=True)
    return profile, suite, output


def interrupted(tmp_path):
    profile, suite, output = inputs(tmp_path)
    calls = []

    def completion(messages):
        calls.append(messages[0]["content"])
        if len(calls) == 3:
            raise RuntimeError("interrupted")
        return "wrong" if len(calls) == 1 else "4"

    client = SimpleNamespace(completion=completion)
    with pytest.raises(RuntimeError, match="interrupted"):
        evaluation.evaluate_suite(client, profile, suite, output)
    assert calls == ["0", "1", "2"]
    assert not output.exists()
    return profile, suite, output


def test_interrupted_baseline_resumes_without_replaying_cases_or_erasing_failures(tmp_path):
    profile, suite, original = interrupted(tmp_path)
    output = original.parent.with_name("run-next") / original.name
    assert evaluation.resume_baseline(tmp_path, profile, suite, output) == original.with_suffix(
        ".partial.json"
    )
    calls = []
    client = SimpleNamespace(
        completion=lambda messages: calls.append(messages[0]["content"]) or "4"
    )
    report = evaluation.evaluate_suite(client, profile, suite, output)
    assert calls == ["2"]
    assert [case["passed"] for case in report["cases"]] == [False, True, True]
    progress = json.loads(output.with_suffix(".progress.json").read_text(encoding="utf-8"))
    assert (
        progress["completed"] == 3 and progress["passed"] == 2 and progress["state"] == "finished"
    )
    assert (
        report["cases"][:2]
        == json.loads(original.with_suffix(".partial.json").read_text(encoding="utf-8"))["report"][
            "cases"
        ]
    )


@pytest.mark.parametrize("field", ["suite", "model", "execution", "generation", "memory_mode"])
def test_changed_identity_is_not_reused(tmp_path, field):
    profile, suite, original = interrupted(tmp_path)
    checkpoint = original.with_suffix(".partial.json")
    value = json.loads(checkpoint.read_text(encoding="utf-8"))
    key = dict(
        suite="suite_sha256",
        model="model_sha256",
        execution="execution_sha256",
        generation="generation",
        memory_mode="memory_mode",
    )[field]
    value["report"][key] = "changed"
    value["sha256"] = evaluation.checkpoint_digest(value["report"])
    atomic_json(checkpoint, value)
    output = original.parent.with_name("run-next") / original.name
    assert evaluation.resume_baseline(tmp_path, profile, suite, output) is None
    assert not output.with_suffix(".partial.json").exists()


@pytest.mark.parametrize(
    "mutation",
    ["checksum", "duplicate", "order", "missing_field", "fake_grade", "counts_only", "nonfinite"],
)
def test_invalid_checkpoint_is_never_admitted(tmp_path, mutation):
    profile, suite, output = interrupted(tmp_path)
    checkpoint = output.with_suffix(".partial.json")
    value = json.loads(checkpoint.read_text(encoding="utf-8"))
    cases = value["report"]["cases"]
    if mutation == "checksum":
        cases[0]["answer"] = "changed"
    elif mutation == "duplicate":
        cases[1] = dict(cases[0])
    elif mutation == "order":
        cases.reverse()
    elif mutation == "missing_field":
        del cases[0]["answer"]
    elif mutation == "fake_grade":
        cases[0]["passed"] = True
    elif mutation == "counts_only":
        value["report"]["cases"] = {"completed": 2, "last_case": "1"}
    elif mutation == "nonfinite":
        cases[0]["seconds"] = -1
    if mutation != "checksum":
        value["sha256"] = evaluation.checkpoint_digest(value["report"])
    atomic_json(checkpoint, value)
    with pytest.raises(ValueError):
        evaluation.evaluate_suite(
            SimpleNamespace(completion=lambda messages: pytest.fail("inference called")),
            profile,
            suite,
            output,
        )
    assert not output.exists()


def test_complete_checkpoint_crash_before_final_report_requires_no_replay(tmp_path):
    profile, suite, output = inputs(tmp_path)
    report = evaluation.evaluate_suite(
        SimpleNamespace(completion=lambda messages: "4"), profile, suite, output
    )
    output.unlink()
    resumed = evaluation.evaluate_suite(
        SimpleNamespace(completion=lambda messages: pytest.fail("replayed")), profile, suite, output
    )
    assert resumed == report
    progress = json.loads(output.with_suffix(".progress.json").read_text(encoding="utf-8"))
    assert progress["state"] == "finished" and progress["completed"] == 3


def test_changed_profile_generation_does_not_reuse_old_answers(tmp_path):
    profile, suite, original = interrupted(tmp_path)
    profile["runtime"]["temperature"] = 0.25
    output = original.parent.with_name("run-next") / original.name
    assert evaluation.resume_baseline(tmp_path, profile, suite, output) is None
    calls = []
    evaluation.evaluate_suite(
        SimpleNamespace(completion=lambda messages: calls.append(messages[0]["content"]) or "4"),
        profile,
        suite,
        output,
    )
    assert calls == ["0", "1", "2"]


def test_no_resume_from_progress_counts_without_full_case_checkpoint(tmp_path):
    profile, suite, output = inputs(tmp_path)
    atomic_json(
        output.with_suffix(".progress.json"),
        {"completed": 2, "passed": 2, "last_case": {"id": "1"}},
    )
    calls = []
    evaluation.evaluate_suite(
        SimpleNamespace(completion=lambda messages: calls.append(messages[0]["content"]) or "4"),
        profile,
        suite,
        output,
    )
    assert calls == ["0", "1", "2"]
