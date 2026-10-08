import json
import random

import pytest

from rlm.v100 import curriculum, distributed_compute, fresh_audit, insights


@pytest.mark.parametrize(
    "task,answer",
    [
        ({"operation": "sort", "values": [9, -4, 9, 2]}, "[-4,2,9,9]"),
        ({"operation": "unique_sorted", "values": [9, -4, 9, 2]}, "[-4,2,9]"),
        ({"operation": "reverse", "values": [9, -4, 9, 2]}, "[2,9,-4,9]"),
    ],
)
def test_sequence_references_are_exact_and_forged_answers_rejected(task, answer):
    record = insights.verified_record(
        {"kind": "sequence_transform", "expression": json.dumps(task)}
    )
    assert record["messages"][-1]["content"] == answer
    insights.verify_record(record)
    record["messages"][-1]["content"] = "[0]"
    with pytest.raises(ValueError, match="independently calculated"):
        insights.verify_record(record)


@pytest.mark.parametrize(
    "value,expected", [("hello", '"hello"'), (False, "false"), (None, "null"), (7, "7")]
)
def test_extraction_preserves_scalar_types(value, expected):
    task = {
        "kind": "structured_extraction",
        "expression": json.dumps(
            {"data": {"selected": value, "distractor": "ignore"}, "field": "selected"}
        ),
    }
    assert insights.verified_record(task)["messages"][-1]["content"] == expected


@pytest.mark.parametrize(
    "domain,payload",
    [
        ("sequence_transform", {"operation": "eval", "values": [1]}),
        ("sequence_transform", {"operation": "sort", "values": [True]}),
        ("structured_extraction", {"data": {"a": 2}, "field": "missing"}),
        ("structured_extraction", {"data": {"a": [1]}, "field": "a"}),
        ("structured_extraction", {"data": {"a": 10001}, "field": "a"}),
    ],
)
def test_structured_proofs_never_guess_or_execute_unbounded_tasks(domain, payload):
    with pytest.raises(ValueError):
        insights.verified_record({"kind": domain, "expression": json.dumps(payload)})


@pytest.mark.parametrize("domain", insights.PROOF_DOMAINS)
def test_generated_examples_fit_owned_worker_context_without_truncation(domain):
    rng = random.Random(79)
    for _ in range(100):
        row = insights.verified_record(curriculum.draw_task(domain, rng))
        text = row["messages"][0]["content"] + "\nAnswer: " + row["messages"][1]["content"]
        assert len(text.encode()) + 2 <= 256


@pytest.mark.parametrize("domain", ["sequence_transform", "structured_extraction"])
def test_new_domains_reach_training_and_owned_compute(domain, tmp_path):
    receipt = curriculum.request(tmp_path, "A", domain, 8)
    assert receipt["new_verified_examples"] == 8 and receipt["weights_changed"] is False
    queue = insights.InsightQueue(tmp_path)
    try:
        assert len(queue.records()) == 8
    finally:
        queue.close()
    rows = distributed_compute.fresh_records(domain, 32)
    assert len(rows) == 32 and len({row["group"] for row in rows}) == 32
    for row in rows:
        insights.verify_record(row)


def test_postfreeze_audit_covers_all_proof_domains_without_training_labels(tmp_path):
    parent, candidate = tmp_path / "parent.gguf", tmp_path / "child.gguf"
    parent.write_bytes(b"parent")
    candidate.write_bytes(b"child")
    folder = fresh_audit.create(
        tmp_path,
        {"server": {"model": str(parent)}, "training": {}},
        candidate,
        tmp_path / "audit",
        12,
    )
    rows = [json.loads(line) for line in (folder / "suite.jsonl").read_text().splitlines()]
    assert {row["skill"] for row in rows} == set(insights.PROOF_DOMAINS)
    assert all(all(message["role"] != "assistant" for message in row["messages"]) for row in rows)
    assert json.loads((folder / "manifest.json").read_text())["state"] == "sealed"
