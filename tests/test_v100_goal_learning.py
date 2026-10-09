import copy
import json
from types import SimpleNamespace

import pytest

from rlm.v100 import goal_learning as lab
from rlm.v100.goals import load_goal, set_goal
from rlm.v100.training import load_records


@pytest.fixture
def world(tmp_path, monkeypatch):
    clock = [1000.0]
    values = {"https://example.org/metric": 100, "https://other.example/metric": 30}
    monkeypatch.setattr(lab, "time", SimpleNamespace(time=lambda: clock[0]))
    monkeypatch.setattr(lab.shutil, "disk_usage", lambda root: SimpleNamespace(free=8 * 2**30))
    from rlm.v100 import research_tools

    def download(url, max_bytes):
        return url, json.dumps(
            {"value": values[url]}
        ) if url in values else "A newly observed event"

    monkeypatch.setattr(research_tools, "download_page", download)
    suite = tmp_path / "suite.jsonl"
    suite.write_text(
        json.dumps(
            {
                "id": "fixed",
                "skill": "retention",
                "match": "exact",
                "expected": "4",
                "messages": [{"role": "user", "content": "2+2"}],
            }
        )
        + "\n"
    )
    goal = set_goal(tmp_path, "Improve decisions toward the operator objective", suite)
    evidence = lab.observe(tmp_path, "https://signals.example/event")
    target = lab.observe(tmp_path, "https://example.org/metric", "value")
    spec = {
        "question": "Does this fresh event predict a change?",
        "rationale": "Test association, not causality",
        "evidence": [evidence["id"]],
        "target": target["id"],
        "horizon_seconds": 60,
        "threshold": 1,
        "probabilities": [0.1, 0.1, 0.8],
    }
    return tmp_path, clock, values, spec, goal


def test_precommit_failed_forecast_becomes_verified_correction_without_future_inputs(world):
    root, clock, values, spec, goal = world
    forecast = lab.predict(root, "A", spec)
    assert forecast["weights_changed"] is False
    assert lab.records(root) == []
    clock[0] = 1060
    values["https://example.org/metric"] = 90
    after = lab.observe(root, "https://example.org/metric", "value")
    result = lab.settle(root, forecast["id"], after["id"])
    assert result["correct"] is False
    assert result["brier"] > result["uniform_baseline_brier"]
    records = lab.records(root)
    assert records[0]["messages"][-1]["content"] == "down"
    prompt = json.loads(records[0]["messages"][0]["content"])
    assert prompt["target_at_prediction"]["value"] == 100
    assert "actual" not in prompt and "delta" not in prompt
    assert records[0]["verification"]["goal_id"] == goal["id"]
    assert load_goal(root) == goal
    assert lab.settle(root, forecast["id"], after["id"]) == result


@pytest.mark.parametrize(
    "change",
    [
        {"probabilities": [0.2, 0.2, 0.2]},
        {"probabilities": [float("nan"), 0, 1]},
        {"horizon_seconds": 0},
        {"threshold": 0},
        {"threshold": True},
        {"evidence": []},
    ],
)
def test_unbounded_or_unscorable_forecast_rejected(world, change):
    root, _, _, spec, _ = world
    with pytest.raises(ValueError):
        lab.predict(root, "A", {**spec, **change})


def test_stale_target_and_early_late_or_different_outcomes_rejected(world):
    root, clock, _, spec, _ = world
    clock[0] = 1061
    with pytest.raises(ValueError, match="60 seconds"):
        lab.predict(root, "A", spec)
    clock[0] = 1000
    forecast = lab.predict(root, "B", spec)
    early = lab.observe(root, "https://example.org/metric", "value")
    with pytest.raises(ValueError, match="horizon"):
        lab.settle(root, forecast["id"], early["id"])
    clock[0] = 1060
    other = lab.observe(root, "https://other.example/metric", "value")
    with pytest.raises(ValueError, match="series"):
        lab.settle(root, forecast["id"], other["id"])
    clock[0] = 1361
    late = lab.observe(root, "https://example.org/metric", "value")
    with pytest.raises(ValueError, match="horizon"):
        lab.settle(root, forecast["id"], late["id"])
    assert lab.status(root)["forecasts"][0]["state"] == "expired"


def test_observer_resolves_forward_forecast_and_training_rejects_forged_labels(world):
    root, clock, values, spec, _ = world
    lab.predict(root, "A", spec)
    clock[0] = 1060
    values["https://example.org/metric"] = 110
    lab.tick(root)
    record = lab.records(root)[0]
    assert record["messages"][-1]["content"] == "up"
    assert lab.status(root)["recent_resolved"] == 1
    forged = copy.deepcopy(record)
    forged["messages"][-1]["content"] = "down"
    path = root / "forged.jsonl"
    path.write_text(json.dumps(forged) + "\n")
    with pytest.raises(ValueError, match="host-observed"):
        load_records(path, root / "research/state/splits.sqlite3")


def test_archived_evidence_and_commitment_tampering_detected(world):
    root, _, _, spec, _ = world
    archive = root / "research/goal-learning/sources" / (spec["target"] + ".json")
    stored = json.loads(archive.read_text())
    stored["body"] = '{"value": 999}'
    archive.write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="archive changed"):
        lab.predict(root, "A", spec)


def test_query_variants_share_source_group_and_disk_headroom_enforced(world, monkeypatch):
    root, _, _, _, _ = world
    assert lab.source_key("https://example.org/metric?asset=A") == lab.source_key(
        "https://example.org/metric?asset=B&nonce=3"
    )
    monkeypatch.setattr(lab.shutil, "disk_usage", lambda root: SimpleNamespace(free=1))
    with pytest.raises(ValueError, match="headroom"):
        lab.observe(root, "https://example.org/metric", "value")


def test_goal_labels_join_pool_and_stay_source_disjoint(world):
    root, clock, values, spec, _ = world
    lab.predict(root, "A", spec)
    clock[0] = 1060
    values["https://example.org/metric"] = 120
    lab.tick(root)
    rows = lab.records(root)
    rows.extend(
        {
            "id": f"human-{i}",
            "group": f"human-{i}",
            "messages": [
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "answer"},
            ],
            "verification": {"kind": "human_feedback", "accepted": True},
        }
        for i in range(8)
    )
    pool = root / "pool.jsonl"
    pool.write_text("".join(json.dumps(row) + "\n" for row in rows))
    train, validation = load_records(pool, root / "research/state/splits.sqlite3")
    assert len(train) + len(validation) == 9
    assert {r["group"] for r in train}.isdisjoint({r["group"] for r in validation})


def test_missing_operator_goal_cannot_be_invented_by_predictor(world):
    root, _, _, spec, _ = world
    (root / "research/goal.json").unlink()
    with pytest.raises(ValueError, match="Operator long-term"):
        lab.predict(root, "A", spec)


def test_goal_development_uses_only_validation_and_current_operator_goal(world, monkeypatch):
    root, _, _, _, goal = world
    from rlm.v100 import training

    def row(index, goal_id):
        return {
            "id": str(index),
            "group": str(index % 2),
            "verification": {"kind": "goal_observation", "goal_id": goal_id},
            "messages": [
                {"role": "user", "content": "before outcome"},
                {"role": "assistant", "content": "up"},
            ],
        }

    training_only = row(999, goal["id"])
    validation = [row(i, goal["id"]) for i in range(4)] + [row(20, "other-goal")]
    monkeypatch.setattr(training, "load_records", lambda *args: ([training_only], validation))
    destination = root / "development.jsonl"
    lab.development_suite(root, root / "pool", root / "ledger", destination)
    cases = [json.loads(line) for line in destination.read_text().splitlines()]
    assert len(cases) == 4 and {case["id"] for case in cases} == {str(i) for i in range(4)}
    assert all(case["messages"][-1]["role"] == "user" for case in cases)


def test_goal_gate_rejects_regression_incomplete_reports_and_wrong_weights(tmp_path):
    from rlm.v100.protection import file_hash

    parent, candidate = tmp_path / "old.gguf", tmp_path / "new.gguf"
    parent.write_bytes(b"parent")
    candidate.write_bytes(b"candidate")
    suite = tmp_path / "goal-development.jsonl"
    suite.write_text(json.dumps({"expected": "up"}) + "\n")
    report = {
        "schema": "v100-quality-v1",
        "generation": {},
        "memory_mode": "fixed",
        "suite_sha256": file_hash(suite),
        "cases": [{"id": "1", "passed": True}],
    }
    previous = {**report, "model_sha256": file_hash(parent)}
    current = {
        **report,
        "model_sha256": file_hash(candidate),
        "cases": [{"id": "1", "passed": False}],
    }
    (tmp_path / "goal-parent.json").write_text(json.dumps(previous))
    path = tmp_path / "goal-candidate.json"
    path.write_text(json.dumps(current))
    assert lab.verified_gate(tmp_path, parent, candidate)["passed"] is False
    current["cases"] = [{"id": "1", "passed": True, "error": "transport failed"}]
    path.write_text(json.dumps(current))
    assert lab.verified_gate(tmp_path, parent, candidate)["complete"] is False
    candidate.write_bytes(b"other weights")
    with pytest.raises(ValueError, match="different weights"):
        lab.verified_gate(tmp_path, parent, candidate)


def test_long_examples_are_deferred_before_training_without_modifying_evidence(world, monkeypatch):
    root, _, _, _, _ = world
    from transformers import AutoTokenizer

    class Tokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            value = "".join(row["role"] + ":" + row["content"] + "\n" for row in messages)
            if add_generation_prompt:
                value += "assistant:"
            return list(value.encode())

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    short = {
        "id": "short",
        "messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "up"}],
    }
    long = {
        "id": "long",
        "messages": [
            {"role": "user", "content": "x" * 200},
            {"role": "assistant", "content": "down"},
        ],
    }
    original = copy.deepcopy([short, long])
    profile = {"training": {"base_model": "local", "max_length": 50}}
    assert lab.admit(root, [short, long], profile) == [short]
    assert [short, long] == original
    admission = lab.status(root)["training_admission"]
    assert admission["admitted"] == 1 and admission["deferred_count"] == 1
    assert "shorten sources" in admission["deferred"][0]["reason"]


@pytest.mark.parametrize(
    "labels, expected", [(["flat"] * 4, False), (["up", "down", "up", "down"], True)]
)
def test_goal_gate_requires_more_than_a_constant_answer(tmp_path, labels, expected):
    from rlm.v100.protection import file_hash

    parent, candidate = tmp_path / "old", tmp_path / "new"
    parent.write_bytes(b"old")
    candidate.write_bytes(b"new")
    suite = tmp_path / "goal-development.jsonl"
    suite.write_text("".join(json.dumps({"expected": label}) + "\n" for label in labels))
    report = {
        "schema": "v100-quality-v1",
        "generation": {},
        "memory_mode": "fixed",
        "suite_sha256": file_hash(suite),
    }
    previous = {
        **report,
        "model_sha256": file_hash(parent),
        "cases": [{"id": str(i), "passed": i < 2} for i in range(4)],
    }
    current = {
        **report,
        "model_sha256": file_hash(candidate),
        "cases": [{"id": str(i), "passed": True} for i in range(4)],
    }
    (tmp_path / "goal-parent.json").write_text(json.dumps(previous))
    (tmp_path / "goal-candidate.json").write_text(json.dumps(current))
    gate = lab.verified_gate(tmp_path, parent, candidate)
    assert gate["passed"] is expected
    assert gate["beats_best_constant_baseline"] is expected
