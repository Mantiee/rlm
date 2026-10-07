import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlm.v100.challenge import case, prepare_challenge, tool_cases
from rlm.v100.common import load_profile
from rlm.v100.competition import helper_client
from rlm.v100.evaluation import evaluate_suite
from rlm.v100.fixture_tools import fixture_answer, visible_sources
from rlm.v100.inference import generation_conditions, prepare_thinking
from rlm.v100.insights import verified_record
from rlm.v100.protection import file_hash, reserve_audit_sources
from rlm.v100.training import load_records


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def bootstrap(root):
    original = load_profile(Path(__file__).parents[1] / "profiles/v100.toml", root)
    for name in ("model", "binary"):
        path = Path(original["server"][name])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixed native/model identity")
    profile = load_profile(prepare_thinking(original, root), root)
    rows = [
        verified_record({"kind": "arithmetic", "expression": f"{index}+13"}) for index in range(160)
    ]
    controls = [
        {
            "id": row["group"],
            "skill": "arithmetic",
            "messages": row["messages"][:-1],
            "match": "exact",
            "expected": row["messages"][-1]["content"],
        }
        for row in rows[128:]
    ] + [case(f"control-{index:02d}", "logic", "2+2?", "4") for index in range(9)]
    folder = root / "research/income-bootstrap-v1"
    jsonl(folder / "pool.jsonl", rows[:128])
    jsonl(folder / "development.jsonl", controls)
    ledger = Path(profile["training"]["split_ledger"])
    reserve_audit_sources(ledger, [row["id"] for row in controls])
    load_records(folder / "pool.jsonl", ledger)
    baseline = {
        "schema": "v100-quality-v1",
        "suite_sha256": file_hash(folder / "development.jsonl"),
        "model_sha256": file_hash(Path(profile["server"]["model"])),
        "generation": generation_conditions(profile),
        "cases": [{"id": row["id"], "passed": True} for row in controls],
    }
    (folder / "baseline-thinking.json").write_text(json.dumps(baseline))
    return profile


def test_curriculum_preserves_replay_and_reserves_disjoint_development(tmp_path):
    profile = bootstrap(tmp_path)
    old = tmp_path / "research/income-bootstrap-v1"
    before = {path.name: path.read_bytes() for path in old.iterdir()}
    report = prepare_challenge(profile, tmp_path)
    folder = Path(report["folder"])
    assert report["development_cases"] == 81 and report["new_verified_records"] == 256
    assert report["training_records"] + report["validation_records"] == 384
    assert report["tool_cases"] == report["source_cases"] == 8
    pool = [json.loads(line) for line in (folder / "pool.jsonl").read_text().splitlines()]
    suite = [json.loads(line) for line in (folder / "development.jsonl").read_text().splitlines()]
    assert not {row["group"] for row in pool} & {row["id"] for row in suite}
    assert {path.name: path.read_bytes() for path in old.iterdir()} == before
    assert prepare_challenge(profile, tmp_path) == report
    with sqlite3.connect(profile["training"]["split_ledger"]) as db:
        assert all(
            dict(db.execute("SELECT * FROM roles"))[row["id"]] == "audit" for row in suite[41:]
        )
    held_out = suite[41]
    leaked = verified_record(
        {
            "kind": held_out["skill"],
            "expression": held_out["messages"][-1]["content"].splitlines()[-1],
        }
    )
    jsonl(tmp_path / "leaked.jsonl", pool + [leaked])
    with pytest.raises(ValueError, match="Audit"):
        load_records(tmp_path / "leaked.jsonl", Path(profile["training"]["split_ledger"]))


def test_changed_baseline_or_existing_challenge_is_never_silently_replaced(tmp_path):
    profile = bootstrap(tmp_path)
    report = prepare_challenge(profile, tmp_path)
    path = Path(report["folder"]) / "pool.jsonl"
    path.write_text("changed\n")
    with pytest.raises(FileExistsError, match="not overwritten"):
        prepare_challenge(profile, tmp_path)
    assert path.read_text() == "changed\n"


@pytest.mark.parametrize("change", ["failed", "weights", "generation"])
def test_preparation_requires_the_actual_completed_baseline(tmp_path, change):
    profile = bootstrap(tmp_path)
    if change == "weights":
        Path(profile["server"]["model"]).write_bytes(b"changed weights")
    else:
        path = tmp_path / "research/income-bootstrap-v1/baseline-thinking.json"
        baseline = json.loads(path.read_text())
        if change == "failed":
            baseline["cases"][0]["passed"] = False
        else:
            baseline["generation"]["thinking"] = False
        path.write_text(json.dumps(baseline))
    with pytest.raises(ValueError, match="41/41"):
        prepare_challenge(profile, tmp_path)
    assert not (tmp_path / "research/income-challenge-v1").exists()


def call(name, arguments):
    return {
        "id": "call-1",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def fake_client(responder):
    requests = []

    def request(endpoint, payload):
        requests.append((endpoint, payload))
        if endpoint == "/apply-template":
            return {"prompt": "fixed prompt"}
        return {"choices": [{"finish_reason": "stop", "message": responder(payload)}]}

    client = SimpleNamespace(
        sampling_args={"max_tokens": 2048},
        template_args=lambda: {},
        request=request,
        count_text=lambda *args, **kwargs: 10,
        context_window=8192,
        model_name="fixture",
    )
    return client, requests


def test_calculator_executes_checked_arithmetic_and_reports_the_actual_trace():
    row = tool_cases()[0]

    def respond(payload):
        if payload["messages"][-1]["role"] != "tool":
            return {
                "tool_calls": [call("calculate", {"expression": "17*(6013-5347)-319-367-113-79"})]
            }
        result = json.loads(payload["messages"][-1]["content"])
        return {"content": json.dumps(result)}

    client, _ = fake_client(respond)
    result = fixture_answer(client, row)
    assert result["answer"] == row["expected"] == "10444"
    assert [step["tool"] for step in result["trace"]] == ["calculate"]


def test_reopened_memory_returns_original_passage_but_never_future_evidence():
    row = tool_cases()[4]

    def respond(payload):
        messages = payload["messages"]
        if messages[-1]["role"] != "tool":
            return {"tool_calls": [call("search_memory", {"query": "ORIONMEM0"})]}
        result = json.loads(messages[-1]["content"])
        if isinstance(result, list):
            assert "4217" not in result[0]["preview"]
            return {"tool_calls": [call("read_source", {"source_id": result[0]["id"]})]}
        assert "4217" in result["text"] and "999999" not in result["text"]
        return {"content": '{"answer":"4217"}'}

    client, requests = fake_client(respond)
    result = fixture_answer(client, row)
    assert result["answer"] == "4217"
    assert [step["tool"] for step in result["trace"]] == ["search_memory", "read_source"]
    assert "999999" not in json.dumps(requests)
    assert len(visible_sources(row["tool_fixture"])) == 1


@pytest.mark.parametrize(
    "name,args",
    [
        ("calculate", {"expression": "__import__('os').system('id')"}),
        ("calculate", {"expression": "1//0"}),
        ("calculate", {"expression": "2**100000"}),
        ("calculate", ["1+1"]),
        ("read_source", {"source_id": "not-retrieved"}),
        ("shell", {"command": "id"}),
    ],
)
def test_fixture_tool_cannot_execute_code_or_read_arbitrary_sources(name, args):
    client, _ = fake_client(lambda payload: {"tool_calls": [call(name, args)]})
    with pytest.raises(ValueError):
        fixture_answer(client, tool_cases()[0])


def test_correct_guess_without_required_tool_does_not_pass_quality(tmp_path):
    profile = bootstrap(tmp_path)
    row = tool_cases()[0]
    fake, _ = fake_client(lambda payload: {"content": json.dumps({"answer": row["expected"]})})
    client = helper_client(profile)
    client.request = fake.request
    client.count_text = fake.count_text
    # Stale metadata from a previous raw completion must not taint native tool evaluation.
    client.get_response_info = lambda: {"finish_reason": "length"}
    suite = tmp_path / "tools.jsonl"
    jsonl(suite, [row])
    report = evaluate_suite(client, profile, suite, tmp_path / "quality.json")
    assert not report["cases"][0]["passed"]
    assert report["cases"][0]["answer"] == row["expected"]
    assert report["cases"][0]["seconds"] >= 0


@pytest.mark.parametrize(
    "expression,passed",
    [
        ("17*(6013-5347)-319-367-113-79", True),
        ("2+2", False),
    ],
)
def test_quality_requires_a_calculator_result_supporting_the_final_answer(
    tmp_path, expression, passed
):
    profile = bootstrap(tmp_path)
    row = tool_cases()[0]

    def respond(payload):
        if payload["messages"][-1]["role"] != "tool":
            return {"tool_calls": [call("calculate", {"expression": expression})]}
        return {"content": json.dumps({"answer": row["expected"]})}

    fake, _ = fake_client(respond)
    client = helper_client(profile)
    client.request, client.count_text = fake.request, fake.count_text
    client.get_response_info = lambda: {"finish_reason": "length"}
    suite = tmp_path / "suite.jsonl"
    jsonl(suite, [row])
    report = evaluate_suite(client, profile, suite, tmp_path / "quality.json")
    assert report["cases"][0]["passed"] is passed
    assert report["cases"][0]["finish_reason"] == "stop"
    assert len(report["cases"][0]["tool_trace"]) == 1


def test_infinite_tool_requests_have_a_fixed_budget():
    client, requests = fake_client(
        lambda payload: {"tool_calls": [call("calculate", {"expression": "2+2"})]}
    )
    with pytest.raises(ValueError, match="budget exhausted"):
        fixture_answer(client, tool_cases()[0])
    assert sum(endpoint == "/v1/chat/completions" for endpoint, _ in requests) == 6
