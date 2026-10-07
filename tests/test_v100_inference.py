import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlm.v100.agent import native_turn
from rlm.v100.cli import client_for
from rlm.v100.common import load_profile
from rlm.v100.competition import helper_client
from rlm.v100.evaluation import evaluate_suite
from rlm.v100.inference import generation_conditions, prepare_thinking, sampling_settings


def profile(tmp_path):
    value = load_profile(Path(__file__).parents[1] / "profiles/v100.toml", tmp_path)
    model = Path(value["server"]["model"])
    model.parent.mkdir(parents=True)
    model.write_bytes(b"Synthetic model identity for fixed generation tests")
    binary = Path(value["server"]["binary"])
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"Synthetic server identity for fixed generation tests")
    return value


def suite(tmp_path):
    path = tmp_path / "suite.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "math",
                "skill": "arithmetic",
                "expected": "4",
                "match": "exact",
                "messages": [{"role": "user", "content": "2+2?"}],
            }
        )
    )
    return path


def test_separate_thinking_profile_preserves_original_and_clients_use_it(tmp_path):
    original = profile(tmp_path)
    before = json.dumps(original, sort_keys=True)
    path = prepare_thinking(original, tmp_path)
    chosen = load_profile(path, tmp_path)
    assert json.dumps(original, sort_keys=True) == before
    assert chosen["server"] == original["server"]
    assert prepare_thinking(original, tmp_path) == path
    for client in (client_for(chosen, tmp_path), helper_client(chosen, tmp_path)):
        assert client.enable_thinking is True
        assert client.sampling_args == {
            "temperature": 1.0,
            "seed": 42,
            "max_tokens": 2048,
            "top_p": 0.95,
            "top_k": 64,
        }
    assert helper_client(original, tmp_path).enable_thinking is False
    raw = json.loads(path.read_text())
    raw["runtime"]["temperature"] = 0.5
    path.write_text(json.dumps(raw))
    with pytest.raises(FileExistsError, match="not overwritten"):
        prepare_thinking(original, tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        {"temperature": float("nan")},
        {"temperature": 3},
        {"temperature": True},
        {"top_p": 0},
        {"top_k": 0},
        {"top_k": 1.5},
        {"seed": -1},
        {"max_output_tokens": 8192},
        {"max_output_tokens": False},
    ],
)
def test_invalid_sampling_cannot_become_a_quality_configuration(change):
    value = {"runtime": {"max_output_tokens": 512, "context_window": 8192, **change}}
    with pytest.raises(ValueError):
        sampling_settings(value)


def test_native_json_turn_preserves_explicit_sampling(tmp_path):
    requests = []

    def request(endpoint, payload):
        requests.append((endpoint, payload))
        if endpoint == "/apply-template":
            return {"prompt": "synthetic prompt"}
        return {"choices": [{"finish_reason": "stop", "message": {"content": '{"result":4}'}}]}

    client = SimpleNamespace(
        sampling_args={
            "max_tokens": 2048,
            "temperature": 1.0,
            "seed": 42,
            "top_p": 0.95,
            "top_k": 64,
        },
        template_args=lambda: {"chat_template_kwargs": {"enable_thinking": True}},
        request=request,
        count_text=lambda *a, **kw: 10,
        context_window=8192,
        model_name="test",
    )
    native_turn(client, [{"role": "user", "content": "2+2?"}])
    data = requests[-1][1]
    assert data["temperature"] == 1.0
    assert data["top_p"] == 0.95 and data["top_k"] == 64
    assert data["chat_template_kwargs"]["enable_thinking"] is True


def test_quality_report_records_actual_thinking_and_rejects_profile_mismatch(tmp_path):
    value = load_profile(prepare_thinking(profile(tmp_path), tmp_path), tmp_path)
    client = helper_client(value)
    client.completion = lambda messages: "4"
    client.get_response_info = lambda: {"finish_reason": "stop", "reasoning_chars": 100}
    report = evaluate_suite(client, value, suite(tmp_path), tmp_path / "quality.json")
    assert report["generation"] == generation_conditions(value)
    assert report["generation"]["thinking"] is True
    assert report["cases"][0]["passed"]
    assert report["cases"][0]["reasoning_chars"] == 100
    client.enable_thinking = False
    with pytest.raises(ValueError, match="thinking mode differs"):
        evaluate_suite(client, value, tmp_path / "suite.jsonl", tmp_path / "bad.json")
    assert not (tmp_path / "bad.json").exists()
    client.enable_thinking = True
    client.sampling_args["temperature"] = 0.0
    with pytest.raises(ValueError, match="sampling differs"):
        evaluate_suite(client, value, tmp_path / "suite.jsonl", tmp_path / "bad.json")


@pytest.mark.parametrize("mode", ["length", "missing"])
def test_truncated_or_missing_final_answer_never_passes_quality(tmp_path, mode):
    value = profile(tmp_path)
    client = helper_client(value)
    client.get_response_info = lambda: {"finish_reason": "length", "reasoning_chars": 200}

    def completion(messages):
        if mode == "missing":
            raise ValueError("No final answer")
        return "4"

    client.completion = completion
    report = evaluate_suite(client, value, suite(tmp_path), tmp_path / "quality.json")
    assert not report["cases"][0]["passed"]
    if mode == "missing":
        assert report["cases"][0]["error"] == "No final answer"


def test_legacy_generation_conditions_keep_existing_baselines_comparable(tmp_path):
    value = profile(tmp_path)
    assert generation_conditions(value) == {
        "temperature": 0.0,
        "seed": 42,
        "max_tokens": 512,
        "context_window": 8192,
        "thinking": False,
    }
