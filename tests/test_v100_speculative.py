import copy
import json
import subprocess
import tomllib
from pathlib import Path

import pytest

import rlm.v100.speculative as speculative
from rlm.v100.common import load_profile
from rlm.v100.speculative import (
    ASSISTANT_REVISION,
    benchmark_prompts,
    compare_reports,
    mtp_profile_text,
    prepare_mtp,
)
from rlm.v100.speculative import (
    test_mtp as run_mtp,
)


def test_experiment_profiles_preserve_target_and_memory(tmp_path):
    source = (Path(__file__).parents[1] / "profiles/v100.toml").read_text()
    # Compatibility with already-installed v100.2 profiles without spec_type.
    source = source.replace('spec_type = "draft-simple"\n', "")
    original = tomllib.loads(source)
    experiment = tmp_path / "experiment.toml"
    experiment.write_text(mtp_profile_text(source, tmp_path / "draft.gguf", 2))
    data = tomllib.loads(experiment.read_text())
    assert data["runtime"]["base_url"] == "http://127.0.0.1:8089"
    assert data["server"]["spec_type"] == "draft-mtp"
    assert data["server"]["draft_tokens"] == 2
    assert data["server"]["model"] == original["server"]["model"]
    assert data["memory"] == original["memory"]
    assert data["training"] == original["training"]
    assert load_profile(experiment, tmp_path)["runtime"]["context_window"] == 8192
    assert tomllib.loads(mtp_profile_text(source, None, 2))["server"]["draft_model"] == ""


def test_speed_prompts_are_fixed_and_include_long_workload():
    first, second = benchmark_prompts(), benchmark_prompts()
    assert first == second
    assert len(first[-1][1]) > len(first[0][1]) * 100
    assert first[1][0] == "code"


def benchmark(tmp_path):
    return {
        "profile": load_profile(Path(__file__).parents[1] / "profiles/v100.toml", tmp_path),
        "suite": True,
        "enable_thinking": False,
        "prompts_sha256": "fixed",
        "cases": {"short": {"median_tokens_per_second": 48, "median_seconds": 10}},
        "runs": [
            {
                "case": "short",
                "repeat": 0,
                "answer": "A",
                "timings": {"draft_n": 8, "draft_n_accepted": 6},
            }
        ],
    }


def test_comparison_records_speed_and_differences_without_quality_claim(tmp_path):
    old = benchmark(tmp_path)
    new = copy.deepcopy(old)
    new["cases"]["short"] = {"median_tokens_per_second": 72, "median_seconds": 8}
    new["runs"][0]["answer"] = "B"
    report = compare_reports(old, new)
    assert report["median_case_throughput_ratio"] == 1.5
    assert report["cases"]["short"]["latency_ratio"] == 1.25
    assert report["draft_acceptance"] == 0.75
    assert report["answers_exact_match"] is False
    assert report["quality_evaluated"] is False
    assert report["promote_automatically"] is False


@pytest.mark.parametrize("field", ["enable_thinking", "suite", "prompts_sha256"])
def test_comparison_refuses_different_workloads(tmp_path, field):
    old = benchmark(tmp_path)
    new = copy.deepcopy(old)
    new[field] = "different"
    with pytest.raises(ValueError, match=field):
        compare_reports(old, new)


def test_comparison_refuses_missing_drafts_and_changed_target(tmp_path):
    old = benchmark(tmp_path)
    new = copy.deepcopy(old)
    new["profile"]["server"]["model"] = "different.gguf"
    with pytest.raises(ValueError, match="settings"):
        compare_reports(old, new)
    new = copy.deepcopy(old)
    new["runs"][0]["timings"] = {}
    with pytest.raises(ValueError, match="activation"):
        compare_reports(old, new)


def prepare_fixture(tmp_path):
    profile_path = tmp_path / "research/v100.toml"
    profile_path.parent.mkdir()
    profile_path.write_text((Path(__file__).parents[1] / "profiles/v100.toml").read_text())
    profile = load_profile(profile_path, tmp_path)
    server = Path(profile["server"]["binary"])
    paths = [
        server,
        server.with_name("llama-quantize"),
        server.parents[2] / "convert_hf_to_gguf.py",
        tmp_path / "venvs/convert/bin/python",
        tmp_path / "venvs/train/bin/python",
        tmp_path / "venvs/mtp-tools/bin/python",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    conversion = server.parents[2] / "conversion/gemma.py"
    conversion.parent.mkdir()
    conversion.write_text("Gemma4UnifiedAssistantForCausalLM")
    base = Path(profile["training"]["base_model"])
    base.mkdir(parents=True)
    target = Path(profile["server"]["model"])
    target.parent.mkdir(parents=True)
    target.write_text("target")
    (base / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Gemma4UnifiedForConditionalGeneration"],
                "text_config": {"hidden_size": 3840, "vocab_size": 262144},
            }
        )
    )
    tokenizer = {"model": {"vocab": {"x": 1}}, "added_tokens": []}
    (base / "tokenizer.json").write_text(json.dumps(tokenizer))
    assistant = tmp_path / "models/gemma4-12b-assistant" / ASSISTANT_REVISION
    assistant.mkdir(parents=True)
    (assistant / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Gemma4UnifiedAssistantForCausalLM"],
                "backbone_hidden_size": 3840,
                "text_config": {"vocab_size": 262144},
            }
        )
    )
    (assistant / "tokenizer.json").write_text(json.dumps(tokenizer))
    (assistant / "model.safetensors").write_bytes(b"fixture")
    return profile_path


def test_preparation_atomic_export_provenance_and_profile_preservation(tmp_path, monkeypatch):
    profile_path = prepare_fixture(tmp_path)
    baseline_text = profile_path.read_text()
    existing = tmp_path / "research/v100-mtp4.toml"
    existing.write_text("user-owned profile")
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if "--help" in command:
            return subprocess.CompletedProcess(command, 0, "draft-mtp", "")
        if command[1:2] == ["-c"] and "metadata" in command[2]:
            return subprocess.CompletedProcess(command, 0, "1.33.0\n", "")
        if "--outtype" in command:
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
            Path(command[command.index("--outfile") + 1]).write_bytes(b"GGUF" + b"x" * 2048)
        if "Q8_0" in command:
            Path(command[2]).write_bytes(b"GGUF" + b"y" * 2048)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(speculative.subprocess, "run", fake_run)
    prepare_mtp(profile_path, tmp_path)
    output = (
        tmp_path
        / "models/gguf/gemma4-12b-assistant"
        / ASSISTANT_REVISION
        / "gemma4-12b-assistant-Q8_0.gguf"
    )
    assert output.read_bytes().startswith(b"GGUF")
    assert not list(output.parent.glob("*.partial.gguf"))
    assert profile_path.read_text() == baseline_text
    assert existing.read_text() == "user-owned profile"
    prepare_mtp(profile_path, tmp_path)
    assert sum("--outtype" in c for c in commands) == 1
    output.write_bytes(b"GGUF" + b"corrupt" * 300)
    with pytest.raises(ValueError, match="hash changed"):
        prepare_mtp(profile_path, tmp_path)


def test_preparation_rejects_token_mismatch_before_conversion(tmp_path, monkeypatch):
    profile_path = prepare_fixture(tmp_path)
    assistant = tmp_path / "models/gemma4-12b-assistant" / ASSISTANT_REVISION
    (assistant / "tokenizer.json").write_text(
        json.dumps({"model": {"vocab": {"y": 1}}, "added_tokens": []})
    )
    monkeypatch.setattr(
        speculative.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(
            cmd, 0, "draft-mtp" if "--help" in cmd else "1.33.0", ""
        ),
    )
    with pytest.raises(ValueError, match="token vocabulary"):
        prepare_mtp(profile_path, tmp_path)


def test_runner_refuses_busy_gpu_without_starting_processes(tmp_path, monkeypatch):
    profile_path = prepare_fixture(tmp_path)
    source = profile_path.read_text()
    for name, draft in (
        ("baseline", None),
        ("mtp2", tmp_path / "draft.gguf"),
        ("mtp4", tmp_path / "draft.gguf"),
    ):
        (tmp_path / f"research/v100-{name}.toml").write_text(
            mtp_profile_text(source, draft, 2 if name != "mtp4" else 4)
        )
    (tmp_path / "draft.gguf").write_text("draft")
    monkeypatch.setattr(
        speculative.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "10232\n", ""),
    )
    monkeypatch.setattr(
        speculative.subprocess,
        "Popen",
        lambda *a, **kw: pytest.fail("Must not start a child on a busy GPU"),
    )
    with pytest.raises(ValueError, match="Stop the inference server"):
        run_mtp(tmp_path, 3)


def test_runner_cleans_up_only_own_children_after_benchmark_failure(tmp_path, monkeypatch):
    profile_path = prepare_fixture(tmp_path)
    source = profile_path.read_text()
    for name, draft in (
        ("baseline", None),
        ("mtp2", tmp_path / "draft.gguf"),
        ("mtp4", tmp_path / "draft.gguf"),
    ):
        (tmp_path / f"research/v100-{name}.toml").write_text(
            mtp_profile_text(source, draft, 2 if name != "mtp4" else 4)
        )
    (tmp_path / "draft.gguf").write_text("draft")
    children = []

    class Child:
        def __init__(self, *args, **kwargs):
            self.terminated = False
            children.append(self)

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

    def fake_run(cmd, **kwargs):
        if "--query-gpu=memory.used" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "1\n", "")
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(speculative.subprocess, "run", fake_run)
    monkeypatch.setattr(speculative.subprocess, "Popen", Child)
    monkeypatch.setattr(
        speculative.requests.Session,
        "get",
        lambda *a, **kw: type("Response", (), {"status_code": 200})(),
    )
    with pytest.raises(subprocess.CalledProcessError):
        run_mtp(tmp_path, 3)
    assert len(children) == 2
    assert all(child.terminated for child in children)
    assert profile_path.read_text() == source
