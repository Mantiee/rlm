import json

import pytest
import requests

from rlm.v100 import competition, model_candidates, model_policy, model_roles, remote_helper
from rlm.v100.common import atomic_json


def test_current_candidates_exclude_retired_families():
    assert all(not model_policy.retired(name) for _, name in model_policy.WINDOWS_ROLES)
    assert all(not model_policy.retired(name) for name in model_policy.NODE_MODELS)
    assert model_candidates.CATALOG["planner"][0] == "unsloth/Qwen3.8-27B-GGUF"


def test_failed_quality_cannot_admit_a_new_name():
    assert not model_policy.admitted_panel({"passed": 19, "critical_passed": True})
    assert not model_policy.admitted_panel({"passed": 20, "critical_passed": False})
    assert model_policy.admitted_panel({"passed": 20, "critical_passed": True})


def test_retired_remote_production_routes_to_unavailability(tmp_path):
    atomic_json(tmp_path / "research/model-roles/model-policy.json", {"version": 90})
    profile = {"runtime": {"model_name": "qwen3.5:9b-q8_0"}, "resources": {"device": "remote"}}
    with pytest.raises(requests.ConnectionError, match="Retired remote"):
        competition.helper_client(profile, tmp_path)


@pytest.mark.parametrize("name", ["qwen3.5:2b-q4_K_M", "qwen3.6:35b-a3b-q4_K_M"])
def test_previously_admitted_old_role_is_archived_not_selected(tmp_path, name):
    folder = tmp_path / "research/model-roles"
    profile = {"runtime": {"model_name": name}}
    atomic_json(folder / "profile.json", profile)
    atomic_json(folder / "report.json", {"admitted": True})
    atomic_json(
        folder / "active.json",
        {
            "roles": {
                "fast": {
                    "profile": str(folder / "profile.json"),
                    "report": str(folder / "report.json"),
                }
            }
        },
    )
    assert model_roles.selected_profile(tmp_path, "fast") is None
    assert json.loads((folder / "report.json").read_text())["admitted"]


def test_gemma_identity_does_not_accept_qwen_tokenizer():
    info = {
        "details": {"family": "gemma4", "quantization_level": "Q4_K_M"},
        "model_info": {"tokenizer.ggml.model": "gemma4"},
        "template": "fixed template",
    }
    client = remote_helper.OllamaResearchClient(
        base_url="http://192.168.0.61:11435",
        model_name="gemma4:e2b-it-q4_K_M",
        model_digest="a" * 64,
        metadata_sha256="",
        max_vram_gib=7,
        context_window=32768,
        max_output_tokens=256,
    )
    client.remote_request = (
        lambda path, data=None: {"models": [{"name": client.model_name, "digest": "a" * 64}]}
        if path == "/api/tags"
        else info
    )
    assert client.identity() == info
    assert client.count_text("a b") == 5
    info["model_info"]["tokenizer.ggml.model"] = "gpt2"
    with pytest.raises(ValueError, match="metadata"):
        client.identity()
