import copy
import json
from pathlib import Path

import pytest

from rlm.v100 import (
    architecture_promotion,
    architectures,
    competition,
    fresh_audit,
    scratch_master,
    serving,
)
from rlm.v100.common import atomic_json
from rlm.v100.protection import ExpertRegistry, execution_hash, file_hash
from tests.test_v100_architecture_activation import evidence
from tests.test_v100_architectures import workload
from tests.test_v100_campaign import goal
from tests.test_v100_continual import profile


def candidate(root):
    goal(root)
    parent = profile(root)
    parent["runtime"].update(tool_protocol="json", context_window=8192, max_output_tokens=32)
    code = "raise RuntimeError('must never be imported on host')\ndef build(config): return None\n"
    architectures.create_candidate(root, "A", "new", code, "Different network")
    folder = architectures.candidate_path(root, "new")
    (folder / "trial/weights").mkdir(parents=True)
    (folder / "trial/weights/weights.safetensors").write_bytes(b"frozen test weights")
    budget = {**architectures.DEFAULT_BUDGET, "context_window": 8192}
    atomic_json(folder / "trial/budget.json", budget)
    return parent, scratch_master.profile_for(root, parent, "new", budget), budget


def test_scratch_freeze_does_not_import_code_and_detects_edits(tmp_path):
    _, chosen, budget = candidate(tmp_path)
    assert scratch_master.verify(chosen) == budget
    Path(chosen["resources"]["scratch_source"]).write_text("forged source")
    with pytest.raises(ValueError, match="source changed"):
        scratch_master.verify(chosen)


def test_scratch_registry_relocation_binds_source_budget_and_weights(tmp_path):
    _, chosen, _ = candidate(tmp_path)
    registry = ExpertRegistry(tmp_path / "research/experts")
    entry = registry.register("scratch-one", chosen, "Immutable scratch expert")
    selected = entry["profile"]
    assert execution_hash(selected) == execution_hash(chosen)
    scratch_master.verify(selected)
    Path(chosen["resources"]["scratch_source"]).write_text("changed original")
    scratch_master.verify(selected)
    budget = Path(selected["resources"]["scratch_budget"])
    budget.chmod(0o644)
    budget.write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"):
        registry.get("scratch-one")


def test_scratch_client_uses_isolated_transport_and_preserves_finish(tmp_path, monkeypatch):
    _, chosen, _ = candidate(tmp_path)
    monkeypatch.setattr("rlm.v100.researchers.available_ram_gib", lambda: 32)
    calls = []

    def isolated(output, run, inputs, budget, mode):
        calls.append((mode, json.loads((inputs / "data.json").read_text())))
        architectures.verify_candidate(output)
        (run / "infer.log").write_text(
            json.dumps(
                {"id": "turn", "answer": "4", "completion_tokens": 1, "finish_reason": "stop"}
            )
            + "\n"
        )

    monkeypatch.setattr(scratch_master, "phase", isolated)
    client = competition.helper_client(chosen, tmp_path)
    serving.assert_served_expert(client, chosen, tmp_path)
    assert client.completion("2+2") == "4"
    assert client.get_response_info()["finish_reason"] == "stop"
    assert calls[0][0] == "infer"
    assert calls[0][1][0]["messages"] == [{"role": "user", "content": "2+2"}]
    assert not list((tmp_path / "research/scratch-runtime").glob("*/source"))


def test_scratch_never_silently_falls_back_when_sandbox_missing(tmp_path, monkeypatch):
    _, chosen, _ = candidate(tmp_path)
    monkeypatch.setattr("rlm.v100.researchers.available_ram_gib", lambda: 32)

    def unavailable(*args):
        raise FileNotFoundError("bubblewrap unavailable")

    monkeypatch.setattr(scratch_master, "phase", unavailable)
    with pytest.raises(FileNotFoundError, match="bubblewrap"):
        competition.helper_client(chosen).completion("2+2")


def test_scratch_cannot_certify_another_clients_execution(tmp_path):
    _, chosen, _ = candidate(tmp_path)
    client = competition.helper_client(chosen)
    edited = copy.deepcopy(chosen)
    edited["runtime"]["max_output_tokens"] += 1
    with pytest.raises(ValueError, match="another frozen execution"):
        serving.assert_served_expert(client, edited, tmp_path)


def test_scratch_queue_budget_is_frozen_and_short_context_retains_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(serving, "process_identity", lambda pid: "same-process")
    parent, _, budget = candidate(tmp_path)
    budget["context_window"] = 1024
    scratch_master.propose(tmp_path, "A", "new", budget)
    with pytest.raises(ValueError, match="different settings"):
        scratch_master.propose(tmp_path, "A", "new", {**budget, "steps": 41})
    pool, suite = workload(tmp_path)
    result = scratch_master.trial(
        tmp_path, parent, pool, suite, [], scratch_master.pending(tmp_path)[0]
    )
    assert not result["eligible"] and "context/output" in result["detail"]
    assert not scratch_master.pending(tmp_path)


def test_scratch_activation_and_rollback_keep_native_predecessor(tmp_path, monkeypatch):
    parent, chosen, _ = candidate(tmp_path)
    parent_model = tmp_path / "native.gguf"
    parent_model.write_bytes(b"native predecessor")
    parent["server"]["model"] = str(parent_model)
    before, after = evidence(parent, [True, False]), evidence(chosen, [True, True])
    registry = ExpertRegistry(tmp_path / "research/experts")
    registry.register("scratch-active", chosen, "Custom architecture", before, after)
    audit = tmp_path / "fresh"
    atomic_json(audit / "manifest.json", {"parent_sha256": file_hash(parent_model)})
    monkeypatch.setattr(fresh_audit, "verified_gate", lambda *args: {"passed": True})
    monkeypatch.setattr(architecture_promotion, "probe", lambda *args: None)
    selected, receipt = architecture_promotion.activate(
        tmp_path,
        parent,
        {"eligible": True, "expert_id": "scratch-active", "fresh_audit": str(audit)},
        [before],
    )
    assert receipt["activated"] and scratch_master.is_scratch(selected)
    scratch_master.verify(selected)
    restored, rollback = architecture_promotion.rollback(tmp_path, selected, "Sandbox unavailable")
    assert rollback["restored"] and not scratch_master.is_scratch(restored)
    assert Path(restored["server"]["model"]).read_bytes() == b"native predecessor"


def test_interrupted_scratch_audit_is_not_reused_after_restart(tmp_path, monkeypatch):
    parent, _, budget = candidate(tmp_path)
    scratch_master.propose(tmp_path, "A", "new", budget)
    path = scratch_master.pending(tmp_path)[0]
    proposal = json.loads(path.read_text())
    proposal.update(state="running", pid=123, process_start="old")
    atomic_json(path, proposal)
    monkeypatch.setattr(serving, "process_identity", lambda pid: "different")
    assert scratch_master.pending(tmp_path) == []
    assert json.loads(path.read_text())["state"] == "interrupted; predecessor retained"


def test_full_weight_worker_continuation_and_inference_are_real(tmp_path, monkeypatch, capsys):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file, save_file

    from rlm.v100 import architecture_worker

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(257, 8)
            self.projection = torch.nn.Linear(8, 257)

        def forward(self, tokens):
            return self.projection(self.embedding(tokens))

    initial = Tiny()
    initial.projection.bias.data.fill_(-100)
    initial.projection.bias.data[52] = 100
    start = tmp_path / "initial.safetensors"
    save_file(initial.state_dict(), str(start))
    config = {
        **architectures.DEFAULT_BUDGET,
        "steps": 2,
        "init_weights": str(start),
        "init_weights_sha256": file_hash(start),
    }
    atomic_json(tmp_path / "config.json", config)
    atomic_json(
        tmp_path / "data.json",
        [{"messages": [{"role": "user", "content": "2+2"}, {"role": "assistant", "content": "4"}]}],
    )
    weights = tmp_path / "weights"
    weights.mkdir()
    monkeypatch.setattr(architecture_worker, "load_model", lambda settings: Tiny())
    monkeypatch.setattr(
        "sys.argv",
        [
            "worker",
            "train",
            str(tmp_path / "config.json"),
            str(tmp_path / "data.json"),
            str(weights),
        ],
    )
    architecture_worker.main()
    trained = load_file(str(weights / "weights.safetensors"))
    assert any(
        not torch.equal(trained[name], value) for name, value in initial.state_dict().items()
    )
    capsys.readouterr()
    config["max_new_tokens"] = 2
    atomic_json(tmp_path / "config.json", config)
    atomic_json(
        tmp_path / "data.json",
        [
            {
                "id": "one",
                "messages": [{"role": "user", "content": "2+2"}],
                "sampling": {"temperature": 0},
            }
        ],
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "worker",
            "infer",
            str(tmp_path / "config.json"),
            str(tmp_path / "data.json"),
            str(weights),
        ],
    )
    architecture_worker.main()
    result = json.loads(capsys.readouterr().out)
    assert result["answer"] == "44" and result["finish_reason"] == "length"
    assert result["completion_tokens"] == 2
