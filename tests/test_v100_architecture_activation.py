import copy
import json
from pathlib import Path

import pytest

from rlm.v100 import architecture_promotion, fresh_audit, lineages
from rlm.v100.common import atomic_json
from rlm.v100.protection import ExpertRegistry, execution_hash, file_hash
from tests.test_v100_duel import profile


def evidence(chosen, passes):
    return {
        "schema": "v100-quality-v1",
        "model_sha256": file_hash(Path(chosen["server"]["model"])),
        "execution_sha256": execution_hash(chosen),
        "suite_sha256": "fixed-suite",
        "generation": {},
        "memory_mode": "fixed",
        "cases": [
            {"id": str(i), "passed": value, "finish_reason": "stop", "error": None}
            for i, value in enumerate(passes)
        ],
    }


def ready(tmp_path, monkeypatch, improves=True):
    parent = profile(tmp_path)
    model = tmp_path / "parent.gguf"
    model.write_bytes(b"predecessor")
    parent["server"]["model"] = str(model)
    parent["training"]["init_adapter"] = ""
    candidate = copy.deepcopy(parent)
    child = tmp_path / "alternative.gguf"
    child.write_bytes(b"different architecture")
    candidate["server"]["model"] = str(child)
    candidate["training"]["base_model"] = str(tmp_path / "alternative-base")
    candidate["runtime"]["model_version"] = "alternative"
    old, new = evidence(parent, [True, False]), evidence(candidate, [True, improves])
    registry = ExpertRegistry(tmp_path / "research/experts")
    registry.register("foundation-test", candidate, "Distinct useful architecture", old, new)
    audit = tmp_path / "fresh"
    atomic_json(audit / "manifest.json", {"parent_sha256": file_hash(model)})
    monkeypatch.setattr(fresh_audit, "verified_gate", lambda *args: {"passed": True})
    monkeypatch.setattr(architecture_promotion, "probe", lambda *args: None)
    result = {"eligible": True, "expert_id": "foundation-test", "fresh_audit": str(audit)}
    return parent, result, [old], registry


def test_improved_alternate_base_activated_old_weights_and_gates_survive(tmp_path, monkeypatch):
    parent, result, gates, registry = ready(tmp_path, monkeypatch)
    original = copy.deepcopy(parent)
    candidate, receipt = architecture_promotion.activate(tmp_path, parent, result, gates)
    assert receipt["activated"] and receipt["improved_cases"] == 1
    assert parent == original
    assert candidate["training"]["base_model"] != parent["training"]["base_model"]
    assert not candidate["resources"].get("branch_lineages")
    parents, retained = lineages.read(candidate)
    assert parents == {} and retained[0] == gates[0] and len(retained) == 2
    Path(parent["server"]["model"]).unlink()
    predecessor = registry.get(receipt["predecessor_expert"])
    assert Path(predecessor["profile"]["server"]["model"]).read_bytes() == b"predecessor"
    restored, rollback = architecture_promotion.rollback(
        tmp_path, candidate, "Native server failed twice"
    )
    assert rollback["restored"]
    assert restored["training"]["base_model"] == parent["training"]["base_model"]
    assert result["expert_id"] in restored["resources"]["quarantined_foundations"]
    selected, verdict = architecture_promotion.activate(tmp_path, restored, result, gates)
    assert selected is restored and not verdict["activated"]


def test_equal_quality_does_not_trigger_architecture_churn(tmp_path, monkeypatch):
    parent, result, gates, registry = ready(tmp_path, monkeypatch, improves=False)
    selected, receipt = architecture_promotion.activate(tmp_path, parent, result, gates)
    assert selected is parent and not receipt["activated"]
    assert len(registry.list()) == 1


def test_architecture_boot_failure_does_not_change_parent(tmp_path, monkeypatch):
    parent, result, gates, _ = ready(tmp_path, monkeypatch)
    original = copy.deepcopy(parent)

    def unavailable(*args):
        raise RuntimeError("Native launch failed")

    monkeypatch.setattr(architecture_promotion, "probe", unavailable)
    with pytest.raises(RuntimeError, match="launch"):
        architecture_promotion.activate(tmp_path, parent, result, gates)
    assert parent == original


@pytest.mark.parametrize("fault", ["weights", "fresh", "evidence"])
def test_architecture_proof_tampering_rejected(tmp_path, monkeypatch, fault):
    parent, result, gates, registry = ready(tmp_path, monkeypatch)
    if fault == "weights":
        target = Path(registry.get(result["expert_id"])["profile"]["server"]["model"])
        target.chmod(0o644)
        target.write_bytes(b"changed")
    elif fault == "fresh":
        monkeypatch.setattr(fresh_audit, "verified_gate", lambda *args: {"passed": False})
    else:
        gates[0]["cases"][1]["passed"] = True
        gates[0]["model_sha256"] = "missing live predecessor"
    with pytest.raises(ValueError):
        architecture_promotion.activate(tmp_path, parent, result, gates)


def test_retained_architecture_gate_hash_checked_after_restart(tmp_path, monkeypatch):
    parent, result, gates, _ = ready(tmp_path, monkeypatch)
    candidate, _ = architecture_promotion.activate(tmp_path, parent, result, gates)
    path = Path(candidate["resources"]["protected_architecture_gates"]["path"])
    path.chmod(0o644)
    path.write_text(json.dumps({"reports": []}))
    with pytest.raises(ValueError, match="ancestor"):
        lineages.read(candidate)
