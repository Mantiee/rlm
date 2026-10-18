import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("mtp_gate_pkg", HERE / "mtp_gate.py")
gate = importlib.util.module_from_spec(spec)
sys.modules["mtp_gate_pkg"] = gate
spec.loader.exec_module(gate)


def report(passed, model="m", generation=None):
    return {
        "schema": "v100-quality-v1",
        "suite_sha256": "s",
        "model_sha256": model,
        "generation": generation or {"temperature": 1.0, "seed": 42},
        "memory_mode": "none",
        "cases": [{"id": f"c{i}", "passed": p} for i, p in enumerate(passed)],
    }


def test_without_control_any_regression_fails():
    base = report([True, True, False])
    cand = report([True, False, True])
    assert gate.quality_verdict(base, cand)["passed"] is False
    assert gate.quality_verdict(base, report([True, True, True]))["passed"] is True


def test_deterministic_control_keeps_strict_rule():
    base = report([True, True, False])
    verdict = gate.quality_verdict(base, report([True, False, True]), report([True, True, False]))
    assert verdict["control_flips"] == 0 and verdict["passed"] is False


def test_noise_within_control_flips_passes():
    base = report([True, True, True, False, False])
    control = report([True, False, True, True, False])  # 2 flips by sampling alone
    cand = report([False, True, True, True, False])  # 1 regression, same pass count
    verdict = gate.quality_verdict(base, cand, control)
    assert verdict["passed"] is True and verdict["strict_passed"] is False


def test_more_regressions_than_noise_fails():
    base = report([True, True, True, True, False])
    control = report([True, True, True, False, True])  # 2 flips
    cand = report([False, False, False, True, True])  # 3 regressions
    assert gate.quality_verdict(base, cand, control)["passed"] is False


def test_lower_pass_count_than_weaker_baseline_fails():
    base = report([True, True, True, False])
    control = report([True, True, False, True])  # 2 flips, 3 passes each
    cand = report([True, False, False, False])  # 2 regressions but only 1 pass
    assert gate.quality_verdict(base, cand, control)["passed"] is False


def test_control_must_match_conditions_and_model():
    base = report([True])
    with pytest.raises(ValueError):
        gate.quality_verdict(base, report([True]), report([True], generation={"temperature": 0}))
    with pytest.raises(ValueError):
        gate.quality_verdict(base, report([True]), report([True], model="other"))


def test_valid_accepts_control_proof_and_rejects_tampering(tmp_path, monkeypatch):
    model = tmp_path / "target.gguf"
    draft = tmp_path / "draft.gguf"
    model.write_bytes(b"target")
    draft.write_bytes(b"draft")
    binary = tmp_path / "bin" / "llama-server"
    binary.parent.mkdir()
    binary.write_bytes(b"server")
    sha = gate.file_hash(model)
    files = {
        "comparison.json": {"speed": 1},
        "quality-baseline.json": report([True, True, False], model=sha),
        "quality-candidate.json": report([True, False, True], model=sha),
        "quality-control.json": report([False, True, True], model=sha),
    }
    for name, value in files.items():
        (tmp_path / name).write_text(json.dumps(value))
    monkeypatch.setattr(
        gate, "recommend_mtp", lambda *a: {"recommended_speed_profile": "mtp2"}
    )
    profile = {"server": {"model": str(model), "binary": str(binary), "draft_model": str(draft), "draft_tokens": 2,
                          "spec_type": "draft-mtp"}, "runtime": {}, "resources": {}}
    proof = {
        "execution": gate.identity(profile),
        "draft_hash": gate.file_hash(draft),
        "draft_tokens": 2,
        "reports": {str(tmp_path / n): gate.file_hash(tmp_path / n) for n in files},
    }
    profile["resources"]["mtp_validation"] = proof
    assert gate.valid(profile) is True
    # Legacy strict 3-report proof with a regression stays invalid.
    legacy = dict(proof, reports={k: v for k, v in proof["reports"].items()
                                  if not k.endswith("quality-control.json")})
    profile["resources"]["mtp_validation"] = legacy
    assert gate.valid(profile) is False
    # A changed report invalidates the proof.
    profile["resources"]["mtp_validation"] = proof
    (tmp_path / "quality-control.json").write_text(json.dumps(report([True, True, False], model=sha)))
    assert gate.valid(profile) is False
