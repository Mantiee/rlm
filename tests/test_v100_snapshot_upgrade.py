import copy
import json
from pathlib import Path

import pytest

from rlm.v100 import campaign, mission, public_benchmarks, self_code
from rlm.v100.common import atomic_json
from rlm.v100.protection import file_hash


def snapshot(root):
    folder = root / "research/public-benchmarks/old"
    atomic_json(folder / "questions.json", {"cases": "pinned"})
    atomic_json(folder / "references.json", {"references": "pinned"})
    manifest = {
        "questions_sha256": file_hash(folder / "questions.json"),
        "references_sha256": file_hash(folder / "references.json"),
        "worker_sha256": file_hash(
            Path(public_benchmarks.__file__).with_name("benchmark_worker.py")
        ),
    }
    atomic_json(folder / "manifest.json", manifest)
    atomic_json(root / "research/public-benchmarks/current.json", {"snapshot": str(folder)})
    return folder, manifest


def test_matching_snapshot_never_redownloads_or_changes_evidence(tmp_path, monkeypatch):
    folder, _ = snapshot(tmp_path)
    monkeypatch.setattr(
        public_benchmarks, "prepare", lambda *args: pytest.fail("Unexpected benchmark rebuild")
    )
    assert public_benchmarks.prepare_current(tmp_path) == folder


def test_adapter_upgrade_creates_separate_snapshot_and_preserves_old_proof(tmp_path, monkeypatch):
    folder, manifest = snapshot(tmp_path)
    manifest["worker_sha256"] = "old-adapter"
    atomic_json(folder / "manifest.json", manifest)
    before = file_hash(folder / "manifest.json")
    calls = []
    monkeypatch.setattr(
        public_benchmarks,
        "prepare",
        lambda root, limit: calls.append((root, limit)) or root / "new-snapshot",
    )
    assert public_benchmarks.prepare_current(tmp_path) == tmp_path / "new-snapshot"
    assert calls == [(tmp_path, 20)] and file_hash(folder / "manifest.json") == before


def test_tampered_questions_are_not_masked_as_a_valid_upgrade(tmp_path):
    folder, _ = snapshot(tmp_path)
    atomic_json(folder / "questions.json", {"cases": "changed"})
    with pytest.raises(ValueError, match="inputs changed"):
        public_benchmarks.prepare_current(tmp_path)


@pytest.mark.parametrize("changed", [False, True])
def test_campaign_refreshes_only_public_baseline_and_keeps_accepted_weights(
    tmp_path, monkeypatch, changed
):
    folder, _ = snapshot(tmp_path)
    old = tmp_path / "accepted-public-baseline.json"
    atomic_json(
        old, {"snapshot_sha256": "old-snapshot" if changed else file_hash(folder / "manifest.json")}
    )
    selected = {
        "server": {"model": "accepted-model.gguf"},
        "runtime": {},
        "training": {"init_adapter": "accepted-adapter"},
        "resources": {"public_baseline": str(old), "public_baseline_sha256": file_hash(old)},
    }
    monkeypatch.setattr(mission, "status", lambda *args: {"running": False})
    monkeypatch.setattr(campaign, "load_profile", lambda *args: copy.deepcopy(selected))
    monkeypatch.setattr(campaign, "measured_speed", lambda root, profile: (profile, None))
    monkeypatch.setattr(campaign, "configure_helper", lambda *args: {})
    monkeypatch.setattr(self_code, "prepare", lambda *args: {})
    monkeypatch.setattr(public_benchmarks, "prepare_current", lambda *args: folder)
    path = campaign.prepare(tmp_path, path=tmp_path / "input.json", desktop=False)
    value = json.loads(path.read_text())
    assert value["server"] == selected["server"] and value["training"] == selected["training"]
    assert old.exists() and file_hash(old) == selected["resources"]["public_baseline_sha256"]
    if changed:
        assert "public_baseline" not in value["resources"]
        assert value["resources"]["public_baseline_requires_refresh"]["previous_report"] == str(old)
    else:
        assert value["resources"]["public_baseline"] == str(old)
