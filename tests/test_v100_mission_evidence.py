import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlm.v100 import competition, mission, mission_chat, mission_evidence, research_tools
from rlm.v100.common import atomic_json


def running(root, monkeypatch):
    run = root / "research/mission/run-test"
    run.mkdir(parents=True)
    monkeypatch.setattr(
        mission,
        "status",
        lambda root: {
            "running": True,
            "run": str(run),
            "state": {"phase": "baseline-before-weight-updates"},
            "learning": {"completed_cycles": 0},
        },
    )
    return run


def test_missing_training_evidence_is_unknown_not_zero(tmp_path, monkeypatch):
    run = running(tmp_path, monkeypatch)
    atomic_json(run / "baseline-131072.json", {"cases": [{"passed": True}, {"passed": False}]})
    value = research_tools.ResearchTools(tmp_path, {}).execute("mission_evidence", {})
    assert value["optimizer_updates_observed"] is None
    assert value["accepted_weight_updates_this_run"] is None
    assert value["reports"][0]["cases"] == 2
    assert value["reports"][0]["passed"] == 1
    assert "GUI readiness is not a prerequisite" in value["scope"]


def test_successful_optimizer_steps_differ_from_attempts_and_acceptance(tmp_path, monkeypatch):
    run = running(tmp_path, monkeypatch)
    health = {
        "attempted_steps": 7,
        "optimizer_updates": 5,
        "amp_skipped_steps": 2,
        "finite_gradient_steps": 5,
    }
    atomic_json(run / "learning/update-0001/A/training/training_health.json", health)
    atomic_json(
        run / "learning/state.json",
        {
            "cycles": [
                {
                    "status": "upgrade failed; previous version retained",
                    "detail": "quality regression",
                },
                {"status": "selected for next serving phase after finite quality gates"},
            ]
        },
    )
    atomic_json(
        tmp_path / "research/training-calibration/old/training_health.json",
        {**health, "optimizer_updates": 1000},
    )
    value = mission_evidence.collect(tmp_path)
    assert value["optimizer_updates_observed"] == 5
    assert value["accepted_weight_updates_this_run"] == 1
    assert len(value["training_runs"]) == 1
    assert value["errors"][0]["detail"] == "quality regression"


@pytest.mark.parametrize("counter", [True, -1, 6, None])
def test_invalid_or_inconsistent_counts_never_become_training_evidence(
    tmp_path, monkeypatch, counter
):
    run = running(tmp_path, monkeypatch)
    atomic_json(
        run / "learning/update-0001/A/training_health.json",
        {
            "attempted_steps": 7,
            "optimizer_updates": counter,
            "amp_skipped_steps": 2,
            "finite_gradient_steps": 5,
        },
    )
    value = mission_evidence.collect(tmp_path)
    assert value["optimizer_updates_observed"] is None
    assert value["errors"] and value["training_runs"] == []


def test_large_baseline_and_historical_error_do_not_break_chat_evidence(tmp_path, monkeypatch):
    run = running(tmp_path, monkeypatch)
    (run / "baseline-131072.json").write_text("x" * (2 * 2**20 + 1))
    path = tmp_path / "research/public-benchmarks/results/old.error.json"
    atomic_json(path, {"key": "old-task", "error": "IndexError"})
    value = mission_evidence.collect(tmp_path)
    assert value["reports"][0]["summary"] == "Unavailable within read budget"
    assert value["errors"][-1]["scope"].startswith("Historical")


def test_archive_paging_reads_complete_unicode_without_path_escape(tmp_path):
    original = {"text": "zażółć gęślą jaźń " * 800}
    preview = json.loads(research_tools.bounded_tool_result(tmp_path, original, 2048))
    tool = research_tools.ResearchTools(tmp_path, {})
    offset, chunks = 0, []
    while offset is not None:
        page = tool.execute(
            "read_tool_result", {"sha256": preview["full_result_sha256"], "offset": offset}
        )
        chunks.append(page["content"])
        assert len(json.dumps(page, ensure_ascii=False).encode()) < 2048
        offset = page["next_offset"]
    assert json.loads("".join(chunks)) == original
    with pytest.raises(ValueError, match="Invalid archive identity"):
        tool.execute("read_tool_result", {"sha256": "../../secret", "offset": 0})
    Path(preview["full_result_path"]).write_text("tampered")
    with pytest.raises(ValueError, match="Archived tool result changed"):
        tool.execute("read_tool_result", {"sha256": preview["full_result_sha256"], "offset": 0})


def test_escaped_archive_pages_fit_the_helper_without_recursive_preview(tmp_path):
    preview = json.loads(research_tools.bounded_tool_result(tmp_path, {"text": '\\"' * 4000}, 2048))
    tool = research_tools.ResearchTools(tmp_path, {})
    page = tool.execute(
        "read_tool_result", {"sha256": preview["full_result_sha256"], "offset": 512}
    )
    visible = json.loads(research_tools.bounded_tool_result(tmp_path, page, 2048))
    assert visible == page and "truncated" not in visible


def test_archive_offset_cannot_split_utf8_or_pass_end(tmp_path):
    preview = json.loads(research_tools.bounded_tool_result(tmp_path, {"text": "ą" * 2000}, 2048))
    raw = Path(preview["full_result_path"]).read_bytes()
    index = raw.index("ą".encode()) + 1
    tool = research_tools.ResearchTools(tmp_path, {})
    with pytest.raises(ValueError, match="UTF-8"):
        tool.execute("read_tool_result", {"sha256": preview["full_result_sha256"], "offset": index})
    with pytest.raises(ValueError, match="exceeds the archive"):
        tool.execute(
            "read_tool_result", {"sha256": preview["full_result_sha256"], "offset": len(raw) + 1}
        )


def test_ordinary_chat_gets_host_evidence_before_model_can_guess(tmp_path, monkeypatch):
    run = running(tmp_path, monkeypatch)
    atomic_json(run / "serving-active.json", {})
    model = tmp_path / "model.gguf"
    model.write_bytes(b"test")
    monkeypatch.setattr(
        mission_chat, "load_profile", lambda *args: {"runtime": {}, "server": {"model": str(model)}}
    )
    seen = []

    def reply(endpoint, payload=None):
        if endpoint == "/props":
            return {"model_path": str(model)}
        seen.extend(payload["messages"])
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "Counters unavailable; no training claim."},
                }
            ]
        }

    client = SimpleNamespace(
        request=reply,
        template_args=lambda: {},
        model_name="v100",
        timeout=120,
        sampling_args={"max_tokens": 2048},
        activity_context={},
    )
    monkeypatch.setattr(competition, "helper_client", lambda *args: client)
    result = mission_chat.respond(tmp_path, run, {"message": "Czy trenujesz?"})
    evidence = json.loads(seen[0]["content"].split("Verified current facts: ", 1)[1])
    assert evidence["learning"]["completed_cycles"] == 0
    assert evidence["phase"] == "baseline-before-weight-updates"
    assert "Missing optimizer/profit counters are unknown" in seen[0]["content"]
    assert result["actions"] == [] and result["applied"] == []
