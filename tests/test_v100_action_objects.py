import pytest

from rlm.v100.research_tools import TOOLS
from rlm.v100.tool_protocol import validate_value


def test_goal_forecast_specification_accepts_its_dynamic_object_fields():
    schema = next(
        row["function"]["parameters"]
        for row in TOOLS
        if row["function"]["name"] == "predict_goal_pattern"
    )
    specification = {
        "question": "Does a source signal precede an observed change?",
        "rationale": "Compare independently timestamped observations",
        "evidence": ["host-observation"],
        "target": "host-target",
        "horizon_seconds": 3600,
        "threshold": 1,
        "probabilities": [0.2, 0.5, 0.3],
    }
    validate_value({"specification": specification}, schema)
    with pytest.raises(ValueError, match="fields"):
        validate_value({"specification": specification, "unauthorized": "extra"}, schema)
    with pytest.raises(ValueError, match="fields"):
        validate_value({}, schema)
    with pytest.raises(ValueError, match="type"):
        validate_value({"specification": "not an object"}, schema)


def test_closed_nested_objects_still_reject_unknown_fields():
    schema = {
        "type": "object",
        "properties": {
            "nested": {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
                "additionalProperties": False,
            }
        },
        "required": ["nested"],
        "additionalProperties": False,
    }
    validate_value({"nested": {"count": 2}}, schema)
    with pytest.raises(ValueError, match="fields"):
        validate_value({"nested": {"count": 2, "unexpected": 3}}, schema)
    with pytest.raises(ValueError, match="type"):
        validate_value({"nested": {"count": True}}, schema)


def test_additional_properties_schema_is_validated():
    schema = {"type": "object", "additionalProperties": {"type": "integer", "minimum": 0}}
    validate_value({"a": 1, "b": 2}, schema)
    with pytest.raises(ValueError, match="type"):
        validate_value({"a": "1"}, schema)
    with pytest.raises(ValueError, match="budget"):
        validate_value({"a": -1}, schema)


def test_open_objects_validate_known_properties_without_rejecting_other_fields():
    schema = {"type": "object", "properties": {"count": {"type": "integer"}}}
    validate_value({"count": 1, "context": {"signal": "example"}}, schema)
    with pytest.raises(ValueError, match="type"):
        validate_value({"count": "bad", "context": {}}, schema)


def test_status_question_does_not_call_model_or_change_plans(tmp_path, monkeypatch):
    from rlm.v100 import chat_progress, mission_chat, mission_evidence

    evidence = {
        "mission_running": True,
        "phase": "official-public-baseline",
        "run": "current",
        "learning": {"completed_cycles": 0},
        "optimizer_updates_observed": None,
        "accepted_weight_updates_this_run": None,
        "reports": [],
        "goal_learning": {"forecasts": []},
        "official_benchmark": {
            "completed": 3,
            "total": 20,
            "state": "generating",
            "report": "current/public-baseline.json",
            "current_case": "case-4",
        },
    }
    monkeypatch.setattr(mission_evidence, "collect", lambda root: evidence)
    result = mission_chat.respond(
        tmp_path, tmp_path, {"message": "co teraz robisz, jaki jest postep, nie widze tego w html"}
    )
    assert result["actions"] == [] and result["applied"] == []
    assert "3/20" in result["answer"] and "generating" in result["answer"]
    assert "brak potwierdzonych danych" in result["answer"]
    monkeypatch.setattr(
        mission_chat,
        "submit",
        lambda *args: pytest.fail("Status queued instead of reading evidence"),
    )
    mission_chat.chat(tmp_path, "co teraz robisz, jaki jest postep")
    assert not chat_progress.requested("Dodaj postęp do HTML, zrob to")


def test_public_progress_records_first_case_before_completion(tmp_path):
    import json

    from rlm.v100.public_benchmarks import write_progress

    report = {"model_version": "test-model", "cases": []}
    write_progress(
        tmp_path, tmp_path / "run/public-baseline.json", report, 20, "generating", "case-1"
    )
    value = json.loads((tmp_path / "research/public-benchmarks/progress.json").read_text())
    assert value["completed"] == 0 and value["total"] == 20
    assert value["current_case"] == "case-1" and value["state"] == "generating"
    assert value["updated"] > 0


def test_official_evidence_excludes_previous_run_progress(tmp_path, monkeypatch):
    from rlm.v100 import goal_learning, mission, mission_evidence
    from rlm.v100.common import atomic_json

    run = tmp_path / "research/mission/current"
    run.mkdir(parents=True)
    monkeypatch.setattr(
        mission,
        "status",
        lambda root: {
            "run": str(run),
            "running": True,
            "state": {"phase": "official-public-baseline"},
            "learning": {},
        },
    )
    monkeypatch.setattr(goal_learning, "status", lambda root: {})
    path = tmp_path / "research/public-benchmarks/progress.json"
    atomic_json(
        path,
        {
            "report": str(tmp_path / "research/mission/old/public-baseline.json"),
            "completed": 19,
            "total": 20,
        },
    )
    assert "official_benchmark" not in mission_evidence.collect(tmp_path)
    atomic_json(path, {"report": str(run / "public-baseline.json"), "completed": 2, "total": 20})
    assert mission_evidence.collect(tmp_path)["official_benchmark"]["completed"] == 2


def test_local_benchmark_progress_survives_concurrent_helper_progress(tmp_path, monkeypatch):
    from rlm.v100 import goal_learning, mission, mission_evidence
    from rlm.v100.common import atomic_json

    run = tmp_path / "research/mission/current"
    run.mkdir(parents=True)
    monkeypatch.setattr(
        mission,
        "status",
        lambda root: {"run": str(run), "running": True, "state": {}, "learning": {}},
    )
    monkeypatch.setattr(goal_learning, "status", lambda root: {})
    atomic_json(
        run / "public-baseline.progress.json",
        {"report": str(run / "public-baseline.json"), "completed": 3, "total": 20},
    )
    atomic_json(
        tmp_path / "research/public-benchmarks/progress.json",
        {"report": str(tmp_path / "helper.json"), "completed": 19, "total": 20},
    )
    assert mission_evidence.collect(tmp_path)["official_benchmark"]["completed"] == 3


def test_continue_learning_does_not_become_dashboard_instruction():
    from rlm.v100.chat_progress import continue_requested

    assert continue_requested("pracuj dalej, ucz się")
    assert continue_requested("kontynuuj naukę")
    assert not continue_requested("zrób dashboard")
    assert not continue_requested("zmień cel na naukę")


def test_dashboard_repair_reports_written_but_not_published(tmp_path, monkeypatch):
    from rlm.v100 import dashboard_editor, mission_chat

    monkeypatch.setattr(dashboard_editor, "write", lambda root, html: {"written": True})
    monkeypatch.setattr(
        dashboard_editor, "status", lambda root: {"valid": True, "published": False}
    )
    result = mission_chat.respond(tmp_path, tmp_path, {"message": "napraw dashboard"})
    assert result["applied"] == [{"written": True}]
    assert not result["dashboard"]["published"]
    assert "nie potwierdził" in result["answer"]
    assert result["actions"] == []
