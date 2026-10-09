"""Oversized research data and catalogs retain evidence without widening context."""

import json
from types import SimpleNamespace

import pytest

from rlm.v100 import research_tools
from rlm.v100.agent import tool_schema
from rlm.v100.tool_protocol import json_tool_turn


def test_rejected_schema_has_one_correction_and_no_transport_retry(tmp_path, monkeypatch):
    from rlm.v100.activity import ActivityLog

    calls = []

    def route(client, messages, **kwargs):
        calls.append(messages)
        if len(calls) == 1:
            raise ValueError("Action argument fields differ from the schema")
        return {"role": "assistant", "content": "Corrected"}

    monkeypatch.setattr(research_tools, "tool_turn", route)
    client = SimpleNamespace(tool_protocol="json")
    journal = ActivityLog(tmp_path)
    assert research_tools.route_research_tool(client, [], [], journal)["content"] == "Corrected"
    assert len(calls) == 2 and "BEFORE execution" in calls[1][-1]["content"]

    def transport(client, messages, **kwargs):
        calls.append(messages)
        raise ValueError("Remote model metadata changed")

    monkeypatch.setattr(research_tools, "tool_turn", transport)
    calls.clear()
    with pytest.raises(ValueError, match="metadata"):
        research_tools.route_research_tool(client, [], [], journal)
    assert len(calls) == 1

    def bad_schema(client, messages, **kwargs):
        calls.append(messages)
        raise ValueError("Action argument fields differ from the schema")

    monkeypatch.setattr(research_tools, "tool_turn", bad_schema)
    calls.clear()
    with pytest.raises(ValueError, match="schema"):
        research_tools.route_research_tool(client, [], [], journal)
    assert len(calls) == 2


def test_large_tool_result_is_explicitly_partial_and_full_evidence_is_preserved(tmp_path):
    result = {"rows": [{"id": i, "text": "źródło" * 300} for i in range(30)]}
    preview = json.loads(research_tools.bounded_tool_result(tmp_path, result, 2048))
    assert preview["truncated"] is True
    assert preview["original_bytes"] > 2048
    from pathlib import Path

    full = Path(preview["full_result_path"])
    assert json.loads(full.read_text()) == result
    assert len(json.dumps(preview, ensure_ascii=False).encode()) < 2048
    full.write_text("changed evidence")
    with pytest.raises(ValueError, match="Archived tool result changed"):
        research_tools.bounded_tool_result(tmp_path, result, 2048)


@pytest.mark.parametrize("selection", ["small", "not-in-task", "answer", {"name": "small"}])
def test_oversized_catalog_uses_small_selector_and_keeps_argument_validation(selection):
    generated = []

    def request(endpoint, payload):
        if endpoint == "/apply-template":
            return {"prompt": json.dumps(payload)}
        generated.append(payload)
        if len(generated) == 1:
            content = json.dumps(
                {"answer": "No tool required"} if selection == "answer" else {"tool": selection}
            )
        else:
            content = '{"tool":"small","arguments":{"expression":"2+2"}}'
        return {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}

    client = SimpleNamespace(
        request=request,
        count_text=lambda text, **kwargs: len(text.encode()),
        context_window=8192,
        sampling_args={"max_tokens": 1024},
        model_name="helper",
        template_args=lambda: {},
    )
    tools = [
        tool_schema(
            "large", "A complex experiment", {"data": {"type": "string", "description": "x" * 6000}}
        ),
        tool_schema("small", "Calculate an expression", {"expression": {"type": "string"}}),
    ]
    original = dict(client.sampling_args)
    messages = [{"role": "user", "content": "Check this using an appropriate tool"}]
    if selection == "answer":
        assert json_tool_turn(client, messages, tools)["content"] == "No tool required"
        assert len(generated) == 1
    elif selection != "small":
        with pytest.raises(ValueError, match="outside its task scope"):
            json_tool_turn(client, messages, tools)
        assert len(generated) == 1
    else:
        result = json_tool_turn(client, messages, tools)
        assert result["tool_calls"][0]["function"]["name"] == "small"
        assert len(generated) == 2
        assert generated[0]["max_tokens"] == 128
        assert "large" not in generated[1]["messages"][-1]["content"]
    assert client.sampling_args == original


def test_parallel_workers_archive_identical_evidence_without_partial_reads(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    result = {"rows": [{"id": i, "text": "evidence" * 300} for i in range(20)]}
    with ThreadPoolExecutor(max_workers=6) as workers:
        previews = list(
            workers.map(
                lambda _: json.loads(research_tools.bounded_tool_result(tmp_path, result, 2048)),
                range(24),
            )
        )
    assert len({row["full_result_sha256"] for row in previews}) == 1
    assert len(list((tmp_path / "research/tool-results").glob("*.json"))) == 1
