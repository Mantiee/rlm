import json
import time
from types import SimpleNamespace

import pytest

from rlm.v100 import agent, mission_chat, planning


def client():
    return SimpleNamespace(
        sampling_args={"max_tokens": 2048},
        timeout=60,
        request_deadline=time.monotonic() + 30,
        enable_thinking=False,
    )


def tool(name="get_plan", arguments="{}"):
    return {
        "tool_calls": [
            {"id": "call-real-model", "function": {"name": name, "arguments": arguments}}
        ]
    }


def test_bounded_model_selected_plan_read_and_generated_provenance(tmp_path, monkeypatch):
    turns = []
    reads = []

    def read(root):
        reads.append(root)
        return {"long": {"id": "goal-id", "text": "Learn useful paper strategies"}}

    monkeypatch.setattr(planning, "read", read)

    def native(client, messages, **kwargs):
        turns.append((client, messages, kwargs))
        assert client.sampling_args["max_tokens"] == 48
        assert client.research_token_ceiling == 48
        return (
            tool()
            if len(turns) == 1
            else {"content": json.dumps({"answer": "Learn useful paper strategies."})}
        )

    monkeypatch.setattr(agent, "native_turn", native)
    admitted = client()
    result = mission_chat.verified_plan_response(
        tmp_path, admitted, {"model": "accepted-model", "accepted_model_sha256": "a" * 64}
    )
    assert reads == [tmp_path]
    assert result["actions"] == result["applied"] == []
    assert result["tool_receipts"][0]["result"]["long"]["id"] == "goal-id"
    assert result["responder"]["model"] == "accepted-model"
    assert turns[0][0].request_deadline <= admitted.request_deadline
    assert all(sum(len(row["content"]) for row in messages) < 250 for _, messages, _ in turns)


@pytest.mark.parametrize(
    "selection",
    [
        tool("set_plan"),
        tool(arguments='{"horizon":"long"}'),
        {"tool_calls": []},
        {**tool(), "actions": [{"kind": "goal_long"}]},
    ],
)
def test_wrong_tool_arguments_or_extra_actions_rejected_before_read(
    tmp_path, monkeypatch, selection
):
    monkeypatch.setattr(agent, "native_turn", lambda *a, **kw: selection)
    monkeypatch.setattr(
        planning, "read", lambda *_: pytest.fail("Rejected selection must not execute tool")
    )
    with pytest.raises(ValueError):
        mission_chat.verified_plan_response(tmp_path, client(), {"model": "accepted-model"})


def test_deadline_exhaustion_prevents_tool_and_generation(tmp_path, monkeypatch):
    admitted = client()
    admitted.request_deadline = time.monotonic() - 1
    monkeypatch.setattr(
        agent, "native_turn", lambda *a, **kw: pytest.fail("Expired request must not infer")
    )
    with pytest.raises(TimeoutError):
        mission_chat.verified_plan_response(tmp_path, admitted, {"model": "accepted-model"})


def test_generated_extra_action_fields_rejected(tmp_path, monkeypatch):
    turns = iter(
        [tool(), {"content": json.dumps({"answer": "Goal", "actions": [{"kind": "plan_short"}]})}]
    )
    monkeypatch.setattr(agent, "native_turn", lambda *a, **kw: next(turns))
    monkeypatch.setattr(planning, "read", lambda *_: {"long": {"text": "Goal"}})
    with pytest.raises(ValueError, match="extra actions"):
        mission_chat.verified_plan_response(tmp_path, client(), {"model": "accepted-model"})
