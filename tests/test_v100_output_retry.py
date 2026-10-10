import copy
import json
from types import SimpleNamespace

import pytest

from rlm.v100.agent import native_turn, research_output_limit, tool_turn


def reasoning_client(reasons, context=32768, prompt=100, root=None):
    requests = []
    responses = iter(reasons)

    def request(endpoint, payload):
        if endpoint == "/apply-template":
            return {"prompt": "fixed prompt"}
        requests.append(copy.deepcopy(payload))
        reason, content = next(responses)
        return {
            "choices": [{"finish_reason": reason, "message": {"content": content}}],
            "usage": {"completion_tokens": payload["max_tokens"]},
        }

    return SimpleNamespace(
        sampling_args={"max_tokens": 2048, "temperature": 1.0, "seed": 42, "top_p": 0.95},
        template_args=lambda: {"chat_template_kwargs": {"enable_thinking": True}},
        request=request,
        count_text=lambda *args, **kwargs: prompt,
        context_window=context,
        model_name="v100",
        enable_thinking=True,
        activity_root=root,
        research_owner="A",
        activity_actor="researcher",
    ), requests


def test_retry_preserves_input_sampling_and_client_configuration(tmp_path):
    client, requests = reasoning_client(
        [("length", "partial"), ("length", ""), ("stop", "done")], root=tmp_path
    )
    before = copy.deepcopy(client.sampling_args)
    messages = [{"role": "user", "content": "Research a falsifiable hypothesis"}]
    info = {}
    assert (
        native_turn(client, messages, response_info=info, retry_output_limit=8192)["content"]
        == "done"
    )
    assert [request["max_tokens"] for request in requests] == [2048, 4096, 8192]
    assert all(request["messages"] == messages for request in requests)
    assert all(request["temperature"] == 1.0 and request["seed"] == 42 for request in requests)
    assert all(request["chat_template_kwargs"]["enable_thinking"] for request in requests)
    assert client.sampling_args == before
    assert info["attempts"] == 3 and info["total_completion_tokens"] == 14336
    logs = list((tmp_path / "research/logs/activity").glob("*/timeline.jsonl"))
    events = [json.loads(line) for line in logs[0].read_text().splitlines()]
    assert [event["payload"]["next_max_tokens"] for event in events] == [4096, 8192]


def test_evaluation_default_rejects_length_without_retry():
    client, requests = reasoning_client([("length", "4")])
    with pytest.raises(ValueError, match="no partial answer.*attempts=1"):
        native_turn(client, [{"role": "user", "content": "2+2?"}])
    assert len(requests) == 1


def test_resident_host_ceiling_blocks_initial_expansion_and_retry():
    client, requests = reasoning_client([("length", "partial")] * 3)
    client.research_token_ceiling = 1024
    with pytest.raises(ValueError, match="max_tokens=1024, attempts=1"):
        native_turn(client, [], retry_output_limit=8192)
    assert [request["max_tokens"] for request in requests] == [1024]


def test_resident_retry_may_grow_only_to_host_ceiling():
    client, requests = reasoning_client([("length", "partial"), ("stop", "done")])
    client.sampling_args["max_tokens"] = 512
    client.research_token_ceiling = 1024
    assert native_turn(client, [], retry_output_limit=8192)["content"] == "done"
    assert [request["max_tokens"] for request in requests] == [512, 1024]


def test_retry_rejects_partial_final_after_three_attempts():
    client, requests = reasoning_client([("length", "partial")] * 3)
    with pytest.raises(ValueError, match="max_tokens=8192, attempts=3"):
        native_turn(client, [], retry_output_limit=8192)
    assert len(requests) == 3


def test_retry_respects_remaining_context():
    client, requests = reasoning_client([("length", "partial")] * 3, context=6000, prompt=1000)
    with pytest.raises(ValueError, match="max_tokens=4999"):
        native_turn(client, [], retry_output_limit=8192)
    assert [request["max_tokens"] for request in requests] == [2048, 4096, 4999]
    assert all(request["max_tokens"] + 1001 <= 6000 for request in requests)


def test_initial_overflow_does_not_send_an_inference_request():
    client, requests = reasoning_client([], context=3000, prompt=1000)
    with pytest.raises(ValueError, match="exceeds the working context"):
        native_turn(client, [], retry_output_limit=8192)
    assert not requests


@pytest.mark.parametrize("limit", [True, 0, -1, 8193, "8192"])
def test_retry_limit_is_bounded_and_type_checked(limit):
    client, requests = reasoning_client([])
    with pytest.raises(ValueError, match="retry output limit"):
        native_turn(client, [], retry_output_limit=limit)
    assert not requests


@pytest.mark.parametrize("thinking", [False, None, True])
def test_only_reasoning_research_clients_get_extra_allowance(thinking):
    assert research_output_limit(SimpleNamespace(enable_thinking=thinking)) == (
        8192 if thinking is True else None
    )


def test_json_tools_do_not_return_a_truncated_valid_action():
    client, requests = reasoning_client(
        [("length", '{"tool":"calculate","arguments":{}}'), ("stop", '{"answer":"done"}')]
    )
    client.tool_protocol = "json"
    tool = {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Calculate",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    result = tool_turn(client, [], tools=[tool], retry_output_limit=8192)
    assert result["content"] == "done" and not result.get("tool_calls")
    assert requests[0]["messages"] == requests[1]["messages"]
    assert requests[0]["response_format"] == requests[1]["response_format"]


def test_successful_research_keeps_the_original_budget():
    client, requests = reasoning_client([("stop", "done")])
    assert native_turn(client, [], retry_output_limit=8192)["content"] == "done"
    assert [request["max_tokens"] for request in requests] == [2048]
