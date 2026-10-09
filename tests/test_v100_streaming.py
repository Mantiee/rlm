import json

import pytest

from rlm.v100.streaming import local_reply, ollama_reply


def sse(packet):
    return b"data: " + json.dumps(packet).encode()


def test_local_stream_preserves_live_reasoning_output_usage_and_tool_calls():
    events = []
    packets = [
        sse({"choices": [{"delta": {"reasoning_content": "Check "}}]}),
        sse(
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_content": "evidence.",
                            "content": "4",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call",
                                    "function": {"name": "calculate", "arguments": "{"},
                                }
                            ],
                        }
                    }
                ]
            }
        ),
        sse(
            {
                "choices": [
                    {
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "}"}}]},
                        "finish_reason": "stop",
                    }
                ]
            }
        ),
        sse(
            {
                "choices": [],
                "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
            }
        ),
        b"data: [DONE]",
    ]
    response = local_reply(packets, lambda channel, text: events.append((channel, text)))
    assert events == [("reasoning", "Check "), ("output", "4"), ("reasoning", "evidence.")]
    assert response["choices"][0]["message"]["reasoning_content"] == "Check evidence."
    assert response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == "{}"
    assert response["usage"]["total_tokens"] == 11
    with pytest.raises(ValueError, match="Incomplete"):
        local_reply(packets[:-1], lambda *args: None)


def test_ollama_stream_requires_done_and_preserves_measured_counts():
    packets = [
        {"message": {"thinking": "Consider "}, "done": False},
        {"message": {"thinking": "alternatives.", "content": "4"}, "done": False},
        {
            "model": "owned",
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 8,
            "eval_count": 3,
        },
    ]
    events = []
    response = ollama_reply(
        [json.dumps(p).encode() for p in packets], lambda *args: events.append(args)
    )
    assert response["message"]["thinking"] == "Consider alternatives."
    assert response["message"]["content"] == "4"
    assert response["eval_count"] == 3
    assert len(events) == 3
    with pytest.raises(ValueError, match="Incomplete"):
        ollama_reply([json.dumps(p).encode() for p in packets[:-1]], lambda *args: None)


def test_streaming_requires_both_operator_preferences(tmp_path):
    from rlm.clients.llamacpp import LlamaCppClient

    client = LlamaCppClient(activity_root=tmp_path)
    path = tmp_path / "research/user-preferences.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"stream_local_model_trace": True}))
    assert not client.streaming_enabled()
    path.write_text(
        json.dumps({"stream_local_model_trace": True, "capture_local_model_trace": True})
    )
    assert client.streaming_enabled()
    client.thread_state.activity_request_id = "real-request"
    client.stream_delta("output", "working")
    client.flush_stream()
    from rlm.v100.live_status import agent_views, recent_events

    view = agent_views(recent_events(tmp_path), [])[0]
    assert view["stream_output"] == "working"
    assert view["stream_request"] == "real-request"


def test_readiness_never_counts_heartbeat_as_execution_or_zero_as_training(tmp_path):
    from rlm.v100.readiness import assess

    result = assess(
        tmp_path,
        {"running": True},
        {
            "external_compute": {"workers": [{"stale": False}], "jobs": []},
            "mission_evidence": {
                "optimizer_updates_observed": 0,
                "accepted_weight_updates_this_run": None,
            },
        },
        {"state": "validated layout active"},
    )
    checks = {c["name"]: c["state"] for c in result["checks"]}
    assert checks["Owned worker heartbeat"] == "passed"
    assert checks["Owned worker execution"] == "unknown"
    assert checks["Production optimizer steps"] == "unknown"
    assert checks["Accepted production weights"] == "unknown"


def test_local_http_transport_requests_stream_and_assembles_real_frames(tmp_path, monkeypatch):
    from rlm.clients.llamacpp import LlamaCppClient
    from rlm.v100 import live_status

    folder = tmp_path / "research"
    folder.mkdir()
    (folder / "user-preferences.json").write_text(
        json.dumps({"capture_local_model_trace": True, "stream_local_model_trace": True})
    )
    packets = [
        sse({"choices": [{"delta": {"content": "4"}, "finish_reason": "stop"}]}),
        sse(
            {
                "choices": [],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }
        ),
        b"data: [DONE]",
    ]
    seen = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_lines(self, chunk_size):
            return iter(packets)

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, **kwargs):
            seen.append(kwargs)
            return Response()

    monkeypatch.setattr("rlm.clients.llamacpp.requests.Session", Session)
    client = LlamaCppClient(activity_root=tmp_path)
    result = client.request("/v1/chat/completions", {"model": "v100", "stream": False})
    assert seen[0]["json"]["stream"] is True
    assert seen[0]["json"]["stream_options"]["include_usage"] is True
    assert result["choices"][0]["message"]["content"] == "4"
    assert any(e["kind"] == "inference-delta" for e in live_status.recent_events(tmp_path))
