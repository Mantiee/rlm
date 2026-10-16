"""Streaming keepalives must not extend an inference request's wall-clock budget."""

import json

import pytest
import requests

from rlm.clients import llamacpp


def packet(value):
    return b"data: " + json.dumps(value).encode()


def transport(monkeypatch, frames, now):
    closed = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            closed.append("response")

        def raise_for_status(self):
            pass

        def iter_lines(self, **kwargs):
            for elapsed, raw in frames:
                now[0] = elapsed
                yield raw

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            closed.append("session")

        def post(self, *args, **kwargs):
            assert kwargs["stream"] is True
            assert kwargs["timeout"] <= 2
            return Response()

    monkeypatch.setattr(llamacpp.requests, "Session", Session)
    monkeypatch.setattr(llamacpp.time, "monotonic", lambda: now[0])
    client = llamacpp.LlamaCppClient(timeout=2)
    monkeypatch.setattr(client, "streaming_enabled", lambda: True)
    events = []
    monkeypatch.setattr(client, "stream_delta", lambda *args: events.append(args))
    return client, closed, events


@pytest.mark.parametrize("raw", [b"", b": heartbeat", b"data: [DONE]"])
@pytest.mark.parametrize("request_deadline", [None, 1, 10])
def test_native_stream_checks_deadline_on_every_raw_frame(monkeypatch, raw, request_deadline):
    now = [0]
    frames = [(0.1, packet({"choices": [{"delta": {"content": "partial"}}]})), (2, raw)]
    client, closed, events = transport(monkeypatch, frames, now)
    if request_deadline is not None:
        client.request_deadline = request_deadline

    with pytest.raises(requests.Timeout, match="total time budget"):
        client.http_request("/v1/chat/completions", {"messages": []})

    assert events == [("output", "partial")]
    assert closed == ["response", "session"]
    assert client.get_usage_summary().model_usage_summaries == {}


def test_native_stream_completes_in_order_before_deadline(monkeypatch):
    now = [0]
    frames = [
        (0.1, b": heartbeat"),
        (0.2, b""),
        (0.3, packet({"choices": [{"delta": {"content": "4"}, "finish_reason": "stop"}]})),
        (0.4, packet({"usage": {"prompt_tokens": 2, "completion_tokens": 1}})),
        (0.5, b"data: [DONE]"),
    ]
    client, closed, events = transport(monkeypatch, frames, now)

    result = client.http_request("/v1/chat/completions", {"messages": []})

    assert result["choices"][0]["message"]["content"] == "4"
    assert result["usage"] == {"prompt_tokens": 2, "completion_tokens": 1}
    assert events == [("output", "4")]
    assert closed == ["response", "session"]
