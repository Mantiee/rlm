import json
import threading
from types import SimpleNamespace

import pytest

from rlm.v100 import dashboard_health, live_status
from rlm.v100 import inference_benchmark as bench
from rlm.v100.activity import ActivityLog
from rlm.v100.common import atomic_json
from tests.test_v100_dashboard import dashboard


def test_json_stream_from_actual_redacting_journal_does_not_kill_agent_view(tmp_path):
    journal = ActivityLog(tmp_path, "A", "model")
    journal.write("steps", "inference-start", {})
    journal.write(
        "steps",
        "inference-delta",
        {"channel": "content", "text": '{"action":"hold","api_key":"private"}'},
        request_id="one",
    )
    journal.write(
        "steps", "inference-delta", {"channel": "content", "text": " completed"}, request_id="one"
    )
    events = live_status.recent_events(tmp_path)
    assert isinstance(next(row["delta_text"] for row in events if row.get("delta_text")), dict)
    view = live_status.agent_views(events, [])[0]
    assert '"action": "hold"' in view["stream_output"]
    assert "private" not in view["stream_output"] and "[omitted]" in view["stream_output"]
    assert view["stream_output"].endswith(" completed")


def test_structured_trace_and_text_fragments_keep_request_boundaries():
    events = [
        {"branch": "A", "actor": "model", "request_id": "old", "delta_text": "old"},
        {"branch": "A", "actor": "model", "request_id": "new", "delta_text": ["new"]},
        {
            "branch": "A",
            "actor": "model",
            "request_id": "new",
            "returned_trace": {"status": "done"},
            "trace_part": 0,
        },
    ]
    view = live_status.agent_views(events, [])[0]
    assert view["stream_output"] == '["new"]'
    assert json.loads(view["returned_trace"]) == {"status": "done"}


def test_collector_type_error_is_reported_and_next_cycle_runs(tmp_path, monkeypatch):
    state = dashboard.DashboardState(tmp_path, lambda root: {})
    calls = []

    def refresh():
        calls.append(1)
        if len(calls) == 1:
            raise TypeError("bad old record")
        assert "bad old record" in state.data["errors"][0]
        state.data = {"mission": {"running": True}, "errors": []}
        state.stop.set()

    monkeypatch.setattr(state, "refresh", refresh)
    monkeypatch.setattr(state.stop, "wait", lambda seconds: None)
    state.loop()
    assert len(calls) == 2 and state.data["mission"]["running"]


def test_health_uses_lan_origin_and_waits_for_real_collector_snapshot(tmp_path, monkeypatch):
    atomic_json(
        tmp_path / "research/remote-access/config.json", {"dashboard": "http://192.168.0.68:8765"}
    )
    clock = [0.0]
    monkeypatch.setattr(dashboard_health.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(dashboard_health.time, "time", lambda: 100.0)
    monkeypatch.setattr(
        dashboard_health.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay)
    )
    urls = []
    snapshots = [
        {"collected_at": 100.0},
        {"mission": {}, "collected_at": 1.0},
        {"mission": {"running": True}, "collected_at": 100.0},
    ]

    def get(url, **kwargs):
        urls.append(url)
        return SimpleNamespace(
            text="mission-navigation Trade explorer",
            raise_for_status=lambda: None,
            json=lambda: snapshots.pop(0),
        )

    monkeypatch.setattr(dashboard_health.requests, "get", get)
    result = dashboard_health.wait(5, root=tmp_path)
    assert result["collector_ready"] and result["url"] == "http://192.168.0.68:8765/"
    assert clock[0] == 2.0 and all(url.startswith(result["url"]) for url in urls)


def test_health_service_bind_is_read_without_rebinding_server(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dashboard_health.subprocess,
        "check_output",
        lambda *a,
        **k: "{ path=/python ; argv[]=/python serve-v100-dashboard.py --root /owned --bind 192.168.0.68 --port 8765 ; }",
    )
    assert dashboard_health.endpoint(tmp_path) == "http://192.168.0.68:8765/"
    atomic_json(
        tmp_path / "research/remote-access/config.json", {"dashboard": "http://8.8.8.8:8765"}
    )
    with pytest.raises(ValueError, match="private"):
        dashboard_health.endpoint(tmp_path)


def test_benchmark_structured_probe_has_schema_and_keeps_original_reply(tmp_path, monkeypatch):
    payloads = []

    def request(port, path, data, **kwargs):
        payloads.append(data)
        content = (
            "437"
            if data["messages"][0]["content"] == bench.PROMPTS[0]
            else '{"action":"hold","amount":0}'
        )
        return {
            "choices": [{"finish_reason": "stop", "message": {"content": content}}],
            "usage": {"completion_tokens": 3},
        }

    monkeypatch.setattr(bench, "request", request)
    receipt = tmp_path / "probes.jsonl"
    result = bench.measure(8097, threading.Event(), receipt)
    structured = [payload for payload in payloads if "response_format" in payload]
    assert len(structured) == 2
    schema = structured[0]["response_format"]["json_schema"]["schema"]
    assert schema["required"] == ["action", "amount"]
    assert not schema["additionalProperties"]
    assert len(receipt.read_text().splitlines()) == len(result["probes"]) == 8


def test_non_json_model_content_has_clear_error_and_preserved_evidence(tmp_path, monkeypatch):
    def request(port, path, data, **kwargs):
        content = (
            "437"
            if data["messages"][0]["content"] == bench.PROMPTS[0]
            else '```json\n{"action":"hold","amount":0}\n```'
        )
        return {
            "choices": [{"finish_reason": "stop", "message": {"content": content}}],
            "usage": {"completion_tokens": 3},
        }

    monkeypatch.setattr(bench, "request", request)
    receipt = tmp_path / "probes.jsonl"
    with pytest.raises(ValueError, match="Structured probe model content"):
        bench.measure(8097, threading.Event(), receipt)
    assert "```json" in receipt.read_text()
