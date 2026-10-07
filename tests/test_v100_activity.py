import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from rlm.clients.llamacpp import LlamaCppClient
from rlm.v100.activity import ActivityLog
from rlm.v100.experiments import SharedLab


def rows(root, pattern):
    return [
        json.loads(line) for path in root.glob(pattern) for line in path.read_text().splitlines()
    ]


def test_public_memory_exports_separate_readable_decisions_and_testers(tmp_path):
    lab = SharedLab(tmp_path / "research/state/competition.sqlite3")
    lab.append("A", "plan", {"rationale": "Try a smaller learning rate", "round": "trial-1"})
    lab.append("B", "worker-result", {"observation": "Test passed"})
    assert len(lab.recent()) == 2
    lab.close()
    assert len(rows(tmp_path, "research/logs/activity/*/A/model/decisions.jsonl")) == 1
    assert len(rows(tmp_path, "research/logs/activity/*/B/tester/research.jsonl")) == 1
    readable = next(tmp_path.glob("research/logs/activity/*/A/model/decisions.md")).read_text()
    assert "Try a smaller learning rate" in readable
    timeline = rows(tmp_path, "research/logs/activity/*/timeline.jsonl")
    assert len({event["id"] for event in timeline}) == 2
    assert [event["context"]["sequence"] for event in timeline] == [1, 2]


def test_journal_masks_credentials_and_raw_reasoning_without_hiding_rationale(tmp_path):
    journal = ActivityLog(tmp_path, "A", "model")
    journal.write(
        "decisions",
        "plan",
        {
            "rationale": "Use replay",
            "api_key": "private-key",
            "system_prompt": "private-prompt",
            "reasoning_content": "private-reasoning",
            "content": "<think>private-chain</think>Final answer. Bearer secret-token api_key=some-key hf_12345678912345 sk-123456789123456",
        },
    )
    files = list(tmp_path.rglob("*.jsonl")) + list(tmp_path.rglob("*.md"))
    for path in files:
        text = path.read_text()
        for secret in (
            "private-key",
            "private-prompt",
            "private-reasoning",
            "private-chain",
            "secret-token",
            "some-key",
            "hf_12345678912345",
            "sk-123456789123456",
        ):
            assert secret not in text
        assert "Use replay" in text
        assert "Final answer" in text


def test_concurrent_writers_produce_complete_bounded_rows(tmp_path):
    def write(index):
        return ActivityLog(tmp_path, "A", "tester").write(
            "tools", "tool-result", {"i": index, "data": "x" * 10000}
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        ids = list(executor.map(write, range(40)))
    events = rows(tmp_path, "research/logs/activity/*/A/tester/tools.jsonl")
    assert {event["id"] for event in events} == set(ids)
    assert {event["payload"]["i"] for event in events} == set(range(40))
    assert all(len(event["payload"]["data"]) < 8300 for event in events)
    assert len(rows(tmp_path, "research/logs/activity/*/timeline.jsonl")) == 40


@pytest.mark.parametrize("branch,actor", [("../bad", "model"), ("A", "../bad")])
def test_log_paths_cannot_escape(tmp_path, branch, actor):
    with pytest.raises(ValueError):
        ActivityLog(tmp_path, branch, actor)


def test_local_client_logs_final_decisions_usage_and_failures_without_prompts(
    tmp_path, monkeypatch
):
    client = LlamaCppClient(activity_root=str(tmp_path), activity_branch="B")
    result = {
        "choices": [
            {
                "message": {
                    "content": '{"rationale":"Try replay"}',
                    "reasoning_content": "raw-private-thinking",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"completion_tokens": 10},
        "timings": {"draft_n": 8, "draft_n_accepted": 6},
    }
    monkeypatch.setattr(client, "http_request", lambda *a: result)
    assert (
        client.request(
            "/v1/chat/completions",
            {"messages": [{"role": "system", "content": "private-system"}], "max_tokens": 512},
        )
        is result
    )
    events = rows(tmp_path, "research/logs/activity/*/timeline.jsonl")
    assert [event["kind"] for event in events] == [
        "inference-start",
        "model-output",
        "inference-finished",
    ]
    assert events[1]["context"]["request_id"] == events[0]["id"]
    assert events[2]["payload"]["timings"]["draft_n_accepted"] == 6
    text = "".join(path.read_text() for path in tmp_path.rglob("*.jsonl"))
    assert "private-system" not in text and "raw-private-thinking" not in text

    def fail(*args):
        raise RuntimeError("private-error")

    monkeypatch.setattr(client, "http_request", fail)
    with pytest.raises(RuntimeError):
        client.request("/v1/chat/completions", {})
    errors = rows(tmp_path, "research/logs/activity/*/B/model/errors.jsonl")
    assert errors[-1]["payload"] == {"error": "RuntimeError"}
