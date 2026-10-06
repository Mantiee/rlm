import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest

from rlm import RLM
from rlm.clients.llamacpp import LlamaCppClient
from rlm.v100.cli import main, server_command
from rlm.v100.common import load_profile
from rlm.v100.memory import Memory, split_text
from rlm.v100.training import completed_checkpoint, encode_record, load_records


@pytest.fixture
def native_server():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, data))
            if self.path == "/apply-template":
                result = {"prompt": " ".join(m["content"] for m in data["messages"])}
            elif self.path == "/tokenize":
                result = {"tokens": list(range(len(data["content"].split())))}
            else:
                result = {
                    "choices": [{"message": {"content": "odpowiedź"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 7},
                }
                if data["messages"][-1]["content"] == "reasoning-only":
                    result["choices"][0] = {
                        "message": {"content": None, "reasoning_content": "reasoning"},
                        "finish_reason": "length",
                    }
            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", calls
    server.shutdown()
    thread.join()
    server.server_close()


def test_native_budget_usage_and_async(native_server, tmp_path):
    url, calls = native_server
    client = LlamaCppClient(
        base_url=url,
        context_window=20,
        sampling_args={"max_tokens": 8},
        metrics_path=str(tmp_path / "metrics"),
    )
    assert client.completion("raz dwa") == "odpowiedź"
    assert client.get_last_usage().total_output_tokens == 7
    assert calls[-1][1]["cache_prompt"] is True
    assert asyncio.run(client.acompletion("trzy")) == "odpowiedź"
    assert client.get_usage_summary().model_usage_summaries["v100"].total_calls == 2
    before = len([c for c in calls if c[0] == "/v1/chat/completions"])
    with pytest.raises(ValueError, match="overflow"):
        client.completion("word " * 20)
    assert len([c for c in calls if c[0] == "/v1/chat/completions"]) == before
    assert len((tmp_path / "metrics").read_text().splitlines()) == 2


def test_refuse_remote_endpoint():
    with pytest.raises(ValueError, match="loopback"):
        LlamaCppClient(base_url="http://192.0.2.1:8088")


def test_thinking_controls_match_token_count_and_completion(native_server):
    url, calls = native_server
    client = LlamaCppClient(base_url=url, enable_thinking=False)
    assert client.completion("answer") == "odpowiedź"
    template = next(data for path, data in calls if path == "/apply-template")
    completion = calls[-1][1]
    assert (
        template["chat_template_kwargs"]
        == completion["chat_template_kwargs"]
        == {"enable_thinking": False}
    )
    assert template["reasoning_effort"] == completion["reasoning_effort"] == "none"


def test_benchmark_uses_no_thinking_for_warmup_and_runs(native_server, tmp_path, monkeypatch):
    url, calls = native_server
    directory = tmp_path / "research"
    directory.mkdir()
    profile = (Path(__file__).parents[1] / "profiles/v100.toml").read_text()
    (directory / "v100.toml").write_text(profile.replace("http://127.0.0.1:8088", url))
    monkeypatch.setattr(
        "sys.argv", ["v100-lab", "--root", str(tmp_path), "bench", "--repeats", "1"]
    )
    main()
    completions = [data for path, data in calls if path == "/v1/chat/completions"]
    assert len(completions) == 2
    assert all(data["chat_template_kwargs"]["enable_thinking"] is False for data in completions)
    report = json.loads((directory / "logs/benchmark.json").read_text())
    assert report["enable_thinking"] is False
    assert report["runs"][0]["tokens"] == 7


def test_reasoning_only_response_is_logged_and_rejected(native_server, tmp_path):
    url, _ = native_server
    metrics = tmp_path / "metrics.jsonl"
    client = LlamaCppClient(base_url=url, metrics_path=str(metrics))
    with pytest.raises(ValueError, match="finish_reason=length.*reasoning_chars=9"):
        client.completion("reasoning-only")
    assert client.get_last_usage().total_output_tokens == 7
    row = json.loads(metrics.read_text())
    assert row["answer_chars"] == 0
    assert row["reasoning_chars"] == 9
    assert "reasoning_content" not in row


def test_context_override():
    with patch("rlm.core.rlm.get_client"):
        rlm = RLM(backend="llamacpp", context_window=8192, token_counter=lambda messages: 8000)
    assert rlm._get_compaction_status([]) == (8000, int(rlm.compaction_threshold_pct * 8192), 8192)


def test_sources_tree_resume_version_and_feedback(tmp_path):
    memory = Memory(tmp_path / "memory.sqlite", "v1")
    text = ("Kot mieszka w Warszawie.\n" * 20) + "Koniec."
    chunks = split_text(text, len, 32)
    assert "".join(c[2] for c in chunks) == text
    assert all(len(c[2]) <= 32 for c in chunks)
    document = memory.ingest("source", text, len, 32)
    assert memory.ingest("source", text, len, 32) == document
    calls = []

    def summarize(content):
        calls.append(content)
        return "Kot Warszawa"

    top = memory.build_tree(document, summarize)
    count = len(calls)
    assert memory.build_tree(document, summarize) == top
    assert len(calls) == count
    assert "".join(n["text"] for n in memory.leaves(top)) == text
    assert memory.retrieve('Kot OR " Warszawa *', 2)
    with pytest.raises(ValueError, match="No verified"):
        memory.export_feedback(tmp_path / "data.jsonl")
    run = {
        "messages": [{"role": "user", "content": "Gdzie kot?"}],
        "group": document,
        "document_ids": [document],
        "source_ids": [top],
        "model_version": "v1",
    }
    first = memory.add_feedback(run, "Warszawa")
    assert memory.add_feedback(run, "Warszawa") == first
    assert memory.export_feedback(tmp_path / "data.jsonl") == 1
    memory.model_version = "v2"
    assert memory.build_tree(document, summarize) != top
    assert len(calls) > count
    memory.close()


def test_interrupted_tree_resumes(tmp_path):
    memory = Memory(tmp_path / "memory.sqlite", "v1")
    document = memory.ingest("source", "x" * 160, len, 16)
    calls = 0

    def summarize(text):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted")
        return "summary"

    with pytest.raises(RuntimeError):
        memory.build_tree(document, summarize)
    done = memory.db.execute("SELECT COUNT(*) FROM nodes WHERE level>0").fetchone()[0]
    assert done == 1
    top = memory.build_tree(document, lambda text: "summary")
    assert "".join(node["text"] for node in memory.leaves(top)) == "x" * 160
    memory.close()


def record(group, docs):
    return {
        "group": group,
        "document_ids": docs,
        "messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}],
        "verification": {"kind": "human_feedback", "accepted": True},
    }


def test_no_source_leakage_or_unverified_data(tmp_path):
    path = tmp_path / "records.jsonl"
    rows = [record("a", ["a"]), record("ab", ["a", "b"]), record("b", ["b"]), record("c", ["c"])]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    train, evaluation = load_records(path)
    assert len(train) + len(evaluation) == 4
    assert set(d for r in train for d in r["document_ids"]).isdisjoint(
        d for r in evaluation for d in r["document_ids"]
    )
    rows[0]["verification"]["accepted"] = False
    path.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="verified"):
        load_records(path)


def test_assistant_only_mask_and_no_truncation():
    class Tokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return [1, 2] if add_generation_prompt else [1, 2, 3, 4]

    encoded = encode_record(record("a", ["a"]), Tokenizer(), 4)
    assert encoded["labels"] == [-100, -100, 3, 4]
    with pytest.raises(ValueError, match="shorten"):
        encode_record(record("a", ["a"]), Tokenizer(), 3)


def test_only_complete_checkpoints(tmp_path):
    first = tmp_path / "checkpoint-25"
    first.mkdir()
    for name in (
        "complete.json",
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ):
        (first / name).write_text("{}")
    (tmp_path / "checkpoint-50").mkdir()
    assert completed_checkpoint(tmp_path) == first
    (first / "optimizer.pt").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        completed_checkpoint(tmp_path)


def test_native_server_slot_budget(tmp_path):
    from pathlib import Path

    profile = load_profile(Path(__file__).parents[1] / "profiles/v100.toml", tmp_path)
    profile["server"]["slots"] = 2
    command = server_command(profile)
    assert command[command.index("--ctx-size") + 1] == "16384"
    assert command[command.index("--parallel") + 1] == "2"
    assert "--spec-type" not in command
