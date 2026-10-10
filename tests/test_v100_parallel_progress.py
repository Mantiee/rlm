import copy
import json
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from rlm.v100 import (
    insights,
    mission_memory,
    paper_agents,
    paper_outcomes,
    progress,
    research_tools,
    training,
)
from rlm.v100.common import atomic_json
from rlm.v100.experiments import SharedLab
from rlm.v100.paper import PaperBook


def test_recursive_archive_retained_but_never_retrieved(tmp_path):
    old = mission_memory.archive(tmp_path, "worker:A:old", "Coinbase " * 100)
    external = mission_memory.archive(
        tmp_path, "https://example.org/fees", "Coinbase actual source"
    )
    memo = mission_memory.archive(tmp_path, "memo:A:new", "Coinbase unverified hypothesis")
    repaired = mission_memory.repair(tmp_path)
    assert repaired["excluded_from_retrieval"] == 1
    with sqlite3.connect(repaired["backup"]) as backup:
        assert backup.execute("SELECT count(*) FROM documents").fetchone()[0] == 3
    memory = mission_memory.store(tmp_path)
    try:
        rows = memory.search("Coinbase")
        assert [row["document_id"] for row in rows] == [external, memo]
        assert "unverified model" in rows[1]["provenance"]
        assert memory.db.execute("SELECT text FROM documents WHERE id=?", (old,)).fetchone()[0]
    finally:
        memory.close()


def test_router_is_fast_and_source_ids_are_constrained(tmp_path, monkeypatch):
    mission_memory.archive(tmp_path, "https://example.org/fees", "Coinbase fees")
    client = SimpleNamespace(
        sampling_args={"max_tokens": 4096},
        context_window=32768,
        enable_thinking=True,
        tool_protocol="json",
        research_tool_names={"search_memory", "read_source"},
    )
    seen = []

    def turn(chosen, messages, tools=None, **kwargs):
        if tools:
            assert chosen.enable_thinking is False
            assert chosen.sampling_args["max_tokens"] == 1024
            names = {tool["function"]["name"] for tool in tools}
            seen.append(names)
            if len(seen) == 1:
                assert "read_source" not in names
                return {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "x",
                            "type": "function",
                            "function": {
                                "name": "search_memory",
                                "arguments": '{"query":"Coinbase"}',
                            },
                        }
                    ],
                }
            source = next(tool for tool in tools if tool["function"]["name"] == "read_source")
            assert len(source["function"]["parameters"]["properties"]["source_id"]["enum"]) == 1
            return {"role": "assistant", "content": "done"}
        assert chosen.enable_thinking is True
        return {"content": '{"answer":"test"}'}

    monkeypatch.setattr(research_tools, "tool_turn", turn)
    research_tools.research_turn(client, [], {}, tmp_path)
    assert len(seen) == 2
    assert client.enable_thinking is True and client.sampling_args["max_tokens"] == 4096


def test_remote_and_master_overlap_and_empty_book_skips_trade_generation(tmp_path, monkeypatch):
    book = PaperBook(tmp_path)
    book.initialize()
    master_started, tester_started = threading.Event(), threading.Event()
    parent = {
        "runtime": {
            "base_url": "local",
            "max_output_tokens": 2048,
            "enable_thinking": True,
            "model_version": "gemma",
        }
    }
    helper = {"runtime": {"base_url": "remote", "model_version": "qwen"}, "server": {"slots": 1}}
    monkeypatch.setattr(
        paper_agents,
        "helper_client",
        lambda p, *args: SimpleNamespace(
            endpoint=p["runtime"]["base_url"],
            sampling_args={"max_tokens": 512},
            context_window=32768,
        ),
    )

    def research(client, branch, job, observations, root):
        assert len(job["brief"]) <= 400, "Internal assignment exceeds research_task admission"
        if branch == "A":
            if client.endpoint == "local":
                master_started.set()
                assert tester_started.wait(2), "tester did not overlap master"
            else:
                tester_started.set()
                assert master_started.wait(2), "master did not overlap tester"
        return {
            "role": job["role"],
            "observation": "unverified",
            "hypothesis": "idea",
            "suggested_test": "test",
            "model": client.endpoint,
        }

    monkeypatch.setattr(paper_agents, "research_task", research)
    monkeypatch.setattr(
        paper_agents,
        "review_research",
        lambda *args: {"useful_indices": [], "conclusion": "unverified"},
    )
    monkeypatch.setattr(
        paper_agents,
        "native_turn",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("unnecessary trading inference")),
    )
    try:
        assert len(paper_agents.paper_round(book, parent, helper, 1, income_research=True)) == 2
        assert book.state()["orders"] == []
    finally:
        book.close()


def test_progress_does_not_convert_hypotheses_or_failed_trials_to_profit(tmp_path, monkeypatch):
    book = PaperBook(tmp_path)
    book.initialize()
    book.close()
    run = tmp_path / "research/mission/run-test"
    atomic_json(
        run / "learning/state.json",
        {"cycles": [{"status": "upgrade failed; previous version retained"}]},
    )
    from rlm.v100 import mission

    monkeypatch.setattr(
        mission,
        "status",
        lambda root: {"running": True, "run": str(run), "state": {"phase": "research"}},
    )
    shared = SharedLab(tmp_path / "research/state/competition.sqlite3")
    shared.append(
        "A",
        "worker-result",
        {
            "hypothesis": "guaranteed 1000000",
            "suggested_test": "forward test",
            "model": "qwen",
            "mission_run": str(run),
        },
    )
    shared.close()
    report = progress.report(tmp_path)
    assert report["accepted_weight_updates_this_run"] == 0
    assert report["paper"]["branches"]["A"]["net_pnl_before_personal_tax"] == "0"
    assert report["paper_blockers"]
    assert (
        report["research_hypotheses_not_verified_income"][0]["hypothesis"] == "guaranteed 1000000"
    )
    assert json.loads((tmp_path / "research/mission/latest-report.json").read_text()) == report


@pytest.mark.parametrize("exit_price,result", [("110", "profit"), ("90", "loss")])
def test_realized_paper_results_join_training_and_reject_forged_labels(
    tmp_path, exit_price, result
):
    from tests.test_v100_paper import decide, quote, setup

    book, clock, _ = setup(
        tmp_path, fee_changes={"entry_minimum_commission": "2", "exit_minimum_commission": "3"}
    )
    book.ingest(quote(clock))
    decision = decide(book)
    assert not paper_outcomes.records(tmp_path)
    clock.advance()
    book.ingest(quote(clock))
    decide(book, "close")
    clock.advance()
    event = book.ingest(quote(clock, exit_price))
    book.close()
    rows = paper_outcomes.records(tmp_path)
    assert len(rows) == 1
    label = json.loads(rows[0]["messages"][-1]["content"])
    assert label["result"] == result
    assert label["net_pnl_before_personal_tax"] == event["payload"]["trades"][0]["net_pnl"]
    inputs = json.loads(rows[0]["messages"][0]["content"].split("\n", 1)[1])
    assert inputs["decision_inputs"]["quote"]["observed_at"] <= decision["time"]
    paper_outcomes.verify(rows[0], tmp_path)
    pool = tmp_path / "pool.jsonl"
    originals = [
        insights.verified_record({"kind": "arithmetic", "expression": expression})
        for expression in ("2+3", "6*7")
    ]
    pool.write_text("".join(json.dumps(row) + "\n" for row in originals))
    expanded = tmp_path / "expanded.jsonl"
    assert insights.extend_pool(pool, tmp_path, expanded)
    train, validation = training.load_records(expanded, tmp_path / "research/state/splits.sqlite3")
    assert rows[0] in train + validation
    forged = copy.deepcopy(rows[0])
    forged["messages"][-1]["content"] = (
        '{"result":"profit","net_pnl_before_personal_tax":"1000000"}'
    )
    with pytest.raises(ValueError, match="audited"):
        paper_outcomes.verify(forged, tmp_path)
