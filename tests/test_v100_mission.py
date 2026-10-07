import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlm.v100 import (
    agent,
    competition,
    continuous,
    mission,
    mission_memory,
    paper_learning,
    research_tools,
    serving,
)
from rlm.v100.challenge import prepare_challenge, tool_cases
from rlm.v100.common import atomic_json, load_profile
from rlm.v100.fixture_tools import fixture_answer
from rlm.v100.insights import verified_record
from rlm.v100.paper import PaperBook
from rlm.v100.protection import file_hash
from rlm.v100.tool_protocol import json_object, json_tool_turn, validate_value
from tests.test_v100_challenge import bootstrap, fake_client


def test_json_actions_execute_calculator_without_native_tool_markers():
    def responder(payload):
        results = [
            json.loads(message["content"])["tool_result_data"]
            for message in payload["messages"]
            if message["role"] == "user" and '"tool_result_data"' in message["content"]
        ]
        action = (
            json.loads(results[-1])
            if results
            else {"tool": "calculate", "arguments": {"expression": "17*(6013-5347)-319-367-113-79"}}
        )
        return {"content": json.dumps(action)}

    client, requests = fake_client(responder)
    client.tool_protocol = "json"
    result = fixture_answer(client, tool_cases()[0])
    assert result["answer"] == "10444"
    assert result["trace"][0]["tool"] == "calculate"
    for _, payload in requests:
        assert "tools" not in payload
        assert all(message["role"] != "tool" for message in payload["messages"])
        assert not any("tool_calls" in message for message in payload["messages"])


def test_json_actions_read_reopened_original_memory_without_future_data():
    def responder(payload):
        results = [
            json.loads(message["content"])["tool_result_data"]
            for message in payload["messages"]
            if message["role"] == "user" and '"tool_result_data"' in message["content"]
        ]
        if not results:
            action = {"tool": "search_memory", "arguments": {"query": "ORIONMEM0"}}
        elif len(results) == 1:
            action = {
                "tool": "read_source",
                "arguments": {"source_id": json.loads(results[-1])[0]["id"]},
            }
        else:
            assert "4217" in results[-1]
            action = {"answer": "4217"}
        return {"content": json.dumps(action)}

    client, requests = fake_client(responder)
    client.tool_protocol = "json"
    result = fixture_answer(client, tool_cases()[4])
    assert result["answer"] == "4217"
    assert [step["tool"] for step in result["trace"]] == ["search_memory", "read_source"]
    assert "999999" not in json.dumps(requests)


@pytest.mark.parametrize("value", ["", "The answer is 4", "[1]", '{"answer":"4"} trailing'])
def test_action_parser_never_turns_non_json_into_success(value):
    with pytest.raises(ValueError):
        json_object(value)


@pytest.mark.parametrize(
    "action",
    [
        {"tool": "shell", "arguments": {}},
        {"tool": "calculate", "arguments": {"expression": 4}},
        {"tool": "calculate", "arguments": {"expression": "4", "command": "id"}},
        {"answer": 4},
    ],
)
def test_json_action_whitelist_and_host_schema_validation(action, monkeypatch):
    monkeypatch.setattr(agent, "native_turn", lambda *a, **k: {"content": json.dumps(action)})
    with pytest.raises(ValueError):
        json_tool_turn(
            None,
            [],
            [agent.tool_schema("calculate", "Calculate", {"expression": {"type": "string"}})],
        )


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -1, 101])
def test_action_numbers_must_be_finite_and_within_budget(value):
    with pytest.raises(ValueError):
        validate_value(value, {"type": "number", "minimum": 0, "maximum": 100})


def test_memory_compression_preserves_originals_and_resumes_partial_trees(tmp_path, monkeypatch):
    original = "income costs evidence\n" * 600
    identity = mission_memory.archive(tmp_path, "synthetic-source", original)
    profile = load_profile(Path(__file__).parents[1] / "profiles/v100.toml", tmp_path)
    monkeypatch.setattr(
        competition,
        "helper_client",
        lambda *a: SimpleNamespace(completion=lambda messages: "income costs evidence summary"),
    )
    for _ in range(20):
        if mission_memory.compress(tmp_path, profile, max_summaries=1) == 0:
            break
    else:
        pytest.fail("Compression never finished its cached tree")
    memory = mission_memory.store(tmp_path)
    try:
        assert (
            memory.db.execute("SELECT text FROM documents WHERE id=?", (identity,)).fetchone()[0]
            == original
        )
        leaf_text = "".join(
            row[0]
            for row in memory.db.execute(
                "SELECT text FROM nodes WHERE document_id=? AND level=0 ORDER BY ordinal",
                (identity,),
            )
        )
        assert leaf_text == original
        assert (
            memory.db.execute(
                "SELECT MAX(level) FROM nodes WHERE document_id=?", (identity,)
            ).fetchone()[0]
            >= 2
        )
    finally:
        memory.close()
    reader = research_tools.ResearchTools(tmp_path, {}, "B")
    hits = reader.execute("search_memory", {"query": "income costs"})
    passage = reader.execute("read_source", {"source_id": hits["passages"][0]["id"]})
    assert passage["text"] in original


def test_context_profile_keeps_working_inputs_and_gpu_headroom_policy(tmp_path):
    profile = load_profile(Path(__file__).parents[1] / "profiles/v100.toml", tmp_path)
    old = json.dumps(profile, sort_keys=True)
    for context in (32768, 16384, 8192):
        chosen = mission.setup_profile(profile, context)
        assert (
            chosen["runtime"]["context_window"] == chosen["server"]["context_per_slot"] == context
        )
        assert chosen["runtime"]["tool_protocol"] == "json"
        assert chosen["server"]["draft_model"] == ""
        assert chosen["training"]["max_steps"] == 50
    assert json.dumps(profile, sort_keys=True) == old


def mission_inputs(root):
    profile = bootstrap(root)
    prepare_challenge(profile, root)
    book = PaperBook(root)
    book.initialize()
    book.close()
    helper = json.loads(json.dumps(profile))
    helper["server"]["gpu_layers"] = 0
    helper["runtime"]["base_url"] = "http://127.0.0.1:8090"
    helper["resources"] = {"device": "cpu"}
    import tomli_w

    (root / "research/researcher-cpu.toml").write_text(tomli_w.dumps(helper))
    return profile


def test_background_start_snapshots_profile_and_does_not_reset_paper(tmp_path, monkeypatch):
    profile = mission_inputs(tmp_path)
    path = tmp_path / "input.json"
    atomic_json(path, profile)
    created = []
    monkeypatch.setattr(mission, "require_idle_gpu", lambda: None)
    monkeypatch.setattr(mission, "process_identity", lambda pid: "same-process")
    monkeypatch.setattr(mission, "command", lambda *args: list(map(str, args)))
    monkeypatch.setattr(
        mission.subprocess,
        "Popen",
        lambda args, **kwargs: created.append((args, kwargs)) or SimpleNamespace(pid=12345),
    )
    result = mission.start(tmp_path, path)
    assert result["running"]
    assert created[0][1]["start_new_session"] is True
    assert "mission-loop" in created[0][0]
    assert json.loads((Path(result["run"]) / "input-profile.json").read_text()) == profile
    with pytest.raises(FileExistsError, match="already running"):
        mission.start(tmp_path, path)
    monkeypatch.setattr(mission, "process_identity", lambda pid: "reused-pid")
    assert mission.status(tmp_path)["running"] is False
    book = PaperBook(tmp_path)
    try:
        assert not book.state()["instruments"]
        assert not book.state()["fee_profiles"]
    finally:
        book.close()


@pytest.mark.parametrize("owned", [False, True])
def test_stop_only_signals_verified_mission_session(tmp_path, monkeypatch, owned):
    record = {"running": True, "pid": 12345, "run": str(tmp_path / "run")}
    monkeypatch.setattr(mission, "status", lambda root: record)
    arguments = [b"python", b"mission-loop", record["run"].encode()]
    if not owned:
        arguments[1] = b"unrelated-process"
    monkeypatch.setattr(Path, "read_bytes", lambda path: b"\0".join(arguments))
    monkeypatch.setattr(mission.os, "getpgid", lambda pid: pid)
    signals = []
    monkeypatch.setattr(mission.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    if owned:
        assert mission.stop(tmp_path)["stop_requested"]
        assert signals == [(12345, mission.signal.SIGTERM)]
    else:
        with pytest.raises(ValueError, match="no process was stopped"):
            mission.stop(tmp_path)
        assert not signals


def test_mission_researches_before_auto_baseline_then_enters_infinite_learning(
    tmp_path, monkeypatch
):
    profile = mission_inputs(tmp_path)
    directory = tmp_path / "research/mission/run-test"
    directory.mkdir(parents=True)
    events, capacities = [], iter([3, 20])

    @contextmanager
    def server(path, root, log):
        chosen = load_profile(path, root)
        events.append(("serve", chosen["server"]["context_per_slot"]))
        yield chosen

    monkeypatch.setattr(mission, "managed_server", server)
    monkeypatch.setattr(mission, "require_idle_gpu", lambda: None)
    monkeypatch.setattr(mission, "gpu_free_gib", lambda: next(capacities))
    monkeypatch.setattr(mission, "helper_client", lambda *args: None)
    monkeypatch.setattr(serving, "assert_served_expert", lambda *args: None)
    monkeypatch.setattr(
        research_tools.ResearchTools,
        "execute",
        lambda self, name, args: events.append(("source", args["url"])),
    )
    monkeypatch.setattr(paper_learning.PaperLearning, "__enter__", lambda self: self)
    monkeypatch.setattr(paper_learning.PaperLearning, "__exit__", lambda *args: None)
    monkeypatch.setattr(
        paper_learning.PaperLearning, "research", lambda self, *args: events.append(("research",))
    )
    monkeypatch.setattr(mission_memory, "compress", lambda *args: 0)
    monkeypatch.setattr(
        mission,
        "evaluate_suite",
        lambda *args: events.append(("baseline",)) or {"cases": [{"passed": True}]},
    )
    monkeypatch.setattr(
        continuous, "learn_loop", lambda *args, **kwargs: events.append(("learning", kwargs))
    )
    mission.run(tmp_path, profile, directory)
    assert events.index(("research",)) < events.index(("baseline",))
    assert ("serve", 32768) in events and ("serve", 16384) in events
    assert events[-1][0] == "learning"
    assert events[-1][1]["cycles"] == 0 and events[-1][1]["initial_update"] is True
    assert json.loads((directory / "status.json").read_text())["context_window"] == 16384
    assert json.loads((directory / "income-settings.json").read_text())["crypto"] is False


def test_first_update_can_use_verified_seed_then_failed_upgrade_retains_parent(
    tmp_path, monkeypatch
):
    from rlm.v100 import architectures

    profile = load_profile(Path(__file__).parents[1] / "profiles/v100.toml", tmp_path)
    pool, suite = tmp_path / "pool.jsonl", tmp_path / "suite.jsonl"
    pool.write_text(
        "".join(
            json.dumps(verified_record({"kind": "arithmetic", "expression": expr})) + "\n"
            for expr in ("2+3", "7*4")
        )
    )
    suite.write_text("fixed suite")
    baseline = {
        "schema": "v100-quality-v1",
        "suite_sha256": file_hash(suite),
        "generation": {},
        "memory_mode": "fixed",
        "cases": [{"id": "old", "passed": True}],
    }
    output = tmp_path / "learning"
    original = json.dumps(profile, sort_keys=True)

    @contextmanager
    def server(*args):
        yield profile

    monkeypatch.setattr(continuous, "managed_server", server)
    monkeypatch.setattr(continuous, "require_idle_gpu", lambda: None)
    monkeypatch.setattr(continuous, "helper_client", lambda *args: None)
    monkeypatch.setattr(architectures, "prepare_inputs", lambda *args: None)
    monkeypatch.setattr(continuous, "research_task", lambda *args: {})
    monkeypatch.setattr(continuous, "review_research", lambda *args: None)
    monkeypatch.setattr(
        continuous.shutil, "disk_usage", lambda p: SimpleNamespace(free=200 * 2**30)
    )
    ticks = iter([0, 1000])
    monkeypatch.setattr(continuous.time, "monotonic", lambda: next(ticks))
    trials = []

    def evolve(*args, **kwargs):
        trials.append(args[2].read_bytes())
        raise RuntimeError("synthetic candidate failed")

    monkeypatch.setattr(continuous, "evolve", evolve)
    result = continuous.learn_loop(
        profile,
        tmp_path,
        pool,
        output,
        suite,
        [baseline],
        tmp_path / "helper",
        cycles=2,
        interval=30,
        initial_update=True,
    )
    assert trials == [pool.read_bytes()]
    assert len(result["cycles"]) == 2
    assert result["cycles"][0]["status"] == "upgrade failed; previous version retained"
    assert result["cycles"][1]["status"] == "no new verified and admitted examples"
    assert json.dumps(profile, sort_keys=True) == original
    assert json.loads((output / "live.json").read_text()) == profile
