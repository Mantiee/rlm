import copy
import json
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest
import requests

from rlm.clients.llamacpp import LlamaCppClient
from rlm.v100 import competition, mission, remote_helper
from rlm.v100.agent import native_turn
from rlm.v100.cli import server_command
from rlm.v100.common import atomic_json, load_profile
from rlm.v100.paper_agents import financial_helper_profile
from rlm.v100.tool_protocol import json_tool_turn

DIGEST = "a" * 64
URL = "http://192.168.0.61:11435"
INFO = {
    "details": {"family": "qwen35", "quantization_level": "Q8_0"},
    "model_info": {"tokenizer.ggml.model": "gpt2"},
    "template": "fixed official template",
}
LOADED = {
    "name": remote_helper.MODEL,
    "digest": DIGEST,
    "size": 10 * 2**30,
    "size_vram": 10 * 2**30,
    "context_length": 32768,
}


@pytest.fixture
def transport(monkeypatch):
    monkeypatch.setattr(remote_helper, "helper_boot_id", lambda: "test-boot")
    calls = []
    state = {"info": copy.deepcopy(INFO), "loaded": copy.deepcopy(LOADED), "digest": DIGEST}

    def request(self, endpoint, data=None):
        calls.append((endpoint, data))
        if endpoint == "/api/version":
            return {"version": state.get("version", "0.40.0")}
        if endpoint == "/api/show":
            return state["info"]
        if endpoint == "/api/tags":
            return {"models": [{"name": remote_helper.MODEL, "digest": state["digest"]}]}
        if endpoint == "/api/ps":
            return {"models": [state["loaded"]]}
        assert endpoint == "/api/chat"
        return {
            "model": remote_helper.MODEL,
            "done": True,
            "done_reason": "stop",
            "message": {"content": '{"answer":"4"}', "thinking": "private reasoning"},
            "prompt_eval_count": 128,
            "eval_count": 8,
            "eval_duration": 100_000_000,
            **state.get("response", {}),
        }

    monkeypatch.setattr(remote_helper.OllamaResearchClient, "remote_request", request)
    return calls, state


def client(**kwargs):
    return remote_helper.OllamaResearchClient(
        base_url=URL,
        model_digest=DIGEST,
        metadata_sha256=remote_helper.metadata_sha(INFO),
        context_window=32768,
        **kwargs,
    )


def profile(root):
    return load_profile(Path(__file__).parents[1] / "profiles/v100.toml", root)


def test_remote_thinking_and_larger_budget_reach_ollama(transport):
    instance = client(enable_thinking=True, helper_batch_tokens=16, helper_duty_percent=30)
    instance.http_request(
        "/v1/chat/completions",
        {
            "model": remote_helper.MODEL,
            "messages": [{"role": "user", "content": "Check explicit fees"}],
            "max_tokens": 4096,
        },
    )
    payload = next(data for endpoint, data in transport[0] if endpoint == "/api/chat")
    assert payload["think"] is True
    assert payload["options"]["num_predict"] == 4096 and payload["options"]["num_batch"] == 16


def prepared(root):
    path = remote_helper.prepare_remote(profile(root), root, URL, 32768, DIGEST)
    return path, load_profile(path, root)


def legacy_profile(root):
    path, settings = prepared(root)
    settings["resources"].pop("metadata_hash_scheme")
    settings["resources"].pop("metadata_snapshot")
    settings["resources"]["metadata_sha256"] = "e" * 64
    atomic_json(path, settings)
    return path, settings


def test_canonical_migration_validates_new_loaded_context_without_losing_old_pin(
    transport, tmp_path
):
    transport[1]["info"].update(parameters="temperature 1\ntop_k 20")
    path, settings = prepared(tmp_path)
    settings["runtime"]["context_window"] = 131072
    settings["server"]["context_per_slot"] = 131072
    atomic_json(path, settings)
    transport[1]["info"]["parameters"] = "top_k 20\ntemperature 1"
    report = remote_helper.canonicalize_remote(tmp_path, context_window=32768)
    audit = Path(report["audit"])
    assert json.loads((audit / "profile-before.json").read_text()) == settings
    current = load_profile(path, tmp_path)
    assert current["runtime"]["context_window"] == 32768
    assert current["resources"]["metadata_hash_scheme"] == remote_helper.ORDERED_METADATA
    assert remote_helper.canonicalize_remote(tmp_path)["status"] == "canonical identity verified"


def test_canonical_migration_still_rejects_semantic_changes_or_unloaded_model(transport, tmp_path):
    path, _ = prepared(tmp_path)
    original = path.read_bytes()
    transport[1]["info"]["template"] = "new unapproved template"
    with pytest.raises(ValueError, match="beyond parameter order"):
        remote_helper.canonicalize_remote(tmp_path, context_window=32768)
    assert path.read_bytes() == original
    transport[1]["info"] = copy.deepcopy(INFO)
    transport[1]["loaded"]["context_length"] = 131072
    with pytest.raises(ValueError, match="context"):
        remote_helper.canonicalize_remote(tmp_path, context_window=32768)
    assert path.read_bytes() == original


def test_late_helper_preparation_applies_low_workload_only_after_verified_identity(
    transport, tmp_path
):
    from rlm.v100.campaign import configure_helper

    path, _ = prepared(tmp_path)
    result = configure_helper(tmp_path)
    actual = load_profile(path, tmp_path)
    assert actual["resources"]["helper_batch_tokens"] == 16
    assert actual["resources"]["helper_duty_percent"] == 15
    assert result["loaded"]["context_length"] == 32768
    transport[1]["info"]["template"] = "unauthorized template"
    previous = path.read_bytes()
    with pytest.raises(ValueError, match="metadata"):
        configure_helper(tmp_path)
    assert path.read_bytes() == previous


def test_stable_metadata_ignores_only_date_and_keeps_legacy_semantics(transport):
    first = {**INFO, "modified_at": "before"}
    second = {**INFO, "modified_at": "after"}
    assert remote_helper.metadata_sha(first) != remote_helper.metadata_sha(second)
    assert remote_helper.metadata_sha(first, remote_helper.STABLE_METADATA) == (
        remote_helper.metadata_sha(second, remote_helper.STABLE_METADATA)
    )
    instance = client(metadata_hash_scheme=remote_helper.STABLE_METADATA)
    instance.metadata_sha256 = remote_helper.metadata_sha(first, remote_helper.STABLE_METADATA)
    transport[1]["info"] = second
    instance.identity()


@pytest.mark.parametrize(
    "field,value",
    [
        ("template", "different"),
        ("parameters", "temperature 2"),
        ("renderer", "new-renderer"),
        ("parser", "new-parser"),
        ("modelfile", "FROM other-blob"),
        ("messages", [{"role": "user", "content": "changed"}]),
        ("model_info", {"tokenizer.ggml.model": "gpt2", "new_dimension": 123}),
        ("unrecognized_behavior_field", "changed"),
    ],
)
def test_stable_metadata_still_blocks_other_changes(transport, field, value):
    instance = client(metadata_hash_scheme=remote_helper.STABLE_METADATA)
    transport[1]["info"][field] = value
    with pytest.raises(ValueError, match="metadata"):
        instance.identity()
    assert not any(endpoint == "/api/chat" for endpoint, _ in transport[0])


def test_explicit_migration_keeps_original_profile_and_snapshots_and_is_idempotent(
    transport, tmp_path
):
    path, original = legacy_profile(tmp_path)
    report = remote_helper.migrate_remote_metadata(tmp_path)
    current = json.loads(path.read_text())
    audit = Path(report["audit"])
    assert json.loads((audit / "profile-before.json").read_text()) == original
    assert json.loads((audit / "show.json").read_text()) == INFO
    for section in original:
        if section != "resources":
            assert current[section] == original[section]
    assert current["resources"]["metadata_hash_scheme"] == remote_helper.STABLE_METADATA
    assert report["model_digest"] == DIGEST
    assert report["old_full_hash"] == "e" * 64
    assert report["weights_changed"] is False and report["mission_started"] is False
    transport[1]["info"]["modified_at"] = "another pull"
    assert "already stable" in remote_helper.migrate_remote_metadata(tmp_path)["status"]
    assert json.loads(path.read_text()) == current
    transport[1]["info"]["template"] = "changed"
    with pytest.raises(ValueError, match="metadata"):
        remote_helper.migrate_remote_metadata(tmp_path)
    assert json.loads(path.read_text()) == current
    assert len(list(audit.parent.glob("migration-*"))) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"digest": "b" * 64},
        {"version": "0.41.0"},
        {"info": {**INFO, "system": "custom system"}},
        {"info": {**INFO, "remote_host": "cloud"}},
        {"loaded": {**LOADED, "context_length": 8192}},
        {"loaded": {**LOADED, "size_vram": 13 * 2**30}},
        {"loaded": {**LOADED, "size_vram": 4 * 2**30}},
    ],
)
def test_migration_rejects_changed_identity_runtime_or_residency_without_rebinding(
    transport, tmp_path, change
):
    path, _ = legacy_profile(tmp_path)
    before = path.read_bytes()
    transport[1].update(change)
    with pytest.raises(ValueError):
        remote_helper.migrate_remote_metadata(tmp_path)
    assert path.read_bytes() == before
    assert not list((tmp_path / "research/helper-metadata").glob("migration-*"))


def test_migration_refuses_running_mission_and_midflight_template_change(
    transport, tmp_path, monkeypatch
):
    path, _ = legacy_profile(tmp_path)
    before = path.read_bytes()
    monkeypatch.setattr(mission, "status", lambda root: {"running": True})
    with pytest.raises(RuntimeError, match="Stop the mission"):
        remote_helper.migrate_remote_metadata(tmp_path)
    monkeypatch.setattr(mission, "status", lambda root: {"running": False})
    request = remote_helper.OllamaResearchClient.remote_request

    def swapped(self, endpoint, data=None):
        result = request(self, endpoint, data)
        if endpoint == "/api/ps":
            transport[1]["info"] = {
                **transport[1]["info"],
                "template": "swapped during verification",
            }
        return result

    monkeypatch.setattr(remote_helper.OllamaResearchClient, "remote_request", swapped)
    with pytest.raises(ValueError, match="metadata"):
        remote_helper.migrate_remote_metadata(tmp_path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("context", [65536, 131072])
def test_large_remote_context_still_requires_loaded_context_and_vram_budget(
    tmp_path, transport, context
):
    calls, state = transport
    state["loaded"]["context_length"] = context
    path = remote_helper.prepare_remote(profile(tmp_path), tmp_path, URL, context, DIGEST)
    chosen = load_profile(path, tmp_path)
    assert chosen["runtime"]["context_window"] == context
    state["loaded"]["size_vram"] = 13 * 2**30
    with pytest.raises(ValueError, match="GPU"):
        competition.helper_client(chosen).loaded()


@pytest.mark.parametrize(
    "url",
    [
        "http://8.8.8.8:11435",
        "http://100.78.125.51:11435",
        "http://localhost:11435",
        "http://192.168.0.61",
        "https://192.168.0.61:11435",
        URL + "/api",
        URL + "?other=host",
        "http://u:p@192.168.0.61:11435",
    ],
)
def test_remote_origin_is_explicit_private_ipv4(url):
    with pytest.raises(ValueError):
        remote_helper.private_origin(url)


def test_local_backend_remains_loopback_only():
    with pytest.raises(ValueError, match="loopback"):
        LlamaCppClient(base_url=URL)


def test_native_json_action_schema_and_measured_usage(transport, tmp_path):
    calls, _ = transport
    instance = client(activity_root=str(tmp_path), activity_actor="tester")
    result = json_tool_turn(
        instance,
        [{"role": "user", "content": "2+2?"}],
        [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
    )
    assert result["content"] == "4"
    payload = [item for endpoint, item in calls if endpoint == "/api/chat"][0]
    assert payload["think"] is False and payload["stream"] is False
    assert payload["options"]["num_ctx"] == 32768
    assert payload["format"]["oneOf"][0]["properties"]["tool"]["const"] == "calculate"
    assert "tools" not in payload
    text = "".join(
        path.read_text() for path in (tmp_path / "research/logs/activity").rglob("*.jsonl")
    )
    assert "private reasoning" not in text
    assert '"prompt_tokens": 128' in text
    assert '"remote_vram_gib": 10.0' in text
    assert instance.completion("2+2?") == '{"answer":"4"}'
    assert instance.get_last_usage().total_input_tokens == 128


def test_remote_context_admission_precedes_http(transport):
    calls, _ = transport
    with pytest.raises(ValueError, match="working context"):
        native_turn(client(), [{"role": "user", "content": "ą" * 20000}])
    assert not calls


@pytest.mark.parametrize("batch,duty", [(512, 65), (64, 66), (64, 0), (64, True), (True, 65)])
def test_helper_workload_limits_reject_invalid_settings_before_http(transport, batch, duty):
    with pytest.raises(ValueError, match="batch|wall-time"):
        client(helper_batch_tokens=batch, helper_duty_percent=duty)
    assert not transport[0]


def test_helper_pacing_returns_answer_before_idle_and_survives_new_client(
    transport, tmp_path, monkeypatch
):
    clock = {"now": 0.0}
    starts, sleeps = [], []
    request = remote_helper.OllamaResearchClient.remote_request

    def timed(self, endpoint, data=None):
        if endpoint == "/api/chat":
            starts.append(clock["now"])
            assert data["options"]["num_batch"] == 64
            assert data["options"]["num_thread"] == 4
            assert data["options"]["num_ctx"] == 32768
            clock["now"] += 13
        return request(self, endpoint, data)

    def sleep(seconds):
        assert 0 < seconds <= 1
        sleeps.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(remote_helper.OllamaResearchClient, "remote_request", timed)
    monkeypatch.setattr(remote_helper.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(remote_helper.time, "sleep", sleep)
    data = {
        "model": remote_helper.MODEL,
        "messages": [{"role": "user", "content": "2+2?"}],
        "max_tokens": 128,
    }
    first = client(activity_root=str(tmp_path)).http_request("/v1/chat/completions", data)
    assert clock["now"] == 13 and sleeps == []
    assert first["timings"]["helper_planned_idle_seconds"] == 7
    second = client(activity_root=str(tmp_path)).http_request("/v1/chat/completions", data)
    assert starts == [0, 20] and sum(sleeps) == 7
    assert second["timings"]["helper_waited_seconds"] == 7
    assert 13 / (starts[1] - starts[0]) == 0.65
    journal = next((tmp_path / "research/logs/activity").glob("*/timeline.jsonl"))
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [row["payload"]["batch_tokens"] for row in events] == [64, 64]
    assert "not a GPU utilization" in events[0]["payload"]["scope"]


def test_helper_idle_lease_blocks_another_process(transport, tmp_path):
    instance = client(activity_root=str(tmp_path))
    with instance.workload_slot():
        lock = next((tmp_path / "research/state").glob("helper-*.workload.lock"))
        script = """
import fcntl, sys
with open(sys.argv[1], 'a') as lease:
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('blocked')
    else:
        raise SystemExit('lease was not held')
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(lock)], capture_output=True, text=True, check=True
        )
        assert result.stdout.strip() == "blocked"


def test_previous_boot_deadline_is_discarded_and_wait_is_interruptible(
    transport, tmp_path, monkeypatch
):
    instance = client(activity_root=str(tmp_path))
    with instance.workload_slot() as quota:
        path = quota["path"]
    atomic_json(path, {"boot_id": "previous-boot", "not_before": 1e99})
    monkeypatch.setattr(remote_helper.time, "monotonic", lambda: 0.0)
    with instance.workload_slot() as quota:
        assert quota["waited_seconds"] == 0
        boot = quota["boot_id"]
    atomic_json(path, {"boot_id": boot, "not_before": 7})

    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(remote_helper.time, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        with instance.workload_slot():
            pytest.fail("ignored idle reservation")
    assert not transport[0]


def test_optional_helper_work_preserves_cooldown_and_releases_lock(transport, tmp_path):
    instance = client(activity_root=str(tmp_path))
    with instance.workload_slot() as quota:
        instance.reserve_helper_idle(quota, 600)
        path = quota["path"]
    original = path.read_bytes()
    with pytest.raises(TimeoutError, match="cooling down"):
        with instance.workload_slot(wait=False):
            pytest.fail("Optional vision ignored the cooldown")
    assert path.read_bytes() == original
    assert remote_helper.HELPER_WORKLOAD_LOCK.acquire(blocking=False)
    remote_helper.HELPER_WORKLOAD_LOCK.release()
    assert not transport[0]


def test_optional_helper_work_does_not_wait_for_another_thread(transport):
    with remote_helper.HELPER_WORKLOAD_LOCK:
        with pytest.raises(TimeoutError, match="busy"):
            with client().workload_slot(wait=False):
                pytest.fail("Optional vision bypassed a busy helper")
    assert not transport[0]


def test_failed_helper_request_reserves_idle_without_accepting_an_answer(
    transport, tmp_path, monkeypatch
):
    clock = {"now": 0.0}
    request = remote_helper.OllamaResearchClient.remote_request

    def failed(self, endpoint, data=None):
        if endpoint == "/api/chat":
            clock["now"] += 13
            raise requests.ConnectionError("worker disconnected")
        return request(self, endpoint, data)

    monkeypatch.setattr(remote_helper.OllamaResearchClient, "remote_request", failed)
    monkeypatch.setattr(remote_helper.time, "monotonic", lambda: clock["now"])
    with pytest.raises(requests.ConnectionError, match="disconnected"):
        client(activity_root=str(tmp_path)).http_request(
            "/v1/chat/completions",
            {
                "model": remote_helper.MODEL,
                "messages": [{"role": "user", "content": "2+2?"}],
                "max_tokens": 128,
            },
        )
    state = json.loads(
        next((tmp_path / "research/state").glob("helper-*.workload.json")).read_text()
    )
    assert state["not_before"] == 20
    assert clock["now"] == 13


def test_existing_remote_profile_gets_pacing_defaults_without_mutation(transport, tmp_path):
    _, settings = prepared(tmp_path)
    settings["resources"].pop("helper_batch_tokens")
    settings["resources"].pop("helper_duty_percent")
    original = copy.deepcopy(settings)
    remote_helper.validate_remote(settings)
    instance = competition.helper_client(settings, tmp_path)
    assert instance.helper_batch_tokens == 64 and instance.helper_duty_percent == 65
    assert settings == original


@pytest.mark.parametrize(
    "change",
    [
        {"digest": "b" * 64},
        {"info": {**INFO, "system": "changed"}},
        {"info": {**INFO, "template": "changed"}},
        {"loaded": {**LOADED, "context_length": 8192}},
        {"loaded": {**LOADED, "size_vram": 13 * 2**30}},
        {"loaded": {**LOADED, "size_vram": 4 * 2**30}},
    ],
)
def test_changed_model_context_or_memory_blocks_research(transport, change):
    calls, state = transport
    state.update(change)
    with pytest.raises(ValueError):
        client().completion("2+2?")
    assert not any(endpoint == "/api/chat" for endpoint, _ in calls)


@pytest.mark.parametrize(
    "response",
    [
        {"done": False},
        {"prompt_eval_count": 50000},
        {"eval_count": 2000},
        {"model": "cloud:model"},
        {"done_reason": "unexpected"},
    ],
)
def test_incomplete_or_invalid_usage_not_accepted(transport, response):
    _, state = transport
    state["response"] = response
    with pytest.raises(ValueError):
        client().completion("2+2?")


def test_exhausted_output_not_accepted_as_native_turn(transport):
    _, state = transport
    state["response"] = {"done_reason": "length"}
    with pytest.raises(ValueError, match="exhausted|budget|truncated"):
        native_turn(client(), [{"role": "user", "content": "2+2?"}])


def test_remote_profile_keeps_context_and_never_spawns_or_stops_windows(
    transport, tmp_path, monkeypatch
):
    path, settings = prepared(tmp_path)
    selected = remote_helper.selected_helper(tmp_path)
    assert selected == path
    assert mission.setup_helper(settings) == settings
    snapshot = financial_helper_profile(path, tmp_path)
    assert load_profile(snapshot, tmp_path)["runtime"]["context_window"] == 32768
    assert settings["server"]["model"] == ""

    def forbidden(*args, **kwargs):
        raise AssertionError("Remote server must never launch or stop a local process")

    monkeypatch.setattr(competition.subprocess, "Popen", forbidden)
    monkeypatch.setattr(competition, "available_ram_gib", forbidden)
    with competition.managed_server(snapshot, tmp_path, tmp_path / "helper.log") as actual:
        instance = competition.helper_client(actual, tmp_path)
        assert isinstance(instance, remote_helper.OllamaResearchClient)
        assert instance.tool_protocol == "json"
        assert instance.activity_actor == "tester"
    assert URL in (tmp_path / "helper.log").read_text()
    with pytest.raises(ValueError, match="externally"):
        server_command(settings)


def test_remote_profile_failure_never_replaces_existing(transport, tmp_path):
    path, _ = prepared(tmp_path)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        prepared(tmp_path)
    assert path.read_bytes() == before
    changed = json.loads(before)
    changed["runtime"]["model_name"] = "qwen3.5:cloud"
    atomic_json(path, changed)
    with pytest.raises(ValueError, match="pinned"):
        load_profile(path, tmp_path)


def test_cpu_selected_when_remote_not_configured(tmp_path):
    path = tmp_path / "research/researcher-cpu.toml"
    path.parent.mkdir()
    path.touch()
    assert remote_helper.selected_helper(tmp_path) == path


def test_training_runner_accepts_pinned_remote_but_rejects_local_gpu(transport, tmp_path):
    _, settings = prepared(tmp_path)
    competition.validate_concurrent_researcher(settings)
    local = profile(tmp_path)
    with pytest.raises(ValueError, match="Concurrent researcher"):
        competition.validate_concurrent_researcher(local)
    for field, value in (("slots", 2), ("model", "/local/model.gguf")):
        changed = copy.deepcopy(settings)
        changed["server"][field] = value
        with pytest.raises(ValueError, match="pinned"):
            competition.validate_concurrent_researcher(changed)
    cpu = copy.deepcopy(local)
    cpu["resources"] = {"device": "cpu"}
    cpu["server"].update(gpu_layers=0, slots=1, threads=4, draft_model="")
    competition.validate_concurrent_researcher(cpu)
    cpu["server"]["threads"] = 8
    with pytest.raises(ValueError, match="four threads"):
        competition.validate_concurrent_researcher(cpu)


def test_remote_duel_reaches_branch_execution_without_launching_windows(
    transport, tmp_path, monkeypatch
):
    helper, settings = prepared(tmp_path)
    output = tmp_path / "duel"
    output.mkdir()
    suite = tmp_path / "suite.jsonl"
    suite.write_text("fixed")
    main = output / "main.json"
    atomic_json(main, profile(tmp_path))
    bundle = {
        "root": str(tmp_path),
        "branches": {"A": {"profile": str(main)}, "B": {"profile": str(main)}},
    }
    from rlm.v100 import inference, protection

    baseline = {
        "suite_sha256": protection.file_hash(suite),
        "generation": {},
        "memory_mode": "fixed prompt fixtures; no live retrieval",
    }
    monkeypatch.setattr(competition, "load_duel", lambda *args: bundle)
    monkeypatch.setattr(competition, "require_idle_gpu", lambda: None)
    monkeypatch.setattr(protection, "compare_reports", lambda *args: {"passed": True})
    monkeypatch.setattr(inference, "generation_conditions", lambda *args: {})
    observed = []

    def branches(out, root, selected_suite, selected_bundle, researcher, timeout, code):
        observed.append(researcher)
        return {"A": {}, "B": {}}

    monkeypatch.setattr(competition, "run_branches", branches)
    monkeypatch.setattr(competition, "judge_duel", lambda *args: {"winner": None})
    monkeypatch.setattr(
        competition.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("launched Windows")
    )
    assert competition.run_duel(output, tmp_path, suite, baseline, helper) == {"winner": None}
    assert observed == [settings]


def test_remote_adviser_runs_during_training_without_cpu_ram_reservation(
    transport, tmp_path, monkeypatch
):
    _, helper = prepared(tmp_path)
    output = tmp_path / "duel"
    branch = output / "A"
    branch.mkdir(parents=True)
    chosen = profile(tmp_path)
    chosen["training"]["output"] = str(branch / "training")
    metrics = branch / "training/metrics.jsonl"
    metrics.parent.mkdir()
    metrics.write_text('{"step":1,"loss":2.1}\n')
    path = branch / "profile.json"
    atomic_json(path, chosen)
    item = {
        "profile": str(path),
        "dataset": str(branch / "data.jsonl"),
        "decision": {"research_jobs": [{"role": "tester", "brief": "Check training metrics"}]},
    }
    completed = threading.Event()

    class Learner:
        returncode = None

        def poll(self):
            if completed.wait(0.001):
                self.returncode = 0
            return self.returncode

        def terminate(self):
            self.returncode = -15

    learner = Learner()
    monkeypatch.setattr(competition.subprocess, "Popen", lambda *args, **kwargs: learner)
    monkeypatch.setattr(competition, "available_ram_gib", lambda: 1)

    def advise(client, selected_branch, job, observed, root):
        assert isinstance(client, remote_helper.OllamaResearchClient)
        assert learner.returncode is None
        assert observed[-1]["loss"] == 2.1
        completed.set()
        return {"status": "unverified hypothesis"}

    monkeypatch.setattr(competition, "research_task", advise)
    result = competition.train_branch(tmp_path, output, "A", item, helper, train_timeout=5)
    assert completed.is_set() and result == [{"status": "unverified hypothesis"}]
    assert json.loads((branch / "workers.json").read_text())["submitted"] == 1


def test_reconnect_preserves_phase_and_rechecks_identity(transport, tmp_path, monkeypatch):
    path, _ = prepared(tmp_path)
    run = tmp_path / "research/mission/run-reconnect"
    original = {
        "phase": "research-and-learning-loop",
        "context_window": 131072,
        "baseline_passed": 58,
    }
    atomic_json(run / "status.json", original)
    atomic_json(tmp_path / "research/mission/active.json", {"pid": os.getpid(), "run": str(run)})
    checkpoint = run / "learning/live.json"
    checkpoint.parent.mkdir()
    checkpoint.write_text("preserved accepted model")
    saved = checkpoint.read_bytes()
    request = remote_helper.OllamaResearchClient.remote_request
    failed = []

    def transient(self, endpoint, data=None):
        if not failed:
            failed.append(endpoint)
            raise requests.ConnectionError("No route to host")
        assert self.timeout <= 15
        return request(self, endpoint, data)

    monkeypatch.setattr(remote_helper.OllamaResearchClient, "remote_request", transient)
    sleeps = []

    def wait(seconds):
        sleeps.append(seconds)
        waiting = json.loads((run / "status.json").read_text())
        assert waiting["phase"] == "waiting-for-remote-helper"
        assert waiting["context_window"] == 131072
        assert waiting["endpoint"] == URL
        assert checkpoint.read_bytes() == saved

    monkeypatch.setattr(competition.time, "sleep", wait)
    with competition.waiting_researcher(path, tmp_path, run / "helper.log"):
        assert json.loads((run / "status.json").read_text()) == original
    assert sleeps == [30]
    events = [
        json.loads(line)
        for p in (tmp_path / "research/logs/activity").glob("*/timeline.jsonl")
        for line in p.read_text().splitlines()
    ]
    assert [event["kind"] for event in events] == [
        "remote-helper-unavailable",
        "remote-helper-reconnected",
    ]
    assert checkpoint.read_bytes() == saved


def test_reconnect_never_replays_work_inside_context(transport, tmp_path, monkeypatch):
    path, _ = prepared(tmp_path)
    monkeypatch.setattr(
        competition.time, "sleep", lambda *args: pytest.fail("replayed active work")
    )
    with pytest.raises(requests.ConnectionError, match="during work"):
        with competition.waiting_researcher(path, tmp_path, tmp_path / "helper.log"):
            raise requests.ConnectionError("lost connection during work")


def test_reconnect_identity_mismatch_fails_without_retry(transport, tmp_path, monkeypatch):
    path, _ = prepared(tmp_path)
    transport[1]["digest"] = "b" * 64
    monkeypatch.setattr(
        competition.time, "sleep", lambda *args: pytest.fail("retried changed model")
    )
    with pytest.raises(ValueError, match="changed"):
        with competition.waiting_researcher(path, tmp_path, tmp_path / "helper.log"):
            pytest.fail("used changed model")


def test_reconnect_wait_is_interruptible_and_cpu_errors_are_not_retried(
    transport, tmp_path, monkeypatch
):
    remote, _ = prepared(tmp_path)

    @contextmanager
    def unavailable(*args):
        raise requests.ConnectionError("offline")
        yield

    monkeypatch.setattr(competition, "managed_server", unavailable)
    sleeps = []

    def interrupt(seconds):
        sleeps.append(seconds)
        raise KeyboardInterrupt

    monkeypatch.setattr(competition.time, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        with competition.waiting_researcher(remote, tmp_path, tmp_path / "helper.log"):
            pytest.fail("used unavailable helper")
    cpu = tmp_path / "cpu.json"
    local = profile(tmp_path)
    local["resources"] = {"device": "cpu"}
    atomic_json(cpu, local)
    with pytest.raises(requests.ConnectionError):
        with competition.waiting_researcher(cpu, tmp_path, tmp_path / "cpu.log"):
            pytest.fail("used unavailable CPU helper")
    assert sleeps == [30]


def test_remote_transport_rejects_redirect_and_ignores_proxy(monkeypatch):
    class Response:
        status_code = 302

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url, **kwargs):
            assert url == URL + "/api/tags"
            assert self.trust_env is False and kwargs["allow_redirects"] is False
            return Response()

    monkeypatch.setattr(remote_helper.requests, "Session", Session)
    with pytest.raises(ValueError, match="redirected"):
        client().remote_request("/api/tags")
    with pytest.raises(ValueError, match="management"):
        client().remote_request("/api/pull", {"model": remote_helper.MODEL})
