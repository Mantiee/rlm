"""Optional disabled remote research cannot prevent accepted local training."""

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlm.v100 import architectures, continuous
from rlm.v100.common import atomic_json, load_profile
from rlm.v100.insights import verified_record
from rlm.v100.protection import file_hash


@pytest.mark.parametrize(
    "origin,preferences,remote,disabled",
    [
        ("http://192.168.0.61:11435", {}, True, True),
        ("http://192.168.0.9:11435", {"remote_helper_enabled": False}, True, True),
        (
            "http://192.168.0.9:11435",
            {"remote_origin_enabled": {"http://192.168.0.9:11435": False}},
            True,
            True,
        ),
        ("http://192.168.0.9:11435", {}, True, False),
        ("http://127.0.0.1:8095", {"remote_helper_enabled": False}, False, False),
    ],
)
def test_training_researcher_omits_only_explicitly_disabled_remote(
    tmp_path, monkeypatch, origin, preferences, remote, disabled
):
    path = tmp_path / "helper.json"
    path.write_text("{}")
    helper = {
        "runtime": {"base_url": origin},
        "resources": {"device": "remote" if remote else "cpu"},
    }
    monkeypatch.setattr(continuous, "load_profile", lambda *a: helper)
    atomic_json(tmp_path / "research/user-preferences.json", preferences)
    original = path.read_bytes()
    assert continuous.training_researcher(tmp_path, path) == (None if disabled else path)
    assert path.read_bytes() == original
    assert json.loads((tmp_path / "research/user-preferences.json").read_text()) == preferences


def test_unrelated_profile_failure_is_not_swallowed(tmp_path, monkeypatch):
    path = tmp_path / "helper.json"
    path.write_text("{}")
    monkeypatch.setattr(
        continuous,
        "load_profile",
        lambda *a: (_ for _ in ()).throw(RuntimeError("different fault")),
    )
    with pytest.raises(RuntimeError, match="different fault"):
        continuous.training_researcher(tmp_path, path)


@pytest.mark.parametrize("disabled", [True, False])
def test_learn_loop_passes_only_eligible_helper_to_evolve(tmp_path, monkeypatch, disabled):
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
    helper = tmp_path / "helper.json"
    helper.write_text("{}")
    origin = "http://192.168.0.61:11435" if disabled else "http://192.168.0.9:11435"
    monkeypatch.setattr(
        continuous,
        "load_profile",
        lambda *a: {"runtime": {"base_url": origin}, "resources": {"device": "remote"}},
    )
    servers, captured = [], []

    @contextmanager
    def server(path, *args, **kwargs):
        servers.append(path)
        yield profile

    @contextmanager
    def waiting(*args, **kwargs):
        yield profile  # Existing serving-phase policy fallback; never start remote inference.

    monkeypatch.setattr(continuous, "managed_server", server)
    monkeypatch.setattr(continuous, "waiting_researcher", waiting)
    monkeypatch.setattr(continuous, "require_idle_gpu", lambda: None)
    monkeypatch.setattr(continuous, "helper_client", lambda *args: None)
    monkeypatch.setattr(architectures, "prepare_inputs", lambda *args: None)
    monkeypatch.setattr(continuous, "research_task", lambda *args: {})
    monkeypatch.setattr(continuous, "review_research", lambda *args: None)
    monkeypatch.setattr(
        continuous.shutil, "disk_usage", lambda p: SimpleNamespace(free=200 * 2**30)
    )

    class TrialObserved(BaseException):
        pass

    def evolve(*args, **kwargs):
        captured.append(args[6])
        assert args[0]["server"]["model"] == profile["server"]["model"]
        raise TrialObserved

    monkeypatch.setattr(continuous, "evolve", evolve)
    output = tmp_path / "learning"
    with pytest.raises(TrialObserved):
        continuous.learn_loop(
            profile,
            tmp_path,
            pool,
            output,
            suite,
            [baseline],
            helper,
            cycles=1,
            initial_update=True,
        )
    assert captured == [None if disabled else helper]
    assert helper not in servers
    assert servers  # Accepted local serving/planner path remains intact.
    deferred = output / "optional-helper-deferred-0001.json"
    assert deferred.exists() is disabled
    if disabled:
        assert "no remote model loaded" in json.loads(deferred.read_text())["scope"]
