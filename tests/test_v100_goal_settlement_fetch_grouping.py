"""Shared HTTP observations retain independent commitment settlement windows."""

import json
from types import SimpleNamespace

import pytest

from rlm.v100 import goal_learning as lab
from rlm.v100 import research_tools
from tests.test_v100_recovery import financial_world


@pytest.fixture
def world(tmp_path, monkeypatch):
    financial_world(tmp_path)
    clock, values, calls = [1000.0], {"a": 100, "b": 200}, []
    monkeypatch.setattr(lab, "time", SimpleNamespace(time=lambda: clock[0]))
    monkeypatch.setattr(lab.shutil, "disk_usage", lambda root: SimpleNamespace(free=8 * 2**30))

    def download(url, max_bytes):
        calls.append(url)
        return url, json.dumps(values)

    monkeypatch.setattr(research_tools, "download_page", download)
    return tmp_path, clock, values, calls


def commit(root, field="a", branch="A", horizon=60, question="forward change"):
    target = lab.observe(root, "https://example.org/metric", field)
    return lab.predict(
        root,
        branch,
        {
            "question": question,
            "rationale": "Precommitted source observation",
            "target": target["id"],
            "evidence": [target["id"]],
            "horizon_seconds": horizon,
            "threshold": 1,
            "probabilities": [0.1, 0.1, 0.8],
        },
    )


def outcomes(root):
    with lab.database(root) as db:
        return {
            row[0]: json.loads(row[1]) if row[1] else None
            for row in db.execute("SELECT id,outcome FROM forecasts")
        }


def test_same_url_field_fetches_once_and_settles_two_real_commitments(world):
    root, clock, values, calls = world
    first = commit(root)
    second = commit(root, branch="B")
    calls.clear()
    clock[0], values["a"] = 1061.0, 105
    lab.tick(root)
    settled = outcomes(root)
    assert len(calls) == 1
    assert settled[first["id"]]["observation"] == settled[second["id"]]["observation"]
    observed = lab.observation(root, settled[first["id"]]["observation"])
    assert observed["value"] == 105 and observed["acquired"] == 1061.0
    assert len(lab.records(root)) == 2


def test_same_url_different_numeric_fields_are_separate_fetches(world):
    root, clock, values, calls = world
    first = commit(root, field="a")
    second = commit(root, field="b")
    calls.clear()
    clock[0], values["a"], values["b"] = 1061.0, 105, 190
    lab.tick(root)
    settled = outcomes(root)
    assert len(calls) == 2
    assert settled[first["id"]]["actual"] == 2
    assert settled[second["id"]]["actual"] == 0
    assert settled[first["id"]]["observation"] != settled[second["id"]]["observation"]


def test_late_shared_fetch_rejects_only_the_forecast_outside_its_window(world, monkeypatch):
    root, clock, values, calls = world
    early = commit(root, horizon=60)
    later = commit(root, branch="B", horizon=70)
    clock[0], values["a"] = 1300.0, 105
    calls.clear()
    original = research_tools.download_page

    def delayed(url, max_bytes):
        clock[0] = 1365.0
        return original(url, max_bytes)

    monkeypatch.setattr(research_tools, "download_page", delayed)
    lab.tick(root)
    settled = outcomes(root)
    assert len(calls) == 1
    assert settled[early["id"]] is None
    assert settled[later["id"]]["actual"] == 2
    assert lab.observation(root, settled[later["id"]]["observation"])["acquired"] == 1365.0
    calls.clear()
    lab.tick(root)
    assert not calls  # Expired early forecast is not backfilled or repaired.
    assert outcomes(root)[early["id"]] is None


def test_four_forecast_budget_is_preserved_even_when_sources_are_shared(world):
    root, clock, values, calls = world
    for index in range(5):
        commit(root, question=f"predeclared forecast{index}")
    clock[0], values["a"] = 1061.0, 105
    calls.clear()
    lab.tick(root)
    assert len(calls) == 1
    assert sum(value is not None for value in outcomes(root).values()) == 4
    lab.tick(root)
    assert len(calls) == 2
    assert sum(value is not None for value in outcomes(root).values()) == 5


def test_shared_source_failure_is_not_retried_inside_same_tick(world, monkeypatch):
    root, clock, _, calls = world
    commit(root)
    commit(root, branch="B")
    clock[0] = 1061.0
    calls.clear()

    def fail(url, max_bytes):
        calls.append(url)
        raise ValueError("Source unavailable")

    monkeypatch.setattr(research_tools, "download_page", fail)
    lab.tick(root)
    assert len(calls) == 1
    assert all(value is None for value in outcomes(root).values())
    lab.tick(root)
    assert len(calls) == 2  # A later tick may retry while windows remain eligible.
