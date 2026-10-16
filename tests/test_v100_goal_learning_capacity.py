"""Explicit bounded expansion preserves all archived goal evidence."""

import json
from types import SimpleNamespace

import pytest

from rlm.v100 import goal_learning as lab
from rlm.v100 import research_tools


def configure(root, value):
    path = root / "research/goal-learning/capacity.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def seed(root, count):
    with lab.database(root) as db:
        db.executemany(
            "INSERT INTO observations VALUES(?,?,?)",
            ((str(i), i, "{}") for i in range(count)),
        )


def no_fetch(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("No source request may occur before capacity and disk admission")

    monkeypatch.setattr(research_tools, "download_page", forbidden)


def test_capacity_defaults_preserve8192_limit(tmp_path):
    assert lab.evidence_capacity(tmp_path) == {
        "max_observations": 8192,
        "minimum_free_disk_gib": 2,
    }
    assert not (tmp_path / "research").exists()


@pytest.mark.parametrize(
    "value",
    [
        [],
        None,
        {"unknown": 1},
        {"max_observations": True},
        {"max_observations": 8191},
        {"max_observations": 131073},
        {"max_observations": 65536.0},
        {"max_observations": "65536"},
        {"minimum_free_disk_gib": True},
        {"minimum_free_disk_gib": 1.99},
        {"minimum_free_disk_gib": 65},
        {"minimum_free_disk_gib": float("nan")},
        {"minimum_free_disk_gib": float("inf")},
    ],
)
def test_invalid_budget_fails_before_fetch(tmp_path, monkeypatch, value):
    configure(tmp_path, value)
    no_fetch(monkeypatch)
    with pytest.raises(ValueError):
        lab.observe(tmp_path, "https://example.org/metric", "price")


def test_oversized_budget_read_is_bounded(tmp_path):
    path = configure(tmp_path, {})
    path.write_bytes(b" " * 4097)
    with pytest.raises(ValueError, match="4 KiB"):
        lab.evidence_capacity(tmp_path)


def test_symlink_budget_is_rejected(tmp_path):
    path = configure(tmp_path, {})
    target = tmp_path / "real.json"
    target.write_text("{}", encoding="utf-8")
    path.unlink()
    try:
        path.symlink_to(target)
    except OSError:
        pytest.skip("Host does not permit symlink creation")
    with pytest.raises(ValueError, match="escaped"):
        lab.evidence_capacity(tmp_path)


def test_exhausted_default_preserves_evidence_and_never_fetches(tmp_path, monkeypatch):
    seed(tmp_path, 8192)
    no_fetch(monkeypatch)
    with pytest.raises(ValueError, match="8192/8192"):
        lab.observe(tmp_path, "https://example.org/metric", "price")
    with lab.database(tmp_path) as db:
        assert db.execute("SELECT count(*) FROM observations").fetchone()[0] == 8192
    status = lab.status(tmp_path)["evidence_capacity"]
    assert status["state"] == "exhausted" and status["remaining"] == 0


def test_explicit_expansion_allows_new_real_observation_with_same_source_budget(
    tmp_path, monkeypatch
):
    seed(tmp_path, 8192)
    configure(tmp_path, {"max_observations": 65536, "minimum_free_disk_gib": 4})
    monkeypatch.setattr(lab.shutil, "disk_usage", lambda root: SimpleNamespace(free=8 * 2**30))
    calls = []

    def download(url, max_bytes):
        calls.append(max_bytes)
        return url, '{"price":12.5}'

    monkeypatch.setattr(research_tools, "download_page", download)
    result = lab.observe(tmp_path, "https://example.org/metric", "price")
    assert result["value"] == 12.5 and calls == [256 * 1024]
    assert lab.observation(tmp_path, result["id"])["body_sha256"] == result["body_sha256"]
    capacity = lab.status(tmp_path)["evidence_capacity"]
    assert capacity["observations"] == 8193 and capacity["remaining"] == 65536 - 8193


def test_configured_disk_reserve_blocks_before_source_request(tmp_path, monkeypatch):
    configure(tmp_path, {"max_observations": 65536, "minimum_free_disk_gib": 4})
    monkeypatch.setattr(lab.shutil, "disk_usage", lambda root: SimpleNamespace(free=3 * 2**30))
    no_fetch(monkeypatch)
    with pytest.raises(ValueError, match="4 GiB"):
        lab.observe(tmp_path, "https://example.org/metric", "price")


def test_capacity_rechecked_under_write_transaction_after_fetch(tmp_path, monkeypatch):
    seed(tmp_path, 8191)
    monkeypatch.setattr(lab.shutil, "disk_usage", lambda root: SimpleNamespace(free=8 * 2**30))

    def download(url, max_bytes):
        with lab.database(tmp_path) as db:
            db.execute("INSERT INTO observations VALUES('concurrent',0,'{}')")
        return url, '{"price":12.5}'

    monkeypatch.setattr(research_tools, "download_page", download)
    with pytest.raises(ValueError, match="8192/8192"):
        lab.observe(tmp_path, "https://example.org/metric", "price")
    assert not list((tmp_path / "research/goal-learning/sources").glob("*.json"))
    with lab.database(tmp_path) as db:
        assert db.execute("SELECT count(*) FROM observations").fetchone()[0] == 8192
