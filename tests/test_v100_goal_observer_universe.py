"""Predeclared genuinely different source series, without changing split assignments."""

import json
import sqlite3
from types import SimpleNamespace

import pytest

from rlm.v100 import goal_learning, goal_observer, research_tools
from tests.test_v100_recovery import financial_world


def configure(root, value):
    path = root / "research/goal-learning/universe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_default_universe_preserves_original_eight_and_order(tmp_path):
    assert goal_observer.reference_products(tmp_path) == (
        "BTC-USD",
        "ETH-USD",
        "LTC-USD",
        "BCH-USD",
        "ETC-USD",
        "LINK-USD",
        "SOL-USD",
        "AVAX-USD",
    )
    assert not (tmp_path / "research").exists()


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        {"products": []},
        {"products": "ADA-USD"},
        {"products": ["ADA-USD", "ADA-USD"]},
        {"products": ["ada-USD"]},
        {"products": ["ADA-EUR"]},
        {"products": ["ADA-USD?variant=1"]},
        {"products": ["../ADA-USD"]},
        {"products": ["https://example.org"]},
        {"products": [True]},
        {"products": ["X" * 17 + "-USD"]},
        {"products": ["ADA-USD"], "validation_role": True},
        {"products": [f"ASSET{i}-USD" for i in range(65)]},
    ],
)
def test_invalid_universe_rejected_before_any_observation(tmp_path, monkeypatch, value):
    financial_world(tmp_path)
    configure(tmp_path, value)

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid universe cannot request source data")

    monkeypatch.setattr(goal_learning, "observe", forbidden)
    with pytest.raises(ValueError):
        goal_observer.tick(tmp_path)


def test_numeric_leading_products_and_64_product_bound(tmp_path):
    products = ["00-USD", "1INCH-USD", "2Z-USD"] + [f"ASSET{i}-USD" for i in range(61)]
    configure(tmp_path, {"products": products})
    assert goal_observer.reference_products(tmp_path) == tuple(products)


def test_configuration_read_size_bound(tmp_path):
    path = configure(tmp_path, {"products": ["ADA-USD"]})
    path.write_bytes(b" " * 4097)
    with pytest.raises(ValueError, match="4 KiB"):
        goal_observer.reference_products(tmp_path)


def test_linked_universe_is_rejected(tmp_path):
    path = configure(tmp_path, {"products": ["ADA-USD"]})
    target = tmp_path / "other.json"
    target.write_text('{"products":["ADA-USD"]}', encoding="utf-8")
    path.unlink()
    try:
        path.symlink_to(target)
    except OSError:
        pytest.skip("Host does not permit symlink creation")
    with pytest.raises(ValueError, match="escaped"):
        goal_observer.reference_products(tmp_path)


def test_new_series_are_precommitted_then_settled_without_split_reassignment(tmp_path, monkeypatch):
    financial_world(tmp_path)
    configure(tmp_path, {"products": ["ADA-USD", "DOT-USD"]})
    ledger = tmp_path / "research/state/splits.sqlite3"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(ledger) as db:
        db.execute("CREATE TABLE roles(source TEXT PRIMARY KEY,role TEXT NOT NULL)")
        db.executemany(
            "INSERT INTO roles VALUES(?,?)",
            [("existing-train", "train"), ("existing-validation", "validation")],
        )
    original_roles = ledger.read_bytes()
    clock, price, urls = [1000.0], [100.0], []
    monkeypatch.setattr(goal_learning, "time", SimpleNamespace(time=lambda: clock[0]))
    monkeypatch.setattr(goal_observer, "time", SimpleNamespace(time=lambda: clock[0]))
    monkeypatch.setattr(
        goal_learning.shutil, "disk_usage", lambda root: SimpleNamespace(free=8 * 2**30)
    )

    def download(url, max_bytes):
        urls.append(url)
        return url, json.dumps({"data": {"amount": str(price[0])}})

    monkeypatch.setattr(research_tools, "download_page", download)
    receipt = goal_observer.tick(tmp_path)
    assert receipt["universe"] == ["ADA-USD", "DOT-USD"]
    assert receipt["universe_sha256"] == goal_learning.digest(receipt["universe"])
    assert [r["product"] for r in receipt["results"]] == ["ADA", "DOT"]
    assert urls == [
        "https://api.coinbase.com/v2/prices/ADA-USD/spot",
        "https://api.coinbase.com/v2/prices/DOT-USD/spot",
    ]
    assert goal_learning.records(tmp_path) == []
    clock[0], price[0] = 1301.0, 105.0
    goal_learning.tick(tmp_path)
    rows = goal_learning.records(tmp_path)
    assert len(rows) == 4 and len({row["group"] for row in rows}) == 2
    assert all(row["messages"][-1]["content"] == "up" for row in rows)
    assert ledger.read_bytes() == original_roles


def test_single_product_universe_does_not_duplicate_same_tick(tmp_path, monkeypatch):
    financial_world(tmp_path)
    configure(tmp_path, {"products": ["ADA-USD"]})
    monkeypatch.setattr(
        goal_learning.shutil, "disk_usage", lambda root: SimpleNamespace(free=8 * 2**30)
    )
    monkeypatch.setattr(
        research_tools, "download_page", lambda url, max_bytes: (url, '{"data":{"amount":"100"}}')
    )
    receipt = goal_observer.tick(tmp_path)
    assert len(receipt["results"]) == 1
    assert goal_learning.status(tmp_path)["pending_total"] == 2
