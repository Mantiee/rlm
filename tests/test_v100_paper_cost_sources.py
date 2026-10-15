import copy
import hashlib
import json

import pytest

from rlm.v100 import paper_cost_sources as sources

NOW = "2026-10-15T04:00:00+00:00"


def payload():
    return {
        "error": [],
        "result": {
            "BTC/USD": {
                "bids": [["99.0000000001", "1.250", 1792036799]],
                "asks": [["100.0000000001", "1.250", 1792036799]],
            }
        },
    }


def normalize(data=None, **kwargs):
    raw = json.dumps(payload() if data is None else data).encode()
    return sources.normalize_depth(
        raw,
        base="BTC",
        count=100,
        source_url=kwargs.pop("source_url", sources.depth_url("BTC")),
        source_sha256=kwargs.pop("source_sha256", hashlib.sha256(raw).hexdigest()),
        available_at=kwargs.pop("available_at", NOW),
        **kwargs,
    )


def test_preserve_decimal_strings_pair_and_row_timestamps():
    book = normalize()
    assert book["bids"] == [["99.0000000001", "1.250"]]
    assert book["quote_currency"] == "USD"
    assert book["bids_level_times"] == ["1792036799"]
    assert book["available_at"] == NOW


@pytest.mark.parametrize(
    "change",
    [
        "error",
        "pair",
        "many",
        "unsorted",
        "future",
        "crossed",
        "malformed",
        "float",
        "nan",
        "empty",
    ],
)
def test_malformed_or_mismatched_exchange_data(change):
    data = copy.deepcopy(payload())
    rows = data["result"]["BTC/USD"]
    if change == "error":
        data["error"] = ["EGeneral:fail"]
    elif change == "pair":
        data["result"] = {"ETH/USD": rows}
    elif change == "many":
        data["result"]["ETH/USD"] = rows
    elif change == "unsorted":
        rows["asks"].append(["99.5", "1", 1792036799])
    elif change == "future":
        rows["asks"][0][2] = 1792036801
    elif change == "crossed":
        rows["asks"][0][0] = "98"
    elif change == "malformed":
        rows["asks"][0] = "123"
    elif change == "float":
        rows["asks"][0][0] = 100.1
    elif change == "nan":
        rows["asks"][0][0] = "NaN"
    elif change == "empty":
        rows["asks"] = []
    with pytest.raises((ValueError, KeyError, TypeError)):
        normalize(data)


def test_binding_and_allowlist():
    with pytest.raises(ValueError, match="hash mismatch"):
        normalize(source_sha256="b" * 64)
    with pytest.raises(ValueError, match="URL"):
        normalize(source_url="https://api.kraken.com/0/public/Depth?pair=ETHUSD")
    with pytest.raises(ValueError):
        sources.depth_url("../BTC")
    with pytest.raises(ValueError):
        sources.depth_url("BTC", 500)


def test_collect_shadow_only_usd_estimate_pln_unknown_and_idempotent(tmp_path, monkeypatch):
    raw = json.dumps(payload()).encode()
    digest = hashlib.sha256(raw).hexdigest()

    def fetch(root, url):
        assert root == tmp_path / "research/paper-cost-shadow"
        assert url == sources.depth_url("BTC")
        path = root / "research/paper/sources" / f"{digest}.json"
        sources.immutable(path, raw)
        return payload(), digest, NOW

    monkeypatch.setattr(sources, "fetch_public", fetch)
    fee = {
        "source_url": "https://example.org/fees",
        "source_sha256": "b" * 64,
        "available_at": "2026-10-01T00:00:00+00:00",
        "currency": "USD",
        "taker_bps": "40",
        "minimum": "0",
        "valid_until": "2026-10-16T00:00:00+00:00",
    }
    usd = sources.collect(tmp_path, fee, reporting_currency="USD", decision_at=NOW)
    assert usd["receipt"]["state"] == "SHADOW_ESTIMATE"
    assert usd == sources.collect(tmp_path, fee, reporting_currency="USD", decision_at=NOW)
    pln = sources.collect(tmp_path, fee, decision_at=NOW)
    assert pln["receipt"]["state"] == "UNKNOWN"
    assert "FX UNKNOWN" in pln["receipt"]["reason"]
    assert not list(tmp_path.rglob("*.sqlite3"))
    assert not (tmp_path / "research/paper").exists()
    assert len(list((tmp_path / "research/paper-cost-shadow/receipts").glob("*.json"))) == 2
    assert not usd["receipt"]["execution_eligible"]


def test_immutable_collision_does_not_overwrite(tmp_path):
    path = tmp_path / "receipt.json"
    sources.immutable(path, b"original")
    with pytest.raises(ValueError):
        sources.immutable(path, b"changed")
    assert path.read_bytes() == b"original"
