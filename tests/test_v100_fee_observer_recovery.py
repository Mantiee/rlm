import json
from types import SimpleNamespace

import pytest

from rlm.v100 import paper_learning, spot_bootstrap
from rlm.v100.common import atomic_json
from tests.test_v100_paper_learning import settings


@pytest.fixture
def observer(tmp_path, monkeypatch):
    from rlm.v100 import market_adapters, provider_registry, reward_policy

    ledger = tmp_path / "research/paper/ledger.sqlite3"
    ledger.parent.mkdir(parents=True)
    ledger.touch()
    state = {
        "instruments": {
            "BTC": {
                "feed_id": "kraken:XBTUSD",
                "fee_profile": "old",
                "market": "crypto",
                "product": "spot",
            }
        },
        "fee_profiles": {"old": {"valid_until": "2020-01-01T00:00:00+00:00"}},
    }
    notes, quotes = [], []

    class Book:
        def __init__(self, root):
            pass

        def state(self):
            return state

        def note(self, branch, payload):
            notes.append(payload)

        def close(self):
            pass

    monkeypatch.setattr(paper_learning, "PaperBook", Book)
    monkeypatch.setattr(paper_learning, "poll_crypto", lambda *a: quotes.append("price"))
    monkeypatch.setattr(provider_registry, "poll", lambda *a: None)
    monkeypatch.setattr(market_adapters, "poll", lambda *a: None)
    monkeypatch.setattr(reward_policy, "train", lambda *a: {"state": "no labels"})
    monkeypatch.setattr(paper_learning, "scheduled_reports", lambda *a: [])
    now = [1791748000.0]
    monkeypatch.setattr(paper_learning, "time", SimpleNamespace(time=lambda: now[0]))
    config = tmp_path / "settings.json"
    atomic_json(config, {**settings(), "crypto": True})
    session = paper_learning.PaperLearning(tmp_path, config)
    return session, state, notes, quotes, now


def test_fee_source_failure_preserves_observations_and_backs_off(observer, monkeypatch):
    session, state, notes, quotes, now = observer
    calls = []

    def unavailable(*args, **kwargs):
        calls.append(1)
        raise spot_bootstrap.FeeSourceUnavailable("Archived unrecognized fee table")

    monkeypatch.setattr(spot_bootstrap, "prepare", unavailable)
    session.tick()
    session.tick()
    session.check()
    assert calls == [1] and quotes == ["price", "price"]
    assert state["fee_profiles"]["old"]["valid_until"] == "2020-01-01T00:00:00+00:00"
    assert session.failure is None and notes[0]["status"] == "fee-refresh-blocked"
    assert session.fee_retry_after == now[0] + 3600
    now[0] += 3601
    session.tick()
    assert len(calls) == 2


def test_non_fee_integrity_errors_are_not_downgraded(observer, monkeypatch):
    session, _, _, quotes, _ = observer

    def invalid(*a, **k):
        raise ValueError("Spot execution rules changed")

    monkeypatch.setattr(spot_bootstrap, "prepare", invalid)
    with pytest.raises(ValueError, match="rules changed"):
        session.tick()
    assert not quotes


def test_repaired_fee_source_is_retried_without_resetting_session(observer, monkeypatch):
    session, _, _, quotes, now = observer
    calls = []

    def refresh(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise spot_bootstrap.FeeSourceUnavailable("Temporary layout failure")
        return {"status": "paper spot sources configured"}

    monkeypatch.setattr(spot_bootstrap, "prepare", refresh)
    session.tick()
    now[0] += 3601
    session.tick()
    session.check()
    assert len(calls) == 2 and len(quotes) == 2 and session.failure is None


def test_public_fee_locale_and_redirect_receipt_are_exact(tmp_path, monkeypatch):
    state = {"currency": "PLN", "instruments": {}, "fee_profiles": {}}

    class Book:
        def __init__(self, root):
            pass

        def state(self):
            return state

        def close(self):
            pass

    monkeypatch.setattr(spot_bootstrap, "PaperBook", Book)
    seen = []

    def page(url, **kwargs):
        seen.append((url, kwargs))
        return "https://www.kraken.com/features/fee-schedule", "<p>Unrecognized publication</p>"

    monkeypatch.setattr(spot_bootstrap, "download_page", page)
    with pytest.raises(spot_bootstrap.FeeSourceUnavailable, match="archived source"):
        spot_bootstrap.prepare(tmp_path, refresh=True)
    assert seen[0][1]["language"] == "en-US"
    receipt = json.loads((tmp_path / "research/paper/fee-source-status.json").read_text())
    assert receipt["source_url"] == "https://www.kraken.com/features/fee-schedule"
    assert receipt["state"] == "blocked"
