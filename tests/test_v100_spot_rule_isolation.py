"""A changed exchange rule must block spot fills, not independent learning."""

import json

from rlm.v100 import (
    market_adapters,
    paper_learning,
    provider_registry,
    reward_policy,
    spot_bootstrap,
)
from rlm.v100.common import atomic_json


def test_changed_spot_rules_suspend_crypto_polls_without_killing_observer(tmp_path, monkeypatch):
    root = tmp_path
    ledger = root / "research/paper/ledger.sqlite3"
    ledger.parent.mkdir(parents=True)
    ledger.touch()
    settings = {
        "schema": "v100-paper-learning-v1",
        "objective": "Independent bounded learning",
        "crypto": True,
        "ciks": [],
        "sec_contact": "",
        "observer_interval": 30,
        "research_rounds": 1,
        "other_income_rnd": True,
    }
    settings_path = root / "settings.json"
    atomic_json(settings_path, settings)
    state = {
        "instruments": {
            "BTCUSD": {
                "market": "crypto",
                "product": "spot",
                "feed_id": "kraken:XBTUSD",
                "fee_profile": "fee",
            }
        },
        "fee_profiles": {"fee": {"valid_until": "2020-01-01T00:00:00+00:00"}},
    }

    class Book:
        def __init__(self, _root):
            pass

        def state(self):
            return state

        def close(self):
            pass

    monkeypatch.setattr(paper_learning, "PaperBook", Book)
    monkeypatch.setattr(provider_registry, "poll", lambda *args: None)
    monkeypatch.setattr(market_adapters, "poll", lambda *args: None)
    monkeypatch.setattr(reward_policy, "train", lambda *args: {})
    monkeypatch.setattr(paper_learning, "scheduled_reports", lambda *args: [])
    polls = []
    monkeypatch.setattr(paper_learning, "poll_crypto", lambda *args: polls.append(1))

    def changed(*args, **kwargs):
        raise ValueError(
            "Spot execution rules changed; retain positions and inspect before updating"
        )

    monkeypatch.setattr(spot_bootstrap, "prepare", changed)
    session = paper_learning.PaperLearning(root, settings_path)
    monkeypatch.setattr(session, "note", lambda *args: None)
    session.tick()
    session.check()
    assert polls == []
    status = json.loads(session.spot_rules_status_path.read_text())
    assert status["state"] == "blocked"
    session.tick()
    assert polls == []
    assert session.spot_rules_blocked

    resumed = paper_learning.PaperLearning(root, settings_path)
    monkeypatch.setattr(resumed, "note", lambda *args: None)
    assert resumed.spot_rules_blocked
    monkeypatch.setattr(spot_bootstrap, "prepare", lambda *args, **kwargs: {})
    resumed.tick()
    assert polls == [1]
    assert json.loads(resumed.spot_rules_status_path.read_text())["state"] == "cleared"
