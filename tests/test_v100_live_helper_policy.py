"""Deliberately disabled hardware is not an unavailable connection."""

import pytest
import requests

from rlm.v100 import common, competition, live_status, model_roles
from rlm.v100.common import atomic_json


@pytest.mark.parametrize("enabled", [False, True])
def test_disabled_origin_skips_probe_but_enabled_failure_remains_unavailable(
    tmp_path, monkeypatch, enabled
):
    origin = "http://192.168.0.61:11435"
    atomic_json(
        tmp_path / "research/user-preferences.json", {"remote_origin_enabled": {origin: enabled}}
    )
    monkeypatch.setattr(model_roles, "selected_profile", lambda *a: tmp_path / "profile.json")
    monkeypatch.setattr(
        common,
        "load_profile",
        lambda *a: {
            "runtime": {"backend": "ollama", "base_url": origin},
            "resources": {"device": "remote"},
        },
    )
    calls = []

    def failed_client(*args, **kwargs):
        calls.append(1)
        raise requests.ConnectionError("Actual enabled helper connection failure")

    monkeypatch.setattr(competition, "helper_client", failed_client)
    result = live_status.snapshot(tmp_path, {"running": True, "state": {}}, {}, {})
    card = next(row for row in result["actors"] if row["id"] == "rtx")
    assert card["state"] == ("unavailable" if enabled else "disabled")
    assert card["stale"] is enabled
    assert len(calls) == int(enabled)
    if not enabled:
        assert not card["jobs"] and "resource policy" in card["detail"]


def test_unconfigured_helper_is_not_reported_as_intentionally_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(model_roles, "selected_profile", lambda *a: tmp_path / "profile.json")
    monkeypatch.setattr(common, "load_profile", lambda *a: {"resources": {"device": "cpu"}})
    monkeypatch.setattr(competition, "helper_client", lambda *a: pytest.fail("Unconfigured probe"))
    result = live_status.snapshot(tmp_path, {"running": True, "state": {}}, {}, {})
    card = next(row for row in result["actors"] if row["id"] == "rtx")
    assert card["state"] == "unavailable" and card["stale"]
    assert "not configured" in card["detail"]
