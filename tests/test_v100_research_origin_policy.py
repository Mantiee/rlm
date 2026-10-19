"""Optional research helpers respect per-origin policy before any server/client work."""

import pytest

from rlm.v100 import competition
from rlm.v100.common import atomic_json
from rlm.v100.remote_helper import WINDOWS_3090_ORIGIN


def disabled_origin(tmp_path, monkeypatch):
    profile = {"resources": {"device": "remote"}, "runtime": {"base_url": WINDOWS_3090_ORIGIN}}
    monkeypatch.setattr(competition, "load_profile", lambda *args: profile)
    monkeypatch.setattr(
        competition,
        "managed_server",
        lambda *args, **kwargs: pytest.fail("disabled origin server/client attempted"),
    )
    atomic_json(
        tmp_path / "research/user-preferences.json",
        {"remote_helper_enabled": True, "remote_origin_enabled": {WINDOWS_3090_ORIGIN: False}},
    )


def test_disabled_origin_research_uses_existing_accepted_master_without_network(
    tmp_path, monkeypatch
):
    disabled_origin(tmp_path, monkeypatch)
    fallback = {"runtime": {"model_version": "accepted-local-master"}}
    with competition.waiting_researcher(
        tmp_path / "remote.json", tmp_path, tmp_path / "log", fallback=fallback
    ) as actual:
        assert actual is fallback
    assert not (tmp_path / "log").exists()


def test_disabled_origin_without_accepted_fallback_fails_before_network(tmp_path, monkeypatch):
    disabled_origin(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="origin policy; no accepted serving master"):
        with competition.waiting_researcher(tmp_path / "remote.json", tmp_path, tmp_path / "log"):
            pytest.fail("disabled origin entered")
    assert not (tmp_path / "log").exists()
