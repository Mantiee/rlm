import json

import pytest

from rlm.v100.release_selector import SCHEMA, current


def test_dedicated_update_cannot_change_controller_or_daily_use_pc_release(tmp_path):
    folder = tmp_path / "research/private-sync"
    folder.mkdir(parents=True)
    dedicated = {
        "schema": SCHEMA,
        "repository": "Mantiee/rlm",
        "mode": "pinned",
        "revision": "a" * 40,
    }
    (folder / "3060-release.json").write_text(json.dumps(dedicated))
    assert current(tmp_path, "192.168.0.9")["revision"] == "a" * 40
    for peer in (None, "192.168.0.61", "192.168.0.68", "127.0.0.1", "192.168.0.9.evil"):
        assert current(tmp_path, peer)["mode"] == "follow-published"
        assert not current(tmp_path, peer)["configured"]


def test_invalid_dedicated_pin_fails_closed_without_changing_global_release(tmp_path):
    folder = tmp_path / "research/private-sync"
    folder.mkdir(parents=True)
    (folder / "3060-release.json").write_text(
        json.dumps({"schema": SCHEMA, "repository": "other/repo", "mode": "pinned"})
    )
    with pytest.raises(ValueError):
        current(tmp_path, "192.168.0.9")
    assert current(tmp_path, "192.168.0.61")["mode"] == "follow-published"


def test_dedicated_pin_directory_is_not_silently_ignored(tmp_path):
    (tmp_path / "research/private-sync/3060-release.json").mkdir(parents=True)
    with pytest.raises(ValueError, match="Invalid private release"):
        current(tmp_path, "192.168.0.9")
