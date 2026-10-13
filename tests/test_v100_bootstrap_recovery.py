import hashlib
import json
from pathlib import Path

import pytest

from rlm.v100 import managed_bootstrap


def test_bootstrap_entrypoint_is_replaced_after_all_dependencies(tmp_path, monkeypatch):
    folder = tmp_path / "verified"
    source = folder / "rlm/v100"
    source.mkdir(parents=True)
    for name in ("managed_bootstrap", "inference_benchmark", "release_selector"):
        (source / (name + ".py")).write_text("pass\n")
    replaced = []
    actual = managed_bootstrap.os.replace

    def replace(before, after):
        replaced.append(Path(after).name)
        actual(before, after)

    monkeypatch.setattr(managed_bootstrap.os, "replace", replace)
    managed_bootstrap.refresh_bootstrap(tmp_path, folder)
    assert replaced == ["inference_benchmark.py", "release_selector.py", "managed_bootstrap.py"]


def test_incomplete_bootstrap_bundle_does_not_replace_working_runner(tmp_path):
    folder = tmp_path / "verified/rlm/v100"
    folder.mkdir(parents=True)
    (folder / "managed_bootstrap.py").write_text("pass\n")
    bootstrap = tmp_path / "tools/managed-bootstrap"
    bootstrap.mkdir(parents=True)
    old = bootstrap / "managed_bootstrap.py"
    old.write_text("old entrypoint")
    with pytest.raises(FileNotFoundError, match="dependency"):
        managed_bootstrap.refresh_bootstrap(tmp_path, tmp_path / "verified")
    assert old.read_text() == "old entrypoint"


def missing_selector(tmp_path, monkeypatch, body):
    monkeypatch.setattr(managed_bootstrap, "__package__", None)
    monkeypatch.setattr(managed_bootstrap, "__file__", str(tmp_path / "managed_bootstrap.py"))

    def unavailable(name):
        raise ModuleNotFoundError("Missing selector", name="release_selector")

    monkeypatch.setattr(managed_bootstrap.importlib, "import_module", unavailable)
    monkeypatch.setattr(
        managed_bootstrap,
        "release",
        lambda: (
            "a" * 40,
            {
                "files": {
                    "rlm/v100/release_selector.py": hashlib.sha256(body).hexdigest(),
                }
            },
        ),
    )
    monkeypatch.setattr(managed_bootstrap, "fetch", lambda url: body)


def test_missing_standalone_selector_is_recovered_and_private_pin_is_honored(tmp_path, monkeypatch):
    body = (Path(__file__).parents[1] / "rlm/v100/release_selector.py").read_bytes()
    missing_selector(tmp_path, monkeypatch, body)
    control = tmp_path / "research/private-sync/approved-release.json"
    control.parent.mkdir(parents=True)
    control.write_text(
        json.dumps(
            {
                "schema": "synta-approved-release-v1",
                "repository": "Mantiee/rlm",
                "mode": "pinned",
                "revision": "b" * 40,
            }
        )
    )
    assert managed_bootstrap.bootstrap_selection(tmp_path) == "b" * 40
    assert (tmp_path / "release_selector.py").read_bytes() == body


def test_recovery_does_not_execute_a_checksum_mismatch(tmp_path, monkeypatch):
    missing_selector(tmp_path, monkeypatch, b"pass\n")
    monkeypatch.setattr(
        managed_bootstrap, "fetch", lambda url: b"raise RuntimeError('unverified code')\n"
    )
    with pytest.raises(ValueError, match="checksum"):
        managed_bootstrap.bootstrap_selection(tmp_path)
    assert not (tmp_path / "release_selector.py").exists()


def test_other_missing_dependency_is_reported_without_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(managed_bootstrap, "__package__", None)

    def unavailable(name):
        raise ModuleNotFoundError("Different dependency", name="unrelated_module")

    monkeypatch.setattr(managed_bootstrap.importlib, "import_module", unavailable)
    monkeypatch.setattr(
        managed_bootstrap, "release", lambda: pytest.fail("Unrelated import triggered download")
    )
    with pytest.raises(ModuleNotFoundError, match="Different"):
        managed_bootstrap.bootstrap_selection(tmp_path)
