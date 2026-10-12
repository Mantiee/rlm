"""Release acquisition must include serving code before stopping the helper."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def bootstrap(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "rlm/v100"))
    spec = importlib.util.spec_from_file_location(
        "standalone_bootstrap", ROOT / "rlm/v100/managed_bootstrap.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_proxy_is_declared_and_downloaded(tmp_path, monkeypatch):
    manifest = json.loads((ROOT / "tools/managed-release.json").read_text())
    name = "rlm/v100/windows_resource_proxy.py"
    content = (ROOT / name).read_bytes()
    assert manifest["files"][name] == hashlib.sha256(content).hexdigest()
    module = bootstrap(monkeypatch)
    monkeypatch.setattr(module, "fetch", lambda url: content)
    module.materialize(tmp_path, "a" * 40, {"files": {name: manifest["files"][name]}})
    assert (tmp_path / name).read_bytes() == content


def test_corrupt_proxy_never_marked_complete(tmp_path, monkeypatch):
    module = bootstrap(monkeypatch)
    monkeypatch.setattr(module, "fetch", lambda url: b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.materialize(
            tmp_path, "a" * 40, {"files": {"rlm/v100/windows_resource_proxy.py": "0" * 64}}
        )
    assert not (tmp_path / "release.json").exists()


def test_validation_precedes_every_disruptive_operation():
    script = (ROOT / "tools/refresh-synta-windows.ps1").read_text()
    validation = script.index("Get-FileHash")
    for action in ("/api/synta-maintenance", "Stop-ScheduledTask", "Stop-Process"):
        assert validation < script.index(action)
