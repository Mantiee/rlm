from pathlib import Path
from types import SimpleNamespace

import pytest

from rlm.v100 import sandbox_package


def test_private_apt_is_signed_and_does_not_load_host_hooks(tmp_path):
    keyring = tmp_path / "debian.gpg"
    keyring.write_bytes(b"fixture")
    root = tmp_path / "tools"
    env = sandbox_package.private_apt(root, "trixie", keyring, "marek")
    config = Path(env["APT_CONFIG"]).read_text()
    assert f'Dir::Etc::parts "{root}/etc/apt.conf.d";' in config
    assert f'Dir::Etc::main "{root}/etc/empty.conf";' in config
    assert f'Dir::State::status "{root}/state/status";' in config
    assert f'Dir::Log "{root}/log";' in config
    assert "trusted=yes" not in (root / "etc/sources.list").read_text()
    assert f"signed-by={keyring}" in (root / "etc/sources.list").read_text()
    assert "AllowUnauthenticated" not in config and "AllowInsecure" not in config
    assert (root / "state/status").read_text() == ""
    assert not list((root / "etc/apt.conf.d").iterdir())


def test_bubblewrap_download_only_uses_private_fresh_lists(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(
        sandbox_package.platform,
        "freedesktop_os_release",
        lambda: {"ID": "debian", "VERSION_CODENAME": "trixie"},
    )
    monkeypatch.setattr(sandbox_package, "private_apt", lambda *args: {"APT_CONFIG": "private"})

    def run(argv, **kwargs):
        commands.append((argv, kwargs))
        if argv[:2] == ["apt-get", "download"]:
            (tmp_path / "bubblewrap_0.11_amd64.deb").write_bytes(b"fixture")
        if argv[0] == "dpkg-deb":
            binary = tmp_path / "extracted/usr/bin/bwrap"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"fixture")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sandbox_package.subprocess, "run", run)
    assert sandbox_package.download_bubblewrap(tmp_path).name == "bwrap"
    assert [c[0][:2] for c in commands] == [
        ["apt-get", "update"],
        ["apt-get", "download"],
        ["dpkg-deb", "-x"],
    ]
    assert all(c[1]["env"] == {"APT_CONFIG": "private"} for c in commands[:2])


def test_private_apt_rejects_configuration_injection(tmp_path):
    keyring = tmp_path / "debian.gpg"
    keyring.touch()
    with pytest.raises(ValueError):
        sandbox_package.private_apt(tmp_path, "trixie;malicious", keyring, "marek")
