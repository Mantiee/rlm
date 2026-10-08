"""Download a signed Debian sandbox package with private APT state, never install it."""

import os
import platform
import pwd
import re
import subprocess
from pathlib import Path


def private_apt(folder: Path, codename: str, keyring: Path, user: str) -> dict:
    if not re.fullmatch(r"[a-z]+", codename) or not keyring.is_file():
        raise ValueError("Need a Debian codename and the existing Debian archive keyring")
    folder = folder.resolve()
    for directory in (
        "etc/apt.conf.d",
        "etc/trusted.gpg.d",
        "state/lists/partial",
        "cache/archives/partial",
        "log",
    ):
        (folder / directory).mkdir(parents=True, exist_ok=True)
    (folder / "state/status").write_text("")
    (folder / "etc/sources.list").write_text(
        f"deb [signed-by={keyring}] https://deb.debian.org/debian {codename} main\n"
        f"deb [signed-by={keyring}] https://deb.debian.org/debian {codename}-updates main\n"
        f"deb [signed-by={keyring}] https://security.debian.org/debian-security {codename}-security main\n"
    )
    options = {
        "Dir::Etc": str(folder / "etc"),
        "Dir::Etc::parts": str(folder / "etc/apt.conf.d"),
        "Dir::Etc::main": str(folder / "etc/empty.conf"),
        "Dir::Etc::sourcelist": str(folder / "etc/sources.list"),
        "Dir::Etc::sourceparts": "-",
        "Dir::Etc::preferences": "-",
        "Dir::Etc::preferencesparts": "-",
        "Dir::Etc::trusted": str(folder / "etc/trusted.gpg"),
        "Dir::Etc::trustedparts": str(folder / "etc/trusted.gpg.d"),
        "Dir::State": str(folder / "state"),
        "Dir::State::status": str(folder / "state/status"),
        "Dir::Cache": str(folder / "cache"),
        "Dir::Cache::pkgcache": "",
        "Dir::Cache::srcpkgcache": "",
        "Dir::Log": str(folder / "log"),
        "APT::Sandbox::User": user,
        "APT::Update::Error-Mode": "any",
        "Acquire::Retries": "3",
        "Acquire::Languages": "none",
    }
    if any('"' in value or "\\" in value or "\n" in value for value in options.values()):
        raise ValueError("Invalid private APT configuration value")
    config = folder / "apt.conf"
    config.write_text("".join(f'{key} "{value}";\n' for key, value in options.items()))
    return {**os.environ, "APT_CONFIG": str(config)}


def download_bubblewrap(folder: Path) -> Path:
    release = platform.freedesktop_os_release()
    if release.get("ID") != "debian":
        raise ValueError("Private sandbox-package bootstrap supports Debian only")
    environment = private_apt(
        folder,
        release["VERSION_CODENAME"],
        Path("/usr/share/keyrings/debian-archive-keyring.gpg"),
        pwd.getpwuid(os.getuid()).pw_name,
    )
    print("Refreshing signed Debian package lists in the isolated tools folder...", flush=True)
    subprocess.run(["apt-get", "update"], cwd=folder, env=environment, check=True, timeout=300)
    subprocess.run(
        ["apt-get", "download", "bubblewrap"], cwd=folder, env=environment, check=True, timeout=120
    )
    packages = list(folder.glob("bubblewrap_*.deb"))
    if len(packages) != 1:
        raise ValueError("Expected one APT-verified sandbox package")
    subprocess.run(
        ["dpkg-deb", "-x", str(packages[0]), str(folder / "extracted")], check=True, timeout=60
    )
    binary = folder / "extracted/usr/bin/bwrap"
    if not binary.is_file():
        raise FileNotFoundError(binary)
    return binary
