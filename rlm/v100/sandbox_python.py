"""Read-only interpreter mounts, including uv venv-to-venv executable links."""

import os
import sys
from pathlib import Path


def runtime_mounts(python: Path | None = None) -> list[Path]:
    executable = python or Path(sys.executable)
    if not executable.is_absolute():
        raise ValueError("Sandbox interpreter must use an absolute path")
    prefixes = [
        Path("/usr"),
        Path("/lib"),
        Path("/lib64"),
        Path(sys.base_prefix),
        executable.parent.parent if python else Path(sys.prefix),
    ]
    current = executable
    seen = set()
    while current.is_symlink():
        if current in seen or len(seen) >= 40:
            raise ValueError("Sandbox interpreter symlink chain is cyclic or too long")
        seen.add(current)
        target = Path(os.readlink(current))
        current = Path(os.path.abspath(target if target.is_absolute() else current.parent / target))
        # Bind only the interpreter's bin directories, never the whole home or
        # project root just to resolve an intermediate train-venv Python link.
        prefixes.append(current.parent)
    if not current.is_file():
        raise ValueError("Sandbox interpreter symlink target is missing")
    return [path for path in dict.fromkeys(prefixes) if path.exists()]
