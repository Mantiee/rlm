import json
import os
import tomllib
from pathlib import Path
from typing import Any


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def load_profile(path: Path, root: Path) -> dict[str, Any]:
    profile = tomllib.loads(path.read_text())
    runtime, server = profile["runtime"], profile["server"]
    if runtime["context_window"] != server["context_per_slot"]:
        raise ValueError("Client context_window must equal context_per_slot")
    if server["slots"] < 1 or server["context_per_slot"] < 512:
        raise ValueError("Invalid server context or slots")
    if (
        profile["memory"]["chunk_tokens"] * profile["memory"]["fanout"] + 1024
        > runtime["context_window"]
    ):
        raise ValueError("Summary fanout cannot fit the configured working context")
    if (
        profile["memory"]["retrieve_count"] * profile["memory"]["chunk_tokens"] + 1024
        > runtime["context_window"]
    ):
        raise ValueError("Retrieved chunks cannot fit the configured working context")
    for section, names in (
        ("server", ("binary", "model", "draft_model")),
        ("memory", ("database",)),
        ("training", ("base_model", "output", "init_adapter")),
    ):
        for name in names:
            raw = profile[section][name]
            if raw:
                path_value = Path(raw).expanduser()
                profile[section][name] = str(
                    path_value if path_value.is_absolute() else root / path_value
                )
    return profile
