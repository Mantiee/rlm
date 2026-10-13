"""Explicit October 2026 candidates, independent of downloaded or admitted state."""

import json
from pathlib import Path

WINDOWS_ROLES = (
    ("fast", "gemma4:e2b-it-q4_K_M"),
    ("fast", "gemma4:e4b-it-q4_K_M"),
    ("planner", "qwen3.8:27b-q4_K_M"),
    ("planner", "gemma4:26b-a4b-it-q4_K_M"),
)
NODE_MODELS = ("gemma4:e2b-it-q4_K_M", "gemma4:e4b-it-q4_K_M")


def retired(name: str) -> bool:
    return name.startswith(("gpt-oss:", "qwen3.5:", "qwen3.6:"))


def metadata_contract(name: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if name.startswith("gemma4:"):
        return ("gemma4",), ("gemma4",)
    if name.startswith("qwen3.8:"):
        return ("qwen35", "qwen38"), ("gpt2",)
    return ("qwen35",), ("gpt2",)


def admitted_panel(result: dict) -> bool:
    """Task correctness gate, not a claim of superiority or profitability."""
    return result.get("passed") == 20 and result.get("critical_passed") is True


def deployment(root: Path) -> list[dict]:
    """Read receipts only; configured or qualified never means currently serving."""
    rows = []
    for host, relative in (
        ("Windows RTX 3090", "research/model-roles/active.json"),
        ("Debian RTX 3060 Ti", "research/owned-node/status.json"),
        ("Debian V100 candidates", "research/managed-update/preparation.json"),
    ):
        path = root / relative
        try:
            if path.stat().st_size > 2 * 2**20:
                raise ValueError("Deployment receipt too large")
            value = json.loads(path.read_text())
            if host == "Windows RTX 3090":
                if not value.get("roles"):
                    rows.append(
                        {
                            "host": host,
                            "role": "fast / planner",
                            "model": "none admitted",
                            "state": "qualification pending or blocked",
                            "receipt": str(path),
                        }
                    )
                for role, entry in value.get("roles", {}).items():
                    profile_path = Path(entry["profile"]).resolve()
                    if not profile_path.is_relative_to((root / "research/model-roles").resolve()):
                        raise ValueError("Role profile escaped its owned folder")
                    name = json.loads(profile_path.read_text())["runtime"]["model_name"]
                    rows.append(
                        {
                            "host": host,
                            "role": role,
                            "model": name,
                            "state": "retired"
                            if retired(name)
                            else "metadata compatibility blocked; historical admission retained"
                            if role in value.get("unresolved_compatibility", [])
                            else "qualified, current serving unverified",
                            "receipt": str(path),
                        }
                    )
            elif host == "Debian RTX 3060 Ti":
                rows.append(
                    {
                        "host": host,
                        "role": "fast",
                        "model": value.get("model", "unknown"),
                        "state": "retired, not routed"
                        if retired(value.get("model", ""))
                        else value.get("phase", "unknown"),
                        "receipt": str(path),
                    }
                )
            else:
                for entry in value.get("assets", []):
                    rows.append(
                        {
                            "host": host,
                            "role": entry["role"],
                            "model": entry.get("repo", "unknown"),
                            "state": entry.get("state", "unknown"),
                            "receipt": str(path),
                        }
                    )
        except (OSError, ValueError, KeyError, TypeError) as error:
            rows.append(
                {
                    "host": host,
                    "role": "unknown",
                    "model": "unknown",
                    "state": "receipt unavailable: " + str(error)[:160],
                    "receipt": str(path),
                }
            )
    if not rows:
        rows.append(
            {
                "host": "All hosts",
                "role": "unknown",
                "model": "unknown",
                "state": "No admission receipts",
            }
        )
    return rows
