"""Explicit, comparable inference conditions for serving and quality evaluation."""

import copy
import json
import math
from pathlib import Path


def thinking_enabled(profile: dict, override: bool | None = None) -> bool:
    value = profile["runtime"].get("enable_thinking", False) if override is None else override
    if type(value) is not bool:
        raise ValueError("enable_thinking must be boolean")
    return value


def sampling_settings(profile: dict, max_tokens: int | None = None) -> dict:
    runtime = profile["runtime"]
    temperature = runtime.get("temperature", 0.0)
    if (
        type(temperature) not in (int, float)
        or not math.isfinite(temperature)
        or not 0 <= temperature <= 2
    ):
        raise ValueError("temperature must be finite and between zero and two")
    seed = runtime.get("seed", 42)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be a nonnegative 32-bit integer")
    maximum = runtime["max_output_tokens"] if max_tokens is None else max_tokens
    if type(maximum) is not int or not 1 <= maximum < runtime["context_window"]:
        raise ValueError("Output budget must be positive and below the context window")
    result = {"temperature": float(temperature), "seed": seed, "max_tokens": maximum}
    if "top_p" in runtime:
        value = runtime["top_p"]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError("top_p must be finite and in (0, 1]")
        result["top_p"] = float(value)
    if "top_k" in runtime:
        value = runtime["top_k"]
        if type(value) is not int or not 1 <= value <= 1000:
            raise ValueError("top_k must be an integer between 1 and 1000")
        result["top_k"] = value
    return result


def generation_conditions(profile: dict) -> dict:
    result = sampling_settings(profile)
    result.update(
        context_window=profile["runtime"]["context_window"],
        thinking=thinking_enabled(profile),
    )
    if "tool_protocol" in profile["runtime"]:
        result["tool_protocol"] = profile["runtime"]["tool_protocol"]
    return result


def assert_client_conditions(client, profile: dict) -> None:
    if getattr(client, "tool_protocol", "native") != profile["runtime"].get(
        "tool_protocol", "native"
    ):
        raise ValueError("Quality client tool protocol differs from its recorded profile")
    intended = sampling_settings(profile)
    if hasattr(client, "sampling_args"):
        actual = {"temperature": 0.0, "max_tokens": 512, "seed": 42, **client.sampling_args}
        if any(actual.get(key) != value for key, value in intended.items()) or any(
            key in actual and key not in intended for key in ("top_p", "top_k")
        ):
            raise ValueError("Quality client sampling differs from its recorded profile")
    if hasattr(client, "enable_thinking"):
        if client.enable_thinking is None or client.enable_thinking != thinking_enabled(profile):
            raise ValueError("Quality client thinking mode differs from its recorded profile")
    if (
        hasattr(client, "context_window")
        and client.context_window != profile["runtime"]["context_window"]
    ):
        raise ValueError("Quality client context differs from its recorded profile")


def prepare_thinking(profile: dict, root: Path) -> Path:
    chosen = copy.deepcopy(profile)
    chosen["runtime"].update(
        enable_thinking=True,
        max_output_tokens=2048,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        seed=42,
    )
    if chosen["runtime"]["context_window"] < 4096:
        raise ValueError("Thinking profile requires at least 4096 tokens of context")
    sampling_settings(chosen)
    destination = root / "research/v100-thinking.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(chosen, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if destination.exists():
        if json.loads(destination.read_text()) != chosen:
            raise FileExistsError("Existing thinking profile differs; it was not overwritten")
    else:
        with destination.open("x") as file:
            file.write(content)
    return destination
