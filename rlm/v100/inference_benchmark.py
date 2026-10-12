"""Same-model, bounded native inference experiments for Linux and Windows.

Standalone execution uses only stdlib; Windows pressure monitoring needs the
already installed psutil. Results do not certify task quality or profitability.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

PROMPTS = (
    "Return only the integer: 19 * 23.",
    "Return only valid JSON with keys action and amount: action is hold, amount is zero.",
    "Repeat exactly: capital costs execution evidence capital costs execution evidence.",
    "Rewrite this exact Python function with a different argument name and no explanation: def twice(x): return x+x",
)


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".new")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def variants(profile: dict, help_text: str) -> list[dict]:
    base = copy.deepcopy(profile["server"])
    base["spec_type"], base["draft_model"] = "none", ""
    values = [{"name": "baseline", "server": base}]
    if (
        "--spec-type" in help_text
        and "ngram-simple" in help_text
        and "--spec-draft-n-max" in help_text
    ):
        values.append(
            {"name": "ngram", "server": {**base, "spec_type": "ngram-simple", "draft_tokens": 4}}
        )
    if base.get("flash_attention") == "on" and base.get("cache_type") != "q8_0":
        values.append({"name": "q8-cache", "server": {**base, "cache_type": "q8_0"}})
    if len(values) > 2:
        values.append(
            {
                "name": "ngram-q8-cache",
                "server": {**values[-1]["server"], "spec_type": "ngram-simple", "draft_tokens": 4},
            }
        )
    # CPU layer offload is tested, not assumed faster. Context/model stay fixed.
    values.append({"name": "cpu-offload", "server": {**base, "gpu_layers": 16}})
    return values


def command(server: dict, port: int) -> list[str]:
    args = [
        server["binary"],
        "--model",
        server["model"],
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--alias",
        "synta-perf",
        "--gpu-layers",
        str(server.get("gpu_layers", 999)),
        "--ctx-size",
        str(server["context_per_slot"] * server["slots"]),
        "--parallel",
        str(server["slots"]),
        "--threads",
        str(server["threads"]),
        "--threads-batch",
        str(server["threads"]),
        "--batch-size",
        str(server["batch_size"]),
        "--ubatch-size",
        str(server["ubatch_size"]),
        "--flash-attn",
        server["flash_attention"],
        "--cache-type-k",
        server["cache_type"],
        "--cache-type-v",
        server["cache_type"],
        "--no-context-shift",
    ]
    if server.get("jinja") is True:
        args.append("--jinja")
    if server.get("spec_type", "none") != "none":
        args += [
            "--spec-type",
            server["spec_type"],
            "--spec-draft-n-max",
            str(server["draft_tokens"]),
        ]
    return args


def request(port: int, path: str, data: dict | None = None, timeout: float = 120) -> dict:
    body = json.dumps(data).encode() if data is not None else None
    value = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(value, timeout=timeout) as reply:
        raw = reply.read().decode("utf-8")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"Native HTTP {path} returned non-JSON: {raw[:240]!r}") from error


def measure(port: int, interrupted: threading.Event, evidence: Path | None = None) -> dict:
    rows = []
    for prompt in PROMPTS:
        for repeat in range(2):
            if interrupted.is_set():
                raise RuntimeError("Host pressure interrupted the owned experiment")
            started = time.monotonic()
            format_options = {}
            if prompt == PROMPTS[1]:
                format_options["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "hold_instruction",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "enum": ["hold"]},
                                "amount": {"type": "integer", "enum": [0]},
                            },
                            "required": ["action", "amount"],
                            "additionalProperties": False,
                        },
                    },
                }
            reply = request(
                port,
                "/v1/chat/completions",
                {
                    **format_options,
                    "model": "synta-perf",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "seed": 42,
                    "max_tokens": 128,
                    "stream": False,
                    "cache_prompt": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            elapsed = time.monotonic() - started
            if evidence is not None:
                with evidence.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "prompt": prompt,
                                "repeat": repeat,
                                "seconds": elapsed,
                                "response": reply,
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
            choice = reply["choices"][0]
            tokens = reply.get("usage", {}).get("completion_tokens")
            if type(tokens) is not int or tokens <= 0 or not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("Missing measured completion token count")
            if choice.get("finish_reason") not in ("stop", "eos"):
                raise ValueError("Probe output was incomplete; throughput cannot admit it")
            content = choice["message"].get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Probe output has no user-visible content")
            if prompt == PROMPTS[0] and content.strip() != "437":
                raise ValueError("Deterministic arithmetic correctness probe failed")
            if prompt == PROMPTS[1]:
                try:
                    structured = json.loads(content)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Structured probe model content was not JSON: {content[:240]!r}; inspect the probe receipt"
                    ) from error
                if (
                    structured != {"action": "hold", "amount": 0}
                    or type(structured.get("amount")) is not int
                ):
                    raise ValueError("Structured instruction correctness probe failed")
            rows.append(
                {
                    "repeat": repeat,
                    "seconds": elapsed,
                    "tokens": tokens,
                    "output": choice["message"],
                    "timings": reply.get("timings", {}),
                }
            )
    if interrupted.is_set():
        raise RuntimeError("Experiment interrupted before acceptance")
    return {
        "probes": rows,
        "seconds": sum(r["seconds"] for r in rows),
        "tokens_per_second": sum(r["tokens"] for r in rows) / sum(r["seconds"] for r in rows),
    }


def select(rows: list[dict]) -> dict:
    baseline = next((r for r in rows if r["name"] == "baseline" and r["state"] == "measured"), None)
    if baseline is None:
        return {"state": "blocked", "reason": "No complete baseline", "winner": None}
    outputs = [r["output"] for r in baseline["measurement"]["probes"]]
    admitted = []
    for row in rows:
        if row["state"] != "measured":
            continue
        row["same_probe_outputs"] = [r["output"] for r in row["measurement"]["probes"]] == outputs
        row["speedup"] = baseline["measurement"]["seconds"] / row["measurement"]["seconds"]
        if row["same_probe_outputs"] and row["speedup"] >= 1.1:
            admitted.append(row)
    winner = max(admitted, key=lambda r: r["speedup"]) if admitted else baseline
    return {
        "state": "measured",
        "winner": winner["name"],
        "server": winner["server"],
        "speedup": winner.get("speedup", 1),
        "scope": "Same outputs on eight bounded deterministic probes; not broad quality, financial skill or future speed guarantee. Context/model fixed. No production service changed.",
    }


def pressure_reason(preflight: bool = False) -> str | None:
    telemetry = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,memory.free,power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=3,
    )
    temperature, free, watts = map(float, telemetry.stdout.splitlines()[0].split(","))
    if not all(math.isfinite(value) for value in (temperature, free, watts)):
        return "GPU resource sensors are not finite"
    minimum = 12288 if preflight and os.name == "nt" else 3072
    if temperature >= 75:
        return f"GPU temperature {temperature:.1f} C exceeds admission threshold 75 C"
    if free < minimum:
        return f"Free VRAM {free:.0f} MiB is below admission requirement {minimum} MiB"
    if watts > 250:
        return f"GPU board draw {watts:.1f} W exceeds admission threshold 250 W"
    if os.name != "nt":
        return None
    import psutil

    helper_status = Path.home() / "ai-v100-helper/logs/resource-status.json"
    if helper_status.is_file() and time.time() - helper_status.stat().st_mtime < 10:
        if json.loads(helper_status.read_text(encoding="utf-8-sig")).get("generating"):
            return "Existing Synta helper request has priority"

    games = {
        "cs2.exe",
        "cyberpunk2077.exe",
        "fortniteclient-win64-shipping.exe",
        "valorant-win64-shipping.exe",
        "eldenring.exe",
        "gta5.exe",
    }
    names = {str(p.info.get("name", "")).casefold() for p in psutil.process_iter(["name"])}
    if (
        psutil.cpu_percent(interval=0.2) > 65
        or psutil.virtual_memory().available < 6 * 2**30
        or games & names
    ):
        return "Windows host CPU, RAM or game has priority"
    return None


def pressure_stop(process, interrupted: threading.Event, finished: threading.Event) -> None:
    while not finished.wait(2):
        try:
            reason = pressure_reason()
        except Exception:
            reason = "Resource telemetry unavailable"
        if reason:
            interrupted.set()
            if process.poll() is None:
                process.terminate()
            return


def wait_admission(seconds: int = 60) -> str | None:
    deadline = time.monotonic() + seconds
    reported = None
    while True:
        reason = pressure_reason(preflight=True)
        if reason is None:
            return None
        if reason != reported:
            print(
                json.dumps(
                    {
                        "phase": "between-variants-cooldown",
                        "reason": reason,
                        "scope": "No next native process started; existing host work has priority",
                    }
                ),
                flush=True,
            )
            reported = reason
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return reason
        time.sleep(min(2, remaining))


def run(profile: dict, output: Path, port: int = 8097) -> dict:
    if output.exists():
        raise FileExistsError("Use a new experiment directory")
    with socket.socket() as probe:
        if os.name != "nt":
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
        probe.listen(1)
    binary = Path(profile["server"]["binary"])
    model = Path(profile["server"]["model"])
    if not binary.is_file() or not model.is_file():
        raise FileNotFoundError("Pinned native binary/model unavailable")
    if profile["server"].get("draft_model"):
        raise ValueError(
            "Benchmark of an existing draft model needs a separate comparison; no silent removal"
        )
    reason = pressure_reason(preflight=True)
    if reason:
        raise RuntimeError(reason + "; no native benchmark process started")
    if os.name != "nt":
        processes = subprocess.check_output(["ps", "-eo", "args="], text=True)
        if any(
            "rlm.v100.cli" in line
            and any(
                " " + mode + " " in line + " "
                for mode in (
                    "train",
                    "mission-loop",
                    "mission-prepare",
                    "optimize-mtp",
                    "calibrate-training",
                )
            )
            for line in processes.splitlines()
        ):
            raise RuntimeError("Stop the mission only after its exclusive experiment finishes")
    help_reply = subprocess.run(
        [str(binary), "--help"], capture_output=True, text=True, timeout=30, check=True
    )
    help_text = help_reply.stdout + help_reply.stderr
    output.mkdir(parents=True)
    with model.open("rb") as source:
        model_hash = hashlib.file_digest(source, "sha256").hexdigest()
    if profile["server"].get("model_sha256") not in (None, model_hash):
        raise ValueError("Local GGUF checksum differs from the pinned model manifest")
    supported = variants(profile, help_text)
    rows = (
        []
        if any(row["name"] == "ngram" for row in supported)
        else [
            {
                "name": "ngram",
                "state": "blocked",
                "error": "Installed native help does not advertise ngram-simple and its draft flags",
            }
        ]
    )
    for variant in supported:
        reason = wait_admission()
        if reason:
            rows.append(
                {**variant, "state": "blocked", "error": reason + "; next variant not started"}
            )
            selection = select(rows)
            selection["benchmark_complete"] = False
            save(
                output / "results.json",
                {"variants": rows, "selection": selection, "model_sha256": model_hash},
            )
            return json.loads((output / "results.json").read_text())
        record = {**variant, "state": "starting"}
        save(output / "progress.json", record)
        environment = {
            key: value for key, value in os.environ.items() if not key.startswith("LLAMA_ARG_")
        }
        if profile["server"].get("library_path"):
            environment["LD_LIBRARY_PATH"] = profile["server"]["library_path"]
        interrupted, finished = threading.Event(), threading.Event()
        with (output / (variant["name"] + ".log")).open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command(variant["server"], port),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
            )
            if os.name == "nt":
                import psutil

                try:
                    psutil.Process(process.pid).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                except psutil.Error:
                    process.terminate()
                    process.wait(timeout=15)
                    raise
            monitor = threading.Thread(
                target=pressure_stop, args=(process, interrupted, finished), daemon=True
            )
            monitor.start()
            try:
                deadline = time.monotonic() + 180
                while time.monotonic() < deadline and process.poll() is None:
                    try:
                        if request(port, "/health", timeout=2).get("status") == "ok":
                            break
                    except (urllib.error.URLError, TimeoutError):
                        pass
                    time.sleep(0.5)
                else:
                    raise RuntimeError("Native server did not become ready; inspect variant log")
                record.update(
                    state="measured",
                    measurement=measure(
                        port, interrupted, output / (variant["name"] + "-probes.jsonl")
                    ),
                )
            except (ValueError, RuntimeError, OSError) as error:
                record.update(state="blocked", error=str(error))
            finally:
                finished.set()
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                monitor.join(timeout=5)
        rows.append(record)
        selection = select(rows)
        selection["benchmark_complete"] = len(
            [row for row in rows if row["state"] == "measured"]
        ) == len(supported)
        save(
            output / "results.json",
            {"variants": rows, "selection": selection, "model_sha256": model_hash},
        )
        print(
            json.dumps(
                {"variant": variant["name"], "state": record["state"], "error": record.get("error")}
            ),
            flush=True,
        )
        if interrupted.is_set():
            break
    return json.loads((output / "results.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8097)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        raise ValueError("Use an unprivileged isolated port")
    result = run(json.loads(args.profile.read_text(encoding="utf-8-sig")), args.output, args.port)
    print(json.dumps(result["selection"], indent=2))


if __name__ == "__main__":
    main()
