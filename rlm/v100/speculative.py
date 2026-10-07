"""Pinned Gemma MTP preparation, without changing the working server or weights."""

import hashlib
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

import requests

from rlm.v100.common import atomic_json, load_profile

ASSISTANT_MODEL = "google/gemma-4-12B-it-assistant"
ASSISTANT_REVISION = "46d4c6f13f0ac0ad827b915669b8df9b81c64c51"
DEFAULT_DRAFT_TOKENS = (2, 4, 8, 16)


def file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def benchmark_prompts() -> list[tuple[str, str]]:
    # Fixed public speed workloads, never a hidden quality evaluation or training data.
    context = "\n".join(
        f"Rekord {i}: magazyn przechowuje {i % 17 + 3} skrzynek, a inspekcja odbywa się w środę."
        for i in range(120)
    )
    return [
        ("short", "Wyjaśnij po polsku działanie pamięci komputera w 300 słowach."),
        (
            "code",
            "Napisz w Pythonie funkcję łączącą nakładające się przedziały liczbowe. "
            "Dodaj przykłady dla pustej listy, przedziałów rozłącznych i zagnieżdżonych. Wyjaśnij złożoność.",
        ),
        (
            "long",
            f"Dane:\n{context}\n\nNa podstawie danych napisz około 300 słów po polsku: "
            "opisz wzorzec liczby skrzynek, dzień inspekcji i sposób sprawdzenia spójności rekordów. "
            "Nie wymyślaj dodatkowych faktów.",
        ),
    ]


def mtp_profile_text(source: str, draft_model: Path | None, tokens: int) -> str:
    if type(tokens) is not int or not 1 <= tokens <= 16:
        raise ValueError("draft_tokens must be between 1 and 16")
    # Preserve the user's comments, settings and formatting. Change only isolated experiment fields.
    lines = source.splitlines(keepends=True)
    section = ""
    found_spec_type = False
    for index, line in enumerate(lines):
        match = re.match(r"\s*\[([^]]+)\]", line)
        if match:
            section = match[1]
        key = line.split("=", 1)[0].strip()
        if section == "runtime" and key == "base_url":
            lines[index] = 'base_url = "http://127.0.0.1:8089"\n'
        if section == "server":
            if key == "draft_model":
                lines[index] = (
                    f"draft_model = {json.dumps(str(draft_model) if draft_model else '')}\n"
                )
            elif key == "draft_tokens":
                lines[index] = f"draft_tokens = {tokens}\n"
            elif key == "spec_type":
                lines[index] = 'spec_type = "draft-mtp"\n'
                found_spec_type = True
    if not found_spec_type:
        for index, line in enumerate(lines):
            if line.startswith("draft_model ="):
                lines.insert(index + 1, 'spec_type = "draft-mtp"\n')
                break
    return "".join(lines)


def validate_gguf(path: Path) -> None:
    with path.open("rb") as handle:
        if handle.read(4) != b"GGUF" or path.stat().st_size < 1024:
            raise ValueError(f"Invalid or incomplete GGUF: {path}")


def compare_reports(baseline: dict, candidate: dict) -> dict:
    for key in ("enable_thinking", "suite", "prompts_sha256"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"Benchmarks differ in {key}")
    for section, keys in (
        ("runtime", ("model_version", "context_window", "max_output_tokens")),
        (
            "server",
            (
                "binary",
                "model",
                "context_per_slot",
                "slots",
                "batch_size",
                "ubatch_size",
                "cache_type",
                "flash_attention",
                "threads",
            ),
        ),
    ):
        if any(
            baseline["profile"][section][key] != candidate["profile"][section][key] for key in keys
        ):
            raise ValueError(f"Target/runtime settings differ in {section}")
    before = {(r["case"], r["repeat"]): r for r in baseline["runs"]}
    after = {(r["case"], r["repeat"]): r for r in candidate["runs"]}
    if not before or before.keys() != after.keys():
        raise ValueError("Benchmark workloads/repeats differ")
    drafted = sum(r["timings"].get("draft_n", 0) for r in after.values())
    accepted = sum(r["timings"].get("draft_n_accepted", 0) for r in after.values())
    if not drafted:
        raise ValueError("No draft tokens reported: MTP activation has not been confirmed")
    cases = {}
    for name in baseline["cases"]:
        old = baseline["cases"][name]
        new = candidate["cases"][name]
        cases[name] = {
            "baseline_tokens_per_second": old["median_tokens_per_second"],
            "candidate_tokens_per_second": new["median_tokens_per_second"],
            "throughput_ratio": new["median_tokens_per_second"] / old["median_tokens_per_second"],
            "latency_ratio": old["median_seconds"] / new["median_seconds"],
        }
    return {
        "cases": cases,
        "draft_acceptance": accepted / drafted,
        "drafted_tokens": drafted,
        "accepted_tokens": accepted,
        "answers_exact_match": all(before[key]["answer"] == after[key]["answer"] for key in before),
        "median_case_throughput_ratio": statistics.median(
            c["throughput_ratio"] for c in cases.values()
        ),
        "quality_evaluated": False,
        "promote_automatically": False,
    }


def recommend_mtp(comparison: dict) -> dict:
    # Conservative speed candidate only; fixed text equality is not a quality suite.
    eligible = {
        name: result
        for name, result in comparison.items()
        if result["answers_exact_match"]
        and result["median_case_throughput_ratio"] >= 1.05
        and all(c["latency_ratio"] >= 0.95 for c in result["cases"].values())
    }
    selected = (
        max(eligible, key=lambda name: eligible[name]["median_case_throughput_ratio"])
        if eligible
        else "baseline"
    )
    return {
        "recommended_speed_profile": selected,
        "quality_evaluated": False,
        "promote_automatically": False,
        "requires": "Independent quality gate before serving; repeat sweep after target weights, draft or runtime changes",
    }


def test_mtp(
    root: Path, repeats: int, draft_tokens: tuple[int, ...] = DEFAULT_DRAFT_TOKENS
) -> None:
    from rlm.v100.cli import server_command

    if repeats < 1:
        raise ValueError("repeats must be positive")
    if (
        not draft_tokens
        or len(set(draft_tokens)) != len(draft_tokens)
        or any(type(n) is not int or not 1 <= n <= 16 for n in draft_tokens)
    ):
        raise ValueError("Choose distinct draft token counts between 1 and 16")
    names = ("baseline", *(f"mtp{n}" for n in draft_tokens))
    profiles = {name: root / f"research/v100-{name}.toml" for name in names}
    loaded = {name: load_profile(path, root) for name, path in profiles.items()}
    for name, profile in loaded.items():
        if profile["runtime"]["base_url"] != "http://127.0.0.1:8089":
            raise ValueError("MTP experiment runner requires loopback port 8089")
        if (name == "baseline") != (not profile["server"]["draft_model"]):
            raise ValueError("Baseline must have no draft, MTP profiles must have a draft")
        if name != "baseline" and profile["server"].get("spec_type") != "draft-mtp":
            raise ValueError("MTP experiment profile must specify draft-mtp")
        if name != "baseline" and profile["server"]["draft_tokens"] != int(name[3:]):
            raise ValueError("MTP profile name and draft_tokens differ")
        for key in ("binary", "model", "draft_model"):
            value = profile["server"][key]
            if value and not Path(value).is_file():
                raise FileNotFoundError(value)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 8089))
    memory = (
        subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .splitlines()
    )
    if len(memory) != 1 or int(memory[0]) > 256:
        raise ValueError(
            "Stop the inference server first (Ctrl+C); experiment requires one idle GPU"
        )
    destination = (
        root
        / "research/logs"
        / ("mtp-ab-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    )
    destination.mkdir(parents=True)
    reports = {}
    failures = {}
    for name, profile in loaded.items():
        log_path = destination / f"{name}.server.log"
        report_path = destination / f"{name}.json"
        print(f"Starting {name}. Native log: {log_path}", flush=True)
        with log_path.open("w") as log, (destination / f"{name}.gpu.csv").open("w") as gpu_log:
            process = subprocess.Popen(
                server_command(profile), stdout=log, stderr=subprocess.STDOUT
            )
            monitor = None
            try:
                deadline, last_notice = time.monotonic() + 240, 0.0
                with requests.Session() as session:
                    session.trust_env = False
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError(
                                f"{name} server exited with {process.returncode}; inspect {log_path}"
                            )
                        now = time.monotonic()
                        if now >= deadline:
                            raise TimeoutError(f"Server startup timed out; inspect {log_path}")
                        if now - last_notice >= 15:
                            print(f"Waiting for {name} model load...", flush=True)
                            last_notice = now
                        try:
                            response = session.get("http://127.0.0.1:8089/health", timeout=2)
                        except requests.ConnectionError:
                            time.sleep(0.25)
                            continue
                        if response.status_code == 200:
                            break
                        time.sleep(0.25)
                monitor = subprocess.Popen(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
                        "--format=csv",
                        "-l",
                        "1",
                    ],
                    stdout=gpu_log,
                    stderr=subprocess.STDOUT,
                )
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "rlm.v100.cli",
                        "--root",
                        str(root),
                        "--profile",
                        str(profiles[name]),
                        "bench",
                        "--suite",
                        "--repeats",
                        str(repeats),
                        "--output",
                        str(report_path),
                    ],
                    check=True,
                )
                reports[name] = json.loads(report_path.read_text())
            except (RuntimeError, TimeoutError, OSError, subprocess.CalledProcessError) as error:
                if name == "baseline":
                    raise
                failures[name] = {
                    "error": type(error).__name__,
                    "detail": str(error)[:400],
                    "server_log": str(log_path),
                }
                print(f"{name} failed; retaining logs and testing remaining sizes", flush=True)
            finally:
                # Terminate only child processes created by this runner, never another user's server.
                for child in (monitor, process):
                    if child is not None and child.poll() is None:
                        child.terminate()
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
    comparison = {}
    for name, report in reports.items():
        if name == "baseline":
            continue
        try:
            comparison[name] = compare_reports(reports["baseline"], report)
        except ValueError as error:
            failures[name] = {"error": type(error).__name__, "detail": str(error)[:400]}
    atomic_json(destination / "comparison.json", comparison)
    atomic_json(destination / "failures.json", failures)
    recommendation = recommend_mtp(comparison)
    selected = recommendation["recommended_speed_profile"]
    recommendation["profile_path"] = str(profiles[selected])
    recommendation["target_sha256"] = file_digest(Path(loaded[selected]["server"]["model"]))
    atomic_json(destination / "recommendation.json", recommendation)
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)
    print("MTP TEST OK. Reports:", destination)
    print("No profile was promoted. Restore the usual server with v100-lab serve.")


def prepare_mtp(profile_path: Path, root: Path) -> None:
    profile = load_profile(profile_path, root)
    server = Path(profile["server"]["binary"])
    source = server.parents[2]
    converter = source / "convert_hf_to_gguf.py"
    converter_python = root / "venvs/convert/bin/python"
    quantizer = server.with_name("llama-quantize")
    train_python = root / "venvs/train/bin/python"
    for path in (server, converter, converter_python, quantizer, train_python):
        if not path.is_file():
            raise FileNotFoundError(path)
    if profile["server"]["draft_model"]:
        raise ValueError("Prepare from a no-draft baseline profile")
    base_config = json.loads((Path(profile["training"]["base_model"]) / "config.json").read_text())
    if "Gemma4UnifiedForConditionalGeneration" not in base_config.get("architectures", []):
        raise ValueError("This assistant is for Gemma 4 Unified 12B, not an arbitrary target")
    help_result = subprocess.run(
        [str(server), "--help"], check=True, capture_output=True, text=True
    )
    if "draft-mtp" not in help_result.stdout + help_result.stderr:
        raise ValueError("Existing llama-server has no draft-mtp support; working binary unchanged")
    if "Gemma4UnifiedAssistantForCausalLM" not in (source / "conversion/gemma.py").read_text():
        raise ValueError("Existing converter does not support the Gemma Unified assistant")
    environment = {
        **os.environ,
        "HF_HUB_DOWNLOAD_TIMEOUT": "600",
        "HF_HUB_ETAG_TIMEOUT": "60",
        "HF_HUB_DISABLE_XET": "1",
        "UV_HTTP_TIMEOUT": "600",
        "UV_HTTP_RETRIES": "5",
        "UV_LINK_MODE": "copy",
        "PYTHONNOUSERSITE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
    }
    downloader_python = root / "venvs/mtp-tools/bin/python"
    if not downloader_python.exists():
        subprocess.run(
            [
                "uv",
                "--no-config",
                "venv",
                "--python",
                str(train_python),
                str(downloader_python.parents[1]),
            ],
            check=True,
            env=environment,
        )
    hub_version = subprocess.run(
        [
            str(train_python),
            "-c",
            "from importlib.metadata import version; print(version('huggingface-hub'))",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout.strip()
    subprocess.run(
        [
            "uv",
            "--no-config",
            "pip",
            "install",
            "--python",
            str(downloader_python),
            "--index-url",
            "https://pypi.org/simple",
            f"huggingface-hub=={hub_version}",
        ],
        check=True,
        env=environment,
    )
    assistant = root / "models/gemma4-12b-assistant" / ASSISTANT_REVISION
    print("Downloading pinned MTP assistant:", ASSISTANT_MODEL, ASSISTANT_REVISION, flush=True)
    subprocess.run(
        [
            str(downloader_python),
            "-c",
            "import sys; from huggingface_hub import snapshot_download; "
            "snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], "
            "allow_patterns=['*.json', '*.safetensors', '*.model'], max_workers=2)",
            ASSISTANT_MODEL,
            ASSISTANT_REVISION,
            str(assistant),
        ],
        check=True,
        env=environment,
    )
    config = json.loads((assistant / "config.json").read_text())
    if config.get("architectures") != ["Gemma4UnifiedAssistantForCausalLM"]:
        raise ValueError("Unexpected assistant architecture")
    if config["backbone_hidden_size"] != base_config["text_config"]["hidden_size"]:
        raise ValueError("Assistant/target hidden dimensions mismatch")
    if config["text_config"]["vocab_size"] != base_config["text_config"]["vocab_size"]:
        raise ValueError("Assistant/target vocab dimensions mismatch")
    # Token IDs must mean the same thing, not merely have equal vocabulary sizes.
    target_tokenizer = Path(profile["training"]["base_model"]) / "tokenizer.json"
    for name in ("model", "added_tokens"):
        if (
            json.loads(target_tokenizer.read_text())[name]
            != json.loads((assistant / "tokenizer.json").read_text())[name]
        ):
            raise ValueError("Assistant/target token vocabulary mismatch")
    destination = root / "models/gguf/gemma4-12b-assistant" / ASSISTANT_REVISION
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "gemma4-12b-assistant-Q8_0.gguf"
    manifest_path = destination / "manifest.json"
    manifest = {
        "model": ASSISTANT_MODEL,
        "revision": ASSISTANT_REVISION,
        "config_sha256": file_digest(assistant / "config.json"),
        "weights_sha256": file_digest(assistant / "model.safetensors"),
        "converter_sha256": file_digest(converter),
        "gemma_converter_sha256": file_digest(source / "conversion/gemma.py"),
        "server_sha256": file_digest(server),
        "quantizer_sha256": file_digest(quantizer),
        "quantization": "Q8_0",
    }
    if output.exists():
        saved = json.loads(manifest_path.read_text())
        if saved != {**manifest, "gguf_sha256": file_digest(output)}:
            raise ValueError("Existing MTP export provenance/hash changed; refusing to overwrite")
        validate_gguf(output)
    else:
        full = destination / "gemma4-12b-assistant-F16.partial.gguf"
        quantized = destination / "gemma4-12b-assistant-Q8_0.partial.gguf"
        # Only unfinished temporary files owned by this preparation may be recreated.
        for path in (full, quantized):
            path.unlink(missing_ok=True)
        print("Converting assistant to F16, then Q8_0 (CPU only)...", flush=True)
        subprocess.run(
            [
                str(converter_python),
                str(converter),
                str(assistant),
                "--outtype",
                "f16",
                "--outfile",
                str(full),
            ],
            check=True,
            env=environment,
        )
        validate_gguf(full)
        subprocess.run(
            [str(quantizer), str(full), str(quantized), "Q8_0", "8"],
            check=True,
            env=environment,
        )
        validate_gguf(quantized)
        atomic_json(manifest_path, {**manifest, "gguf_sha256": file_digest(quantized)})
        quantized.replace(output)
        full.unlink()
    original = profile_path.read_text()
    for name, draft, tokens in (
        ("baseline", None, 2),
        *((f"mtp{n}", output, n) for n in DEFAULT_DRAFT_TOKENS),
    ):
        experiment = root / f"research/v100-{name}.toml"
        if not experiment.exists():
            text = mtp_profile_text(original, draft, tokens)
            # Validate before writing; never rewrite the user's original or existing experiment.
            temporary = experiment.with_suffix(".toml.tmp")
            temporary.write_text(text)
            load_profile(temporary, root)
            temporary.replace(experiment)
        print("Profile:", experiment)
    print(
        "MTP PREPARE OK. Working server and target weights unchanged. Test profiles use port 8089."
    )
