"""Exclusive short V100 training pilots; no automatic serving promotion."""

import copy
import json
import math
import subprocess
import threading
import time
import uuid
from pathlib import Path

from rlm.v100.common import atomic_json, load_profile
from rlm.v100.protection import file_hash


def run_pilot(argv: list[str], path: Path, timeout: int = 3600) -> int | None:
    """Mirror the child log into the existing preparation console."""
    stopped = threading.Event()

    def follow():
        with path.open("rb") as reader:
            while True:
                chunk = reader.read(65536)
                if chunk:
                    print(chunk.decode(errors="replace"), end="", flush=True)
                elif stopped.is_set():
                    return
                else:
                    stopped.wait(0.25)

    with path.open("w") as log:
        thread = threading.Thread(target=follow, daemon=True)
        thread.start()
        try:
            return subprocess.run(
                argv, stdout=log, stderr=subprocess.STDOUT, timeout=timeout
            ).returncode
        except subprocess.TimeoutExpired:
            return None
        finally:
            log.flush()
            stopped.set()
            thread.join(timeout=5)


def calibrate(root: Path, path: Path, pool: Path) -> Path:
    from rlm.v100.competition import command, require_idle_gpu
    from rlm.v100.mission import status
    from rlm.v100.training import load_records

    if status(root)["running"]:
        raise ValueError("Stop the mission before exclusive training calibration")
    require_idle_gpu()
    source = load_profile(path, root)
    if source["training"].get("init_adapter"):
        source["training"]["rank"] = json.loads(
            (Path(source["training"]["init_adapter"]) / "adapter_config.json").read_text()
        )["r"]
    train, validation = load_records(pool, Path(source["training"]["split_ledger"]))
    folder = root / "research/training-calibration" / ("run-" + uuid.uuid4().hex[:12])
    folder.mkdir(parents=True)
    selected = (
        sorted(train, key=lambda r: len(json.dumps(r["messages"])), reverse=True)[:32]
        + sorted(validation, key=lambda r: len(json.dumps(r["messages"])), reverse=True)[:8]
    )
    if not train or not validation:
        raise ValueError("Need disjoint independently verified training and validation")
    dataset = folder / "pool.jsonl"
    dataset.write_text("".join(json.dumps(r) + "\n" for r in selected))
    results = {}
    for precision in ("nf4", "fp16"):
        require_idle_gpu()
        pilot = copy.deepcopy(source)
        pilot["training"].update(
            retention_mode=0,
            retention_rank_growth=1,
            retention_reference_groups=[],
            precision=precision,
            output=str(folder / precision),
            max_steps=25,
            eval_steps=25,
            save_steps=25,
            gradient_accumulation=1,
            microbatch=1,
            seed=42,
        )
        config = folder / (precision + ".json")
        atomic_json(config, pilot)
        print(
            json.dumps(
                {
                    "phase": "training-calibration",
                    "precision": precision,
                    "log": str(folder / (precision + ".log")),
                }
            ),
            flush=True,
        )
        started, code = time.monotonic(), None
        code = run_pilot(
            command(root, config, "train", str(dataset)), folder / (precision + ".log")
        )
        rows = []
        metrics = folder / precision / "metrics.jsonl"
        if metrics.exists():
            rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
        losses = [
            row.get("loss", row.get("eval_loss"))
            for row in rows
            if "loss" in row or "eval_loss" in row
        ]
        for name in ("baseline_eval.json", "candidate_eval.json"):
            evaluation = folder / precision / name
            losses.append(
                json.loads(evaluation.read_text()).get("eval_loss") if evaluation.exists() else None
            )
        peak = max((row.get("peak_vram_gib", 0) for row in rows), default=0)
        health_path = folder / precision / "training_health.json"
        health = json.loads(health_path.read_text()) if health_path.exists() else {}
        eligible = (
            code == 0
            and 0 < peak <= 27.5
            and all(type(loss) in (int, float) and math.isfinite(loss) for loss in losses)
            and health.get("schema") == "v100-training-health-v1"
            and health.get("eligible") is True
            and health.get("optimizer_updates", 0) > 0
        )
        results[precision] = {
            "eligible": eligible,
            "seconds": time.monotonic() - started,
            "exit_code": code,
            "peak_allocated_vram_gib": peak,
            "training_health": health,
            "losses": [
                loss if type(loss) in (int, float) and math.isfinite(loss) else None
                for loss in losses
            ],
        }
        print(
            json.dumps(
                {
                    "phase": "training-calibration-result",
                    "precision": precision,
                    **results[precision],
                }
            ),
            flush=True,
        )
    atomic_json(folder / "results.json", results)
    allowed = [name for name, result in results.items() if result["eligible"]]
    if not allowed:
        raise RuntimeError(f"No eligible pilot; previous profile retained. Inspect {folder}")
    fastest = min(allowed, key=lambda p: results[p]["seconds"])
    source["training"]["precision"] = fastest
    source.setdefault("resources", {})["training_budget"] = {
        "max_rank": source["training"]["rank"],
        "max_length": source["training"]["max_length"],
        "report": str(folder / "results.json"),
        "report_sha256": file_hash(folder / "results.json"),
        "scope": "Observed examples, not a worst-case memory guarantee; independent quality still required",
    }
    output = folder / "calibrated-profile.json"
    atomic_json(output, source)
    return output
