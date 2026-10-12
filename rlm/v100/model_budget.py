"""Conservative per-host admission before downloading a large local MoE."""

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

SIZES = {"gpt-oss:20b": 14_000_000_000, "gpt-oss:120b": 65_000_000_000}


def admission(
    model: str, ram_available_gib: float, vram_free_gib: float, disk_free_gib: float
) -> dict:
    if model not in SIZES or not all(
        math.isfinite(v) and v >= 0 for v in (ram_available_gib, vram_free_gib, disk_free_gib)
    ):
        raise ValueError("Require a declared local model and finite measured resources")
    weights = SIZES[model] / 2**30
    # Existing helper reserves board headroom; this does not pool VRAM with another machine.
    usable_gpu = min(12, max(0, vram_free_gib - 3))
    available = max(0, ram_available_gib - 6) + usable_gpu
    reasons = []
    if available < weights + 4:
        reasons.append("Local RAM/GPU residency budget below weights plus 4 GiB runtime allowance")
    if disk_free_gib < weights + 8:
        reasons.append("Disk budget below model plus 8 GiB reserve")
    return {
        "model": model,
        "admitted": not reasons,
        "reasons": reasons,
        "weights_gib": weights,
        "ram_available_gib": ram_available_gib,
        "vram_free_gib": vram_free_gib,
        "usable_gpu_gib": usable_gpu,
        "disk_free_gib": disk_free_gib,
        "scope": "Conservative local download admission only; actual residency, quality and speed must still pass host tests. No memory pooled across hosts.",
    }


def main() -> None:
    import psutil

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    reply = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    rows = reply.stdout.strip().splitlines()
    if len(rows) != 1:
        raise ValueError("One explicitly owned GPU required for this budget")
    values = [
        admission(
            model,
            psutil.virtual_memory().available / 2**30,
            float(rows[0]) / 1024,
            shutil.disk_usage(args.root).free / 2**30,
        )
        for model in SIZES
    ]
    print(json.dumps({"candidates": values}))


if __name__ == "__main__":
    main()
