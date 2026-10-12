"""Headroom admission, not synthetic activity or a board power limit."""

import json
import math
import os
from pathlib import Path


def available_cpus(
    cgroup_root: Path = Path("/sys/fs/cgroup"), membership: Path = Path("/proc/self/cgroup")
) -> int:
    """Count process affinity and Linux cgroup v2 quota; admission reserves CPU separately."""
    count = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    )
    if membership.exists():
        for line in membership.read_text().splitlines():
            if line.startswith("0::"):
                base = cgroup_root.resolve()
                folder = (base / line[3:].lstrip("/")).resolve()
                if not folder.is_relative_to(base):
                    # A cgroup namespace can report a path outside the mounted subtree.
                    # Its mounted root still exposes the namespace's effective root quota.
                    folder = base
                while folder.is_relative_to(base):
                    quota = folder / "cpu.max"
                    if quota.exists():
                        limit, period = quota.read_text().split()
                        if limit != "max":
                            count = min(count, max(1, math.floor(int(limit) / int(period))))
                    if folder == base:
                        break
                    folder = folder.parent
    return max(1, count)


def local_budget(root: Path | None = None) -> dict:
    import psutil

    cpu = psutil.cpu_percent(interval=0.05)
    free = psutil.virtual_memory().available / 2**30
    maximum = 2
    reserve_ram, ram_per_slot = 0, 2
    path = root / "research/state/resource-policy.json" if root is not None else None
    if path is not None and path.exists():
        policy = json.loads(path.read_text())
        maximum = policy["max_cpu_slots"]
        if type(maximum) is not int or not 1 <= maximum <= 16:
            raise ValueError("Local CPU policy must allow 1-16 slots")
        reserve_ram, ram_per_slot = (
            policy.get("reserve_ram_gib", 0),
            policy.get("ram_per_slot_gib", 2),
        )
        if (
            type(reserve_ram) is not int
            or not 0 <= reserve_ram <= 8
            or type(ram_per_slot) is not int
            or not 2 <= ram_per_slot <= 4
        ):
            raise ValueError("Invalid local RAM reservation")
    cpus = available_cpus()
    maximum = min(maximum, max(1, (cpus - 1) // 2))
    ram_slots = max(0, int((free - reserve_ram) // ram_per_slot))
    slots = 0 if cpu >= 85 else min(1 if cpu >= 65 else maximum, ram_slots)
    return {
        "cpu_slots": slots,
        "max_cpu_slots": maximum,
        "available_cpus": cpus,
        "reserve_ram_gib": reserve_ram,
        "ram_per_slot_gib": ram_per_slot,
        "host_cpu_percent": cpu,
        "free_ram_gib": round(free, 2),
        "reason": "Host pressure; queued work retained"
        if not slots
        else "Measured CPU/RAM headroom",
        "scope": "Bounded local CPU/network admission; one model job. GPU serving/training keep exclusive gates. Windows keeps its own pressure, RAM and game guards. Slots are concurrency limits, not measured speedup.",
    }


def cpu_assignment(active: list[list[int]]) -> list[int]:
    """Allocate distinct affinity slots for concurrent owned Python experiments."""
    cpus = sorted(os.sched_getaffinity(0))[: available_cpus()]
    if len(cpus) > 2:
        cpus = cpus[1:]
    used = {cpu for assignment in active for cpu in assignment}
    return [cpu for cpu in cpus if cpu not in used][:2]
