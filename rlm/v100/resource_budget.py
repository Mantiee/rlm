"""Headroom admission, not synthetic activity or a board power limit."""


def local_budget() -> dict:
    import psutil

    cpu = psutil.cpu_percent(interval=0.05)
    free = psutil.virtual_memory().available / 2**30
    slots = 0 if cpu >= 85 or free < 2 else 1 if cpu >= 65 or free < 4 else 2
    return {
        "cpu_slots": slots,
        "host_cpu_percent": cpu,
        "free_ram_gib": round(free, 2),
        "reason": "Host pressure; queued work retained"
        if not slots
        else "Measured CPU/RAM headroom",
        "scope": "At most two local CPU/network drones. GPU serving/training use existing exclusive admission. Windows uses its own lower limits; RTX remains operator-disabled.",
    }
