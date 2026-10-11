"""Bounded dashboard startup checks after asynchronous systemd restart."""

import time

import requests


def wait(seconds: int = 60) -> dict:
    if type(seconds) is not int or not 1 <= seconds <= 90:
        raise ValueError("Dashboard readiness budget must be 1..90 seconds")
    deadline = time.monotonic() + seconds
    detail = "No HTTP attempt completed"
    while time.monotonic() < deadline:
        try:
            response = requests.get(
                "http://127.0.0.1:8765/", timeout=min(2, max(0.1, deadline - time.monotonic()))
            )
            response.raise_for_status()
            if "mission-navigation" not in response.text or "Trade explorer" not in response.text:
                detail = "HTTP response is not the expected host dashboard renderer"
            else:
                return {
                    "dashboard_ready": True,
                    "url": "http://127.0.0.1:8765/",
                    "scope": "HTTP renderer readiness only; not proof of model progress or phone routing",
                }
        except requests.RequestException as error:
            detail = str(error)[:300]
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    raise RuntimeError("Dashboard did not become ready within the budget: " + detail)
