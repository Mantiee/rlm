"""Bounded renderer and collector readiness at the configured private bind."""

import ipaddress
import json
import math
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests


def endpoint(root: Path | None) -> str:
    if root is None:
        return "http://127.0.0.1:8765/"
    config = root / "research/remote-access/config.json"
    if config.is_file():
        origin = json.loads(config.read_text())["dashboard"]
    else:
        command = subprocess.check_output(
            [
                "systemctl",
                "--user",
                "show",
                "v100-dashboard.service",
                "--property=ExecStart",
                "--value",
            ],
            text=True,
            timeout=5,
        )
        found = re.search(r"--bind(?:=|\s+)([0-9.]+)(?:\s|;|$)", command)
        if found is None:
            raise ValueError("Cannot identify the owned dashboard bind; no address guessed")
        origin = "http://" + found.group(1) + ":8765"
    parsed = urlsplit(origin)
    address = ipaddress.ip_address(parsed.hostname or "")
    if (
        parsed.scheme != "http"
        or address.version != 4
        or not address.is_private
        or address.is_unspecified
        or address.is_multicast
        or parsed.port != 8765
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
    ):
        raise ValueError(
            "Dashboard readiness requires the configured private IPv4 origin on port 8765"
        )
    return origin.rstrip("/") + "/"


def wait(seconds: int = 60, *, root: Path | None = None) -> dict:
    if type(seconds) is not int or not 1 <= seconds <= 90:
        raise ValueError("Dashboard readiness budget must be 1..90 seconds")
    url = endpoint(root)
    deadline = time.monotonic() + seconds
    detail = "No HTTP attempt completed"
    while time.monotonic() < deadline:
        try:
            response = requests.get(url, timeout=min(2, max(0.1, deadline - time.monotonic())))
            response.raise_for_status()
            if "mission-navigation" not in response.text or "Trade explorer" not in response.text:
                raise ValueError("HTTP response is not the expected host dashboard renderer")
            response = requests.get(
                url + "api/status", timeout=min(2, max(0.1, deadline - time.monotonic()))
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("Dashboard status must be a JSON object")
            collected = data.get("collected_at")
            if (
                not isinstance(data.get("mission"), dict)
                or type(collected) not in (int, float)
                or not math.isfinite(collected)
                or not -5 <= time.time() - collected <= 30
                or data.get("report_freshness", {}).get("source")
                == "previous snapshot after collector error"
            ):
                raise ValueError(
                    "Dashboard HTTP is available but its collector has no fresh successful snapshot"
                )
            return {
                "dashboard_ready": True,
                "collector_ready": True,
                "url": url,
                "scope": "Fresh local collector snapshot and renderer; not proof of model learning or phone routing",
            }
        except (requests.RequestException, ValueError, TypeError) as error:
            detail = str(error)[:300]
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    raise RuntimeError("Dashboard did not become ready within the budget: " + detail)
