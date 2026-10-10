"""Print this authenticated device's actual private URLs and service blockers."""

import getpass
import json
import subprocess
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def links(status: dict) -> dict:
    if status.get("BackendState") != "Running":
        raise ValueError("Tailscale is not connected on Debian; run sudo tailscale up")
    host = status.get("Self", {}).get("DNSName", "").rstrip(".")
    if not host.endswith(".ts.net") or any(c in host for c in "/@ :\r\n"):
        raise ValueError("This device has no valid Tailscale DNS name; check MagicDNS")
    return {
        "dashboard": "https://" + host,
        "chat": "https://" + host + "/chat",
        "ssh": "ssh " + getpass.getuser() + "@" + host,
    }


def status(root: Path) -> dict:
    state = json.loads(
        subprocess.check_output(["tailscale", "status", "--json"], text=True, timeout=10)
    )
    result = links(state)
    gateway = subprocess.run(
        ["systemctl", "--user", "is-active", "synta-remote.service"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    result["gateway_service"] = gateway.stdout.strip() or "unknown"
    serve = json.loads(
        subprocess.check_output(["tailscale", "serve", "status", "--json"], text=True, timeout=10)
    )
    host = result["dashboard"].removeprefix("https://")
    result["private_https_proxy_configured"] = (
        serve.get("Web", {}).get(host + ":443", {}).get("Handlers", {}).get("/", {}).get("Proxy")
        == "http://127.0.0.1:8786"
    )
    result["public_funnel_enabled"] = any(serve.get("AllowFunnel", {}).values())
    config = root / "research/remote-access/config.json"
    if config.exists():
        from rlm.v100.remote_access import validate_config

        value = validate_config(json.loads(config.read_text()))
        result["configured_origin_matches"] = value["origin"] == result["dashboard"]
        try:
            with urlopen(value["dashboard"] + "/api/status", timeout=5) as response:
                result["dashboard_upstream_http"] = response.status
        except (OSError, URLError) as error:
            result["dashboard_upstream_error"] = str(error)[:200]
    else:
        result["setup_required"] = (
            "Run tools/install-synta-remote.sh from the pinned revision; login alone does not install the web gateway"
        )
    result["next_step"] = (
        "Turn on Tailscale on phone/laptop using the same account, open the HTTPS dashboard/chat above. SSH uses existing Debian credentials."
    )
    result["blockers"] = []
    if not result["private_https_proxy_configured"]:
        result["blockers"].append(
            "Private HTTPS Serve proxy is not configured. Signing in on iPhone does not enable Serve."
        )
        result["next_step"] = (
            "On Debian run: sudo tailscale serve --bg --https=443 http://127.0.0.1:8786 . Open and approve the enable-HTTPS link printed by Tailscale, then rerun this command."
        )
    if result["gateway_service"] != "active":
        result["blockers"].append(
            "Owner gateway service is not active; check systemctl --user status synta-remote.service"
        )
    if result["public_funnel_enabled"]:
        result["blockers"].append("Public Funnel is enabled; owner-only access is not ready")
    if not result.get("configured_origin_matches"):
        result["blockers"].append(
            "Owner gateway configuration is missing or has a different DNS origin"
        )
    if result.get("dashboard_upstream_http") != 200:
        result["blockers"].append("Existing LAN dashboard upstream is not responding with HTTP 200")
    result["ready"] = not result["blockers"]
    result["scope"] = (
        "Local configuration and upstream checks; successful access from this iPhone is not verified here. Use the HTTPS DNS name, not the home LAN address or :8765."
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(status(parser.parse_args().root), indent=2))
