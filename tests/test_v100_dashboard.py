import importlib.util
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

spec = importlib.util.spec_from_file_location(
    "v100_dashboard_tool", Path(__file__).parents[1] / "tools/serve-v100-dashboard.py"
)
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


@pytest.mark.parametrize("address", ["0.0.0.0", "8.8.8.8", "224.0.0.1", "::1"])
def test_dashboard_rejects_unspecified_public_and_non_ipv4_binds(address):
    with pytest.raises(ValueError):
        dashboard.validate_bind(address)


def test_dashboard_private_bind_and_readonly_http_routes(tmp_path):
    assert dashboard.validate_bind("192.168.0.68") == "192.168.0.68"
    state = SimpleNamespace(root=tmp_path, data={"mission": {"running": True}})
    with dashboard.DashboardServer(("127.0.0.1", 0), state) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        origin = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(origin + "/api/status", timeout=3) as response:
                assert json.load(response)["mission"]["running"] is True
                assert response.headers["Cache-Control"] == "no-store"
            with urlopen(origin, timeout=3) as response:
                assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
                assert b"/api/status" in response.read()
            for path in ("/env.sh", "/paper/%2e%2e/env.sh", "/api/stop"):
                with pytest.raises(HTTPError) as error:
                    urlopen(origin + path, timeout=3)
                assert error.value.code == 404
            with pytest.raises(HTTPError) as error:
                urlopen(Request(origin + "/api/status", method="POST"), timeout=3)
            assert error.value.code == 405
        finally:
            server.shutdown()
            thread.join(timeout=3)


def test_dashboard_serves_only_allowed_report_assets_and_refuses_symlinks(tmp_path):
    name = "20261009-061201-all-04c7a1f8"
    folder = tmp_path / "research/paper/reports" / name
    folder.mkdir(parents=True)
    (folder / "report.html").write_text("<p>paper report</p>")
    path, mime = dashboard.paper_asset(tmp_path, f"/paper/{name}/report.html")
    assert path.read_text() == "<p>paper report</p>"
    assert mime.startswith("text/html")
    private = tmp_path / "secret.json"
    private.write_text("private")
    (folder / "report.json").symlink_to(private)
    for route in (f"/paper/{name}/secret.json", f"/paper/{name}/report.json"):
        with pytest.raises(ValueError):
            dashboard.paper_asset(tmp_path, route)


def test_dashboard_does_not_attribute_previous_run_reports_to_current_run(tmp_path, monkeypatch):
    folder = tmp_path / "research/mission"
    folder.mkdir(parents=True)
    (folder / "latest-report.json").write_text(
        json.dumps(
            {
                "mission_evidence": {"run": "old-run"},
                "accepted_weight_updates_this_run": 7,
            }
        )
    )

    def fake_run(command, **kwargs):
        if command[0] == "nvidia-smi":
            return SimpleNamespace(stdout="91, 14758, 32768, 56, 204\n", returncode=0)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(dashboard.subprocess, "run", fake_run)
    state = dashboard.DashboardState(tmp_path, lambda root: {"run": "new-run", "running": True})
    state.refresh()
    assert state.data["mission"]["run"] == "new-run"
    assert state.data["report"] == {}
    assert state.data["gpu"]["memory_used"] == 14758
    assert "poprzedniego przebiegu" in state.data["errors"][0]


def test_dashboard_report_timeout_is_visible_and_does_not_hide_live_mission(tmp_path, monkeypatch):
    def unavailable(command, **kwargs):
        raise dashboard.subprocess.TimeoutExpired(command, 20)

    monkeypatch.setattr(dashboard.subprocess, "run", unavailable)
    state = dashboard.DashboardState(tmp_path, lambda root: {"running": True})
    state.refresh()
    assert state.data["mission"]["running"] is True
    assert state.data["report"] == {}
    assert state.data["gpu"] == {}
    assert len(state.data["errors"]) == 2
