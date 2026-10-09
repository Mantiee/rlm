import contextlib
import importlib.util
import io
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
    state = dashboard.DashboardState(tmp_path, lambda root: {})
    state.data = {"mission": {"running": True}}
    with dashboard.DashboardServer(("127.0.0.1", 0), state) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        origin = f"http://127.0.0.1:{server.server_port}"
        try:
            from rlm.v100.activity import ActivityLog

            event_id = ActivityLog(tmp_path).write("tools", "tool-start", {"tool": "calculate"})
            with urlopen(origin + "/api/actions?limit=1", timeout=3) as response:
                actions = json.load(response)
                assert actions["events"][0]["id"] == event_id
                assert len(actions["days"]) == 1
            with pytest.raises(HTTPError) as error:
                urlopen(origin + "/api/actions?day=../secret", timeout=3)
            assert error.value.code == 404
            with urlopen(origin + "/api/live-inference", timeout=3) as response:
                live = json.load(response)
                assert live["events"][0]["kind"] == "tool-start"
                assert live["events"][0]["tool"] == "calculate"
                assert isinstance(live["agents"], list)
            with urlopen(origin + "/api/readiness", timeout=3) as response:
                readiness = json.load(response)
                assert readiness["schema"] == "synta-readiness-v1"
                assert "Accepted production weights" in readiness["unverified"]
            with urlopen(origin + "/api/status", timeout=3) as response:
                assert json.load(response)["mission"]["running"] is True
                assert response.headers["Cache-Control"] == "no-store"
            with urlopen(origin, timeout=3) as response:
                assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
                assert "script-src 'sha256-" in response.headers["Content-Security-Policy"]
                assert (
                    "script-src 'unsafe-inline'" not in response.headers["Content-Security-Policy"]
                )
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
    assert (
        sum("timed out" in error or "TimeoutExpired" in error for error in state.data["errors"])
        == 2
    )


@pytest.mark.parametrize(
    "addition",
    [
        "<script>alert(1)</script>",
        '<div onclick="alert(1)">x</div>',
        '<iframe src="http://192.168.0.1"></iframe>',
        '<style>@import "https://example.com/leak";</style>',
        "<style>body{background:u/**/rl(https://example.com)}</style>",
        '<meta http-equiv="refresh" content="0;url=https://example.com">',
        '<a href="https://example.com">outside</a>',
    ],
)
def test_master_layout_refuses_active_content_and_external_resources(addition):
    content = dashboard.BASE_TEMPLATE.replace("</html>", addition + "</html>")
    with pytest.raises(ValueError):
        dashboard.validate_template(content)


def test_master_layout_preserves_data_components_and_fixed_renderer():
    content = dashboard.BASE_TEMPLATE.replace("background:#101721", "background:#172b35")
    identity = dashboard.validate_template(content)
    page = dashboard.render_template(content, identity)
    assert "background:#172b35" in page
    assert page.count("<script>") == 1
    assert f"const layoutVersion='{identity}'" in page
    assert "setInterval(refresh,5000)" in page
    with pytest.raises(ValueError, match="component IDs"):
        dashboard.validate_template(content.replace('id="cards"', 'id="other"'))


def test_master_invalid_edit_keeps_previous_layout_and_survives_restart(tmp_path, monkeypatch):
    state = dashboard.DashboardState(tmp_path, lambda root: {})
    changed = dashboard.BASE_TEMPLATE.replace("V100 - postęp misji", "Mój pulpit V100")
    monkeypatch.setattr(dashboard, "guest_layout", lambda root: changed)
    state.sync_layout()
    accepted = state.page
    identity = state.layout["active_sha256"]
    assert "Mój pulpit V100" in accepted
    monkeypatch.setattr(dashboard, "guest_layout", lambda root: "<script>bad()</script>")
    state.sync_layout()
    assert state.page == accepted
    assert state.layout["active_sha256"] == identity
    assert state.layout["error"]
    restarted = dashboard.DashboardState(tmp_path, lambda root: {})
    assert restarted.page == accepted
    assert len(list((tmp_path / "research/dashboard/layouts").glob("*.html"))) == 2


def test_master_oversized_layout_is_not_published():
    with pytest.raises(ValueError, match="UTF-8 bytes"):
        dashboard.validate_template(dashboard.BASE_TEMPLATE + (" " * 262144))


def test_guest_layout_seed_preserves_master_edits_and_bounds_transfer(tmp_path, monkeypatch):
    from rlm.v100 import dashboard_layout, desktop

    path = tmp_path / "guest/dashboard/index.html"
    monkeypatch.setattr(dashboard_layout, "GUEST_TEMPLATE", str(path))

    def guest_run(root, script, **kwargs):
        assert len(script) <= 16000
        assert kwargs == {"seconds": 8, "output_limit": 360000}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(script.split("\n", 1)[1].rsplit("V100_LAYOUT", 1)[0], {})
        return {"exit_code": 0, "stdout": output.getvalue(), "stderr": ""}

    monkeypatch.setattr(desktop, "run", guest_run)
    assert dashboard.guest_layout(tmp_path) == dashboard.BASE_TEMPLATE
    changed = dashboard.BASE_TEMPLATE.replace("V100 - postęp misji", "Nowy wygląd")
    path.write_text(changed)
    assert dashboard.guest_layout(tmp_path) == changed
    path.write_bytes(b"x" * 262145)
    with pytest.raises(ValueError, match="256 KiB"):
        dashboard.guest_layout(tmp_path)


def test_dashboard_sync_retries_guest_repair_after_boot_without_publishing_invalid_html(
    tmp_path, monkeypatch
):
    from rlm.v100 import dashboard_editor, dashboard_layout

    state = dashboard.DashboardState(tmp_path, lambda root: {})
    reads = iter(
        [
            RuntimeError("guest booting"),
            dashboard_layout.BASE_TEMPLATE.replace("</html>", "<script>bad()</script></html>"),
            dashboard_layout.BASE_TEMPLATE,
        ]
    )

    def read(root):
        value = next(reads)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(dashboard, "guest_layout", read)
    repairs = []
    monkeypatch.setattr(
        dashboard_editor, "repair", lambda root: repairs.append(root) or {"repaired": True}
    )
    state.sync_layout()
    assert not repairs and state.layout["state"] == "previous validated layout retained"
    state.sync_layout()
    assert repairs == [tmp_path] and state.layout["state"] == "validated layout active"
    assert state.layout["repair"]["repaired"]
