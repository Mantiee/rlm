import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

from rlm.v100 import github_dashboard as github

SHA = "a" * 40


class Response:
    def __init__(self, status=200, payload=None, headers=None):
        self.status_code = status
        self.payload = payload or {}
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def iter_content(self, size):
        yield json.dumps(self.payload).encode()


class Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.trust_env = True
        self.closed = False

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)

    def close(self):
        self.closed = True


def pull_request(number):
    return {
        "number": number,
        "title": f"PR {number}",
        "state": "open",
        "draft": False,
        "head": {"sha": SHA, "ref": f"branch-{number}"},
    }


def checks(conclusion="success"):
    return {
        "total_count": 1,
        "check_runs": [
            {
                "name": "Test",
                "status": "completed",
                "conclusion": conclusion,
                "started_at": "2026-10-15T10:00:00Z",
                "completed_at": "2026-10-15T10:01:00Z",
                "details_url": "https://github.com/Mantiee/rlm/actions/runs/123",
            }
        ],
    }


def cached_snapshot(stamp=None):
    observed = datetime.fromtimestamp(stamp, UTC) if stamp is not None else datetime.now(UTC)
    timestamp = observed.isoformat()
    row = checks()["check_runs"][0]
    return {
        "schema": "synta-github-dashboard-v1",
        "repository": "Mantiee/rlm",
        "source": "unauthenticated GitHub REST GET; read-only status evidence",
        "state": "available",
        "observed_at": timestamp,
        "pull_requests": [
            {
                "number": 1,
                "title": "PR 1",
                "state": "open",
                "draft": False,
                "head_sha": SHA,
                "head_ref": "branch-1",
                "requested_at": timestamp,
                "observed_at": timestamp,
                "check_state": "success",
                "check_run_count": 1,
                "check_runs": [row],
            }
        ],
    }


def test_refresh_is_pinned_unauthenticated_and_caches_bounded_read_only_status(tmp_path):
    session = Session(
        [
            Response(payload=pull_request(1)),
            Response(payload=checks()),
            Response(payload=pull_request(18)),
            Response(payload=checks()),
        ]
    )
    result = github.refresh(tmp_path, session=session)

    assert result["repository"] == "Mantiee/rlm"
    assert result["state"] == "available"
    assert [(row["number"], row["check_state"]) for row in result["pull_requests"]] == [
        (1, "success"),
        (18, "success"),
    ]
    assert session.closed is False
    assert len(session.calls) == 4
    for url, options in session.calls:
        assert url.startswith(github.API_ROOT + "/")
        assert options["timeout"] == github.REQUEST_TIMEOUT
        assert options["allow_redirects"] is False
        assert options["auth"] is None
        assert "Authorization" not in options["headers"]
        assert options["headers"]["X-GitHub-Api-Version"] == "2022-11-28"
    cached = github.cached(tmp_path)
    assert cached["state"] == "available"
    assert cached["pull_requests"][1]["head_sha"] == SHA
    assert github.CACHE_RELATIVE.as_posix() == "research/github-dashboard/latest.json"


def test_default_session_disables_environment_credentials_and_closes(monkeypatch, tmp_path):
    session = Session(
        [
            Response(payload=pull_request(1)),
            Response(payload=checks()),
            Response(payload=pull_request(18)),
            Response(payload=checks()),
        ]
    )
    monkeypatch.setattr(github.requests, "Session", lambda: session)

    result = github.refresh(tmp_path)

    assert result["state"] == "available"
    assert session.trust_env is False
    assert session.closed is True


def test_failed_check_and_empty_checks_never_report_green(tmp_path):
    failed = Session(
        [
            Response(payload=pull_request(1)),
            Response(payload=checks("failure")),
            Response(payload=pull_request(18)),
            Response(payload={"total_count": 0, "check_runs": []}),
        ]
    )
    result = github.refresh(tmp_path, session=failed)
    assert [row["check_state"] for row in result["pull_requests"]] == ["failure", "unknown"]
    assert result["state"] == "partial"


def test_credentials_and_rate_limit_failures_are_explicit(tmp_path):
    session = Session(
        [
            Response(status=401),
            Response(status=403, headers={"X-RateLimit-Remaining": "0"}),
        ]
    )
    result = github.refresh(tmp_path, session=session)
    assert result["state"] == "unavailable"
    assert [row["error"]["kind"] for row in result["pull_requests"]] == [
        "credentials_required",
        "rate_limited",
    ]
    assert all(row["check_state"] == "unknown" for row in result["pull_requests"])
    assert all(row["observed_at"] for row in result["pull_requests"])


def test_config_rejects_repository_override_and_credentials(tmp_path):
    config = tmp_path / "research/github-dashboard/config.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"repository": "somebody/else", "pull_requests": [1]}))
    result = github.refresh(tmp_path, session=Session([]))
    assert result["state"] == "unavailable"
    assert result["error"]["kind"] == "invalid_configuration"
    assert json.loads((tmp_path / github.CACHE_RELATIVE).read_text())["state"] == "unavailable"

    config.write_text(
        json.dumps({"repository": "Mantiee/rlm", "pull_requests": [1], "token": "secret"})
    )
    result = github.refresh(tmp_path, session=Session([]))
    assert result["error"]["kind"] == "invalid_configuration"
    assert "secret" not in (tmp_path / github.CACHE_RELATIVE).read_text()


def test_cache_expiry_and_future_timestamps_become_unknown(tmp_path):
    path = tmp_path / github.CACHE_RELATIVE
    path.parent.mkdir(parents=True)
    stamp = datetime(2026, 10, 15, tzinfo=UTC).timestamp()
    path.write_text(json.dumps(cached_snapshot(stamp)))
    assert github.cached(tmp_path, now=stamp + github.REFRESH_SECONDS - 1)["state"] == "available"
    expired = github.cached(tmp_path, now=stamp + github.REFRESH_SECONDS + 1)
    assert expired["state"] == "stale"
    assert expired["pull_requests"][0]["check_state"] == "unknown"
    assert expired["pull_requests"][0]["previous_check_state"] == "success"
    assert expired["pull_requests"][0]["check_runs"] == []
    assert expired["pull_requests"][0]["check_run_count"] is None
    future = github.cached(tmp_path, now=stamp - 1)
    assert future["state"] == "unknown"
    assert future["error"]["kind"] == "future_snapshot_timestamp"


def test_oversized_and_wrong_identity_cache_are_unavailable(tmp_path):
    path = tmp_path / github.CACHE_RELATIVE
    path.parent.mkdir(parents=True)
    path.write_text("x" * (github.MAX_CACHE_BYTES + 1))
    assert github.cached(tmp_path)["state"] == "unavailable"
    path.write_text(
        json.dumps(
            {
                "schema": "synta-github-dashboard-v1",
                "repository": "alexzhang13/rlm",
                "state": "available",
                "observed_at": datetime.now(UTC).isoformat(),
                "pull_requests": [],
            }
        )
    )
    result = github.cached(tmp_path)
    assert result["state"] == "unavailable"
    assert result["error"]["kind"] == "invalid_cache"


def test_cache_timestamp_without_timezone_is_unavailable(tmp_path):
    path = tmp_path / github.CACHE_RELATIVE
    path.parent.mkdir(parents=True)
    snapshot = cached_snapshot()
    snapshot["observed_at"] = "2026-10-15T10:00:00"
    path.write_text(json.dumps(snapshot))
    result = github.cached(tmp_path)
    assert result["state"] == "unavailable"
    assert result["error"]["kind"] == "invalid_cache"


def test_malformed_cached_snapshot_never_preserves_green_status(tmp_path):
    path = tmp_path / github.CACHE_RELATIVE
    path.parent.mkdir(parents=True)
    corruptions = (
        lambda value: value.update(state="green"),
        lambda value: value["pull_requests"][0].update(number=True),
        lambda value: value["pull_requests"][0].update(state="unknown"),
        lambda value: value["pull_requests"][0].update(check_state="green"),
        lambda value: value["pull_requests"][0].update(check_run_count=2),
        lambda value: value["pull_requests"][0]["check_runs"][0].update(status=[]),
        lambda value: value["pull_requests"][0].update(head_sha="bad-sha"),
        lambda value: value["pull_requests"][0].update(observed_at="yesterday"),
        lambda value: value["pull_requests"][0].update(error=[]),
    )

    for corrupt in corruptions:
        snapshot = cached_snapshot()
        corrupt(snapshot)
        path.write_text(json.dumps(snapshot))

        result = github.cached(tmp_path)

        assert result["state"] == "unavailable"
        assert result["error"]["kind"] == "invalid_cache"
        assert result["pull_requests"] == []


def test_dashboard_collector_exposes_only_cached_github_status_in_api_payload(
    tmp_path, monkeypatch
):
    spec = importlib.util.spec_from_file_location(
        "v100_dashboard_github_test",
        Path(__file__).parents[1] / "tools/serve-v100-dashboard.py",
    )
    dashboard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dashboard)
    state = dashboard.DashboardState(tmp_path, lambda root: {})
    saved = {
        "schema": "synta-github-dashboard-v1",
        "repository": "Mantiee/rlm",
        "state": "available",
        "observed_at": datetime.now(UTC).isoformat(),
        "pull_requests": [{"number": 18, "check_state": "success"}],
    }
    monkeypatch.setattr(github, "cached", lambda root: saved)
    monkeypatch.setattr(dashboard, "read_json", lambda path: {})
    from rlm.v100 import dashboard_snapshot, live_status, trade_explorer

    monkeypatch.setattr(dashboard_snapshot, "supplement", lambda root, mission, report, errors: {})
    monkeypatch.setattr(live_status, "snapshot", lambda root, mission, report, gpu: {})
    monkeypatch.setattr(trade_explorer, "snapshot", lambda root: {})

    state.refresh()

    assert state.data["report"]["github_ci"] == saved


def test_rendered_dashboard_shows_pr_head_and_ci_state_safely():
    from rlm.v100 import dashboard_layout

    html = dashboard_layout.render_template(dashboard_layout.BASE_TEMPLATE, "test")
    start = html.index("function githubCiPanel(data){")
    end = html.index("\nfunction modelDeploymentView", start)
    panel = html[start:end]

    assert "githubCiPanel(data);" in html
    assert "github-ci-panel" in panel
    assert "GitHub PRs and CI checks" in panel
    assert "Snapshot state" in panel and "Snapshot age" in panel
    assert "row.check_state||'unknown'" in panel
    assert "check.name||'unknown'" in panel
    assert "No check runs; status unknown" in panel
    assert "No cached PR/check snapshot; status unknown" in panel
    assert "titleCell.textContent" in panel
    assert "checksCell.textContent" in panel
    assert "titleCell.innerHTML" not in panel and "checksCell.innerHTML" not in panel
    assert "do not prove deployment, runtime correctness, or revenue" in panel


def test_response_size_and_check_run_page_truncation_do_not_claim_success(monkeypatch):
    class Oversized(Response):
        def iter_content(self, size):
            yield b"x" * (github.MAX_RESPONSE_BYTES + 1)

    oversize = Session([Oversized()])
    _, error = github._request_json(oversize, f"{github.API_ROOT}/pulls/1")
    assert error["kind"] == "request_failed"
    assert "exceeds 512 KiB" in error["detail"]

    assert (
        github._check_state({"total_count": 101, "check_runs": [checks()["check_runs"][0]]})
        == "unknown"
    )


def test_check_run_count_must_match_complete_nonempty_page():
    row = checks()["check_runs"][0]
    incomplete_or_invalid = (
        {"check_runs": [row]},  # Missing count cannot imply completeness.
        {"total_count": 0, "check_runs": [row]},
        {"total_count": -1, "check_runs": [row]},
        {"total_count": 2, "check_runs": [row]},  # Truncated page.
        {"total_count": 1, "check_runs": [row, row]},  # Inconsistent payload.
        {"total_count": 0, "check_runs": []},
        {"total_count": 1, "check_runs": []},
        {"total_count": True, "check_runs": [row]},  # bool is not an integer count.
    )

    assert all(github._check_state(payload) == "unknown" for payload in incomplete_or_invalid)

    assert github._check_state({"total_count": 1, "check_runs": [row]}) == "success"


def test_non_object_live_check_run_forces_unknown_and_hides_rows(tmp_path):
    config = tmp_path / "research/github-dashboard/config.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"repository": "Mantiee/rlm", "pull_requests": [1]}))
    malformed_checks = {
        "total_count": 2,
        "check_runs": [checks()["check_runs"][0], "malformed row"],
    }
    session = Session([Response(payload=pull_request(1)), Response(payload=malformed_checks)])

    result = github.refresh(tmp_path, session=session)

    assert result["state"] == "partial"
    row = result["pull_requests"][0]
    assert row["check_state"] == "unknown"
    assert row["check_run_count"] is None
    assert row["check_runs"] == []
    assert github.cached(tmp_path)["pull_requests"][0]["check_runs"] == []


def test_unrecognized_pull_request_state_is_unavailable_without_check_fetch(tmp_path):
    config = tmp_path / "research/github-dashboard/config.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"repository": "Mantiee/rlm", "pull_requests": [1]}))
    invalid_pr = pull_request(1)
    invalid_pr["state"] = "merging"
    session = Session([Response(payload=invalid_pr)])

    result = github.refresh(tmp_path, session=session)

    assert result["state"] == "unavailable"
    assert len(session.calls) == 1
    row = result["pull_requests"][0]
    assert row["state"] == "unavailable"
    assert row["check_state"] == "unknown"
    assert row["error"]["kind"] == "invalid_pull_request_response"
    assert github.cached(tmp_path)["state"] == "unavailable"
