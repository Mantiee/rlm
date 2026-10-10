import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlm.v100 import competition, goal_learning, income_opportunities, planning, research_policy
from rlm.v100.common import atomic_json
from rlm.v100.serving import assert_served_expert, receipt_path
from tests.test_v100_continual import profile
from tests.test_v100_recovery import financial_world


def specification(identity):
    return {
        "title": "Public software bounty",
        "domain": "software",
        "mechanism": "Prepare and test a useful fix against public acceptance criteria",
        "eligibility": "Operator account and eligibility still unverified",
        "blockers": "Payment and maintainer acceptance unverified; no submission authorized",
        "next_test": "Reproduce the issue and prepare a locally tested patch",
        "failure_condition": "Reject if terms require upfront spending or criteria cannot be reproduced",
        "gross_pln_low": 0,
        "gross_pln_high": 1000,
        "total_cost_pln": 20,
        "labor_hours": 8,
        "first_income_days": 7,
        "upfront_spend_pln": 0,
        "evidence": [identity],
    }


def evidence(tmp_path, monkeypatch):
    financial_world(tmp_path)
    monkeypatch.setattr(
        goal_learning.shutil, "disk_usage", lambda root: SimpleNamespace(free=8 * 2**30)
    )
    from rlm.v100 import research_tools

    monkeypatch.setattr(
        research_tools,
        "download_page",
        lambda url, max_bytes: (url, "Public bounty terms. Acceptance and payment conditional."),
    )
    return goal_learning.observe(tmp_path, "https://example.org/public-bounty")["id"]


def test_income_hypothesis_uses_real_host_archive_but_never_certifies_payment(
    tmp_path, monkeypatch
):
    identity = evidence(tmp_path, monkeypatch)
    row = income_opportunities.register(tmp_path, "A", specification(identity))
    assert row["estimated_net_pln_low"] == -20 and row["actual_income_pln"] is None
    assert "unverified" in row["specification"]["eligibility"]
    assert income_opportunities.status(tmp_path)["candidates"][0]["id"] == row["id"]
    # Updating the same mechanism keeps a single candidate and immutable previous versions.
    updated = specification(identity) | {"labor_hours": 10}
    assert income_opportunities.register(tmp_path, "B", updated)["id"] == row["id"]
    assert len(income_opportunities.status(tmp_path)["candidates"]) == 1
    assert (
        len(list((tmp_path / "research/income-opportunities/versions" / row["id"]).glob("*.json")))
        == 2
    )


@pytest.mark.parametrize(
    "change",
    [
        {"upfront_spend_pln": 5},
        {"labor_hours": 0},
        {"gross_pln_high": float("nan")},
        {"gross_pln_low": 5, "gross_pln_high": 4},
        {"evidence": ["a" * 64]},
        {"claimed_profit": 500},
    ],
)
def test_income_registration_rejects_fabricated_evidence_and_invalid_costs(
    tmp_path, monkeypatch, change
):
    identity = evidence(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        income_opportunities.register(tmp_path, "A", specification(identity) | change)
    assert income_opportunities.status(tmp_path)["candidates"] == []


def test_income_registration_rejects_stale_sources(tmp_path, monkeypatch):
    identity = evidence(tmp_path, monkeypatch)
    acquired = goal_learning.observation(tmp_path, identity)["acquired"]
    monkeypatch.setattr(
        income_opportunities, "time", SimpleNamespace(time=lambda: acquired + 86401)
    )
    with pytest.raises(ValueError, match="24 hours"):
        income_opportunities.register(tmp_path, "A", specification(identity))


def test_income_job_is_bounded_periodic_deduplicated_and_goal_attributed(tmp_path):
    goal, run = financial_world(tmp_path)
    first = income_opportunities.commission(tmp_path)
    assert first["state"] == "scheduled" and first["goal_id"] == goal["id"]
    assert first == income_opportunities.commission(tmp_path)
    from rlm.v100.drones import inspect

    jobs = inspect(tmp_path)
    assert len(jobs) == 1 and jobs[0]["interval"] == 600
    assert "outside markets" in jobs[0]["assignment"]
    atomic_json(run / "input-profile.json", {"resources": {"paper_research_enabled": False}})
    assert income_opportunities.commission(tmp_path)["state"] == "blocked"


def test_income_ui_plans_cannot_sneak_in_by_mentioning_forecasts(tmp_path):
    financial_world(tmp_path)
    with pytest.raises(ValueError, match="UI-only"):
        planning.update(tmp_path, "short", "Fix dashboard layout to show income forecasts", "A")
    planning.realign_income_plans(tmp_path)
    assert all(
        "dashboard" not in planning.read(tmp_path)[key]["text"].lower() for key in ("short", "mid")
    )


def test_helper_research_preserves_operator_50_percent_pacing(tmp_path):
    helper = SimpleNamespace(
        sampling_args={"max_tokens": 1024},
        context_window=32768,
        enable_thinking=False,
        helper_batch_tokens=16,
        helper_duty_percent=50,
    )
    assert research_policy.apply(helper, tmp_path).helper_duty_percent == 50


def test_native_launch_skips_second_cli_and_binds_receipt_to_real_child(tmp_path, monkeypatch):
    settings = profile(tmp_path)
    # Test transport only, not CUDA or model quality. The native process stand-in
    # accepts real argv and serves real HTTP while the controller hashes artifacts.
    binary = Path(settings["server"]["binary"])
    binary.write_text(
        f"#!{sys.executable}\n"
        + """
import sys, json
from http.server import BaseHTTPRequestHandler, HTTPServer
port = int(sys.argv[sys.argv.index('--port')+1])
model = sys.argv[sys.argv.index('--model')+1]
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        data = json.dumps({'model_path': model}).encode()
        self.send_response(200); self.end_headers(); self.wfile.write(data)
HTTPServer(('127.0.0.1', port), Handler).serve_forever()
"""
    )
    binary.chmod(0o755)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    settings["runtime"].update(base_url=f"http://127.0.0.1:{port}", max_timeout=10)
    settings["server"]["library_path"] = str(tmp_path)
    source = tmp_path / "profile.json"
    atomic_json(source, settings)
    from rlm.v100 import serving

    monkeypatch.setattr(serving, "process_identity", lambda pid: "test-process-identity")
    log = tmp_path / "server.log"
    with competition.managed_server(source, tmp_path, log) as actual:
        receipt = json.loads(receipt_path(actual, tmp_path).read_text())
        startup = json.loads(log.with_suffix(".startup.json").read_text())
        assert startup["stage"] == "ready" and startup["pid"] == receipt["pid"]
        assert "Starting native server directly" in log.read_text()
        assert '"rlm.v100.cli"' not in log.read_text()
        import requests

        class Client:
            def request(self, endpoint):
                return requests.get(settings["runtime"]["base_url"] + endpoint, timeout=2).json()

        assert_served_expert(Client(), actual, tmp_path)


def test_dashboard_shows_candidates_and_server_startup():
    from rlm.v100.dashboard_layout import APP_SCRIPT as SCRIPT

    assert "incomePanel(data)" in SCRIPT
    assert "No payment" in SCRIPT or "no payment ledger" in SCRIPT
    assert "server_startup?.stage" in SCRIPT


def test_actual_source_fetches_separate_unchanged_content_from_new_content(tmp_path, monkeypatch):
    evidence(tmp_path, monkeypatch)
    from rlm.v100.source_receipts import record

    record(tmp_path, "https://example.org/other", "a" * 64)
    same = record(tmp_path, "https://example.org/other", "a" * 64)
    new = record(tmp_path, "https://example.org/other", "b" * 64)
    value = json.loads((tmp_path / "research/source-acquisition/status.json").read_text())
    assert same["content_state"] == "unchanged reread"
    assert new["content_state"] == "new content for this goal"
    assert value["distinct_urls"] == 2 and value["distinct_contents"] == 3
    assert value["total_fetches"] == 4 and value["unchanged_rereads"] == 1


def test_concurrent_source_counter_snapshot_cannot_regress(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from rlm.v100.source_receipts import record

    financial_world(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: record(tmp_path, "https://example.org/a", "a" * 64), range(12)))
    value = json.loads((tmp_path / "research/source-acquisition/status.json").read_text())
    assert value["total_fetches"] == 12 and value["unchanged_rereads"] == 11


def test_live_evidence_has_event_identity_and_source_receipt_not_just_shared_journal(tmp_path):
    from rlm.v100.activity import ActivityLog
    from rlm.v100.live_status import recent_events

    event_id = ActivityLog(tmp_path, "A", "researcher").write(
        "tools",
        "tool-result",
        {
            "tool": "observe_goal_source",
            "result": {"id": "a" * 64, "url": "https://example.org/a", "body_sha256": "b" * 64},
        },
        step_id="actual-step",
    )
    event = recent_events(tmp_path)[0]
    assert event["id"] == event_id and event["step_id"] == "actual-step"
    assert event["evidence_ref"]["url"] == "https://example.org/a"
    assert event["evidence_ref"]["body_sha256"] == "b" * 64
