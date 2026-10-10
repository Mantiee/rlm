import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlm.v100 import competition, drones, income_work, research_contract
from rlm.v100.common import atomic_json
from tests.test_v100_income_runtime import evidence, specification
from tests.test_v100_recovery import financial_world


def note(hypothesis, test, trace=None):
    return {"hypothesis": hypothesis, "suggested_test": test, "research_trace": trace or []}


def test_rejects_observed_non_market_hypothesis_with_btc_test(tmp_path):
    financial_world(tmp_path)
    row = note(
        "Micro-tasks yield faster first income than trading", "Run BTC/USD candle simulation"
    )
    verdict = research_contract.assessment(tmp_path, row)
    assert verdict["state"] == "rejected test" and not verdict["eligible_for_review"]
    assert "cannot establish" in verdict["reason"]


def test_repeated_notes_do_not_count_as_work_but_receipts_survive(tmp_path):
    financial_world(tmp_path)
    row = note("Software bounty may pay", "Reproduce software acceptance criteria")
    first = research_contract.assessment(tmp_path, row)
    assert first["state"].startswith("advisory only")
    second = research_contract.assessment(tmp_path, row)
    assert second["state"] == "repeated advisory" and not second["eligible_for_review"]
    row["research_trace"] = [{"tool": "register_income_opportunity", "result": {"id": "candidate"}}]
    assert research_contract.assessment(tmp_path, row)["state"] == "executed research receipt"
    row["research_trace"][0]["result"]["status"] = "failed"
    assert not research_contract.assessment(tmp_path, row)["work_receipts"]


def test_quality_migration_retains_text_and_backup(tmp_path):
    financial_world(tmp_path)
    from rlm.v100.experiments import SharedLab

    lab = SharedLab(tmp_path / "research/state/competition.sqlite3")
    original = note("Affiliate earnings arrive earlier", "Backtest BTC candles")
    lab.append("A", "worker-result", original)
    lab.close()
    result = research_contract.repair_history(tmp_path)
    assert result["annotated"] == 1
    from pathlib import Path

    assert Path(result["backup"]).is_file()
    lab = SharedLab(tmp_path / "research/state/competition.sqlite3")
    row = lab.recent()[0]["payload"]
    lab.close()
    assert row["hypothesis"] == original["hypothesis"]
    assert row["research_quality"]["state"] == "rejected test"
    assert research_contract.repair_history(tmp_path)["annotated"] == 0


def test_operator_disabled_gpu_never_contacts_remote_helper(tmp_path, monkeypatch):
    atomic_json(tmp_path / "research/user-preferences.json", {"remote_helper_enabled": False})
    profile = {"resources": {"device": "remote"}, "runtime": {}}
    atomic_json(tmp_path / "remote.json", profile)
    monkeypatch.setattr(competition, "load_profile", lambda *args: profile)
    monkeypatch.setattr(
        competition, "managed_server", lambda *args: pytest.fail("Disabled RTX contacted")
    )
    fallback = {"accepted": "V100"}
    with competition.waiting_researcher(
        tmp_path / "remote.json", tmp_path, tmp_path / "log", fallback=fallback
    ) as actual:
        assert actual is fallback
    with pytest.raises(RuntimeError, match="disabled"):
        competition.helper_client(profile, tmp_path)
    with pytest.raises(RuntimeError, match="disabled"):
        with competition.waiting_researcher(tmp_path / "remote.json", tmp_path, tmp_path / "log"):
            pass


def income_job(goal, index=0):
    return {"id": f"job-{index}", "branch": "A", "kind": "income", "payload": goal["id"]}


def test_income_pipeline_collects_rotating_sources_even_without_inference(tmp_path, monkeypatch):
    identity = evidence(tmp_path, monkeypatch)
    from rlm.v100.goals import load_goal

    goal = load_goal(tmp_path)
    from rlm.v100.goal_learning import observation

    archived = observation(tmp_path, identity)
    urls = []

    def collect(root, url):
        urls.append(url)
        return {"id": identity, **archived}

    monkeypatch.setattr(income_work, "observe", collect)
    monkeypatch.setattr(
        income_work,
        "accepted_master",
        lambda *args: (_ for _ in ()).throw(RuntimeError("Master busy")),
    )
    first = drones.execute(tmp_path, income_job(goal))
    second = drones.execute(tmp_path, income_job(goal, 1))
    assert first["state"] == second["state"] == "evidence collected; analysis blocked"
    assert len(set(urls)) == 2
    assert second["sources_collected"] == 2 and second["actual_income_pln"] is None
    from pathlib import Path

    assert Path(first["dossier"]).exists()


def test_income_pipeline_registers_real_archived_proposal_with_host_domain(tmp_path, monkeypatch):
    identity = evidence(tmp_path, monkeypatch)
    from rlm.v100.goal_learning import observation
    from rlm.v100.goals import load_goal

    goal = load_goal(tmp_path)
    monkeypatch.setattr(
        income_work, "observe", lambda *args: {"id": identity, **observation(tmp_path, identity)}
    )
    monkeypatch.setattr(income_work, "accepted_master", lambda *args: SimpleNamespace())
    proposal = specification(identity)
    proposal.pop("evidence")
    proposal["domain"] = "model cannot relabel this source"
    monkeypatch.setattr(
        income_work, "native_turn", lambda *args, **kwargs: {"content": json.dumps(proposal)}
    )
    result = drones.execute(tmp_path, income_job(goal))
    assert result["state"] == "feasibility dossier prepared" and result["proposals_registered"] == 1
    from rlm.v100.income_opportunities import status

    candidate = status(tmp_path)["candidates"][0]
    assert candidate["specification"]["domain"] == income_work.STARTING_SOURCES[0][0]
    assert candidate["specification"]["evidence"] == [identity]
    assert candidate["actual_income_pln"] is None


@pytest.mark.parametrize(
    "change",
    [{"next_test": "Backtest BTC candles"}, {"gross_pln_low": 10}, {"upfront_spend_pln": 1}],
)
def test_income_pipeline_rejects_bad_tests_or_unearned_payment_floor(tmp_path, monkeypatch, change):
    identity = evidence(tmp_path, monkeypatch)
    from rlm.v100.goal_learning import observation
    from rlm.v100.goals import load_goal

    monkeypatch.setattr(
        income_work, "observe", lambda *args: {"id": identity, **observation(tmp_path, identity)}
    )
    monkeypatch.setattr(income_work, "accepted_master", lambda *args: SimpleNamespace())
    proposal = specification(identity) | change
    proposal.pop("evidence")
    monkeypatch.setattr(
        income_work, "native_turn", lambda *args, **kwargs: {"content": json.dumps(proposal)}
    )
    result = drones.execute(tmp_path, income_job(load_goal(tmp_path)))
    assert result["state"] == "evidence collected; analysis blocked"
    assert result["proposals_registered"] == 0 and result["error"]


def test_disabled_income_or_stale_goal_never_fetches(tmp_path, monkeypatch):
    goal, run = financial_world(tmp_path)
    monkeypatch.setattr(
        income_work, "observe", lambda *args: pytest.fail("Unauthorized source fetched")
    )
    with pytest.raises(ValueError, match="another goal"):
        drones.execute(tmp_path, income_job({"id": "old-goal"}))
    atomic_json(run / "input-profile.json", {"resources": {"paper_research_enabled": False}})
    with pytest.raises(ValueError, match="disabled"):
        drones.execute(tmp_path, income_job(goal))


def test_native_researcher_checks_protected_profile_before_any_generation(tmp_path, monkeypatch):
    _, run = financial_world(tmp_path)
    atomic_json(run / "status.json", {"context_window": 8192})
    atomic_json(run / "profile-8192.json", {"runtime": {"model_name": "accepted"}})
    profile = {"runtime": {"model_name": "accepted"}}
    monkeypatch.setattr(income_work, "load_profile", lambda *args: profile)
    client = SimpleNamespace(timeout=120, sampling_args={"max_tokens": 4096})
    monkeypatch.setattr(competition, "helper_client", lambda *args: client)
    from rlm.v100 import serving

    called = []
    monkeypatch.setattr(serving, "assert_served_expert", lambda c, p, r: called.append(p))
    assert income_work.accepted_master(tmp_path, "A") is client
    assert called == [profile] and client.enable_thinking is False
    assert client.timeout == 40 and client.sampling_args["max_tokens"] == 1024
    monkeypatch.setattr(
        serving,
        "assert_served_expert",
        lambda *args: (_ for _ in ()).throw(ValueError("Candidate serves endpoint")),
    )
    with pytest.raises(ValueError, match="Candidate"):
        income_work.accepted_master(tmp_path, "A")


def benchmark_world(tmp_path, monkeypatch):
    from rlm.v100 import public_benchmarks
    from rlm.v100.protection import file_hash

    snapshot = tmp_path / "research/public-benchmarks/snapshot"
    questions = [
        {"category": "language", "task": "task", "question_id": str(index)} for index in range(3)
    ]
    atomic_json(snapshot / "questions.json", questions)
    atomic_json(snapshot / "references.json", [])
    atomic_json(
        snapshot / "manifest.json",
        {
            "questions_sha256": file_hash(snapshot / "questions.json"),
            "references_sha256": file_hash(snapshot / "references.json"),
        },
    )
    model = tmp_path / "model.gguf"
    model.write_bytes(b"same original model")
    profile = {"server": {"model": str(model)}}
    monkeypatch.setattr(public_benchmarks, "current", lambda root: snapshot)
    monkeypatch.setattr(public_benchmarks, "generation_conditions", lambda p: {"max_tokens": 123})
    identity = {
        "snapshot_sha256": file_hash(snapshot / "manifest.json"),
        "model_sha256": file_hash(model),
        "generation": {"max_tokens": 123},
    }
    report = {
        "identity": identity,
        "cases": [
            {
                "key": public_benchmarks.question_key(questions[0]),
                "category": "language",
                "score": 0,
            }
        ],
    }
    return public_benchmarks, profile, report


def test_restart_resumes_matching_partial_baseline_without_dropping_failure(tmp_path, monkeypatch):
    module, profile, report = benchmark_world(tmp_path, monkeypatch)
    source = tmp_path / "research/mission/run-old/public-baseline.partial.json"
    output = tmp_path / "research/mission/run-new/public-baseline.json"
    atomic_json(source, report)
    assert module.resume_baseline(tmp_path, profile, output) == source
    assert json.loads(output.with_suffix(".partial.json").read_text())["cases"][0]["score"] == 0
    assert source.exists() and not output.exists()
    receipt = json.loads(output.with_suffix(".resume.json").read_text())
    assert receipt["completed"] == 1 and receipt["total"] == 3


@pytest.mark.parametrize("mutation", ["model", "generation", "duplicate", "wrong-category", "nan"])
def test_resume_rejects_different_conditions_and_invalid_rows(tmp_path, monkeypatch, mutation):
    module, profile, report = benchmark_world(tmp_path, monkeypatch)
    if mutation == "model":
        report["identity"]["model_sha256"] = "another model"
    elif mutation == "generation":
        report["identity"]["generation"]["max_tokens"] = 456
    elif mutation == "duplicate":
        report["cases"] *= 2
    elif mutation == "wrong-category":
        report["cases"][0]["category"] = "coding"
    else:
        report["cases"][0]["score"] = float("nan")
    source = tmp_path / "research/mission/run-old/public-baseline.partial.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(json.dumps(report))
    output = tmp_path / "research/mission/run-new/public-baseline.json"
    assert module.resume_baseline(tmp_path, profile, output) is None
    assert not output.with_suffix(".partial.json").exists()


def test_benchmark_finishing_preserves_case_rows_and_writes_progress(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from rlm.v100 import competition, serving

    module, profile, report = benchmark_world(tmp_path, monkeypatch)
    snapshot = module.current(tmp_path)
    questions = json.loads((snapshot / "questions.json").read_text())
    for question in questions:
        question["turns"] = ["prompt"]
    atomic_json(snapshot / "questions.json", questions)
    manifest = json.loads((snapshot / "manifest.json").read_text())
    from rlm.v100.protection import file_hash

    manifest["questions_sha256"] = file_hash(snapshot / "questions.json")
    manifest["scope"] = "pinned official test"
    atomic_json(snapshot / "manifest.json", manifest)
    profile["runtime"] = {"model_version": "original"}
    client = SimpleNamespace(
        completion=lambda messages: "answer", get_response_info=lambda: {"finish_reason": "stop"}
    )
    monkeypatch.setattr(competition, "helper_client", lambda p: client)
    monkeypatch.setattr(serving, "assert_served_expert", lambda *args: None)
    monkeypatch.setattr(module, "grade", lambda *args: {"score": 0.5})
    output = tmp_path / "new/public-baseline.json"
    value = module.evaluate(tmp_path, profile, output, snapshot)
    assert isinstance(value["cases"], list) and len(value["cases"]) == 3
    assert value["case_count"] == 3 and value["complete"]
    assert module.evaluate(tmp_path, profile, output, snapshot)["cases"] == value["cases"]
    progress = json.loads((tmp_path / "research/public-benchmarks/progress.json").read_text())
    assert progress["state"] == "finished" and progress["completed"] == 3


def test_resume_ignores_old_integer_case_bug_retains_matching_partial(tmp_path, monkeypatch):
    module, profile, report = benchmark_world(tmp_path, monkeypatch)
    source = tmp_path / "research/mission/run-old/public-baseline.partial.json"
    atomic_json(source, report)
    atomic_json(source.with_name("public-baseline.json"), {**report, "cases": 20})
    output = tmp_path / "research/mission/run-new/public-baseline.json"
    assert module.resume_baseline(tmp_path, profile, output) == source


def test_deltas_do_not_displace_actions_and_new_request_clears_previous_error(tmp_path):
    from rlm.v100 import live_status

    folder = tmp_path / "research/logs/activity/2026-10-10"
    folder.mkdir(parents=True)
    rows = [
        {
            "time": "2026-10-10T01:00:00Z",
            "actor": "researcher",
            "branch": "A",
            "kind": "tool-result",
            "payload": {"result": {"error": "old failure"}},
        },
        {
            "time": "2026-10-10T01:00:01Z",
            "actor": "researcher",
            "branch": "A",
            "kind": "inference-start",
            "payload": {},
        },
    ]
    rows += [
        {
            "time": f"2026-10-10T01:00:{n:02}Z",
            "actor": "researcher",
            "branch": "A",
            "kind": "inference-delta",
            "payload": {"text": "output", "channel": "content"},
        }
        for n in range(2, 40)
    ]
    (folder / "timeline.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    events = live_status.recent_events(tmp_path, limit=3)
    assert len(live_status.action_events(events)) == 2
    view = live_status.agent_views(events, [])[0]
    assert not view.get("result")
    job = {
        "id": "job",
        "branch": "B",
        "kind": "critic",
        "state": "queued",
        "updated": 1,
        "result": {"error": "previous attempt"},
    }
    card = live_status.agent_views([], [job])[0]
    assert card["result"] == "" and "previous attempt" in card["previous_result"]


def test_disabled_rtx_dashboard_never_probes_remote(tmp_path, monkeypatch):
    import psutil

    from rlm.v100 import competition, live_status

    atomic_json(tmp_path / "research/user-preferences.json", {"remote_helper_enabled": False})
    monkeypatch.setattr(competition, "helper_client", lambda *a, **k: pytest.fail("RTX probe"))
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(available=4 * 2**30, total=8 * 2**30)
    )
    monkeypatch.setattr(psutil, "cpu_percent", lambda: 2)
    value = live_status.snapshot(tmp_path, {"running": True, "state": {}}, {}, {})
    card = next(actor for actor in value["actors"] if actor["id"] == "rtx")
    assert card["state"] == "disabled by operator" and not card["jobs"] and not card["stale"]


def test_fee_parse_failure_archives_page_and_does_not_block_price_poll(tmp_path, monkeypatch):
    from rlm.v100 import spot_bootstrap

    state = {
        "currency": "PLN",
        "instruments": {"BTC-KRAKEN": {"feed_id": "kraken:XBTUSD", "fee_profile": "old"}},
        "fee_profiles": {"old": {"valid_until": "2020-01-01T00:00:00+00:00"}},
    }
    calls = []

    class Book:
        def __init__(self, root):
            pass

        def state(self):
            return state

        def close(self):
            calls.append("close")

    monkeypatch.setattr(spot_bootstrap, "PaperBook", Book)
    monkeypatch.setattr(spot_bootstrap, "poll_crypto", lambda book: calls.append("prices"))
    monkeypatch.setattr(
        spot_bootstrap, "download_page", lambda url, **k: (url, "<p>Unexpected fee layout</p>")
    )
    with pytest.raises(ValueError, match="archived source"):
        spot_bootstrap.prepare(tmp_path, refresh=True)
    assert calls == ["prices", "close"]
    receipt = json.loads((tmp_path / "research/paper/fee-source-status.json").read_text())
    assert receipt["state"] == "blocked" and Path(receipt["snapshot"]).exists()
