import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from rlm.v100 import (
    architectures,
    chat_facts,
    dashboard_layout,
    drones,
    goals,
    income_policy,
    live_status,
    market_research,
    mission_chat,
    ngram_memory,
    progress,
    research_tools,
)
from tests.test_v100_architectures import workload
from tests.test_v100_backtesting import candles
from tests.test_v100_recovery import financial_world


@pytest.mark.parametrize(
    "message",
    [
        "a co z zarabianiem",
        "czemu minus -15% w backtest",
        "gdzie kandydaci i paper",
        "what about profit",
    ],
)
def test_financial_question_needs_no_model_and_cannot_change_plans(tmp_path, monkeypatch, message):
    monkeypatch.setattr(
        progress,
        "snapshot",
        lambda root: {
            "phase": "research",
            "paper": {"executed_fills": 0},
            "paper_blockers": ["expired fees"],
            "recent_exploratory_backtests": [
                {
                    "parameters": {"rule": "momentum"},
                    "development_test": {"net_return": -0.15},
                    "report": "/reports/loss.json",
                }
            ],
        },
    )
    result = mission_chat.respond(tmp_path, tmp_path, {"message": message})
    assert result["actions"] == result["applied"] == []
    assert "-15.00%" in result["answer"] and "/reports/loss.json" in result["answer"]
    assert "expired fees" in result["answer"]
    assert result["responder"]["model"] == "controller-financial-evidence"


@pytest.mark.parametrize(
    "message",
    [
        "nie zmien na kapital",
        "do not enable capital research",
        "czy zmienic na kapital?",
        "example: enable capital research",
    ],
)
def test_policy_rejects_non_authorization(message):
    assert income_policy.setting(message) is None


def test_capital_research_is_not_spending_and_disable_cancels_job(tmp_path):
    goal, _ = financial_world(tmp_path)
    receipt = income_policy.update(tmp_path, "Enable hypothetical capital research")
    assert receipt["capital_research"] and not receipt["spending"] and not receipt["real_orders"]
    jobs = drones.inspect(tmp_path)
    assert jobs[0]["kind"] == "market-research" and jobs[0]["assignment"] == goal["id"]
    income_policy.update(tmp_path, "Disable capital research")
    assert drones.inspect(tmp_path)[0]["state"] == "cancelled"
    assert len(list((tmp_path / "research/income-policy-history").glob("*.json"))) == 2


def test_historical_comparison_same_source_real_returns_and_tamper_rejected(tmp_path, monkeypatch):
    financial_world(tmp_path)
    calls = []
    monkeypatch.setattr(
        research_tools,
        "download_page",
        lambda url: calls.append(url) or (url, json.dumps(candles())),
    )
    value = market_research.compare(tmp_path)
    assert value["state"] == "completed" and len(calls) == 1
    assert len(value["comparisons"]) == 2
    assert value["comparisons"][0]["sources"] == value["comparisons"][1]["sources"]
    assert all(Path(row["report"]).exists() for row in value["comparisons"])
    assert all(not row["triage"]["profit_verified"] for row in value["comparisons"])
    assert market_research.compare(tmp_path)["cache_reused"] and len(calls) == 1
    Path(value["comparisons"][0]["report"]).write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        market_research.compare(tmp_path)


def test_source_failure_has_no_fabricated_roi(tmp_path, monkeypatch):
    financial_world(tmp_path)
    monkeypatch.setattr(
        research_tools, "download_page", lambda url: (_ for _ in ()).throw(ValueError("HTTP 403"))
    )
    result = chat_facts.test_response(tmp_path, "przetestuj teraz strategie")
    assert "HTTP 403" in result["answer"]
    assert result["applied"][0]["comparisons"] == [] and "0.00%" not in result["answer"]


@pytest.mark.parametrize("net,fills", [(0, 0), (-0.15, 10), (None, 1)])
def test_no_trade_loss_and_missing_metrics_are_rejected(net, fills):
    result = market_research.triage({"development_test": {"net_return": net, "fills": fills}})
    assert result["state"] == "rejected" and not result["profit_verified"]


def test_legacy_chat_migration_safe_with_concurrent_readers(tmp_path):
    path = tmp_path / "research/state/user-chat.sqlite3"
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE requests(id TEXT PRIMARY KEY, created TEXT NOT NULL, message TEXT NOT NULL, state TEXT NOT NULL, response TEXT, attempts INTEGER NOT NULL DEFAULT 0, error TEXT, updated REAL NOT NULL DEFAULT 0)"
        )

    def read(_):
        with mission_chat.connect(tmp_path) as db:
            return "wait_started" in {row[1] for row in db.execute("PRAGMA table_info(requests)")}

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(pool.map(read, range(8)))


def test_readiness_wait_releases_next_question_and_has_finite_budget(tmp_path, monkeypatch):
    first = mission_chat.submit(tmp_path, "waiting")
    second = mission_chat.submit(tmp_path, "facts")
    seen = []

    def respond(root, run, request):
        seen.append(request["id"])
        if request["id"] == first:
            raise mission_chat.BackendNotReady("waiting for accepted weights")
        return {"answer": "facts", "actions": [], "applied": []}

    monkeypatch.setattr(mission_chat, "respond", respond)

    class Stop:
        def wait(self, seconds):
            return len(seen) >= 2

    mission_chat.service_loop(tmp_path, tmp_path, Stop())
    assert mission_chat.inspect(tmp_path, first)["wait_started"] > 0
    assert mission_chat.inspect(tmp_path, second)["state"] == "completed"
    with mission_chat.connect(tmp_path) as db:
        db.execute("UPDATE requests SET attempts=19,updated=0 WHERE id=?", (first,))
    seen.clear()

    class One:
        def wait(self, seconds):
            return bool(seen)

    mission_chat.service_loop(tmp_path, tmp_path, One())
    assert mission_chat.inspect(tmp_path, first)["state"] == "failed"


def test_full_assignment_is_distinct_from_response_token_exhaustion(tmp_path):
    drones.schedule(tmp_path, "A", "researcher", "x" * 400, 0)
    assert drones.inspect(tmp_path)[0]["assignment"] == "x" * 400
    events = [
        {"branch": "A", "actor": "master", "kind": "inference-start"},
        {"branch": "A", "actor": "master", "kind": "model-output", "finish_reason": "length"},
        {
            "branch": "A",
            "actor": "master",
            "kind": "inference-finished",
            "usage": {"completion_tokens": 512},
        },
    ]
    card = live_status.agent_views(events, [])[0]
    assert card["state"] == "incomplete response" and card["finish_reason"] == "length"
    assert card["usage"]["completion_tokens"] == 512


def test_publication_cannot_force_reload_and_ui_is_english():
    assert "location.reload();return" not in dashboard_layout.APP_SCRIPT
    assert "pendingLayout?location.reload():refresh()" in dashboard_layout.APP_SCRIPT
    assert "Historical development backtests" in dashboard_layout.BASE_TEMPLATE


def test_action_receipts_distinguish_failed_queued_and_read_only():
    rows = [
        {"tool": "read_dashboard", "result": {"source": "HTML"}},
        {"tool": "schedule_drone", "result": {"status": "queued", "id": "one"}},
        {"tool": "sandbox_run", "result": {"exit_code": 1, "stderr": "error"}},
    ]
    receipts = mission_chat.execution_receipts(rows)
    assert [r["state"] for r in receipts] == ["read-only", "queued; not execution", "failed"]


def test_ngram_pilot_causal_identity_initialization_train_reload_and_goal_binding(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file, save_file

    torch.set_num_threads(1)
    _, suite = workload(tmp_path)
    goal = goals.set_goal(tmp_path, "Learn the recorded task", suite)
    receipt = ngram_memory.propose(
        tmp_path, "A", "memory", 16, 1, 256, 3, "Test useful conditional memory"
    )
    assert receipt["shape"]["goal_id"] == goal["id"] and not receipt["weights_changed"]
    assert not receipt["windows_rtx_started"] and receipt["memory_table_mib"] < 1
    namespace = {}
    exec(
        (architectures.candidate_path(tmp_path, "memory") / "source/model.py").read_text(),
        namespace,
    )
    model = namespace["build"]({}).eval()
    x = torch.tensor([[1, 2, 3, 4]])
    y = torch.tensor([[1, 2, 200, 201]])
    before = model(x).detach()
    model.memory_enabled = False
    assert torch.equal(before, model(x))
    model.memory_enabled = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(3):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x).reshape(-1, 257), x.reshape(-1))
        loss.backward()
        optimizer.step()
    assert model.value.weight.abs().sum() > 0
    assert torch.allclose(model(x)[:, :2], model(y)[:, :2], atol=1e-6)
    file = tmp_path / "memory.safetensors"
    save_file(model.state_dict(), str(file))
    copy = namespace["build"]({})
    copy.load_state_dict(load_file(str(file)))
    assert torch.equal(model(x), copy(x))


@pytest.mark.parametrize(
    "slots,order,width", [(0, 3, 16), (8193, 3, 16), (256, 1, 16), (8192, 4, 128), (True, 3, 16)]
)
def test_ngram_budget_rejected_before_allocation(tmp_path, slots, order, width):
    _, suite = workload(tmp_path)
    goals.set_goal(tmp_path, "Task", suite)
    with pytest.raises(ValueError):
        ngram_memory.propose(tmp_path, "A", "bad", width, 1, slots, order, "Hypothesis")
    assert not (tmp_path / "research/architecture-candidates/bad").exists()


@pytest.mark.parametrize("improved", [True, False])
def test_ngram_trial_measures_ablation_without_exposing_gold(tmp_path, monkeypatch, improved):
    pool, suite = workload(tmp_path)
    goals.set_goal(tmp_path, "Improve held-out task", suite)
    ngram_memory.propose(tmp_path, "A", "pilot", 16, 1, 256, 3, "Measure conditional memory")
    phases = []

    def phase(output, run, inputs, budget, mode):
        config = json.loads((inputs / "config.json").read_text())
        phases.append((mode, config.get("ablate_ngram_memory", False)))
        if mode == "train":
            (run / "weights/weights.safetensors").write_bytes(b"frozen weights")
        else:
            data = json.loads((inputs / "data.json").read_text())
            assert not any("expected" in row for row in data)
            answer = "wrong" if improved and config.get("ablate_ngram_memory") else "17"
            (run / "predict.log").write_text(
                json.dumps({"id": "held-out", "answer": answer}) + "\n"
            )
        return 1.0

    monkeypatch.setattr(architectures, "phase", phase)
    result = architectures.run_candidate(tmp_path, "pilot", pool, suite)
    assert phases == [("train", False), ("predict", False), ("predict", True)]
    assert result["memory_ablation"]["development_improvement"] == improved
    assert result["passed_cases"] == 1
    assert (tmp_path / "research/architecture-candidates/pilot/trial/memory-disabled.log").exists()
