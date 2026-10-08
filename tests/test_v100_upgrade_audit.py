import copy
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlm.v100 import (
    backtesting,
    income_skills,
    mission_memory,
    mtp_gate,
    research_policy,
    research_tools,
    semantic,
    spot_bootstrap,
)
from rlm.v100.calculator import calculate
from rlm.v100.common import atomic_json
from rlm.v100.insights import verified_record, verify_record
from rlm.v100.paper import PaperBook
from rlm.v100.protection import file_hash
from tests.test_v100_backtesting import candles
from tests.test_v100_duel import decision, profile, records
from tests.test_v100_paper import Clock, configuration, decide, quote


@pytest.mark.parametrize(
    "expression",
    ['__import__("os")', "2**1000", "1/0", "1e99", "True+1", "0xFF", "1e-90", "[", "1+"],
)
def test_calculator_rejects_execution_and_unbounded_values(expression):
    with pytest.raises(ValueError):
        calculate(expression)


def test_decimal_reference_is_independent_and_forgery_rejected():
    assert calculate("0.1+0.2")["result"] == "0.3"
    row = verified_record(
        {"kind": "decimal_calculation", "expression": "1000*(1-80/10000)-1000*(1+80/10000)"}
    )
    assert row["messages"][-1]["content"] == "-16.000"
    verify_record(row)
    row["messages"][-1]["content"] = "100"
    with pytest.raises(ValueError):
        verify_record(row)


def test_adaptive_research_does_not_change_fixed_clients(tmp_path):
    client = SimpleNamespace(
        sampling_args={"max_tokens": 2048}, context_window=32768, enable_thinking=True
    )
    research_policy.choose(tmp_path, "master", False, 8192, 256)
    chosen = research_policy.apply(client, tmp_path)
    assert chosen.enable_thinking is False and chosen.sampling_args["max_tokens"] == 8192
    assert client.enable_thinking is True and client.sampling_args["max_tokens"] == 2048
    helper = SimpleNamespace(
        sampling_args={"max_tokens": 1024},
        context_window=32768,
        enable_thinking=False,
        helper_batch_tokens=64,
        helper_duty_percent=65,
    )
    research_policy.choose(tmp_path, "helper", True, 4096, 16)
    selected = research_policy.apply(helper, tmp_path)
    assert (
        selected.enable_thinking
        and selected.helper_batch_tokens == 16
        and selected.helper_duty_percent == 30
    )
    with pytest.raises(ValueError):
        research_policy.choose(tmp_path, "helper", True, 4096, 64)
    client.research_device = "cpu"
    assert research_policy.apply(client, tmp_path).sampling_args["max_tokens"] == 2048


def test_source_drones_are_bounded_and_really_overlap(tmp_path, monkeypatch):
    barrier = threading.Barrier(2)

    def page(url):
        barrier.wait(timeout=2)
        return url, "<html>" + url + "</html>"

    monkeypatch.setattr(research_tools, "download_page", page)
    result = research_tools.ResearchTools(tmp_path, {}).execute(
        "parallel_source_research", {"urls": ["https://example.org/a", "https://example.org/b"]}
    )
    assert result["workers"] == 2 and len(result["sources"]) == 2
    assert all("error" not in source for source in result["sources"])
    with pytest.raises(ValueError):
        research_tools.ResearchTools(tmp_path, {}).execute(
            "parallel_source_research", {"urls": ["https://example.org"] * 9}
        )


def test_cost_curriculum_reserved_heldout_and_idempotence(tmp_path):
    result = income_skills.prepare(tmp_path, profile(tmp_path))
    assert result["training_records"] == 128 and result["reserved_heldout_cases"] == 32
    folder = tmp_path / "research/income-cost-skills-v1"
    train = [json.loads(line)["group"] for line in (folder / "pool.jsonl").read_text().splitlines()]
    heldout = [
        json.loads(line)["id"] for line in (folder / "development.jsonl").read_text().splitlines()
    ]
    assert not set(train) & set(heldout)
    assert income_skills.prepare(tmp_path, profile(tmp_path)) == result


def test_fee_parser_uses_exact_spot_table_and_rejects_ambiguity():
    html = "<p>Spot Crypto</p><p>Tier 1 $0+ N/A 0.40 % 0.80 %</p><p>Spot Maker Rebate</p>"
    assert spot_bootstrap.fee_bps(html) == "80.00"
    with pytest.raises(ValueError):
        spot_bootstrap.fee_bps(
            html.replace("Spot Maker Rebate", "Tier 1 $0+ N/A 0.10% 0.20% Spot Maker Rebate")
        )
    with pytest.raises(ValueError):
        spot_bootstrap.fee_bps("<p>Instant Buy 1%</p>")


@pytest.mark.parametrize("futures", ["", "< 5 mln USD "])
def test_polish_fee_table_preserves_product_and_lowest_volume_scope(futures):
    # Cross-platform and rebate tables must not supply the spot fee.
    row = f"Poziom 1 Ponad 0 USD {futures}NIE DOTYCZY 0,40 % 0,80 %"
    html = (
        "<p>Poziom 1 Ponad 0 USD NIE DOTYCZY 0,10 % 0,20 %</p>"
        "<h2>Krypto spot</h2><p>Maker spot (%) Taker spot (%)</p>"
        f"<p>{row.replace(' ', '&nbsp;')}</p>"
        "<h2>Zwrot opłaty maker w handlu spot</h2>"
        "<p>Poziom 1 Ponad 0 USD NIE DOTYCZY 0,38 % 0,70 %</p>"
    )
    assert spot_bootstrap.fee_bps(html) == "80.00"
    with pytest.raises(ValueError):
        spot_bootstrap.fee_bps(html.replace("<h2>Zwrot", f"<p>{row}</p><h2>Zwrot"))
    with pytest.raises(ValueError):
        spot_bootstrap.fee_bps(html.replace("Ponad", "Od"))


def test_english_fee_table_with_futures_column_and_changed_rates():
    html = (
        "<h2>Spot Crypto</h2><p>Tier 1 $0+ &lt; $5M N/A 0.21 % 0.43 %</p><h2>Spot Maker Rebate</h2>"
    )
    assert spot_bootstrap.fee_bps(html) == "43.00"


def test_spot_prepare_uses_primary_rules_does_not_reset_capital(tmp_path, monkeypatch):
    book = PaperBook(tmp_path)
    book.initialize("10000", "PLN")
    book.close()
    monkeypatch.setattr(
        spot_bootstrap,
        "download_page",
        lambda url, **kwargs: (
            url,
            "<p>Spot Crypto Tier 1 $0+ N/A 0.40 % 0.80 % Spot Maker Rebate</p>",
        ),
    )
    pairs = {
        name: {
            "altname": name,
            "quote": "ZUSD",
            "status": "online",
            "lot_decimals": 8,
            "ordermin": "0.0001",
            "costmin": "0.5",
        }
        for name in ("XBTUSD", "ETHUSD")
    }
    from datetime import UTC, datetime

    monkeypatch.setattr(
        spot_bootstrap,
        "public_json",
        lambda *args: ({"error": [], "result": pairs}, "a" * 64, datetime.now(UTC).isoformat()),
    )
    monkeypatch.setattr(spot_bootstrap, "poll_crypto", lambda book: [])
    result = spot_bootstrap.prepare(tmp_path)
    assert result["real_money_ready"] is False
    book = PaperBook(tmp_path)
    state = book.state()
    assert len(state["instruments"]) == 2
    assert state["branches"]["A"]["cash"] == "10000"
    assert "assumptions" in next(iter(state["fee_profiles"].values()))["cost_evidence"]
    book.close()
    assert spot_bootstrap.prepare(tmp_path)["status"].startswith("existing")


def test_order_below_documented_minimum_is_not_filled(tmp_path):
    clock = Clock()
    book = PaperBook(tmp_path, clock)
    book.initialize()
    config = configuration(clock)
    config["instruments"][0]["minimum_quantity"] = "100"
    book.configure(config)
    book.ingest(quote(clock))
    decide(book)
    clock.advance()
    book.ingest(quote(clock))
    assert not book.state()["branches"]["A"]["positions"]
    book.close()


def test_backtest_cache_and_walk_forward_have_no_future_training(tmp_path, monkeypatch):
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=3600"
    data = candles()
    monkeypatch.setattr(research_tools, "download_page", lambda target: (url, json.dumps(data)))
    args = {
        "price_url": url,
        "event_url": "",
        "rule": "momentum",
        "fee_bps": 40,
        "slippage_bps": 10,
    }
    first = backtesting.run(tmp_path, "A", args)
    second = backtesting.run(tmp_path, "B", args)
    assert second["cache_reused"] and first["report"] == second["report"]
    assert [f["start"] for f in first["walk_forward"]] == [50, 65, 80]
    for row in data[50:]:
        row[3] *= 10
        row[4] *= 10
    later = backtesting.run(tmp_path, "B", args)
    assert (
        later["walk_forward"][0]["lookback_selected_only_on_past"]
        == first["walk_forward"][0]["lookback_selected_only_on_past"]
    )
    Path(later["report"]).write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        backtesting.run(tmp_path, "A", args)


def test_semantic_index_batches_preserve_but_exclude_worker_transcripts(tmp_path):
    mission_memory.archive(tmp_path, "worker:A", "Recursive transcript")
    for i in range(6):
        mission_memory.archive(
            tmp_path, "https://example.org/" + str(i), "Public fee data " + str(i)
        )
    memory = mission_memory.store(tmp_path)
    encoder = SimpleNamespace(identity="toy", encode=lambda text: [[1.0, 0.0]])
    assert semantic.index_memory(memory, encoder, max_nodes=4) == 4
    assert semantic.index_memory(memory, encoder, max_nodes=4) == 2
    sources = memory.db.execute(
        "SELECT d.source FROM embeddings e JOIN nodes n ON n.id=e.node_id JOIN documents d ON d.id=n.document_id"
    ).fetchall()
    assert not any(row["source"].startswith("worker:") for row in sources)
    assert memory.db.execute("SELECT count(*) FROM documents").fetchone()[0] == 7
    memory.close()


def test_independent_parents_are_used_in_next_plan(tmp_path, monkeypatch):
    from rlm.v100 import experiments

    data = profile(tmp_path)
    pool = tmp_path / "pool.jsonl"
    pool.write_text("".join(json.dumps(r) + "\n" for r in records()))
    parents = {}
    for branch, rank in (("A", 8), ("B", 32)):
        directory = tmp_path / ("adapter-" + branch)
        directory.mkdir()
        atomic_json(directory / "adapter_config.json", {"r": rank})
        parent = copy.deepcopy(data)
        parent["training"].update(init_adapter=str(directory), teacher_adapter=str(directory))
        parents[branch] = parent
    seen = {}

    def choose(client, branch, catalog, p, recent):
        seen[branch] = p["training"]["init_adapter"]
        result = decision(catalog[0])
        result["parameters"]["rank"] = 8 if branch == "A" else 32
        return result

    monkeypatch.setattr(experiments, "choose_experiment", choose)
    bundle = experiments.plan_duel(
        None, data, pool, tmp_path / "duel", tmp_path, branch_parents=parents
    )
    for branch in ("A", "B"):
        chosen = json.loads(Path(bundle["branches"][branch]["profile"]).read_text())
        assert (
            chosen["training"]["init_adapter"]
            == seen[branch]
            == parents[branch]["training"]["init_adapter"]
        )
        assert chosen["training"]["teacher_adapter"] == seen[branch]


def test_mtp_requires_pinned_quality_and_invalidates_context(tmp_path):
    data = profile(tmp_path)
    target = Path(data["server"]["model"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"target")
    draft = tmp_path / "draft.gguf"
    draft.write_bytes(b"draft")
    data["server"].update(draft_model=str(draft), draft_tokens=4, spec_type="draft-mtp")
    base = {
        "schema": "v100-quality-v1",
        "suite_sha256": "fixed",
        "generation": {},
        "memory_mode": "fixed prompt fixtures; no live retrieval",
        "model_sha256": file_hash(Path(data["server"]["model"])),
        "cases": [{"id": "x", "passed": True}],
    }
    speed = {
        "mtp4": {
            "answers_exact_match": False,
            "median_case_throughput_ratio": 1.2,
            "cases": {"one": {"latency_ratio": 1.1}},
        }
    }
    paths = []
    for name, value in (
        ("quality-baseline.json", base),
        ("quality-candidate.json", base),
        ("comparison.json", speed),
    ):
        path = tmp_path / name
        atomic_json(path, value)
        paths.append(path)
    data.setdefault("resources", {})["mtp_validation"] = {
        "execution": mtp_gate.identity(data),
        "draft_hash": file_hash(draft),
        "draft_tokens": 4,
        "reports": {str(p): file_hash(p) for p in paths},
    }
    assert mtp_gate.valid(data)
    changed = copy.deepcopy(data)
    changed["server"]["context_per_slot"] *= 2
    assert not mtp_gate.valid(changed)
    bad = copy.deepcopy(base)
    bad["cases"][0]["passed"] = False
    atomic_json(paths[1], bad)
    data["resources"]["mtp_validation"]["reports"][str(paths[1])] = file_hash(paths[1])
    assert not mtp_gate.valid(data)


def test_user_chat_queue_controls_and_alerts_survive_restart(tmp_path):
    from rlm.v100 import mission_chat

    identity = mission_chat.submit(tmp_path, "Nie powtarzaj tych samych backtestów")
    assert mission_chat.inspect(tmp_path, identity)["state"] == "queued"
    action = {
        "kind": "directive",
        "text": "Porównuj też usługi bez wpłaty",
        "target": "master",
        "thinking": False,
        "max_tokens": 256,
        "batch_tokens": 128,
        "enabled": False,
    }
    alert = {**action, "kind": "alerts", "enabled": True, "text": "Sygnały paper kupna i sprzedaży"}
    receipt = mission_chat.apply_actions(tmp_path, [action, alert])
    assert len(receipt) == 2
    assert mission_chat.preferences(tmp_path)["directive"] == action["text"]
    mission_chat.emit_alert(tmp_path, "A", {"action": "hold"}, False)
    assert not (tmp_path / "research/alerts/signals.jsonl").exists()
    mission_chat.emit_alert(tmp_path, "B", {"action": "open", "symbol": "TEST"}, True)
    record = json.loads((tmp_path / "research/alerts/signals.jsonl").read_text())
    assert record["rejected"] is True and record["branch"] == "B"
    assert mission_chat.inspect(tmp_path, identity)["state"] == "queued"


def test_chat_invalid_mixed_plan_applies_nothing(tmp_path):
    from rlm.v100 import mission_chat

    action = {
        "kind": "directive",
        "text": "change",
        "target": "master",
        "thinking": False,
        "max_tokens": 256,
        "batch_tokens": 128,
        "enabled": False,
    }
    bad = {**action, "kind": "budget", "target": "helper", "batch_tokens": 512}
    with pytest.raises(ValueError):
        mission_chat.apply_actions(tmp_path, [action, bad])
    assert mission_chat.preferences(tmp_path)["directive"] == ""


def test_chat_service_acknowledges_once_and_closes(tmp_path, monkeypatch):
    from rlm.v100 import mission_chat

    identity = mission_chat.submit(tmp_path, "Co robisz?")
    stop = threading.Event()
    calls = []

    def reply(root, directory, request):
        calls.append(request["id"])
        stop.set()
        return {"answer": "Badam", "applied": []}

    monkeypatch.setattr(mission_chat, "respond", reply)
    mission_chat.service(tmp_path, tmp_path, stop)
    assert calls == [identity]
    assert mission_chat.inspect(tmp_path, identity)["state"] == "completed"


def test_training_calibration_rejects_nonfinite_pilot(tmp_path, monkeypatch):
    from rlm.v100 import competition, mission, training, training_calibration

    data = profile(tmp_path)
    source = tmp_path / "source.json"
    atomic_json(source, data)
    monkeypatch.setattr(mission, "status", lambda root: {"running": False})
    monkeypatch.setattr(competition, "require_idle_gpu", lambda: None)
    rows = records()
    monkeypatch.setattr(training, "load_records", lambda *args: (rows[:2], rows[2:4]))

    def pilot(argv, **kwargs):
        p = json.loads(Path(argv[3]).read_text())
        output = Path(p["training"]["output"])
        output.mkdir()
        loss = float("nan") if p["training"]["precision"] == "nf4" else 2.0
        (output / "metrics.jsonl").write_text(
            json.dumps({"loss": loss, "peak_vram_gib": 20.0}) + "\n"
        )
        for name in ("baseline_eval.json", "candidate_eval.json"):
            (output / name).write_text(json.dumps({"eval_loss": loss}))
        (output / "training_health.json").write_text(
            json.dumps(
                {"schema": "v100-training-health-v1", "eligible": True, "optimizer_updates": 25}
            )
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(training_calibration.subprocess, "run", pilot)
    monkeypatch.setattr(
        competition,
        "command",
        lambda root, p, *args: ["python", "module", "--profile", str(p), *args],
    )
    result = training_calibration.calibrate(tmp_path, source, tmp_path / "unused")
    assert json.loads(result.read_text())["training"]["precision"] == "fp16"
    assert json.loads(source.read_text()) == data


def test_market_discovery_chooses_more_than_bootstrap_coins(tmp_path, monkeypatch):
    monkeypatch.setattr(
        spot_bootstrap,
        "public_json",
        lambda *args: (
            {
                "error": [],
                "result": {
                    "SOL": {
                        "altname": "SOLUSD",
                        "quote": "ZUSD",
                        "status": "online",
                        "ordermin": "0.1",
                        "costmin": "0.5",
                    },
                    "BTC": {
                        "altname": "XBTUSD",
                        "quote": "ZUSD",
                        "status": "online",
                        "ordermin": "0.0001",
                        "costmin": "0.5",
                    },
                },
            },
            "hash",
            "now",
        ),
    )
    assert [r["altname"] for r in spot_bootstrap.discover(tmp_path)["pairs"]] == [
        "SOLUSD",
        "XBTUSD",
    ]


def test_remote_outage_keeps_verified_current_master_available(tmp_path, monkeypatch):
    from contextlib import contextmanager

    import requests

    from rlm.v100 import competition, remote_helper

    source = {"runtime": {"base_url": "remote"}}
    monkeypatch.setattr(competition, "load_profile", lambda *args: source)
    monkeypatch.setattr(remote_helper, "remote_profile", lambda p: True)

    @contextmanager
    def down(*args):
        raise requests.ConnectionError("helper paused for game")
        yield

    monkeypatch.setattr(competition, "managed_server", down)
    current = {"verified": True}
    with competition.waiting_researcher(
        tmp_path / "profile", tmp_path, tmp_path / "log", fallback=current
    ) as actual:
        assert actual is current


def test_crossbreed_never_replaces_missing_parent_or_oversized_adapters(tmp_path, monkeypatch):
    from rlm.v100 import crossbreeding, lineages

    source = profile(tmp_path)
    monkeypatch.setattr(lineages, "read", lambda p: ({}, []))
    assert crossbreeding.try_child(
        tmp_path, source, tmp_path / "suite", [], tmp_path / "child"
    ) == (source, None)
    parents = {}
    for branch in ("A", "B"):
        adapter = tmp_path / branch
        adapter.mkdir()
        atomic_json(adapter / "adapter_config.json", {"r": 40})
        parents[branch] = {"training": {"init_adapter": str(adapter)}}
    monkeypatch.setattr(lineages, "read", lambda p: (parents, []))
    chosen, verdict = crossbreeding.try_child(
        tmp_path, source, tmp_path / "suite", [], tmp_path / "child"
    )
    assert chosen is source and verdict["status"] == "deferred"
    assert not (tmp_path / "child").exists()


def test_windows_guard_only_touches_isolated_helper_and_pauses_for_lol():
    text = (Path(__file__).parents[1] / "tools/start-rtx3090-helper.ps1").read_text()
    assert "[int]$Context = 32768" in text and "[int]$BatchTokens = 16" in text
    assert "[int]$ActiveTimePercent = 30" in text
    assert "League of Legends" in text and "paused-for-game" in text
    assert "ExecutablePath.StartsWith($prefix" in text
    assert "CreateNoWindow = $true" in text
    assert "nvidia-smi -pl" not in text and "TdrDelay" not in text


@pytest.mark.parametrize("improves", [False, True])
def test_crossbreed_promotes_only_after_independent_strict_improvement(
    tmp_path, monkeypatch, improves
):
    from contextlib import contextmanager

    from rlm.v100 import breeding, competition, crossbreeding, evaluation, lineages

    source = profile(tmp_path)
    parents = {}
    before = {
        "schema": "v100-quality-v1",
        "suite_sha256": "fixed",
        "generation": {},
        "memory_mode": "fixed",
        "cases": [{"id": "old", "passed": True}, {"id": "new", "passed": False}],
    }
    after = copy.deepcopy(before)
    after["cases"][1]["passed"] = improves
    for branch in ("A", "B"):
        directory = tmp_path / branch
        directory.mkdir()
        atomic_json(directory / "adapter_config.json", {"r": 8})
        parents[branch] = {"training": {"init_adapter": str(directory)}}
    monkeypatch.setattr(lineages, "read", lambda p: (parents, [before, before]))
    monkeypatch.setattr(competition, "require_idle_gpu", lambda: None)
    monkeypatch.setattr(
        crossbreeding.shutil, "disk_usage", lambda p: SimpleNamespace(free=200 * 2**30)
    )
    monkeypatch.setattr(breeding, "breed_adapters", lambda *args: args[2].mkdir())
    monkeypatch.setattr(crossbreeding.subprocess, "run", lambda *args, **kwargs: None)

    @contextmanager
    def server(*args):
        yield None

    monkeypatch.setattr(competition, "managed_server", server)
    monkeypatch.setattr(competition, "helper_client", lambda *args: None)
    monkeypatch.setattr(evaluation, "evaluate_suite", lambda *args: after)
    selected, verdict = crossbreeding.try_child(
        tmp_path, source, tmp_path / "suite", [before], tmp_path / "child"
    )
    assert verdict["passed"] is improves
    assert (selected is source) is not improves
