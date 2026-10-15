"""Execution freshness and promotion regressions for the isolated paper lab."""

import datetime
import importlib
import json
import time
from pathlib import Path

import pytest


@pytest.fixture
def lab(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    module = importlib.import_module("synta_live_paper_v3")
    instance = module.Lab(tmp_path)
    instance.fx = 4.0
    instance.fx_date = str(datetime.datetime.now(datetime.UTC).date())
    instance.fx_observed = time.time()
    instance.fx_error = None
    yield instance
    instance.db.close()


def fee_receipt(lab, *, age=0, fee=40):
    path = lab.root / "research/paper/fee-source-status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"state": "parsed", "taker_fee_bps": fee, "updated": time.time() - age})
    )


@pytest.mark.parametrize(
    "age,fee", [(86401, 40), (-3600, 40), (0, float("nan")), (0, float("inf"))]
)
@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_invalid_fee_never_changes_accounts_or_fills(lab, age, fee, side):
    fee_receipt(lab, age=age, fee=fee)
    lab.db.execute("UPDATE accounts SET qty=1,entry=100 WHERE branch='a'")
    before = lab.account("a")
    assert lab.fill("a", side, 100, "test") is False
    assert lab.account("a") == before
    assert lab.db.execute("SELECT COUNT(*) FROM fills").fetchone()[0] == 0


@pytest.mark.parametrize(
    "state", ["expired", "future", "failed_refresh", "old_source", "future_source"]
)
@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_stale_fx_never_executes_cached_conversion(lab, state, side):
    fee_receipt(lab)
    lab.db.execute("UPDATE accounts SET qty=1,entry=100 WHERE branch='a'")
    if state == "expired":
        lab.fx_observed = time.time() - 3601
    elif state == "future":
        lab.fx_observed = time.time() + 3600
    elif state == "failed_refresh":
        lab.fx_error = "provider unavailable"
    else:
        delta = datetime.timedelta(days=-6 if state == "old_source" else 1)
        lab.fx_date = str(datetime.datetime.now(datetime.UTC).date() + delta)
    before = lab.account("a")
    assert lab.fill("a", side, 100, "test") is False
    assert lab.account("a") == before
    assert lab.db.execute("SELECT COUNT(*) FROM fills").fetchone()[0] == 0


def test_mark_to_market_does_not_charge_an_invented_exit(lab):
    lab.db.execute("UPDATE accounts SET cash=9500,qty=2 WHERE branch='a'")
    assert lab.equity("a", 250) == 10000


def test_new_curve_never_mixes_legacy_liquidation_estimates(lab):
    lab.db.execute("INSERT INTO ticks(ts,bid,ask,a,b) VALUES('old',100,101,9990,9980)")
    lab.db.execute(
        "INSERT INTO ticks(ts,bid,ask,a,b,valuation_basis) VALUES('new',100,101,10000,10000,'gross-mark-to-bid-v2')"
    )
    status = lab.status()
    assert status["legacy_curve_points"] == 1
    assert len(status["chart"]) == 1
    assert "v2" in status["chart_scope"]
    assert lab.db.execute("SELECT COUNT(*) FROM ticks").fetchone()[0] == 2


def test_fallback_dashboard_chart_has_legend_axes_last_values_and_resize(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    page = importlib.import_module("synta_live_paper_v3").PAGE
    assert 'aria-label="Legenda wykresu"' in page
    assert 'class="chart-values"' in page
    assert "Oś Y: wycena brutto w PLN" in page
    assert "oś X: kolejność notowań (od najstarszego do najnowszego)" in page
    assert 'id="chart-a"' in page and 'id="chart-b"' in page
    assert "window.devicePixelRatio" in page
    assert "window.addEventListener('resize'" in page
    assert "Number.isFinite(Number(r.a))" in page
    assert "Math.max(1,data.length-1)" in page


def test_served_dashboard_chart_exposes_last_values_and_units():
    dashboard = Path(__file__).resolve().parents[1] / "tools/synta_live_dashboard_v4.html"
    page = dashboard.read_text(encoding="utf-8")
    assert dashboard.name in (dashboard.parent / "synta_live_paper_v3.py").read_text(
        encoding="utf-8"
    )
    assert 'aria-label="Legenda wykresu"' in page
    assert 'id="chartAValue"' in page and 'id="chartBValue"' in page
    assert "Oś X: kolejność notowań od najstarszego do najnowszego" in page
    assert "toLocaleString('pl-PL')+' PLN'" in page
    assert "Number.isFinite(Number(r.a))" in page
    assert "Math.max(1,rows.length-1)" in page
    assert "window.addEventListener('resize'" in page


def test_fresh_evidence_allows_explicit_legacy_paper_scenario(lab):
    fee_receipt(lab)
    assert lab.fill("a", "BUY", 100, "explicit legacy scenario") is True
    assert lab.account("a")[0] == 9500
    assert lab.fill("a", "SELL", 100, "explicit legacy scenario") is True
    assert lab.account("a")[1] == 0
    assert lab.db.execute("SELECT COUNT(*) FROM fills").fetchone()[0] == 2


def test_reference_fx_and_scenario_costs_are_explicit(lab):
    status = lab.status()
    assert status["execution_costs"]["verified"] is False
    assert status["execution_costs"]["promotion_allowed"] is False
    assert "reference" in status["fx_evidence"]["kind"]


def test_historical_cached_candidate_cannot_authorize_promotion(lab):
    module = importlib.import_module("synta_accelerator_v1")
    accelerator = module.Accelerator(lab.root)
    accelerator.state["candidate"] = {"eligible": True, "train": {}, "test": {}}
    assert accelerator.snapshot()["candidate"]["eligible"] is False


def test_saved_historical_promotion_is_preserved_but_not_adopted(lab):
    prior = {"params": {"window": 5}, "source_run": "old"}
    lab.promoted_path.write_text(json.dumps(prior))
    module = importlib.import_module("synta_live_paper_v3")
    restarted = module.Lab(lab.root)
    try:
        assert restarted.promoted is None
        assert restarted.legacy_promotion == prior
        assert json.loads(lab.promoted_path.read_text()) == prior
    finally:
        restarted.db.close()
