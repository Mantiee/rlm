import json
from pathlib import Path

import pytest

from rlm.v100 import backtesting, dashboard_layout, dashboard_snapshot, trade_explorer
from tests.test_v100_backtesting import candles


def test_constant_price_spot_and_liquidation_costs_reconcile(tmp_path):
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=3600"
    source = candles()
    for row in source:
        row[1:5] = [100, 100, 100, 100]
    result = backtesting.run(
        tmp_path,
        "A",
        {"price_url": url, "event_url": "", "rule": "buy_hold", "fee_bps": 50, "slippage_bps": 10},
        source_snapshot=(url, json.dumps(source)),
    )
    path = Path(result["report"])
    before = path.read_bytes()
    series = trade_explorer.historical_series(path)
    buy, sell = series["executions"]
    entry = 1 / (1.001 * 1.005)
    liquidation = entry * 0.999 * 0.995
    assert buy["spot_equity_after"] == pytest.approx(entry)
    assert buy["equity_after"] == pytest.approx(liquidation)
    assert buy["costs"] == pytest.approx(1 - entry)
    assert buy["estimated_exit_cost"] == pytest.approx(entry - liquidation)
    assert sell["equity_before"] == pytest.approx(sell["equity_after"])
    assert series["points"][-1]["spot_equity"] == pytest.approx(liquidation)
    assert series["points"][-1]["equity"] == pytest.approx(liquidation)
    assert series["net_pnl"] == pytest.approx(-series["total_costs"])
    assert series["cost_assumptions"]["fee_bps_per_leg"] == 50
    assert path.read_bytes() == before


def test_host_insights_are_present_on_existing_guest_layout():
    html = dashboard_layout.render_template(dashboard_layout.BASE_TEMPLATE, "test")
    assert "mission-navigation" in html
    assert "Other income" in html
    assert "Maximum drawdown (peak to trough)" in html
    assert "No verified revenue or completed-test time series" in html
    assert "Same-price spot equity change" in html
    assert "Mixed forecast archive" in html
    assert "1 bp = 0.01%" in html
    assert "No recorded fills in this tail" in html
    assert "INSIGHTS_SCRIPT" not in html


def test_cpu_policy_receipt_is_scoped_to_current_goal(tmp_path, monkeypatch):
    from rlm.v100 import goals, planning

    monkeypatch.setattr(goals, "load_goal", lambda root: {"id": "current"})
    monkeypatch.setattr(planning, "read", lambda root: {})
    path = tmp_path / "research/goal-policy/status.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"goal_id": "old", "trained_outcomes": 99}))
    result = dashboard_snapshot.supplement(tmp_path, {}, {}, [])
    assert "cpu_shadow_policy" not in result["mission_evidence"]["goal_learning"]
    path.write_text(json.dumps({"goal_id": "current", "trained_outcomes": 4}))
    result = dashboard_snapshot.supplement(tmp_path, {}, {}, [])
    assert result["mission_evidence"]["goal_learning"]["cpu_shadow_policy"]["trained_outcomes"] == 4


def test_navigation_repairs_headers_and_retains_selected_view():
    import shutil
    import subprocess

    from rlm.v100.dashboard_insights import SCRIPT

    if not shutil.which("node"):
        pytest.skip("Node unavailable")
    functions = SCRIPT.split("function evidenceBars", 1)[0]
    program = r"""
const nodes={};
class Node {
 constructor(tag){this.tag=tag;this.children=[];this.dataset={};this.attrs={};this.hidden=false;this.id=''}
 append(...n){this.children.push(...n);for(const c of n)nodes[c.id]=c}
 setAttribute(k,v){this.attrs[k]=v}
 replaceChildren(...n){this.children=n}
 querySelector(s){return this.ids?.includes(s.slice(1))?nodes[s.slice(1)]:null}
 before(n){nodes[n.id]=n}
 prepend(n){this.children.unshift(n)}
 closest(){return table}
}
const table=new Node('table');table.querySelector=()=>table.children[0];
const overview=new Node('section');overview.id='overview-panel';
const trading=new Node('section');trading.ids=['backtests'];
const income=new Node('section');income.id='income-panel';
const raw=new Node('details');
for(const id of ['cards','backtests'])nodes[id]=new Node('div');
const document={getElementById:id=>nodes[id],createElement:tag=>new Node(tag),querySelectorAll:()=>[overview,trading,income,raw]};
"""
    program += functions
    program += r"""
dashboardNavigation();
if(overview.hidden||!trading.hidden||!income.hidden||!raw.hidden)throw Error('Overview contains other views');
nodes['mission-navigation'].children.find(n=>n.dataset.tab==='trades').onclick();
if(trading.hidden||!overview.hidden||!income.hidden||!nodes.cards.hidden)throw Error('Trading view failed');
dashboardNavigation();
if(dashboardTab!=='trades'||trading.hidden)throw Error('Refresh reset selected view');
if(table.children[0].children[0].children.length!==6)throw Error('Missing table headers');
if(!table.children[0].children[0].children[5].textContent.includes('double costs'))throw Error('Unlabelled stress return');
"""
    # Shared CI can delay Node startup after tensor tests; UI assertions remain unchanged.
    result = subprocess.run(
        ["node", "-"], input=program, text=True, capture_output=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
