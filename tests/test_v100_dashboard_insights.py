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
    assert "Na żywo" in html
    assert "Inne przychody" in html
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
const focused=new Node('section');focused.id='focus-console';
const portfolio=new Node('section');portfolio.id='paper-capital-preview';
const raw=new Node('details');raw.tagName='DETAILS';
for(const id of ['cards','backtests'])nodes[id]=new Node('div');
const document={getElementById:id=>nodes[id],createElement:tag=>new Node(tag),querySelectorAll:()=>[overview,trading,income,raw,focused,portfolio]};
"""
    program += functions
    program += r"""
dashboardNavigation();
if(!overview.hidden||!trading.hidden||!income.hidden||!raw.hidden||!nodes.cards.hidden||focused.hidden||portfolio.hidden)throw Error('Live view must show only the focused console and portfolio');
nodes['mission-navigation'].children.find(n=>n.dataset.tab==='trades').onclick();
if(trading.hidden||!overview.hidden||!income.hidden||!raw.hidden||!nodes.cards.hidden)throw Error('Trading view failed');
dashboardNavigation();
if(dashboardTab!=='trades'||trading.hidden)throw Error('Refresh reset selected view');
if(table.children[0].children[0].children.length!==6)throw Error('Missing table headers');
if(!table.children[0].children[0].children[5].textContent.includes('double costs'))throw Error('Unlabelled stress return');
nodes['mission-navigation'].children.find(n=>n.dataset.tab==='diagnostics').onclick();
if(overview.hidden||trading.hidden||income.hidden||raw.hidden||!focused.hidden||!portfolio.hidden)throw Error('Diagnostics must retain the detailed evidence');
"""
    # Shared CI can delay Node startup after tensor tests; UI assertions remain unchanged.
    result = subprocess.run(
        ["node", "-"],
        input=program,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_live_mission_board_embedded_in_served_html():
    html = dashboard_layout.render_template(dashboard_layout.BASE_TEMPLATE, "test")
    assert "function liveMissionBoard(data)" in html
    assert "Live mission control" in html
    assert "Recent recorded actions" in html
    assert "Simulation, not real income" in html
    assert "Status snapshot collected" in html
    assert "age>300?' — delayed':''" in html
    assert "historical|previous run|may precede current" in html
    assert "liveMissionBoard(data);" in html


def test_focus_console_exposes_recorded_rationale_and_quality_without_inventing_income():
    import shutil
    import subprocess

    from rlm.v100.dashboard_insights import SCRIPT

    if not shutil.which("node"):
        pytest.skip("Node unavailable")
    function = (
        "function focusConsole(data){"
        + SCRIPT.split("function focusConsole(data){", 1)[1].split(
            "function paperEquityTimeline", 1
        )[0]
    )
    program = r"""
class Element {constructor(tag){this.tagName=tag;this.children=[];this.dataset={}}append(...n){this.children.push(...n)}replaceChildren(){this.children=[]}}
const document={createElement:tag=>new Element(tag)};const root=new Element('section');function $(id){return root}function panel(){return root}function textField(p,k,v){const n=new Element('p');n.textContent=k+': '+v;p.append(n)}
FUNCTION
focusConsole({collected_at:Date.now()/1000,mission:{running:true},report:{live_paper:{activity_stream:{entries:[{branch:'A',sequence:1,time:'2026-10-16T00:00:00Z',proposal:{source:'model-proposal',action:'hold',rationale:'Costs exceed the observed signal.',status:'recorded'}}]}},mission_evidence:{goal_learning:{cpu_shadow_policy:{prequential_samples:12,prequential_brier:.3,uniform_baseline_brier:.2}}}}});
const collect=n=>(n.textContent||'')+' '+n.children.map(collect).join(' ');const output=collect(root);
if(!output.includes('Costs exceed the observed signal.')||!output.includes('Brak poprawy')||!output.includes('Niepotwierdzony'))throw Error(output);
if(!root.children.some(n=>n.children.some(c=>c.tagName==='details')))throw Error('Decisions must be expandable');
""".replace("FUNCTION", function)
    result = subprocess.run(
        ["node", "-"], input=program, text=True, encoding="utf-8", capture_output=True, timeout=10
    )
    assert result.returncode == 0, result.stderr


def test_live_mission_board_is_in_combined_live_tab():
    from rlm.v100.dashboard_insights import SCRIPT

    live_group = SCRIPT.split("live:[", 1)[1].split("]", 1)[0]
    for panel in ("live-mission-board", "trade-explorer", "paper-labs-panel", "income-panel"):
        assert f"'{panel}'" in live_group
    assert "dashboardTab='live'" in SCRIPT
    assert "function liveMissionBoard(data)" in SCRIPT


def test_live_board_reads_verified_paper_ledger_and_current_evaluation():
    from rlm.v100.dashboard_insights import SCRIPT

    assert "report.live_paper?.branches||{}" in SCRIPT
    assert "report.current_activity||{}" in SCRIPT
    assert "Candidate evaluation" in SCRIPT
    assert "Paper fills" in SCRIPT
    assert "RTX 3060 Ti" in SCRIPT


def test_100_pln_preview_explains_scaled_model_proposal_and_no_best_claim():
    import shutil
    import subprocess

    from rlm.v100.dashboard_insights import SCRIPT

    if not shutil.which("node"):
        pytest.skip("Node unavailable")
    function = (
        "function paperEquityTimeline(parent,paper,starting,capital){"
        + SCRIPT.split("function paperEquityTimeline(parent,paper,starting,capital){", 1)[1].split(
            "function insightsView(data){", 1
        )[0]
    )
    program = r"""
class Element { constructor(tag){this.tagName=tag;this.children=[];this.style={};this.dataset={};this.attributes={};this.textContent=''} append(...nodes){this.children.push(...nodes)} replaceChildren(...nodes){this.children=nodes} setAttribute(key,value){this.attributes[key]=value} }
const nodes={'live-mission-board':new Element('section')};let created;
const document={createElement:tag=>new Element(tag),createElementNS:(ns,tag)=>new Element(tag)};
function $(id){return nodes[id]}
function panel(id,title,after){created=new Element('section');created.id=id;return created}
function textField(parent,label,value){const node=new Element('p');node.textContent=label+': '+String(value);parent.append(node)}
FUNCTION
const now=new Date(),earlier=new Date(now.getTime()-60000);
const data={report:{
 live_execution_policy:{capital_limit_pln:100,daily_loss_stop_pln:20,max_total_loss_pln:25,leverage_enabled:false},
 live_paper:{initial_capital:'10000',branches:{A:{equity:'10000',net_pnl_before_personal_tax:'0',cash:'10000',positions:{}},B:{equity:'9987.1',net_pnl_before_personal_tax:'-12.9',cash:'9500',positions:{'BTC-KRAKEN':{quantity:'0.0015',side:'long'}}}},equity_series:{A:[{time:earlier.toISOString(),equity:'10000'}],B:[{time:earlier.toISOString(),equity:'9987.1'}]},scenario_inputs:{A:{state:'recorded-current-positions',equity:'10000',net_exposure:'0',as_of:now.toISOString()},B:{state:'recorded-current-positions',equity:'9987.1',net_exposure:'100',as_of:now.toISOString()}},activity_stream:{entries:[
  {branch:'A',category:'decision',time:earlier.toISOString(),proposal:{source:'model-proposal',status:'recorded',action:'open',symbol:'BTC-KRAKEN',budget:500,leverage:1,rationale:'Small evidence-backed paper proposal.',quote:{bid:'64000',ask:'64010'}}},
  {branch:'B',category:'decision',time:now.toISOString(),proposal:{source:'model-proposal',status:'recorded',action:'hold',symbol:'BTC-KRAKEN',budget:0,rationale:'Wait for verified costs.'}}
 ]},blockers:['BTC costs unknown']},paper_blockers:['BTC costs unknown'],recent_exploratory_backtests:[]}};
paperCapitalPreview(data);
const collect=n=>(n.tagName||'')+' '+(n.textContent||'')+' '+JSON.stringify(n.attributes||{})+' '+(n.children||[]).map(collect).join(' ');const output=collect(created);
for(const value of ['Strategia A','OPEN · BTC-KRAKEN','5.00 PLN','Bid/ask przy decyzji: 64000 / 64010','Powód modelu: Small evidence-backed paper proposal.','Strategia B','HOLD','99.87 PLN','95.00 PLN','0.00001500 long','100 PLN','svg','Historia proponowanej ekspozycji','Jak zmienia się wartość obu portfeli paper','Co jeśli cena ruszy się od teraz?','+5% ceny →','Księga nie prowadzi osobnego rachunku 100 PLN'])if(!output.includes(value))throw Error('Missing 100 PLN preview evidence: '+value+'\n'+output);
""".replace("FUNCTION", function)
    result = subprocess.run(
        ["node", "-"], input=program, text=True, encoding="utf-8", capture_output=True, timeout=5
    )
    assert result.returncode == 0, result.stderr


def test_100_pln_timeline_only_uses_model_decisions_with_recorded_budgets():
    from rlm.v100.dashboard_insights import SCRIPT

    assert "proposal.source==='model-proposal'" in SCRIPT
    assert "const proposals=(stream.entries||[]).filter" in SCRIPT
    assert "HOLD = 0 PLN" in SCRIPT
    assert "Wykres pokazuje propozycję, nie wykonanie ani saldo" in SCRIPT
    assert "paper_blockers" in SCRIPT


def test_current_blocker_does_not_reuse_previous_cycle_failure():
    from rlm.v100.dashboard_insights import SCRIPT

    assert "Previous cycle failure (historical, not the current blocker)" in SCRIPT
    blocker = SCRIPT.split("bd.textContent=", 1)[1].split(";", 1)[0]
    assert "learning.last_cycle?.detail" not in blocker


def test_stale_income_error_does_not_mask_fresh_paper_fee_blocker():
    import shutil
    import subprocess

    from rlm.v100.dashboard_insights import SCRIPT

    if not shutil.which("node"):
        pytest.skip("Node unavailable")
    expression = SCRIPT.split("const issueReceipts=", 1)[1].split(
        ";textField(p,'Current blocker / last recorded error'", 1
    )[0]
    program = """
function choose(work,r,workFreshness){
 const issueReceipts=EXPRESSION;
 return currentWorkError;
}
const value=choose({error:'historic income failure'},
 {learning_activity:{},paper_fee_source:{error:'fee publication mismatch'},status_freshness:{paper_fee_source:{state:'fresh'}}},
 {state:'stale'});
if(value!=='Current Paper fee source issue: fee publication mismatch')throw Error('Fresh paper blocker was hidden by stale income receipt: '+value);
const historical=choose({error:'historic income failure'},
 {learning_activity:{},paper_fee_source:{error:'historic fee failure'},status_freshness:{paper_fee_source:{state:'stale'}}},
 {state:'stale'});
if(historical!=='Last recorded (historical) Income work issue: historic income failure')throw Error('Stale income issue was not marked historical: '+historical);
""".replace("EXPRESSION", expression)
    result = subprocess.run(
        ["node", "-"], input=program, text=True, encoding="utf-8", capture_output=True, timeout=5
    )
    assert result.returncode == 0, result.stderr


def test_live_board_distinguishes_completions_failures_and_idle_gpu():
    from rlm.v100.dashboard_insights import SCRIPT

    assert "failed; ${evaluation.state||'unknown'}" in SCRIPT
    assert "Candidate selected for next serving" in SCRIPT
    assert "activity.accepted_updates??'Unknown'" in SCRIPT
    assert "it does not prove serving loaded it or revenue" in SCRIPT
    assert "3060 GPU compute" in SCRIPT
    assert "No recent completed operations" in SCRIPT


def test_live_training_receipts_replace_stale_aggregate_counters(tmp_path):
    run = tmp_path / "research/mission/run-current"
    path = run / "learning/update-0002/generation-01/A/training/training_health.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "attempted_steps": 50,
                "optimizer_updates": 49,
                "amp_skipped_steps": 1,
                "finite_gradient_steps": 49,
            }
        )
    )
    report = {"mission_evidence": {"run": str(run), "optimizer_updates_observed": 3}}
    value = dashboard_snapshot.supplement(tmp_path, {"run": str(run)}, report, [])
    assert value["mission_evidence"]["optimizer_updates_observed"] == 49
    path.unlink()
    value = dashboard_snapshot.supplement(tmp_path, {"run": str(run)}, value, [])
    assert value["mission_evidence"]["optimizer_updates_observed"] is None
    assert value["mission_evidence"]["training_runs"] == []


def test_live_training_never_reuses_previous_run_evidence(tmp_path):
    report = {"mission_evidence": {"run": "old", "optimizer_updates_observed": 99}}
    value = dashboard_snapshot.supplement(tmp_path, {}, report, [])
    assert value["mission_evidence"]["optimizer_updates_observed"] is None


def test_live_training_receipt_preserves_counter_and_writer_timestamp(tmp_path):
    run = tmp_path / "research/mission/run-current"
    path = run / "learning/update-0002/generation-01/A/training/training_health.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "attempted_steps": 50,
                "optimizer_updates": 49,
                "amp_skipped_steps": 1,
                "finite_gradient_steps": 49,
                "updated_at": 1000.0,
            }
        )
    )
    report = {"mission_evidence": {"run": str(run)}}
    value = dashboard_snapshot.supplement(tmp_path, {"run": str(run), "running": True}, report, [])
    assert value["mission_evidence"]["optimizer_updates_observed"] == 49
    assert value["mission_evidence"]["optimizer_receipt_updated_at"] == 1000.0
    assert value["learning_activity_status"]["optimizer_evidence"]["activity_state"] == "stale"


def test_learning_panel_displays_optimizer_receipt_freshness_separately():
    from rlm.v100.dashboard_layout import PAGE

    assert "Last confirmed optimizer receipt" in PAGE
    assert "Optimizer receipt detail" in PAGE
