import json
from pathlib import Path

import pytest

from rlm.v100 import backtesting, mission_chat, trade_explorer
from tests.test_v100_backtesting import candles
from tests.test_v100_paper import decide, quote, setup


def test_forward_chart_matches_real_ledger_fills_and_costs_without_writing(tmp_path):
    book, clock, _ = setup(
        tmp_path, fee_changes={"entry_minimum_commission": "2", "exit_minimum_commission": "3"}
    )
    book.ingest(quote(clock))
    decide(book)
    clock.advance()
    book.ingest(quote(clock))
    decide(book, "close")
    clock.advance()
    closing = book.ingest(quote(clock, "110"))["payload"]["trades"][0]
    before = book.events()
    series = trade_explorer.paper_series(tmp_path)
    branch = next(s for s in series if s["id"] == "paper-A")
    assert [row["action"] for row in branch["executions"]] == ["buy", "close"]
    assert branch["executions"][1]["net_pnl"] == closing["net_pnl"]
    assert branch["net_pnl"] == pytest.approx(float(closing["net_pnl"]))
    assert branch["total_costs"] == 5
    assert book.events() == before
    book.close()


def test_empty_forward_portfolio_does_not_invent_executions(tmp_path):
    book, _, _ = setup(tmp_path)
    rows = trade_explorer.paper_series(tmp_path)
    assert all(not row["executions"] and row["net_pnl"] == 0 for row in rows)
    book.close()


def test_historical_chart_replays_original_metrics_and_retains_losses(tmp_path):
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=3600"
    result = backtesting.run(
        tmp_path,
        "A",
        {
            "price_url": url,
            "event_url": "",
            "rule": "buy_hold",
            "fee_bps": 400,
            "slippage_bps": 100,
        },
        source_snapshot=(url, json.dumps(candles())),
    )
    path = Path(result["report"])
    original = path.read_bytes()
    series = trade_explorer.historical_series(path)
    assert path.read_bytes() == original
    assert [row["action"] for row in series["executions"]] == ["buy", "sell"]
    assert series["executions"][-1]["reason"] == "sample-end liquidation"
    assert series["executions"][-1]["net_pnl"] == pytest.approx(
        result["development_test"]["net_return"]
    )
    assert series["points"][-1]["equity"] - 1 == pytest.approx(series["net_pnl"])
    assert series["total_costs"] > 0
    (path.parent / "source-0.json").write_text("[]")
    chart = trade_explorer.snapshot(tmp_path)
    assert not chart["series"] and "source changed" in chart["errors"][0]


def test_ambiguous_paper_tail_is_rejected(tmp_path):
    book, _, _ = setup(tmp_path)
    book.db.execute("UPDATE state SET payload=? WHERE id=1", ("{}",))
    book.db.commit()
    with pytest.raises(ValueError, match="checksum"):
        trade_explorer.paper_series(tmp_path)
    book.close()


def test_specific_chat_retirement_archives_failure_and_leaves_other_requests(tmp_path):
    identity = mission_chat.submit(tmp_path, "chart work")
    other = mission_chat.submit(tmp_path, "unrelated question")
    receipt = mission_chat.retire_request(tmp_path, identity, "operator replacement")
    assert receipt["retired"]
    assert mission_chat.inspect(tmp_path, identity)["state"] == "cancelled"
    assert mission_chat.inspect(tmp_path, other)["state"] == "queued"
    assert json.loads(Path(receipt["archive"]).read_text())["state"] == "queued"


def test_visualization_command_responds_without_inference_or_guest_ssh(tmp_path, monkeypatch):
    from rlm.v100 import desktop

    monkeypatch.setattr(desktop, "run", lambda *a, **kw: pytest.fail("Chart must not need SSH"))
    response = mission_chat.direct_facts(
        tmp_path, "zrob mi wizualizacje kiedy kupuje kiedy sprzedaje na zywo html"
    )
    assert response["responder"]["model"] == "controller-trade-explorer"
    assert "Trade explorer" in response["answer"]
    assert not response["actions"] and not response["applied"]


def test_chart_receipt_selection_and_zoom_survive_updates():
    import shutil
    import subprocess

    from rlm.v100.dashboard_layout import APP_SCRIPT

    if not shutil.which("node"):
        pytest.skip("Node unavailable")
    functions = APP_SCRIPT.split("let tradeData=", 1)[1].split("async function tradesView", 1)[0]
    functions = "let tradeData=" + functions
    program = r"""
const nodes={};
class Element {
 constructor(tag){this.tag=tag;this.children=[];this.style={};this.dataset={};this.textContent='';this.open=false}
 append(...children){this.children.push(...children)}
 replaceChildren(...children){this.children=children}
 setAttribute(k,v){this[k]=v}
}
const document={getElementById:id=>nodes[id],createElement:t=>new Element(t)};
function svg(tag,attrs,text){const n=new Element(tag);Object.assign(n,attrs);n.textContent=text||'';return n}
for(const id of ['trade-series','trade-start','trade-end','trade-metric','trade-errors','trade-plot','trade-summary','trade-scope','trade-hover','trade-detail','trade-receipt'])nodes[id]=new Element('div');
"""
    program += functions
    program += r"""
const series={id:'real',label:'Forward paper A',currency:'PLN',initial_capital:10000,net_pnl:-10,total_costs:10,scope:'recorded',points:[{time:1,equity:10000},{time:2,equity:9990}],executions:[{time:1,action:'buy',price:100,costs:5,receipt:'buy-proof'},{time:2,action:'close',price:99,costs:5,net_pnl:-10,receipt:'sell-proof'}]};
tradeData={updated:1,series:[series],errors:[]};drawTrades();
const markers=nodes['trade-plot'].children.filter(n=>n.role==='button');
if(markers.length!==2)throw Error('Missing first buy or final sell marker');
markers[1].onclick();if(!nodes['trade-receipt'].open||JSON.parse(nodes['trade-detail'].textContent).net_pnl!==-10)throw Error('Wrong execution receipt');
const retained=nodes['trade-receipt'];tradeStart=20;tradeEnd=90;tradeData={updated:2,series:[{...series,net_pnl:-20}],errors:[]};drawTrades();
if(tradeChoice!=='real'||tradeStart!==20||tradeEnd!==90||nodes['trade-receipt']!==retained||!retained.open||tradeSelected!=='sell-proof')throw Error('Selection or expanded receipt reset');
if(!nodes['trade-summary'].textContent.includes('-20.0000'))throw Error('Live P&L did not update');
"""
    result = subprocess.run(["node", "-"], input=program, text=True, capture_output=True, timeout=5)
    assert result.returncode == 0, result.stderr
