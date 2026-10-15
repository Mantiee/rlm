Warning: truncated output (original token count: 10247)
Total output lines: 669

import base64
import hashlib
import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from rlm.v100 import (
    architecture_goal_gate,
    background_observer,
    dashboard_editor,
    dashboard_layout,
    dashboard_snapshot,
    desktop,
    desktop_proxy,
    drones,
    live_status,
)


def test_invalid_master_script_rejected_before_guest_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(desktop, "run", lambda *a, **k: pytest.fail("Invalid HTML reached VM"))
    with pytest.raises(ValueError, match="script"):
        dashboard_editor.write(tmp_path, dashboard_layout.BASE_TEMPLATE + "<script>bad()</script>")


def test_guest_write_is_not_publication_and_backs_up_previous_html(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        desktop, "run", lambda root, script, **k: calls.append(script) or {"exit_code": 0}
    )
    result = dashboard_editor.write(tmp_path, dashboard_layout.BASE_TEMPLATE)
    assert result["written"] and not result["published"]
    assert "backup.write_bytes(p.read_bytes())" in calls[0]
    assert "os.fsync" in calls[0]
    assert base64.b64encode(dashboard_layout.BASE_TEMPLATE.encode()).decode() in calls[0]


def test_host_receipt_must_match_exact_guest_layout(tmp_path, monkeypatch):
    content = dashboard_layout.BASE_TEMPLATE
    monkeypatch.setattr(dashboard_editor, "guest_layout", lambda root: content)
    directory = tmp_path / "research/dashboard"
    directory.mkdir(parents=True)
    path = directory / "layout-status.json"
    path.write_text(json.dumps({"state": "validated layout active", "active_sha256": "other"}))
    assert not dashboard_editor.status(tmp_path)["published"]
    path.write_text(
        json.dumps(
            {
                "state": "validated layout active",
                "active_sha256": hashlib.sha256(content.encode()).hexdigest(),
            }
        )
    )
    assert dashboard_editor.status(tmp_path)["published"]


def test_old_layout_gains_trusted_live_renderer_without_guest_javascript():
    assert "<script>" not in dashboard_layout.BASE_TEMPLATE
    assert "workView(data)" in dashboard_layout.APP_SCRIPT
    assert "Candidates selected for the next serving phase" in dashboard_layout.APP_SCRIPT
    for identity in ("actors", "goals", "events"):
        assert identity not in dashboard_layout.REQUIRED_IDS
        assert identity in dashboard_layout.APP_SCRIPT


def test_stale_chat_cannot_be_attributed_to_new_mission(tmp_path):
    directory = tmp_path / "research/state"
    directory.mkdir(parents=True)
    (directory / "chat-active.json").write_text(
        json.dumps({"id": "old", "pid": 11, "phase": "inference"})
    )
    assert live_status.current_chat(tmp_path, {"pid": 12, "running": True})["phase"] == "stale"


def test_dashboard_queue_receipt_exposes_bounded_cooldown_retry():
    summary = live_status.public_summary(
        {
            "status": "deferred",
            "error": "Owned GPU cooldown: 2.0 s",
            "retry_after_seconds": 15,
            "retry_attempt": 2,
            "reasoning_content": "must stay private",
        }
    )
    assert "status: deferred" in summary
    assert "retry_after_seconds: 15" in summary
    assert "retry_attempt: 2" in summary
    assert "must stay private" not in summary


def test_observer_feeds_refresh_independent_of_master_phase_and_back_off(tmp_path, monkeypatch):
    from rlm.v100 import goal_learning, spot_bootstrap

    directory = tmp_path / "research/paper"
    directory.mkdir(parents=True)
    (directory / "ledger.sqlite3").touch()
    calls = []
    monkeypatch.setattr(goal_learning, "tick", lambda root: calls.append("goal"))
    monkeypatch.setattr(
        spot_bootstrap,
        "prepare",
        lambda root, refresh: calls.append(refresh) or {"status": "ready"},
    )
    background_observer.tick(tmp_path)
    background_observer.tick(tmp_path)
    assert calls == ["goal", True, "goal"]
    assert (
        json.loads((tmp_path / "research/state/paper-observer.json").read_text())["state"]
        == "ready"
    )


def test_owned_tunnel_binds_guest_loopback_and_pins_host_identity(tmp_path):
    args = desktop_proxy.command(tmp_path)
    assert "StrictHostKeyChecking=yes" in args
    assert "127.0.0.1:3128:127.0.0.1:12230" in args
    assert args[-1] == "root@127.0.0.1"
    assert "ExitOnForwardFailure=yes" in args


def test_tunnel_cleanup_kills_only_its_own_child():
    calls = []
    child = SimpleNamespace(
        poll=lambda: None,
        terminate=lambda: calls.append("terminate"),
        wait=lambda **k: calls.append("wait"),
    )
    desktop_proxy.close(child)
    assert calls == ["terminate", "wait"]


def test_architecture_goal_change_invalidates_promotion(tmp_path, monkeypatch):
    from rlm.v100 import goals

    (tmp_path / "goal-snapshot.json").write_text(json.dumps({"goal_id": "old"}))
    monkeypatch.setattr(goals, "load_goal", lambda root: {"id": "new"})
    with pytest.raises(ValueError, match="goal changed"):
        architecture_goal_gate.verify(tmp_path, tmp_path, {}, {})


def test_drone_status_exposes_bounded_actual_assignment(tmp_path, monkeypatch):
    from rlm.v100 import research_tools

    monkeypatch.setattr(research_tools, "public_origin", lambda url: None)
    drones.schedule(tmp_path, "A", "source", "https://example.com/" + "x" * 800, 0)
    report = drones.inspect(tmp_path)
    assert len(report[0]["assignment"]) == 820
    assert report[0]["state"] == "queued"


def test_goal_observer_failure_does_not_block_quote_refresh(tmp_path, monkeypatch):
    from rlm.v100 import goal_learning, spot_bootstrap

    directory = tmp_path / "research/paper"
    directory.mkdir(parents=True)
    (directory / "ledger.sqlite3").touch()

    def broken(root):
        raise RuntimeError("Goal endpoint temporarily unavailable")

    monkeypatch.setattr(goal_learning, "tick", broken)
    monkeypatch.setattr(
        spot_bootstrap, "prepare", lambda root, refresh: {"status": "quotes refreshed"}
    )
    background_observer.tick(tmp_path)
    assert (
        json.loads((tmp_path / "research/state/paper-observer.json").read_text())["state"]
        == "ready"
    )


def test_dashboard_status_freshness_requires_a_trusted_timestamp():
    now = 1_800_000_000.0
    assert dashboard_snapshot.status_freshness("income_work", {"updated": now - 30}, now) == {
        "source": "income_work",
        "timestamp_field": "updated",
        "stale_after_seconds": 300,
        "observed_at": now - 30,
        "age_seconds": 30.0,
        "state": "fresh",
        "detail": "Status receipt age only; underlying evidence has its own validity rules",
    }
    assert (
        dashboard_snapshot.status_freshness("income_work", {"updated": now - 301}, now)["state"]
        == "stale"
    )
    assert (
        dashboard_snapshot.status_freshness("external_compute", {"updated": now}, now)["state"]
        == "unknown"
    )
    assert (
        dashboard_snapshot.status_freshness("income_work", {"updated": now + 5}, now)["state"]
        == "unknown"
    )
    assert dashboard_snapshot.status_freshness("income_work", None, now)["state"] == "unavailable"


def test_goal_status_freshness_distinguishes_missing_invalid_and_wrong_goal(tmp_path):
    now = 1_800_000_000.0
    path = tmp_path / "status.json"

    value, freshness = dashboard_snapshot.read_goal_status(path, "goal_learning", "active", now)
    assert value is None and freshness["state"] == "unavailable"

    path.write_text("not json")
    value, freshness = dashboard_snapshot.read_goal_status(path, "goal_learning", "active", now)
    assert value is None and freshness["state"] == "unavailable"

    path.write_text(json.dumps({"goal_id": "other", "updated": now - 10}))
    value, freshness = dashboard_snapshot.read_goal_status(path, "goal_learning", "active", now)
    assert value is None and freshness["state"] == "unknown"
    assert freshness["observed_at"] is None and "different goal" in freshness["detail"]

    path.write_text(json.dumps({"updated": now - 10}))
    for name in ("goal_learning", "cpu_shadow_policy", "observer", "owned_cpu_dispatch"):
        value, freshness = dashboard_snapshot.read_goal_status(path, name, "active", now)
        assert value is None and freshness["state"] == "unknown"
        assert freshness["observed_at"] is None
        assert "no explicit goal ID" in freshness["detail"]

    path.write_text(json.dumps({"goal_id": "active", "updated": now - 30}))
    value, freshness = dashboard_snapshot.read_goal_status(path, "goal_learning", "active", now)
    assert value["goal_id"] == "active" and freshness["state"] == "fresh"

    path.write_text(json.dumps({"goal_id": "active", "updated": now - 301}))
    _, freshness = dashboard_snapshot.read_goal_status(path, "goal_learning", "active", now)
    assert freshness["state"] == "stale"

    path.write_text(" " * (2 * 2**20 + 1))
    value, freshness = dashboard_snapshot.read_goal_status(path, "goal_learning", "active", now)
    assert value is None and freshness["state"] == "unavailable"
    assert "exceeds 2 MiB" in freshness["detail"]


def test_snapshot_goal_learning_freshness_map_tracks_receipt_state(tmp_path, monkeypatch):
    import time

    from rlm.v100 import (
        financial_snapshot,
        goals,
        mission_evidence,
        mission_view,
        paper_labs_snapshot,
        planning,
        public_benchmarks,
    )

    monkeypatch.setattr(financial_snapshot, "supplement", lambda root, report, errors: report)
    monkeypatch.setattr(paper_labs_snapshot, "snapshot", lambda root: [])
    monkeypatch.setattr(mission_evidence, "training_receipts", lambda run: {"errors": []})
    monkeypatch.setattr(public_benchmarks, "progress_for_run", lambda root, run: {})
    monkeypatch.setattr(goals, "load_goal", lambda root: {"id": "active"})
    monkeypatch.setattr(planning, "read", lambda root: {})
    monkeypatch.setattr(mission_view, "snapshot", lambda root, mission, report: {})

    path = tmp_path / "research/goal-learning/status.json"
    freshness = dashboard_snapshot.supplement(tmp_path, {}, {}, [])["status_freshness"][
        "goal_learning"
    ]
    assert freshness["state"] == "unavailable"

    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"goal_id": "other", "updated": 1_800_000_000}))
    freshness = dashboard_snapshot.supplement(tmp_path, {}, {}, [])["status_freshness"][
        "goal_learning"
    ]
    assert freshness["state"] == "unknown"

    observed_at = time.time()
    path.write_text(json.dumps({"goal_id": "active", "updated": observed_at}))
    freshness = dashboard_snapshot.supplement(tmp_path, {}, {}, [])["status_freshness"][
        "goal_learning"
    ]
    assert freshness["state"] == "fresh"
    assert freshness["observed_at"] == observed_at

    path.write_text("invalid JSON")
    freshness = dashboard_snapshot.supplement(tmp_path, {}, {}, [])["status_freshness"][
        "goal_learning"
    ]
    assert freshness["state"] == "unavailable"


@pytest.mark.parametrize("paper_labs", [False, True])
def test_trusted_live_renderer_runs_on_older_layout_without_actor_sections(paper_labs):
    import shutil
    import subprocess

    if not shutil.which("node"):
        pytest.skip("Node unavailable for JavaScript runtime verification")
    program = r"""
const nodes={};
class Element {
 constructor(tag){this.tag=tag;this.children=[];this.style={};this.textContent='';this.dataset={}}
 set id(value){this.identity=value;nodes[value]=this}
 get id(){return this.identity}
 append(...items){this.children.push(...items)}
 after(item){this.children.push(item)}
 replaceChildren(...items){this.c…4247 tokens truncated….childNodes.splice(p.childNodes.indexOf(this),1);this.parentNode=null}}
 replaceWith(n){const p=this.parentNode;p.insertBefore(n,this);this.remove()}
}
class Element extends Text {
 constructor(tag){super('');this.nodeType=1;this.nodeName=tag.toUpperCase();this.childNodes=[];this.dataset={};this.style={};this.attrs={};this.scrollTop=0;this.scrollLeft=0}
 get children(){return this.childNodes.filter(n=>n.nodeType===1)}
 get attributes(){return Object.entries(this.attrs).map(([name,value])=>({name,value}))}
 get textContent(){return this.childNodes.map(n=>n.nodeType===3?n.nodeValue:n.textContent).join('')}
 set textContent(v){this.replaceChildren(new Text(String(v)))}
 append(...items){for(const n of items)this.insertBefore(typeof n==='string'?new Text(n):n,null)}
 insertBefore(n,before){n.remove();const i=before?this.childNodes.indexOf(before):this.childNodes.length;this.childNodes.splice(i,0,n);n.parentNode=this}
 replaceChildren(...items){for(const n of [...this.childNodes])n.remove();this.append(...items)}
 after(n){const p=this.parentNode;p.insertBefore(n,p.childNodes[p.childNodes.indexOf(this)+1]||null)}
 setAttribute(k,v){this.attrs[k]=String(v)}
 getAttribute(k){return this.attrs[k]??null}
 hasAttribute(k){return k in this.attrs}
 removeAttribute(k){delete this.attrs[k]}
 querySelectorAll(selector){const all=[];function visit(n){for(const c of n.children||[]){all.push(c);visit(c)}}visit(this);return all.filter(n=>selector[0]==='#'?n.id===selector.slice(1):selector==='pre'?n.nodeName==='PRE':selector==='[data-scroll-key]'?n.dataset.scrollKey:selector==='details[data-key]'?n.nodeName==='DETAILS'&&n.dataset.key:false)}
 querySelector(s){return this.querySelectorAll(s)[0]||null}
 cloneNode(deep){const n=new Element(this.nodeName);n.id=this.id;n.dataset={...this.dataset};n.attrs={...this.attrs};n.style={...this.style};n.open=this.open;if(deep)n.append(...this.childNodes.map(c=>c.cloneNode(true)));return n}
}
const listeners={};const body=new Element('body');const document={addEventListener:(name,callback)=>{listeners[name]=callback},body,getElementById:id=>body.querySelector('#'+id),querySelectorAll:s=>body.querySelectorAll(s),createElement:t=>new Element(t),createElementNS:(ns,t)=>new Element(t),createTextNode:t=>new Text(t)};
const window={scrollX:0,scrollY:400,scrollTo(x,y){this.scrollX=x;this.scrollY=y}};
const location={reload(){throw Error('Unexpected reload')}};const setInterval=()=>{};let fetch=()=>Promise.resolve({ok:true,json:()=>Promise.resolve({})});
"""
    for identity in dashboard_layout.REQUIRED_IDS:
        program += f"{{const n=new Element('div');n.id={json.dumps(identity)};body.append(n)}}\n"
    program += dashboard_layout.APP_SCRIPT
    program += r"""
const readingCard=new Element('article');readingCard.dataset.scrollKey='gpu';readingCard.scrollTop=140;
const detail=new Element('details');detail.dataset.key='gpu-proof';detail.open=true;
const summary=new Element('summary');summary.textContent='Evidence';const pre=new Element('pre');pre.id='probe';pre.textContent='old output';pre.scrollTop=73;
detail.append(summary,pre);readingCard.append(detail);body.append(readingCard);const originalText=pre.childNodes[0];
for(let i=0;i<4;i++)stablePaint(()=>{$('probe').textContent='new output '+i});
if(document.getElementById('probe')!==pre||pre.childNodes[0]!==originalText||pre.textContent!=='new output 3')throw Error('Reader DOM identity lost');
if(!detail.open||readingCard.scrollTop!==140||pre.scrollTop!==73||window.scrollY!==400)throw Error('Reading position lost');
let readerTop=100;detail.isConnected=true;detail.getBoundingClientRect=()=>({top:readerTop,bottom:readerTop+200});window.innerHeight=800;
const originalQuery=document.querySelectorAll;document.querySelectorAll=s=>s==='details[data-key][open]'?[detail]:s==='[id], [data-scroll-key]'?[readingCard,pre]:originalQuery(s);
stablePaint(()=>{readerTop=160;$('probe').textContent='updated while a new event was inserted above'});
if(window.scrollY!==460||readingCard.scrollTop!==140||pre.scrollTop!==73)throw Error('Visible reading anchor moved');
let fieldTop=200;pre.isConnected=true;pre.getBoundingClientRect=()=>({top:fieldTop,bottom:fieldTop+200});window.innerWidth=1920;document.elementFromPoint=(x,y)=>{if(x!==960)throw Error('Anchor sampled outside centered dashboard');return pre};
stablePaint(()=>{fieldTop=260;$('probe').textContent='field inside the same section moved down'});
if(window.scrollY!==520)throw Error('Anchored whole section instead of visible field');
readingCard.contains=n=>n===pre;stablePaint(()=>{fieldTop=320;$('probe').textContent='new row above inside scrolling resource panel'});
if(readingCard.scrollTop!==200||window.scrollY!==520)throw Error('Internal scroll correction moved entire page');


const timeline=new Element('section');timeline.id='timeline-panel';body.append(timeline);stablePaint(()=>archivePanel());
const archive=document.getElementById('action-archive'),select=document.getElementById('archive-day');select.options=select.children;
let archiveCalls=0;fetch=async()=>{archiveCalls++;return {ok:true,json:async()=>({days:['2026-10-10'],day:'2026-10-10',events:[{id:'proof',time:'now',kind:'tool-result',payload:{result:'actual'}}],next_cursor:200,archive_bytes:200})}};
(async()=>{listeners.click({target:document.getElementById('archive-load')});await new Promise(resolve=>setTimeout(resolve,10));const rows=document.getElementById('archive-rows');if(rows.children.length!==1)throw Error('Archive click did not load live DOM');const opened=rows.children[0];opened.open=true;for(let i=0;i<3;i++)stablePaint(()=>archivePanel());if(document.getElementById('action-archive')!==archive||rows.children[0]!==opened||!opened.open)throw Error('Expanded archive reset during refresh');listeners.click({target:document.getElementById('archive-load')});await new Promise(resolve=>setTimeout(resolve,10));if(archiveCalls!==2||rows.children.length!==1)throw Error('Archive stopped responding or duplicated events')})().catch(error=>{console.error(error);process.exitCode=1});
"""
    result = subprocess.run(
        ["node", "-"], input=program, text=True, encoding="utf-8", capture_output=True, timeout=5
    )
    assert result.returncode == 0, result.stderr


def test_decision_and_fill_freshness_require_completed_receipts():
    from rlm.v100.financial_snapshot import model_decision_evidence, paper_fill_evidence

    now = datetime(2026, 10, 15, 12, 0, tzinfo=UTC).timestamp()
    recent = datetime.fromtimestamp(now - 45, UTC).isoformat()
    stale = datetime.fromtimestamp(now - 600, UTC).isoformat()
    attempts = [
        {
            "status": "decision-round-started",
            "time": recent,
            "model_version": "m1",
            "decisions_recorded": 4,
        },
        {"status": "heartbeat", "time": recent, "model_version": "m1", "decisions_recorded": 4},
        {
            "status": "decision-round-completed",
            "time": stale,
            "model_version": "m1",
            "decisions_recorded": 2,
        },
    ]
    evidence = model_decision_evidence(attempts, now=now)
    assert evidence["state"] == "stale"
    assert evidence["confirmed_at"] == stale
    assert evidence["decisions_recorded"] == 2

    unknown = model_decision_evidence(attempts[:2], now=now)
    assert unknown["state"] == "unknown"
    assert unknown["confirmed_at"] is None and unknown["age_seconds"] is None

    fill_audit = paper_fill_evidence(
        {"executed_fills": 1, "updated_at": recent, "last_fill_at": stale}, now=now
    )
    assert fill_audit["state"] == "fresh" and fill_audit["fill_count"] == 1
    assert fill_audit["audit_at"] == recent
    assert fill_audit["last_fill_state"] == "stale"
    assert fill_audit["last_fill_at"] == stale

    absent = paper_fill_evidence(None, now=now)
    assert absent["state"] == "unknown" and absent["fill_count"] is None
    assert absent["audit_at"] is None and absent["last_fill_state"] == "unknown"


def test_paper_activity_stream_separates_quotes_decisions_attempts_and_fills():
    from rlm.v100.financial_snapshot import paper_activity_stream

    stream = paper_activity_stream(
        [
            {
                "sequence": 1,
                "time": "2026-10-15T11:00:00Z",
                "kind": "observation",
                "branch": "controller",
                "payload": {
                    "observation": {"kind": "quote", "symbol": "BTC-KRAKEN"},
                    "trades": [
                        {"action": "buy", "branch": "A", "symbol": "BTC-KRAKEN", "quantity": "0.1"}
                    ],
                },
            },
            {
                "sequence": 2,
                "time": "2026-10-15T11:00:01Z",
                "kind": "decision",
                "branch": "A",
                "payload": {
                    "proposal": {
                        "action": "open",
                        "symbol": "BTC-KRAKEN",
                        "proposal_source": "model-proposal",
                    }
                },
            },
            {
                "sequence": 3,
                "time": "2026-10-15T11:00:02Z",
                "kind": "decision",
                "branch": "B",
                "payload": {
                    "proposal": {
                        "action": "hold",
                        "symbol": "BTC-KRAKEN",
                        "receipt_source": "host-quote-freshness-guard",
                    }
                },
            },
            {
                "sequence": 4,
                "time": "2026-10-15T11:00:03Z",
                "kind": "research",
                "branch": "controller",
                "payload": {"status": "decision-round-started"},
            },
            {
                "sequence": 5,
                "time": "2026-10-15T11:00:04Z",
                "kind": "research",
                "branch": "B",
                "payload": {
                    "status": "proposal-rejected",
                    "error": "costs unknown",
                    "proposal": {"action": "close", "symbol": "BTC-KRAKEN"},
                },
            },
        ],
        pending_orders=1,
    )

    assert stream["counts"] == {
        "observations": 1,
        "model_decisions": 1,
        "automatic_freshness_holds": 1,
        "decision_round_attempts": 2,
        "paper_execution_attempts": 2,
        "confirmed_fills": 1,
        "pending_orders": 1,
    }
    assert [entry["category"] for entry in stream["entries"]] == [
        "observation",
        "confirmed_fill",
        "decision",
        "execution_attempt",
        "automatic_hold",
        "decision_attempt",
        "execution_attempt",
        "decision_attempt",
    ]
    assert stream["entries"][1]["label"] == "Confirmed paper fill"
    assert "fill not implied" in stream["entries"][3]["detail"]
    assert "blocked" in stream["entries"][6]["label"]


def test_live_paper_snapshot_returns_chain_checked_activity_counts(tmp_path):
    from rlm.v100.financial_snapshot import live_ledger
    from rlm.v100.paper import PaperBook

    book = PaperBook(tmp_path)
    book.initialize()
    book.note("controller", {"status": "decision-round-started", "cycle": 1})
    book.close()

    snapshot = live_ledger(tmp_path)
    activity = snapshot["activity_stream"]
    assert activity["state"] == "verified-tail"
    assert activity["window_events"] == 2
    assert activity["counts"]["observations"] == 0
    assert activity["counts"]["model_decisions"] == 0
    assert activity["counts"]["decision_round_attempts"] == 1
    assert activity["counts"]["paper_execution_attempts"] == 0
    assert activity["counts"]["confirmed_fills"] == 0
    assert activity["entries"][-1]["label"] == "Model decision round receipt"

