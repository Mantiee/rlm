import base64
import hashlib
import json
from types import SimpleNamespace

import pytest

from rlm.v100 import (
    architecture_goal_gate,
    background_observer,
    dashboard_editor,
    dashboard_layout,
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
    assert len(report[0]["assignment"]) <= 300
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


def test_trusted_live_renderer_runs_on_older_layout_without_actor_sections():
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
 replaceChildren(...items){this.children=items}
 setAttribute(k,v){this[k]=v}
}
const document={querySelectorAll:()=>[],getElementById:id=>nodes[id],createElement:t=>new Element(t),createElementNS:(ns,t)=>new Element(t),createTextNode:t=>t};
const location={reload(){throw Error('Unexpected reload')}};
const setInterval=()=>{};
const fetch=()=>Promise.resolve({ok:true,json:()=>Promise.resolve(sample)});
const sample={collected_at:1,mission:{running:true,state:{phase:'research'}},report:{},live:{agents:[{label:'A / researcher',state:'tool-start',task:'Fetch public source',declaration:'<script>Unverified hypothesis</script>',result:'status: ready',source:'timeline.jsonl'}],actors:[{label:'Windows CPU',state:'idle',workers:[{name:'windows-cpu',phase:'idle',reason:'ready'}]},{label:'V100',state:'measured',metrics:{utilization:91}}],goals:{long:{text:'Operator goal'}},events:[{actor:'master',kind:'inference-start'}]}};
"""
    for identity in dashboard_layout.REQUIRED_IDS:
        program += f"new Element('div').id={json.dumps(identity)};\n"
    program += dashboard_layout.APP_SCRIPT
    program += "\nrender(sample);if(nodes.actors.children.length!==2||!nodes.goals.textContent.includes('Operator goal')||nodes.events.children.length!==1||nodes['agent-grid'].children.length!==1)throw Error('Live renderer failed');\n"
    program += """
const textNode={nodeType:3,nodeName:'#text',nodeValue:'previous'};const reader={nodeType:1,nodeName:'PRE',childNodes:[textNode],scrollTop:77};const freshReader={nodeType:1,nodeName:'PRE',childNodes:[{nodeType:3,nodeName:'#text',nodeValue:'updated'}]};updateNode(reader,freshReader);if(reader.childNodes[0]!==textNode||textNode.nodeValue!=='updated'||reader.scrollTop!==77)throw Error('Reading node replaced');
const old={dataset:{key:'reader'},open:false};let disclosures=[old];document.querySelectorAll=()=>disclosures;rememberDisclosures();
const next={dataset:{key:'reader'},open:true};disclosures=[next];restoreDisclosures();if(next.open)throw Error('User-closed detail reopened');
next.open=true;rememberDisclosures();const newer={dataset:{key:'reader'},open:false};disclosures=[newer];restoreDisclosures();if(!newer.open)throw Error('User-open detail lost');
"""
    result = subprocess.run(["node", "-"], input=program, text=True, capture_output=True, timeout=5)
    assert result.returncode == 0, result.stderr


def test_invalid_dashboard_source_can_be_read_without_being_published(tmp_path, monkeypatch):
    invalid = dashboard_layout.BASE_TEMPLATE.replace("</html>", "<script>alert(1)</script></html>")
    monkeypatch.setattr(dashboard_editor, "guest_layout", lambda root: invalid)
    value = dashboard_editor.status(tmp_path, include_html=True)
    assert not value["valid"] and not value["published"]
    assert value["html"] == invalid and value["repair_tool"] == "repair_dashboard"
    assert "html" not in dashboard_editor.status(tmp_path)


def test_dashboard_repair_removes_scripts_preserves_design_and_uses_backup(tmp_path, monkeypatch):
    original = dashboard_layout.BASE_TEMPLATE.replace("<h1>", '<h1 style="color:red">').replace(
        "</html>", '<SCRIPT type="text/javascript">bad()</SCRIPT></html>'
    )
    monkeypatch.setattr(dashboard_editor, "guest_layout", lambda root: original)
    writes = []
    monkeypatch.setattr(
        dashboard_editor,
        "write",
        lambda root, content: writes.append(content) or {"written": True, "published": False},
    )
    value = dashboard_editor.repair(tmp_path)
    assert value["repaired"] and not value["published"]
    assert "scripts removed" in value["mode"]
    assert "color:red" in writes[0] and "<SCRIPT" not in writes[0]
    dashboard_layout.validate_template(writes[0])


def test_capabilities_lists_only_the_current_request_tools(tmp_path):
    from rlm.v100.research_tools import ResearchTools

    tools = ResearchTools(tmp_path, {}, "A")
    tools.allowed_tool_names = {"read_dashboard", "repair_dashboard"}
    names = {row["name"] for row in tools.execute("capabilities", {})["tools"]}
    assert names == tools.allowed_tool_names


def test_token_metrics_survive_summary_and_do_not_expose_reasoning(tmp_path):
    folder = tmp_path / "research/logs/activity/2026-10-09"
    folder.mkdir(parents=True)
    rows = [
        {
            "actor": "researcher",
            "branch": "A",
            "kind": "model-output",
            "category": "decisions",
            "payload": {"content": {"answer": "Public conclusion", "thinking": "private"}},
        },
        {
            "actor": "researcher",
            "branch": "A",
            "kind": "inference-finished",
            "time": "now",
            "payload": {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "reasoning": "private",
                },
                "seconds": 2,
            },
        },
    ]
    (folder / "timeline.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    events = live_status.recent_events(tmp_path)
    agent = live_status.agent_views(events, [])[0]
    assert agent["usage"] == {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
    assert agent["seconds"] == 2
    assert "Public conclusion" in agent["declaration"]
    assert "private" not in json.dumps(events)


def test_operator_opt_in_records_returned_local_trace_and_preserves_public_output(
    tmp_path, monkeypatch
):
    from rlm.clients.llamacpp import LlamaCppClient

    root = tmp_path / "research"
    root.mkdir()
    (root / "user-preferences.json").write_text(json.dumps({"capture_local_model_trace": True}))
    client = LlamaCppClient(model_name="owned-model", activity_root=tmp_path)
    trace = "Test a hypothesis. " * 500
    response = {
        "choices": [
            {
                "message": {"content": "Public answer", "reasoning_content": trace},
                "finish_reason": "stop",
            }
        ],
        "usage": {},
    }
    monkeypatch.setattr(client, "http_request", lambda endpoint, data: response)
    assert client.request("/v1/chat/completions", {}) == response
    events = live_status.recent_events(tmp_path)
    agent = live_status.agent_views(events, [])[0]
    assert agent["returned_trace"] == trace
    assert "Public answer" in agent["declaration"]
