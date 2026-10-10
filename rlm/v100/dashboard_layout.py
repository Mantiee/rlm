"""Shared passive dashboard layout contract and host-owned live renderer."""

import base64
import hashlib
import json
import re
from html.parser import HTMLParser

PAGE = r"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Synta - mission progress</title>
<style>
:root{color-scheme:dark}body{font:15px system-ui;background:#101721;color:#e9eef6;margin:0;padding:24px;max-width:1200px;margin:auto}h1{font-size:26px}h2{font-size:19px}small,.muted{color:#adbdd1}a{color:#9ac5ff}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}.card,section{background:#192433;border:1px solid #304259;border-radius:12px;padding:18px;margin:14px 0}.card{margin:0}.value{display:block;font-size:22px;margin-top:8px;overflow-wrap:anywhere}progress{width:100%;height:24px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;max-height:420px;overflow:auto}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #304259}svg{width:100%;max-height:250px}#warning{color:#ffbf86}button{padding:8px 14px;background:#294464;border:0;color:white;border-radius:6px;cursor:pointer}
</style>
<h1>Synta - mission progress</h1><p class="muted">Read-only evidence. Paper P&amp;L and historical backtests are not actual income.</p>
<p id="fresh">Loading data...</p><p id="warning"></p><p id="layout" class="muted"></p><button id="reload">Refresh snapshot</button>
<div id="cards" class="grid"></div>
<section><h2>Current run evaluation</h2><p id="evaluation"></p><progress id="progress" max="1" value="0"></progress><p id="case" class="muted"></p></section>
<section><h2>Historical development backtests</h2><p class="muted">Historical net returns. Negative returns are losses. Assumed costs do not certify available fees or an edge.</p><svg id="chart" viewBox="0 0 800 220" role="img" aria-label="Historical backtest results"></svg><table><thead><tr><th>Trial</th><th>Net return</th><th>Drawdown</th><th>Fills</th><th>Triage</th><th>Double costs</th></tr></thead><tbody id="backtests"></tbody></table></section>
<section><h2>Paper A/B</h2><p>Zero fills do not validate a strategy. Forecasts are not trades.</p><pre id="paper"></pre><h3>Blockers</h3><pre id="blockers"></pre></section>
<section><h2>HTML reports and result files</h2><p class="muted">Reports are archived snapshots with their own timestamps and scope.</p><div id="reports"></div></section>
<section><h2>Observed work</h2><div id="actors" class="grid"></div><h3>Goals and plans</h3><pre id="goals"></pre><h3>Recent events</h3><div id="events"></div></section>
<details><summary>Helpers, GUI and official benchmark - source data</summary><pre id="workers"></pre></details>
<details><summary>Controller log</summary><pre id="log"></pre></details>
<details><summary>Training evidence and recent errors</summary><pre id="evidence"></pre></details>
<script>
let renderRoot=null;const selected=selector=>(renderRoot||document).querySelectorAll(selector);const $=id=>renderRoot?renderRoot.querySelector('#'+id):document.getElementById(id), fmt=x=>x===null||x===undefined?'unknown':String(x), pct=x=>typeof x==='number'&&Number.isFinite(x)?(x*100).toFixed(2)+'%':'unknown';
const layoutVersion='__LAYOUT_SHA__';let latestSnapshot=null,latestInference=null,pendingLayout=false;
function inferenceStamp(live){return Math.max(0,...(live?.events||[]).map(e=>Date.parse(e.time)||0))}
function mergeInference(data,live){return {...data,live:{...data.live,events:live.events,agents:[...(live.agents||[]),...(data.live?.agents||[]).filter(a=>a.source==='research/state/drones.sqlite3')]}}}
function card(label,value){const node=document.createElement('div');node.className='card';node.dataset.key='metric-'+label;const title=document.createElement('small');title.textContent=label;const strong=document.createElement('strong');strong.className='value';strong.textContent=fmt(value);node.append(title,strong);$('cards').append(node)}
function workView(data){
for(const id of ['actors','goals','events']){if(!$(id)){const node=document.createElement(id==='goals'?'pre':'div');node.id=id;if(id==='actors')node.className='grid';$('cards').after(node)}}
agentView(data);
$('actors').replaceChildren();
for(const a of data.live?.actors||[]){
 const node=document.createElement('article');node.className='card';node.dataset.scrollKey=a.label+'-resource-card';const title=document.createElement('h3');title.textContent=a.label;
 const state=document.createElement('strong');state.textContent=a.state+(a.stale?' | stale lub unknown':'');state.style.color=a.stale?'#ffbf86':'#75c5ae';
 const detail=document.createElement('p');detail.textContent=a.detail||'';node.append(title,state,detail);
 if(a.cooldown_seconds){const p=document.createElement('p');p.textContent='Helper cooldown: '+a.cooldown_seconds+' s';node.append(p)}
 const metrics=a.metrics||{};for(const [key,label,unit] of [['utilization','GPU','%'],['memory_used','VRAM',' MiB'],['power','Power',' W'],['temperature','Temperature',' °C'],['percent','CPU','%'],['available_ram_gib','Available RAM',' GiB']]){if(typeof metrics[key]==='number'){const p=document.createElement('p');p.textContent=label+': '+Number(metrics[key].toFixed(2))+unit;node.append(p)}}
 for(const w of a.workers||[]){const p=document.createElement('p');p.textContent=[w.name,w.phase,w.reason,w.stale?'stale':''].filter(Boolean).join(' | ');node.append(p)}
 for(const j of a.jobs||[]){const p=document.createElement('p');p.textContent=[j.kind,j.branch,j.state||j.phase,j.id,j.assignment].filter(Boolean).join(' | ');node.append(p)}
 const proof=document.createElement('details'),summary=document.createElement('summary'),raw=document.createElement('pre');proof.dataset.key=a.label+'-resource-proof';summary.textContent='Source data';raw.textContent=JSON.stringify(a,null,2);proof.append(summary,raw);node.append(proof);$('actors').append(node)
}
restoreDisclosures();const plans=data.live?.goals||{};$('goals').textContent=JSON.stringify(plans,null,2);$('goals').style.display='none';const oldPanel=$('goals').parentElement;if(oldPanel?.tagName==='SECTION')oldPanel.style.display='none';
}

function textField(parent,label,value){if(value===undefined||value===null||value==='')return;const p=document.createElement('p'),b=document.createElement('strong');b.textContent=label+': ';p.append(b,document.createTextNode(String(value)));parent.append(p)}
const disclosureStates=new Map(), disclosureScroll=new Map(), panelScroll=new Map();
function nodeKey(n){return n.nodeType===1?(n.id||n.dataset.key||n.dataset.scrollKey||''):''}
function updateNode(old,fresh){
if(old.nodeType!==fresh.nodeType||old.nodeName!==fresh.nodeName){old.replaceWith(fresh);return}
if(old.nodeType===3){if(old.nodeValue!==fresh.nodeValue)old.nodeValue=fresh.nodeValue;return}
if(old.attributes&&fresh.attributes){for(const a of [...old.attributes])if(!fresh.hasAttribute(a.name))old.removeAttribute(a.name);for(const a of [...fresh.attributes])if(old.getAttribute(a.name)!==a.value)old.setAttribute(a.name,a.value)}
const a=[...old.childNodes],b=[...fresh.childNodes],keyed=new Map(a.filter(nodeKey).map(n=>[nodeKey(n),n])),used=new Set();
for(let i=0;i<b.length;i++){const key=nodeKey(b[i]);let match=key?keyed.get(key):a.find(n=>!used.has(n)&&!nodeKey(n)&&n.nodeType===b[i].nodeType&&n.nodeName===b[i].nodeName);if(match){used.add(match);updateNode(match,b[i]);if(old.childNodes[i]!==match)old.insertBefore(match,old.childNodes[i]||null)}else old.insertBefore(b[i],old.childNodes[i]||null)}
for(const n of a)if(!used.has(n))n.remove();
}
function rememberDisclosures(){for(const n of selected('[data-scroll-key]'))if(n.dataset.scrollKey)panelScroll.set(n.dataset.scrollKey,[n.scrollTop,n.scrollLeft]);for(const n of selected('details[data-key]')){disclosureStates.set(n.dataset.key,n.open);const pre=n.querySelector?.('pre');if(pre)disclosureScroll.set(n.dataset.key,[pre.scrollTop,pre.scrollLeft])}}
function restoreDisclosures(){for(const n of selected('[data-scroll-key]')){const position=panelScroll.get(n.dataset.scrollKey);if(position){n.scrollTop=position[0];n.scrollLeft=position[1]}}for(let n of selected('details[data-key]')){if(disclosureStates.has(n.dataset.key))n.open=disclosureStates.get(n.dataset.key);const pre=n.querySelector?.('pre'),position=disclosureScroll.get(n.dataset.key);if(pre&&position){pre.scrollTop=position[0];pre.scrollLeft=position[1]}}}
function detailsField(parent,label,value,key,open=false){if(!value)return;const d=document.createElement('details'),h=document.createElement('summary'),p=document.createElement('pre');d.dataset.key=key;d.open=open;h.textContent=label;p.textContent=String(value);d.append(h,p);parent.append(d)}
function panel(id,title,after){let p=$(id);if(!p){p=document.createElement('section');p.id=id;const h=document.createElement('h2');h.textContent=title;p.append(h);after.after(p)}return p}
function agentView(data){
rememberDisclosures();
const plansPanel=panel('plan-panel','Mission plan',$('cards'));let planGrid=$('plan-grid');if(!planGrid){planGrid=document.createElement('div');planGrid.id='plan-grid';planGrid.className='plan-grid';plansPanel.append(planGrid)}planGrid.replaceChildren();
for(const [key,label] of [['long','Long-term goal'],['mid','Medium-term plan'],['short','Next steps']]){const box=document.createElement('article');box.className='plan-item';box.dataset.key='plan-'+key;const h=document.createElement('h3');h.textContent=label;box.append(h);const p=document.createElement('p');p.textContent=data.live?.goals?.[key]?.text||'No recorded plan';box.append(p);planGrid.append(box)}
const resources=panel('resource-panel','Compute resources',plansPanel);resources.append($('actors'));$('actors').className='resource-grid';
const agents=panel('agent-panel','Agent activity - tasks, decisions and evidence',resources);let grid=$('agent-grid');if(!grid){const note=document.createElement('p');note.className='muted';note.textContent='Recorded public output, tool results and token counts. Opted-in local output and reasoning stream during inference. Measured token counts arrive at completion. Model reasoning is unverified, not evidence of execution.';grid=document.createElement('div');grid.id='agent-grid';agents.append(note,grid)}
grid.replaceChildren();let history=$('agent-history');if(!history){history=document.createElement('details');history.id='agent-history';history.dataset.key='agent-history';const title=document.createElement('summary');title.textContent='Historical agent events and finished jobs';history.append(title);agents.append(history)}while(history.children.length>1)history.children[history.children.length-1].remove();
for(const a of [...(data.live?.agents||[])].sort((a,b)=>a.label.localeCompare(b.label))){const node=document.createElement('article');node.className='agent-row';node.dataset.key=a.label+'-agent-card';const head=document.createElement('header');const h=document.createElement('h3');h.textContent=a.label;const badge=document.createElement('span');badge.className='state-badge';badge.textContent=a.state||'unknown';head.append(h,badge);node.append(head);
textField(node,'Model / device',[a.model,a.device].filter(Boolean).join(' / '));
const u=a.usage||{};textField(node,'Last response tokens',Object.keys(u).length?`Input ${fmt(u.prompt_tokens)} | Output ${fmt(u.completion_tokens)} | Total ${fmt(u.total_tokens)} | recorded ${a.usage_at||''}`:'Not reported for this actor');
textField(node,'Response completeness',a.completeness);textField(node,'Finish reason',a.finish_reason);textField(node,'Duration',a.seconds===undefined?null:a.seconds+' s');
detailsField(node,'Assigned task',a.task,a.label+'-task');textField(node,'Tool',a.tool);
detailsField(node,'Live partial output - latest recorded window',a.stream_output,a.label+'-live-output',a.state==='inference-delta');detailsField(node,'Live local reasoning - latest recorded window',a.stream_reasoning,a.label+'-live-trace',a.state==='inference-delta');detailsField(node,'Returned local model reasoning (unverified)',a.returned_trace,a.label+'-trace');detailsField(node,'Public conclusion and next step',a.declaration,a.label+'-output');detailsField(node,'Observed result / error',a.result,a.label+'-result');detailsField(node,'Previous attempt, not current result',a.previous_result,a.label+'-previous');detailsField(node,'Source evidence / receipt',a.evidence_ref?JSON.stringify(a.evidence_ref):'No source evidence attached to this event',a.label+'-source');textField(node,'Evidence acquired',a.evidence_at);textField(node,'Event / job ID',a.event_id);textField(node,'History',a.historical?'Historical finished job, not current work':null);detailsField(node,'Journal file',a.source,a.label+'-journal');textField(node,'Last event',typeof a.updated==='number'?new Date(a.updated*1000).toLocaleString():a.updated);(a.historical?history:grid).append(node)}
if(!grid.children.length)textField(grid,'Activity','No recent recorded agent events');
const timeline=panel('timeline-panel','Action timeline - latest 200 recorded events',agents);timeline.append($('events'));$('events').replaceChildren();
for(const e of [...(data.live?.events||[])].reverse()){const d=document.createElement('details');d.className='event-row';d.dataset.key=e.id||e.time+'-'+(e.request_id||'')+'-'+e.actor+'-'+e.kind;const h=document.createElement('summary');h.textContent=[e.time,e.branch,e.actor,e.kind,e.tool].filter(Boolean).join(' | ');d.append(h);textField(d,'Live delta',e.delta_text);textField(d,'Returned local reasoning (unverified)',e.returned_trace);textField(d,'Task / arguments',e.task);textField(d,'Public output / result',e.summary);textField(d,'Error / status',e.detail);if(e.usage&&Object.keys(e.usage).length)textField(d,'Tokens',JSON.stringify(e.usage));textField(d,'Seconds',e.seconds);textField(d,'Event ID',e.id);textField(d,'Source evidence / receipt',Object.keys(e.evidence_ref||{}).length?JSON.stringify(e.evidence_ref):null);textField(d,'Journal file',e.source);$('events').append(d)}
restoreDisclosures();
}
function archivePanel(){
if($('action-archive'))return;
const p=document.createElement('section');p.id='action-archive';const h=document.createElement('h2');h.textContent='Full recorded action archive';const note=document.createElement('p');note.textContent='Browse daily journals from their first event. Live activity above is the latest 200 events. Archives contain recorded actions, not proof that an unlogged action occurred.';const select=document.createElement('select');select.id='archive-day';const button=document.createElement('button');button.textContent='Load action history';const status=document.createElement('p');const rows=document.createElement('div');p.append(h,note,select,button,status,rows);$('timeline-panel').after(p);let cursor=0;
async function load(reset=false){button.disabled=true;try{if(reset){cursor=0;rows.replaceChildren()}const query=new URLSearchParams({cursor:String(cursor),limit:'100'});if(select.value)query.set('day',select.value);const response=await fetch('/api/actions?'+query,{cache:'no-store'});if(!response.ok)throw Error('HTTP '+response.status);const result=await response.json();if(!select.options.length){for(const day of result.days||[]){const option=document.createElement('option');option.value=day;option.textContent=day;select.append(option)}select.value=result.day||''}
for(const e of result.events||[]){const d=document.createElement('details'),title=document.createElement('summary'),body=document.createElement('pre');title.textContent=[e.time,e.branch,e.actor,e.kind].filter(Boolean).join(' | ');body.textContent=JSON.stringify({context:e.context,payload:e.payload,id:e.id},null,2);d.append(title,body);rows.append(d)}cursor=result.next_cursor;status.textContent=`${rows.children.length} events shown | ${cursor} / ${result.archive_bytes||0} archive bytes`;button.textContent=result.has_more?'Load next 100 actions':'Check for new actions';}catch(error){status.textContent='Archive unavailable: '+error.message}finally{button.disabled=false}}
button.onclick=()=>load();select.onchange=()=>load(true);
}
function sourcesPanel(data){const p=panel('source-panel','Public source acquisition - measured work',$('plan-panel'));let body=$('source-view');if(!body){body=document.createElement('div');body.id='source-view';p.append(body)}body.replaceChildren();const s=data.report?.source_acquisition;if(!s){textField(body,'Status','No retrieval counters recorded since instrumentation');return}textField(body,'Distinct public URLs',s.distinct_urls);textField(body,'Distinct source contents',s.distinct_contents);textField(body,'Total actual retrievals',s.total_fetches);textField(body,'Unchanged rereads',s.unchanged_rereads);textField(body,'Last collection',new Date(s.updated*1000).toLocaleString());for(const r of s.latest||[]){const node=document.createElement('article');node.className='card';node.dataset.key='source-'+r.url;textField(node,'URL',r.url);textField(node,'Retrieved',new Date(r.last_fetched*1000).toLocaleString());textField(node,'Retrieval count for identical content',r.fetch_count);detailsField(node,'Content checksum',r.sha256,'source-proof-'+r.url);body.append(node)}}
function incomePanel(data){const p=panel('income-panel','Income opportunities - evidence and next tests',$('plan-panel'));let body=$('income-view');if(!body){body=document.createElement('div');body.id='income-view';p.append(body)}body.replaceChildren();const board=data.report?.income_opportunities||{},dispatch=data.report?.income_dispatch||{};textField(body,'Research policy',board.research_policy?.capital_research?'Hypothetical capital and zero-upfront':'Zero-upfront');textField(body,'Historical comparison',data.report?.market_research?.state);textField(body,'Historical comparison blocker',data.report?.market_research?.error);textField(body,'Actual verified income','Unknown - no payment ledger connected');const work=data.report?.income_work||{},quality=data.report?.research_quality||{};textField(body,'Concrete work / blocker',work.state||'not started');textField(body,'Work source / domain',`${work.url||''} | ${work.domain||''}`);textField(body,'Fresh source dossiers / registered proposals',`${work.sources_collected??'unknown'} / ${work.proposals_registered??'unknown'}`);textField(body,'Analysis blocker',work.error||'none reported');textField(body,'Hypothesis/test audit',`${quality.state||'unknown'} | ${quality.reason||''}`);if(work.dossier)detailsField(body,'Prepared dossier and source receipt',JSON.stringify(work),'income-work-dossier');textField(body,'Research assignment',dispatch.state||'Not observed');textField(body,'Assignment blocker',dispatch.error);const candidates=board.candidates||[];if(!candidates.length)textField(body,'Evidence-backed candidates','None recorded in this snapshot');for(const c of candidates){const s=c.specification||{};const node=document.createElement('article');node.className='card';node.dataset.key='income-'+c.id;const title=document.createElement('h3');title.textContent=s.title||c.id;node.append(title);textField(node,'Domain / state',`${s.domain||''} | ${c.state}`);textField(node,'Estimated net before personal tax',`${c.estimated_net_pln_low} to ${c.estimated_net_pln_high} PLN - unverified estimate`);textField(node,'Labor / time to first income',`${s.labor_hours} hours | ${s.first_income_days} days - estimates`);textField(node,'Mechanism',s.mechanism);textField(node,'Eligibility',s.eligibility);textField(node,'Blockers',s.blockers);textField(node,'Next falsifiable test',s.next_test);textField(node,'Reject if',s.failure_condition);detailsField(node,'Evidence IDs and proposal receipt',JSON.stringify(c),'income-evidence-'+c.id);body.append(node)}}
function learningPanel(data){rememberDisclosures();const p=panel('goal-learning-panel','Goal-directed ML pipeline',$('plan-panel'));let body=$('goal-learning-view');if(!body){body=document.createElement('div');body.id='goal-learning-view';p.append(body)}body.replaceChildren();const evidence=data.report?.mission_evidence||{},g=evidence.goal_learning||{};textField(body,'Active goal ID',g.goal_id||(data.report?.plans?.long?.id?'Goal metrics unavailable':'Goal status unavailable'));for(const [k,v] of Object.entries(g.learning_task||{}))textField(body,k,v);textField(body,'Recent forecasts',fmt(g.forecasts?.length));textField(body,'Recent resolved forecasts',g.recent_resolved);textField(body,'Recent Brier score',g.recent_brier);textField(body,'Admitted training examples',g.training_admission?.admitted);textField(body,'Confirmed optimizer updates',fmt(evidence.optimizer_updates_observed));textField(body,'Accepted weight updates',fmt(evidence.accepted_weight_updates_this_run));detailsField(body,'Training admission evidence',JSON.stringify(g.training_admission||{}),'goal-admission');textField(body,'Verified total outcomes',g.resolved_total);detailsField(body,'Independent observation pipeline',JSON.stringify(g.observer||{}),'goal-observer');const dispatch=g.owned_cpu_dispatch||{};textField(body,'Owned CPU dispatch',dispatch.state||'Not observed');textField(body,'Dispatch blocker',dispatch.reason);for(const branch of dispatch.branches||[])textField(body,'CPU trial',`${branch.id||'Not allocated'} | ${branch.state} | ${branch.reason||branch.scope||''}`);detailsField(body,'Owned CPU provenance and receipts',JSON.stringify(dispatch),'goal-compute');restoreDisclosures()}
function svg(tag,attrs,text){let node=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [key,val] of Object.entries(attrs))node.setAttribute(key,val);if(text!==undefined)node.textContent=text;return node}
function render(data){if(latestInference&&inferenceStamp(latestInference)>=inferenceStamp(data.live))data=mergeInference(data,latestInference);latestSnapshot=data;const m=data.mission||{},r=data.report||{},ev=(m.state?.phase==='official-public-baseline'?r.mission_evidence?.official_benchmark:null)||m.evaluation||{},learn=m.learning||{},gpu=data.gpu||{};
if(data.layout?.active_sha256&&layoutVersion!==data.layout.active_sha256){pendingLayout=true;$('reload').textContent='New layout available - load when ready'}$('layout').textContent='Master HTML: /workspace/dashboard/index.html | '+(data.layout?.state||'waiting')+(data.layout?.error?' | '+data.layout.error:'');
$('fresh').textContent='Collected: '+new Date(data.collected_at*1000).toLocaleString()+' | Live snapshots every 5 s; aggregate audit runs separately | Run: '+fmt(m.run)+(data.report_freshness?.state==='stale'?' | STALE REPORT: '+fmt(data.report_freshness.updated_at):'')+' | Aggregate audit: '+fmt(data.report_collection?.state);
workView(data);learningPanel(data);incomePanel(data);sourcesPanel(data);archivePanel();$('warning').textContent=(data.errors||[]).join('\n');$('cards').replaceChildren();card('Mission',m.running?'running':'stopped');card('Phase',m.state?.phase);card('V100 server startup',m.server_startup?.stage);card('Server blocker',m.server_startup?.error);card('Learning cycles',learn.completed_cycles);card('Accepted weight updates',r.accepted_weight_updates_this_run);card('Confirmed optimizer updates',r.mission_evidence?.optimizer_updates_observed);card('Context',m.state?.context_window);card('GPU utilization',gpu.utilization===undefined?null:gpu.utilization+'%');card('VRAM used / total',gpu.memory_used===undefined?null:gpu.memory_used+' / '+gpu.memory_total+' MiB');card('GPU temperature / power',gpu.temperature===undefined?null:gpu.temperature+' °C / '+gpu.power+' W');
$('evaluation').textContent=ev.total?`${ev.completed}/${ev.total} completed, ${ev.passed===undefined?'official grading':ev.passed+' passed'} (${ev.state})`:'No current evaluation counter';$('progress').max=ev.total||1;$('progress').value=ev.completed||0;$('case').textContent=ev.current_case?('Current case: '+ev.current_case):'';
const ledger=r.paper||{};$('paper').textContent=Object.entries(ledger.branches||{}).map(([name,b])=>`${name} | Equity ${fmt(b.equity)} ${ledger.currency||''} | Net P&L ${fmt(b.net_pnl_before_personal_tax)} | Costs ${fmt(b.modeled_costs)} | Resolved fills ${fmt(b.resolved_fills)}`).join('\n\n')||'Paper ledger status unavailable';$('blockers').textContent=(r.paper_blockers||[]).join('\n')||(Array.isArray(r.paper_blockers)?'No blockers in recorded snapshot':'Paper blockers unavailable');$('workers').textContent=JSON.stringify({drones:r.drones,external_compute:r.external_compute,desktop:r.desktop,official_benchmark:r.official_benchmark},null,2);$('log').textContent=data.controller_log||'Log unavailable';$('evidence').textContent=JSON.stringify({report_freshness:data.report_freshness,report_collection:data.report_collection,last_cycle:learn.last_cycle,evidence:r.mission_evidence,collection_errors:data.errors},null,2);
$('backtests').replaceChildren();$('chart').replaceChildren();const tests=r.recent_exploratory_backtests||[];let largest=Math.max(.01,...tests.map(t=>Math.abs(t.development_test?.net_return||0)));$('chart').append(svg('line',{x1:400,y1:0,x2:400,y2:220,stroke:'#8296b1'}));tests.forEach((test,i)=>{const result=test.development_test||{},tr=document.createElement('tr');[`${i+1}. ${test.parameters?.rule||''}`,pct(result.net_return),pct(result.max_sampled_drawdown),fmt(result.fills),test.triage?.state||'unreviewed',pct(test.double_cost_stress?.net_return)].forEach(v=>{let td=document.createElement('td');td.textContent=v;tr.append(td)});$('backtests').append(tr);if(typeof result.net_return==='number'&&Number.isFinite(result.net_return)){let width=Math.abs(result.net_return)/largest*280,y=15+i*48;$('chart').append(svg('rect',{x:result.net_return<0?400-width:400,y,width,height:26,fill:result.net_return<0?'#e68d80':'#75c5ae'}),svg('text',{x:8,y:y+19,fill:'#dce6f5','font-size':14},`${i+1}. ${pct(result.net_return)}`))}});if(!tests.length)$('chart').append(svg('text',{x:20,y:50,fill:'#adbdd1'},'No recorded results'));
$('reports').replaceChildren();for(const id of data.paper_reports||[]){const row=document.createElement('p');row.append(document.createTextNode(id+' '));for(const [name,label] of [['report.html','HTML and charts'],['report.json','JSON'],['trades.csv','CSV']]){const a=document.createElement('a');a.href='/paper/'+encodeURIComponent(id)+'/'+name;a.target='_blank';a.rel='noopener';a.textContent=label;row.append(a,document.createTextNode(' · '))}$('reports').append(row)}
}
function stablePaint(update){
if(typeof window==='undefined'||!document.body?.cloneNode){update();return}
const body=document.body,draft=body.cloneNode(true),x=window.scrollX,y=window.scrollY;
const containers=[...document.querySelectorAll('[id], [data-scroll-key]')].filter(n=>n.scrollTop||n.scrollLeft).map(n=>[n,n.scrollTop,n.scrollLeft]);
const atPoint=document.elementFromPoint?.(Math.min(80,window.innerWidth/2),Math.min(80,window.innerHeight/2));const anchor=atPoint?.closest?.('[data-key],section,details');const reader=anchor||[...document.querySelectorAll('details[data-key][open]')].find(n=>{const r=n.getBoundingClientRect();return r.bottom>0&&r.top<window.innerHeight});const readerTop=reader?.getBoundingClientRect().top;
try{renderRoot=draft;update()}finally{renderRoot=null}
updateNode(body,draft);for(const [n,top,left] of containers){n.scrollTop=top;n.scrollLeft=left}const shift=reader?.isConnected?reader.getBoundingClientRect().top-readerTop:0;if(window.scrollX!==x||window.scrollY!==y+shift)window.scrollTo(x,y+shift);
}
async function refresh(){try{let response=await fetch('/api/status',{cache:'no-store'});if(!response.ok)throw Error('HTTP '+response.status);const data=await response.json();stablePaint(()=>render(data))}catch(error){$('warning').textContent='Refresh failed: '+error.message}}
async function liveInference(){if(!latestSnapshot)return;try{const response=await fetch('/api/live-inference',{cache:'no-store'});if(!response.ok)throw Error('HTTP '+response.status);const live=await response.json();if(latestInference&&inferenceStamp(live)<inferenceStamp(latestInference))return;latestInference=live;stablePaint(()=>agentView(mergeInference(latestSnapshot,live)))}catch(error){$('warning').textContent='Live inference unavailable: '+error.message}}
async function readinessView(){try{const response=await fetch('/api/readiness',{cache:'no-store'});if(!response.ok)throw Error('HTTP '+response.status);const data=await response.json();stablePaint(()=>{const p=panel('readiness-panel','Commissioning evidence',$('goal-learning-panel')||$('plan-panel'));let body=$('readiness-view');if(!body){body=document.createElement('div');body.id='readiness-view';p.append(body)}body.replaceChildren();for(const c of data.checks||[])textField(body,c.name,`${c.state} | ${c.detail} | ${c.source||''}`)})}catch(error){$('warning').textContent='Readiness unavailable: '+error.message}}
$('reload').onclick=()=>pendingLayout?location.reload():refresh();refresh();setInterval(refresh,5000);setInterval(liveInference,2000);setInterval(readinessView,5000);
</script></html>"""


APP_SCRIPT = PAGE.split("<script>", 1)[1].split("</script>", 1)[0]
BASE_TEMPLATE = PAGE.split("<script>", 1)[0] + "</html>"
GUEST_TEMPLATE = "/workspace/dashboard/index.html"
REQUIRED_IDS = set(re.findall(r"\$\('([a-z]+)'\)", APP_SCRIPT)) - {"actors", "goals", "events"}

HOST_STYLE = """
:root{overflow-anchor:none;color-scheme:dark;--line:#2b4058;--muted:#a6b7cc}
body{max-width:1440px;margin:auto;padding:28px;background:linear-gradient(145deg,#0b1220,#14243b);color:#e7eef9;font:15px/1.6 system-ui}
h1{font-size:32px;letter-spacing:-1px;margin-bottom:8px}h2{font-size:21px;margin-top:0}h3{font-size:16px;overflow-wrap:anywhere}
section,details{box-shadow:0 8px 24px #0003;background:#111e30;border:1px solid var(--line);border-radius:16px;padding:22px;margin:20px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;align-items:start}
.card{background:#16263b;border:1px solid var(--line);border-radius:12px;padding:18px;margin:0;min-width:0}
.agent{border-top:3px solid #75c5ae}.agent p{font-size:13px;overflow-wrap:anywhere}.agent strong{color:#9ac5ff}
#agent-panel{border-color:#487087}#actors .card{border-top:3px solid #82aef9}.muted,small{color:var(--muted)}
.value{font-size:24px}pre{background:#0c1726;border-radius:8px;padding:14px;font-size:12px;max-height:280px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere}
summary{cursor:pointer;color:#9ac5ff}#events p{padding:10px 14px;border-left:2px solid #487087;background:#0c1726;font-size:12px;overflow-wrap:anywhere}
button{background:#365c87;padding:10px 18px;border-radius:8px}progress{accent-color:#75c5ae}td,th{padding:12px;border-bottom:1px solid var(--line)}
#cards .value{font-size:20px}#cards .card{min-height:90px}.plan-grid{display:grid;grid-template-columns:2fr 1fr 1fr;gap:22px}.plan-item{min-width:0}.plan-item p{white-space:pre-wrap;font-size:15px;line-height:1.65}.resource-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;align-items:start}.resource-grid .card{max-height:320px;overflow:auto}#agent-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.agent-row{padding:18px;border:1px solid var(--line);border-radius:12px;background:#16263b;min-width:0}.agent-row header{display:flex;align-items:center;justify-content:space-between;gap:15px}.agent-row h3{margin:0;font-size:18px}.agent-row p{font-size:14px;margin:8px 0;overflow-wrap:anywhere}.agent-row details{margin:8px 0;padding:12px;border-radius:8px}.agent-row pre{max-height:420px;font-size:14px}.state-badge{padding:4px 10px;border:1px solid #487087;border-radius:20px;color:#8edbc9;font-size:12px}.event-row{margin:8px 0;padding:12px;font-size:14px}.event-row p{white-space:pre-wrap;overflow-wrap:anywhere}#plan-panel{border-left:4px solid #82aef9}#timeline-panel{max-height:700px;overflow:auto}@media(max-width:900px){.plan-grid{grid-template-columns:1fr}.resource-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:650px){body{padding:12px}.grid,.resource-grid,#agent-grid{grid-template-columns:1fr}section,details{padding:14px}h1{font-size:26px}}
"""


def validate_css(value):
    value = re.sub(r"/\*.*?\*/", "", value, flags=re.S).lower()
    if "\\" in value or re.search(r"@import|url\s*\(|expression\s*\(|behavior\s*:", value):
        raise ValueError("CSS must not load external resources or execute code")


class LayoutValidator(HTMLParser):
    allowed_tags = set(
        "html head body title meta style h1 h2 h3 h4 h5 h6 p div span section article header footer main nav aside strong small em b i br hr ul ol li a button pre code details summary progress table thead tbody tfoot tr th td svg g rect line polyline polygon circle path text defs lineargradient stop".split()
    )

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids = set()
        self.in_style = False
        self.styles = []

    def handle_starttag(self, tag, attrs):
        if tag not in self.allowed_tags:
            raise ValueError("Unsupported layout element: " + tag)
        if tag == "style":
            self.in_style = True
        for name, value in attrs:
            value = value or ""
            if name.startswith("on") or name in (
                "src",
                "srcdoc",
                "http-equiv",
                "action",
                "formaction",
                "xlink:href",
            ):
                raise ValueError("Active or external layout attribute is not allowed")
            if name == "href" and not (value.startswith("#") or value.startswith("/paper/")):
                raise ValueError("Layout links must target existing paper reports or anchors")
            if name == "href" and any(char in value for char in ("\\", "\n", "\r")):
                raise ValueError("Invalid layout link")
            if name == "style":
                validate_css(value)
            if name == "id":
                if value in self.ids:
                    raise ValueError("Duplicate layout ID: " + value)
                self.ids.add(value)

    def handle_endtag(self, tag):
        if tag not in self.allowed_tags:
            raise ValueError("Unsupported closing element")
        if tag == "style":
            self.in_style = False

    def handle_data(self, data):
        if self.in_style:
            self.styles.append(data)


def validate_template(content):
    if not isinstance(content, str) or not 1 <= len(content.encode()) <= 256 * 1024:
        raise ValueError("Layout must contain 1..262144 UTF-8 bytes")
    parser = LayoutValidator()
    parser.feed(content)
    parser.close()
    if parser.in_style or content.count("<!--") != content.count("-->"):
        raise ValueError("Close style elements and HTML comments")
    validate_css("".join(parser.styles))
    missing = REQUIRED_IDS - parser.ids
    if missing:
        raise ValueError("Keep dashboard component IDs: " + ", ".join(sorted(missing)))
    if not re.search(r"</html\s*>\s*\Z", content, re.I):
        raise ValueError("Keep the closing html element")
    return hashlib.sha256(content.encode()).hexdigest()


def render_template(content, identity):
    for old, new in {
        "V100 - postęp misji": "Synta - mission progress",
        "Testy bieżącego przebiegu": "Current run evaluation",
        "Ostatnie backtesty - eksploracyjne": "Historical development backtests",
        "Blokady": "Blockers",
        "Raporty HTML z wykresami i pliki wyników": "HTML reports and result files",
        "Cele i plany": "Goals and plans",
        "Ostatnie zdarzenia": "Recent events",
        "Aktualny log kontrolera": "Controller log",
        "Dowody treningu i ostatnie błędy": "Training evidence and recent errors",
        "Odśwież widok": "Refresh snapshot",
        "Próba": "Trial",
        "Wynik netto": "Net return",
        "Obsunięcie": "Drawdown",
        "Wykonania": "Fills",
        'lang="pl"': 'lang="en"',
    }.items():
        content = content.replace(old, new)
    content = content.replace("Synta - mission progress", "Synta - mission control")
    script = (
        "<style>"
        + HOST_STYLE
        + "</style><script>"
        + APP_SCRIPT.replace("__LAYOUT_SHA__", identity)
        + "</script>"
    )
    return re.sub(
        r"</html\s*>\s*\Z", lambda match: script + match.group(), content, count=1, flags=re.I
    )


def guest_layout(root):
    from rlm.v100.desktop import run

    seed = base64.b64encode(BASE_TEMPLATE.encode()).decode()
    program = (
        "import base64,json\nfrom pathlib import Path\n"
        f"p=Path({GUEST_TEMPLATE!r})\np.parent.mkdir(parents=True,exist_ok=True)\n"
        f"if not p.exists(): p.write_bytes(base64.b64decode({seed!r}))\n"
        "if p.stat().st_size>262144: raise ValueError('Layout exceeds 256 KiB')\n"
        "print(json.dumps({'html_base64':base64.b64encode(p.read_bytes()).decode()}))\n"
    )
    script = "python3 - <<'V100_LAYOUT'\n" + program + "V100_LAYOUT\n"
    result = run(root, script, seconds=8, output_limit=360000)
    if result["exit_code"]:
        raise RuntimeError(result["stderr"][:300] or "Guest layout unavailable")
    value = json.loads(result["stdout"])
    return base64.b64decode(value["html_base64"], validate=True).decode()
