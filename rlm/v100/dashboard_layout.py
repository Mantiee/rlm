"""Shared passive dashboard layout contract and host-owned live renderer."""

import base64
import hashlib
import json
import re
from html.parser import HTMLParser

from rlm.v100.dashboard_insights import SCRIPT as INSIGHTS_SCRIPT
from rlm.v100.dashboard_insights import STYLE as INSIGHTS_STYLE

PAGE = r"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Synta — centrum na żywo</title>
<style>
:root{color-scheme:dark}body{font:15px system-ui;background:#101721;color:#e9eef6;margin:0;padding:24px;max-width:1200px;margin:auto}h1{font-size:26px}h2{font-size:19px}small,.muted{color:#adbdd1}a{color:#9ac5ff}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}.card,section{background:#192433;border:1px solid #304259;border-radius:12px;padding:18px;margin:14px 0}.card{margin:0}.value{display:block;font-size:22px;margin-top:8px;overflow-wrap:anywhere}progress{width:100%;height:24px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;max-height:420px;overflow:auto}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #304259}svg{width:100%;max-height:250px}#warning{color:#ffbf86;white-space:pre-wrap}#warning:not(:empty){padding:12px;border:1px solid #ffbf86;border-radius:8px;font-weight:600}button{padding:8px 14px;background:#294464;border:0;color:white;border-radius:6px;cursor:pointer}
.attempt-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}.attempt-row{background:#14202e;border-radius:8px;padding:14px;min-width:0}.attempt-row h3{font-size:15px;margin:0 0 8px}.attempt-status{font-weight:600}.attempt-status[data-state="fresh"]{color:#76d7ae}.attempt-status[data-state="stale"],.attempt-status[data-state="blocked"]{color:#ffbf86}.attempt-status[data-state="unknown"]{color:#adbdd1}
</style>
<h1>Synta — centrum na żywo</h1><p class="muted">Misja, paper, eksperymenty i inne źródła w jednym widoku. P&amp;L paper i backtesty nie są realnym przychodem.</p>
<p id="fresh">Loading data...</p><p id="warning"></p><p id="layout" class="muted"></p><button id="reload">Refresh snapshot</button>
<div id="cards" class="grid"></div>
<section><h2>Current run evaluation</h2><p id="evaluation"></p><progress id="progress" max="1" value="0"></progress><p id="case" class="muted"></p></section>
<section><h2>Historical development backtests</h2><p class="muted">Historical net returns. Negative returns are losses. Assumed costs do not certify available fees or an edge.</p><svg id="chart" viewBox="0 0 800 220" role="img" aria-label="Historical backtest results"></svg><table><thead><tr><th>Trial</th><th>Net return</th><th title="Largest equity loss from a prior peak, including assumed exit costs. Not standard deviation.">Maximum drawdown</th><th>Fills</th><th>Triage</th><th title="Net return from a separate simulation with twice the assumed fees and slippage.">Net return, double costs</th></tr></thead><tbody id="backtests"></tbody></table></section>
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
function paperLabsPanel(data){
 const p=panel('paper-labs-panel','Eksperymenty paper — osobna księga i wycena',$('paper').parentElement||$('paper'));
 let body=$('paper-labs-view');if(!body){body=document.createElement('div');body.id='paper-labs-view';p.append(body)}body.replaceChildren();
 const labs=data.report?.paper_labs||[];
 if(!labs.length){textField(body,'Stan','Brak odczytu ksiąg eksperymentalnych');return}
 for(const lab of labs){
  const section=document.createElement('div'),title=document.createElement('h3');title.textContent=lab.id;section.append(title);body.append(section);
  textField(section,'Księga',lab.ledger);textField(section,'Stan odczytu',lab.state);textField(section,'Błąd',lab.error);
  textField(section,'Wiek zapisanego notowania',typeof lab.tick_age_seconds==='number'?Math.round(lab.tick_age_seconds)+' s':'UNKNOWN');
  textField(section,'Aktualność notowania',lab.quote_stale===true?'STALE':lab.quote_stale===false?'Ostatni zapis w ciągu 180 s; źródło i FX wymagają osobnej weryfikacji':'UNKNOWN');
  const money=value=>Number.isFinite(value)?value.toLocaleString('pl-PL',{minimumFractionDigits:2,maximumFractionDigits:2})+' PLN':'UNKNOWN';
  for(const [branch,a] of Object.entries(lab.accounts||{}))textField(section,'Portfel '+branch,`Gotówka ${money(a.cash)} | Jednostki ${Number.isFinite(a.units)?a.units.toFixed(8):'UNKNOWN'} | Wycena brutto ${money(a.gross_equity)}`);
  textField(section,'Podstawa wyceny',lab.valuation);textField(section,'Pełne koszty wykonania','UNKNOWN — promocja zablokowana');
  textField(section,'Kapitał początkowy / P&L','UNKNOWN — baza nie zapisuje pochodzenia kapitału początkowego');textField(section,'Zakres',lab.scope);
  const activity=lab.activity_24h,counts=activity?.counts||{};
  if(activity){
   const count=value=>Number.isInteger(value)?value.toLocaleString('pl-PL'):'UNKNOWN';
   textField(section,'Aktywność zapisów — ostatnie 24 h',`${count(counts.quotes)} notowań · ${count(counts.decisions)} decyzji · ${count(counts.tests)} testów · ${count(counts.scored_hypotheses)} ocenionych hipotez · ${count(counts.fills)} filli`);
   const lifetime=activity.lifetime_counts||{},lastFill=lab.recent_fills?.[0];
   textField(section,'Wykonania paper',`${count(counts.fills)} w 24 h · ${count(lifetime.fills)} łącznie w tej księdze`+(lastFill?.ts?` · ostatnie: ${lastFill.ts} (${lastFill.branch||'?'} ${lastFill.side||''})`:''));
   textField(section,'Jak czytać aktywność','Decyzje HOLD i testy pokazują pracę symulacji; nie są zleceniami ani przychodem. Wykres portfela pokazuje wycenę, więc może pozostać prawie płaski, gdy portfele nie zmieniają pozycji.');
   const hours=Array.isArray(activity.hourly)?activity.hourly:[];
   if(hours.length&&Number.isInteger(counts.quotes)&&Number.isInteger(counts.decisions)){
    const plot=svg('svg',{viewBox:'0 0 800 250',role:'img','aria-label':'Zapisana aktywność eksperymentu w ostatnich 24 godzinach'}),left=74,right=782,top=42,bottom=188,max=Math.max(1,...hours.flatMap(row=>[row.quotes||0,row.decisions||0]));
    const label=(x,y,value,color='#bdcde1',anchor='start')=>{const node=svg('text',{x,y,fill:color,'font-size':12,'text-anchor':anchor});node.textContent=value;plot.append(node)};
    label(left,20,'Zapisane notowania i decyzje na godzinę — ostatnie 24 h');
    for(let i=0;i<=4;i++){const y=top+(bottom-top)*i/4,value=Math.round(max*(1-i/4));plot.append(svg('line',{x1:left,x2:right,y1:y,y2:y,stroke:'#35445a','stroke-dasharray':'4 4'}));label(left-8,y+4,String(value),'#bdcde1','end')}
    const slot=(right-left)/Math.max(1,hours.length),bar=Math.min(9,slot*.28);
    for(let i=0;i<hours.length;i++){const row=hours[i],x=left+i*slot+slot*.18;for(const [key,offset,color] of [['quotes',0,'#57b7ff'],['decisions',bar+2,'#f9c36b']]){const value=Number(row[key]||0),height=(bottom-top)*value/max;if(height>0)plot.append(svg('rect',{x:x+offset,y:bottom-height,width:bar,height,fill:color}));}}
    label(left,bottom+24,hours[0].hour?.slice(11)+' UTC');label(right,bottom+24,hours.at(-1).hour?.slice(11)+' UTC','#bdcde1','end');
    plot.append(svg('line',{x1:left,x2:left+18,y1:228,y2:228,stroke:'#57b7ff','stroke-width':3}));label(left+24,232,'Notowania','#bdcde1');plot.append(svg('line',{x1:left+120,x2:left+138,y1:228,y2:228,stroke:'#f9c36b','stroke-width':3}));label(left+144,232,'Decyzje (w tym HOLD)','#bdcde1');section.append(plot);
   }
   textField(section,'Okno aktywności',`${activity.since} → ${activity.until} | ${activity.scope}`);
  }
  const curve=(lab.gross_curve||[]).filter(row=>Number.isFinite(row.a)&&Number.isFinite(row.b));
  if(curve.length){const chart=svg('svg',{viewBox:'0 0 800 270',role:'img','aria-label':lab.id+' — wycena brutto w PLN w czasie, portfele A i B'}),values=curve.flatMap(row=>[row.a,row.b]);let lo=Math.min(...values),hi=Math.max(...values),pad=Math.max(1,(hi-lo)*.1);lo-=pad;hi+=pad;
   const label=(x,y,text,color='#bdcde1',anchor='start')=>{const n=svg('text',{x,y,fill:color,'font-size':13,'text-anchor':anchor});n.textContent=text;chart.append(n)};
   label(80,18,'Wycena brutto (PLN) — przed niezweryfikowanymi kosztami');
   for(let i=0;i<=4;i++){const y=48+140*i/4,value=hi-(hi-lo)*i/4;chart.append(svg('line',{x1:80,x2:775,y1:y,y2:y,stroke:'#35445a','stroke-dasharray':'4 4'}));label(72,y+4,value.toFixed(2),'#bdcde1','end')}
   for(const [key,color,name] of [['a','#57b7ff','Portfel A'],['b','#f9c36b','Portfel B']]){const points=curve.map((row,i)=>`${80+695*i/Math.max(1,curve.length-1)},${48+140*(hi-row[key])/(hi-lo)}`).join(' ');chart.append(svg('polyline',{points,fill:'none',stroke:color,'stroke-width':3}));const x=key==='a'?80:420;chart.append(svg('line',{x1:x,x2:x+28,y1:245,y2:245,stroke:color,'stroke-width':3}));label(x+38,250,name+' · '+money(curve.at(-1)[key]),color)}
   const clock=ts=>new Date(ts).toLocaleTimeString('pl-PL',{hour:'2-digit',minute:'2-digit',second:'2-digit'});label(80,210,clock(curve[0].ts));label(775,210,clock(curve.at(-1).ts),'#bdcde1','end');label(425,228,'Czas lokalny','#bdcde1','middle');section.append(chart);textField(section,'Krzywa',lab.curve_valuation+' | '+curve[0].ts+' → '+curve.at(-1).ts);if(curve.length===1)textField(section,'Historia','Jeden punkt; za mało danych do pokazania trendu');textField(section,'Interpretacja','Pozioma linia A oznacza portfel w gotówce. Zmiana wyceny B wynika z zapisanej pozycji i notowań; nie potwierdza przewagi ani zysku netto.')
  }else textField(section,'Krzywa',lab.curve_valuation||'Brak porównywalnych punktów');
  detailsField(section,'Ostatnie zapisane wykonania (osobna księga)',JSON.stringify(lab.recent_fills||[]),lab.id+'-fills');
 }
}
function workView(data){
githubCiPanel(data);
paperLabsPanel(data);
modelDeploymentView(data);
statusFreshnessView(data);
for(const id of ['actors','goals','events']){if(!$(id)){const node=document.createElement(id==='goals'?'pre':'div');node.id=id;if(id==='actors')node.className='grid';$('cards').after(node)}}
agentView(data);
$('actors').replaceChildren();
for(const a of data.live?.actors||[]){
 const node=document.createElement('article');node.className='card';node.dataset.scrollKey=a.label+'-resource-card';const title=document.createElement('h3');title.textContent=a.label;
 const state=document.createElement('strong');state.textContent=a.state+(a.stale?' | stale or unknown':'');state.style.color=a.stale?'#ffbf86':'#75c5ae';
 const detail=document.createElement('p');detail.textContent=a.detail||'';node.append(title,state,detail);
 if(a.cooldown_seconds){const p=document.createElement('p');p.textContent='Helper cooldown: '+a.cooldown_seconds+' s';node.append(p)}
 const metrics=a.metrics||{};for(const [key,label,unit] of [['utilization','GPU','%'],['memory_used','VRAM',' MiB'],['power','Power',' W'],['temperature','Temperature',' °C'],['percent','CPU','%'],['available_ram_gib','Available RAM',' GiB']]){if(typeof metrics[key]==='number'){const p=document.createElement('p');p.textContent=label+': '+Number(metrics[key].toFixed(2))+unit;node.append(p)}}
 for(const w of a.workers||[]){const p=document.createElement('p');p.textContent=[w.name,w.phase,w.reason,w.stale?'stale':''].filter(Boolean).join(' | ');node.append(p)}
 for(const j of a.jobs||[]){const p=document.createElement('p');p.textContent=[j.kind,j.branch,j.state||j.phase,j.id,j.assignment].filter(Boolean).join(' | ');node.append(p)}
 const proof=document.createElement('details'),summary=document.createElement('summary'),raw=document.createElement('pre');proof.dataset.key=a.label+'-resource-proof';summary.textContent='Source data';raw.textContent=JSON.stringify(a,null,2);proof.append(summary,raw);node.append(proof);$('actors').append(node)
}
restoreDisclosures();const plans=data.live?.goals||{};$('goals').textContent=JSON.stringify(plans,null,2);$('goals').style.display='none';const oldPanel=$('goals').parentElement;if(oldPanel?.tagName==='SECTION')oldPanel.style.display='none';
}

function statusFreshnessView(data){
const rows=data.report?.status_freshness||{},p=panel('status-freshness-panel','Status receipt freshness',$('cards'));p.replaceChildren();const h=document.createElement('h2');h.textContent='Status receipt freshness';p.append(h);const note=document.createElement('p');note.className='muted';note.textContent='Age of the recorded status receipt only; this does not validate the underlying evidence. Unknown means no trusted timestamp; unavailable means the file could not be read.';p.append(note);const table=document.createElement('table'),head=document.createElement('tr');for(const label of ['Source','State','Observed at','Age']){const th=document.createElement('th');th.textContent=label;head.append(th)}table.append(head);for(const [name,row] of Object.entries(rows)){const tr=document.createElement('tr');for(const value of [row.source||name,row.state||'unknown',typeof row.observed_at==='number'?new Date(row.observed_at*1000).toLocaleString():'unknown',typeof row.age_seconds==='number'?`${Math.round(row.age_seconds)} s`:'unknown']){const td=document.createElement('td');td.textContent=String(value);tr.append(td)}table.append(tr)}if(!Object.keys(rows).length){const tr=document.createElement('tr'),td=document.createElement('td');td.colSpan=4;td.textContent='No status receipts recorded';tr.append(td);table.append(tr)}p.append(table);
}

function githubCiPanel(data){
 const snap=data.report?.github_ci||{},p=panel('github-ci-panel','GitHub PRs and checks',$('cards'));p.replaceChildren();
 const h=document.createElement('h2');h.textContent='GitHub PRs and CI checks';p.append(h);
 const note=document.createElement('p');note.className='muted';note.textContent='Public, read-only check-run snapshots for Mantiee/rlm. Green checks validate only the recorded commit; they do not prove deployment, runtime correctness, or revenue.';p.append(note);
 textField(p,'Repository',snap.repository||'unknown');textField(p,'Snapshot state',snap.state||'unknown');
 textField(p,'Observed at',snap.observed_at?new Date(snap.observed_at).toLocaleString():'unknown');
 if(typeof snap.age_seconds==='number')textField(p,'Snapshot age',Math.max(0,Math.round(snap.age_seconds))+' seconds'+(snap.state==='stale'?' — STALE':''));
 if(snap.error)textField(p,'Feed detail',snap.error.kind||'unknown');
 const table=document.createElement('table'),head=document.createElement('tr');
 for(const label of ['PR','Title','State','Checks','Head SHA']){const th=document.createElement('th');th.textContent=label;head.append(th)}table.append(head);
 const rows=Array.isArray(snap.pull_requests)?snap.pull_requests:[];
 for(const row of rows.slice(0,5)){
  const tr=document.createElement('tr'),prCell=document.createElement('td'),titleCell=document.createElement('td');
  if(Number.isInteger(row.number)&&row.number>0&&row.number<=1000000){const link=document.createElement('a');link.href='https://github.com/Mantiee/rlm/pull/'+row.number;link.rel='noopener';link.target='_blank';link.textContent='#'+row.number;prCell.append(link)}else prCell.textContent='unknown';
  titleCell.textContent=typeof row.title==='string'?row.title:'unknown';
  const stateCell=document.createElement('td');stateCell.textContent=(row.state||'unknown')+' / '+(row.check_state||'unknown');
  const checksCell=document.createElement('td'),checks=Array.isArray(row.check_runs)?row.check_runs:[];
  checksCell.textContent=checks.length?checks.slice(0,100).map(check=>(check.name||'unknown')+': '+(check.status||'unknown')+(check.conclusion?' ('+check.conclusion+')':'')).join('\n'):'No check runs; status unknown';
  const shaCell=document.createElement('td');shaCell.textContent=typeof row.head_sha==='string'?row.head_sha:'unknown';
  tr.append(prCell,titleCell,stateCell,checksCell,shaCell);table.append(tr);
 }
 if(!rows.length){const tr=document.createElement('tr'),cell=document.createElement('td');cell.colSpan=5;cell.textContent='No cached PR/check snapshot; status unknown';tr.append(cell);table.append(tr)}
 p.append(table);
}

function modelDeploymentView(data){
const p=panel('model-deployment-panel','Models by computer and role',$('cards'));p.replaceChildren();const title=document.createElement('h2');title.textContent='Models by computer and role';p.append(title);const note=document.createElement('p');note.textContent='Qualified is a completed test, not proof of current generation. Missing receipts remain unknown. The accepted V100 learner is shown separately in Learning.';p.append(note);const table=document.createElement('table'),head=document.createElement('tr');for(const label of ['Computer','Role','Model','Observed state']){const th=document.createElement('th');th.textContent=label;head.append(th)}table.append(head);for(const row of data.live?.model_deployment||[]){const tr=document.createElement('tr');for(const key of ['host','role','model','state']){const td=document.createElement('td');td.textContent=row[key]||'unknown';tr.append(td)}table.append(tr)}p.append(table);
}

function currentActivityView(data){
const v=data.report?.current_activity||{},e=v.evaluation||{},p=panel('current-activity-panel','Now',$('cards'));p.replaceChildren();const h=document.createElement('h2');h.textContent='What Synta is doing now';p.append(h);textField(p,'Stage',v.stage||data.mission?.state?.phase||'unknown');const age=(Date.now()/1000)-data.collected_at;const badge=document.createElement('p');badge.id='current-snapshot-age';badge.dataset.collected=data.collected_at;badge.textContent=Number.isFinite(age)?`Snapshot collected ${Math.max(0,Math.round(age))} seconds ago${age>300?' — delayed':''}`:'Snapshot time unknown';p.append(badge);if(e.total){const meter=document.createElement('progress');meter.max=e.total;meter.value=e.completed||0;p.append(meter);textField(p,'Independent evaluation',`${e.completed}/${e.total} cases completed; ${e.passed} passed`);textField(p,'Current case',e.current_case);textField(p,'Evaluation activity',`${Math.round(e.age_seconds||0)} seconds since last update${e.stale?' - no recent progress':''}`);if(e.last_case&&!e.last_case.passed)textField(p,'Last failure',e.last_case.error||e.last_case.finish_reason||'Answer did not satisfy the validator; inspect its quality report');}textField(p,'Closed cycles in this run',v.completed_cycles);textField(p,'Closed cycle receipts across retained runs',v.history?.completed_cycles);textField(p,'History coverage',v.history?`${v.history.runs_checked} most recent retained runs; includes failed or rejected cycles`:'unknown');const k=v.knowledge||{};detailsField(p,'Archive indexing and goal-linked retrieval',JSON.stringify(k,null,2),'goal-wikipedia-use');textField(p,'Candidates selected for the next serving phase',v.accepted_updates);textField(p,'Installed training package',v.installed_package||'unknown');textField(p,'Dashboard code package',v.renderer_package||'unknown');textField(p,'Training package observed',v.training_package_probe?.observed_at?new Date(v.training_package_probe.observed_at*1000).toLocaleString():'unknown');textField(p,'Code download / installation stage',v.updater?.stage||'unknown');textField(p,'Package installation confirmed by updater',v.updater?.installation_confirmed===true?'yes':'not confirmed; a verified download alone is not installation');textField(p,'Updater revision',v.updater?.revision);detailsField(p,'Accepted learner and adapter',JSON.stringify(v.learner||{},null,2),'current-learner-identity');detailsField(p,'Current-run receipts',JSON.stringify(v,null,2),'current-stage-receipts');
}
function earningActivityView(data){
 const p=panel('earning-activity-panel','Live earning attempts — recorded evidence',$('current-activity-panel'));p.replaceChildren();
 const h=document.createElement('h2');h.textContent='Live earning attempts — recorded evidence';p.append(h);
 const note=document.createElement('p');note.className='muted';note.textContent='Refreshes with the Mission Control snapshot every 5 seconds. It reports recorded work; paper results are not income, and missing receipts remain unknown.';p.append(note);
 const grid=document.createElement('div');grid.className='attempt-grid';p.append(grid);
 const freshness=data.report?.status_freshness||{};
 const stamp=value=>{const date=typeof value==='number'&&Number.isFinite(value)?new Date(value*1000):typeof value==='string'&&value?new Date(value):null;return date&&Number.isFinite(date.valueOf())?date.toLocaleString():'timestamp unknown'};
 const ageLabel=value=>typeof value==='number'&&Number.isFinite(value)?`${Math.floor(value/60)}m ${Math.floor(value%60)}s ago`:'age unknown';
 const add=(name,state,detail,updated)=>{const row=document.createElement('article');row.className='attempt-row';const title=document.createElement('h3');title.textContent=name;const status=document.createElement('p');status.className='attempt-status';status.dataset.state=state;status.textContent=state.toUpperCase();const body=document.createElement('p');body.textContent=detail;const time=document.createElement('small');time.textContent='Last receipt: '+stamp(updated);row.append(title,status,body,time);grid.append(row)};
 const risk=data.report?.live_execution_policy;if(risk){add('Live capital and debt guard',risk.real_orders_enabled===true?'enabled':'blocked',`Cap ${risk.capital_limit_pln??'unknown'} PLN · daily loss stop ${risk.daily_loss_stop_pln??'unknown'} PLN (${risk.daily_loss_stop_percent??'unknown'}%) · total loss stop ${risk.max_total_loss_pln??'unknown'} PLN (${risk.max_total_loss_percent??'unknown'}%) · leverage ${risk.leverage_enabled?'enabled':'disabled'} · margin ${risk.margin_enabled?'enabled':'disabled'} · shorting ${risk.short_selling_enabled?'enabled':'disabled'} · borrowing ${risk.borrowing_enabled?'enabled':'disabled'} · paid services ${risk.paid_services_enabled?'enabled':'disabled'} · real orders ${risk.real_orders_enabled?'enabled':'disabled'} · ${risk.scope||''}`,null)}else add('Live capital and debt guard','unknown','No capital or debt-policy receipt in this snapshot.');
 const connection=data.report?.broker_connection_readiness;if(connection){add('Broker connection readiness',connection.state||'unknown',`Account read: ${connection.account_reading||'unknown'} · orders: ${connection.order_submission||'unknown'} · first step: ${connection.first_connection_scope||''} · required: ${(connection.required_operator_inputs||[]).join(', ')} · before activation: ${(connection.activation_requirements||[]).join(', ')} · credentials: ${connection.credential_storage||''} · ${connection.scope||''}`,null)}else add('Broker connection readiness','unknown','No broker readiness contract in this snapshot.');
 const sources=data.report?.source_acquisition;
 if(sources){const state=freshness.source_acquisition?.state||'unknown';const latest=sources.latest?.[0];add('Internet source research',state,`${sources.total_fetches??'unknown'} retrievals · ${sources.distinct_urls??'unknown'} URLs · ${sources.distinct_contents??'unknown'} distinct contents${latest?.url?' · latest '+latest.url:''}`,sources.updated)}else add('Internet source research','unknown','No source-acquisition receipt in this snapshot.');
 const decision=data.report?.live_paper?.model_decision_evidence||{};add('Last confirmed model decision',decision.state||'unknown',decision.confirmed_at?`${decision.model_version||'model unknown'} · ${decision.decisions_recorded??'unknown'} paper decisions · ${stamp(decision.confirmed_at)} · ${ageLabel(decision.age_seconds)} · HOLD is not a fill or income.`:'No completed model-decision receipt; observations and HOLD freshness receipts do not count.',decision.confirmed_at);
 const fills=data.report?.paper_fill_evidence||{};const lastFill=fills.last_fill_state==='none-in-audited-period'?'none in audited period':fills.last_fill_at?`${stamp(fills.last_fill_at)} · ${ageLabel(fills.last_fill_age_seconds)}`:'unknown';add('Last confirmed paper fill / full audit',fills.state||'unknown',`${fills.fill_count??'UNKNOWN'} fills in saved full audit · audit ${stamp(fills.audit_at)} · ${ageLabel(fills.audit_age_seconds)} · last fill ${lastFill} · ${fills.scope||'No fresh full fill audit.'}`,fills.audit_at);
 const stream=data.report?.live_paper?.activity_stream||{};const counts=stream.counts||{};const countLabel=key=>stream.state==='verified-tail'&&Number.isInteger(counts[key])?counts[key]:'UNKNOWN';add('Paper activity timeline — verified ledger tail',stream.state||'unknown',`${countLabel('observations')} observations · ${countLabel('model_decisions')} model decisions · ${countLabel('automatic_freshness_holds')} automatic quote HOLDs · ${countLabel('decision_round_attempts')} decision-round attempts · ${countLabel('paper_execution_attempts')} paper order attempts · ${countLabel('confirmed_fills')} fills · ${countLabel('pending_orders')} pending orders. ${stream.window_scope||stream.detail||'No checksum-linked event window.'}`,null);
 const activityEntries=stream.state==='verified-tail'&&Array.isArray(stream.entries)?stream.entries.slice(-30).reverse():[];for(const entry of activityEntries){const state=entry.category==='confirmed_fill'?'verified':entry.category==='execution_attempt'?'recorded':entry.category==='observation'?'observed':'recorded';add(`Activity #${entry.sequence??'?'} · ${entry.label||entry.category||'event'}${entry.branch?' · '+entry.branch:''}`,state,entry.detail||'No additional event detail.',entry.time)}
 const work=data.report?.income_work;
 if(work){const receiptState=freshness.income_work?.state||'unknown';const executionState=String(work.state||'unknown').toLowerCase().includes('blocked')?'blocked':receiptState;add('Income work / experiments',executionState,`${work.state||'state unknown'} · receipt ${receiptState} · ${work.sources_collected??'unknown'} source dossiers · ${work.proposals_registered??'unknown'} proposals (not completed tests or payments)${work.error?' · '+work.error:''}`,work.updated)}else add('Income work / experiments','unknown','No current income-work receipt in this snapshot.');
 add('Verified income outcome','unknown',`UNKNOWN · ${data.report?.income_work?.actual_income_pln??'no verified payment amount'} · no payment ledger is connected; source reads and proposals are not earnings.`,data.report?.income_work?.updated);
 const labs=data.report?.paper_labs||[];
 if(labs.length){for(const lab of labs.slice(0,4)){const state=lab.quote_stale===true?'stale':lab.quote_stale===false?'fresh':'unknown';const fillCount=Array.isArray(lab.recent_fills)?lab.recent_fills.length:'unknown';add('Paper quote / feed '+(lab.id||'unknown'),state,`${lab.tick_age_seconds===null||lab.tick_age_seconds===undefined?'quote age unknown':Math.max(0,Math.round(lab.tick_age_seconds))+' s quote age'} · quote freshness only, not a model decision · ${fillCount} fill receipts in recent window · promotion ${lab.promotion_eligible===true?'eligible':'blocked'}`,lab.quote?.ts)}}else add('Paper experiments','unknown','No separate paper-ledger receipt in this snapshot.');
 const learning=data.report?.learning_activity_status||{};const optimizer=learning.optimizer_evidence||{};const evidence=data.report?.mission_evidence||{};const heartbeat=data.report?.learning_activity||{};
 const heartbeatState=learning.state==='running'?'fresh':learning.state||'unknown';add('Mission loop heartbeat (liveness only)',heartbeatState,`${learning.detail||'Heartbeat status unknown'} · ${stamp(heartbeat.updated)} · ${ageLabel(learning.age_seconds)} · does not prove model training or decisions.`,heartbeat.updated);
 add('Confirmed optimizer activity',optimizer.activity_state||'unknown',`${optimizer.state||'unknown'} cumulative receipt · updates ${optimizer.optimizer_updates??'unknown'} · ${stamp(optimizer.receipt_updated_at)} · ${ageLabel(optimizer.receipt_age_seconds)} · ${optimizer.activity_detail||'No optimizer receipt'}`,optimizer.receipt_updated_at);
 add('Training acceptance / CUDA',evidence.cuda_verified===true?'verified':'unknown',`Accepted weight updates ${evidence.accepted_weight_updates_this_run??'unknown'} · CUDA receipt ${evidence.cuda_verified===true?'verified':'not confirmed'} · optimizer steps do not prove accepted weights or income.`,null);
 const activity=data.report?.current_activity||{};add('Mission process',data.mission?.running===true?'running':data.mission?.running===false?'stopped':'unknown',`${activity.stage||data.mission?.state?.phase||'stage unknown'} · completed cycles ${activity.completed_cycles??'unknown'} · process status is not earning or learning evidence`,activity.updated);
}
function refreshAgeBadge(){const p=document.getElementById('current-snapshot-age');if(!p)return;const age=Math.max(0,Date.now()/1000-Number(p.dataset.collected));p.textContent=`Snapshot collected ${Math.round(age)} seconds ago${age>300?' — delayed':''}`;p.style.color=age>300?'#ffbf86':'';}
function textField(parent,label,value){if(value===undefined||value===null||value==='')return;const p=document.createElement('p'),b=document.createElement('strong');b.textContent=label+': ';p.append(b,document.createTextNode(String(value)));parent.append(p)}
const disclosureStates=new Map(), disclosureScroll=new Map(), panelScroll=new Map();
function nodeKey(n){return n.nodeType===1?(n.id||n.dataset.key||n.dataset.scrollKey||''):''}
function updateNode(old,fresh){
if(['action-archive','trade-explorer'].includes(old.id))return;
// The live user's disclosure state wins over an automatically opened draft.
if(old.nodeName==='DETAILS'&&old.open!==undefined)fresh.open=old.open;
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
const readingLocations=new Map([...selected('.agent-row')].filter(n=>n.querySelector?.('details[open]')).map(n=>[n.dataset.key,n.parentElement.id]));
grid.replaceChildren();let history=$('agent-history');if(!history){history=document.createElement('details');history.id='agent-history';history.dataset.key='agent-history';const title=document.createElement('summary');title.textContent='Historical agent events and finished jobs';history.append(title);agents.append(history)}while(history.children.length>1)history.children[history.children.length-1].remove();
for(const a of [...(data.live?.agents||[])].sort((a,b)=>a.label.localeCompare(b.label))){const node=document.createElement('article');node.className='agent-row';node.dataset.key=a.label+'-agent-card';const head=document.createElement('header');const h=document.createElement('h3');h.textContent=a.label;const badge=document.createElement('span');badge.className='state-badge';badge.textContent=a.state||'unknown';head.append(h,badge);node.append(head);
textField(node,'Model / device',[a.model,a.device].filter(Boolean).join(' / '));
const u=a.usage||{};textField(node,'Last response tokens',Object.keys(u).length?`Input ${fmt(u.prompt_tokens)} | Output ${fmt(u.completion_tokens)} | Total ${fmt(u.total_tokens)} | recorded ${a.usage_at||''}`:'Not reported for this actor');
textField(node,'Response completeness',a.completeness);textField(node,'Finish reason',a.finish_reason);textField(node,'Duration',a.seconds===undefined?null:a.seconds+' s');
detailsField(node,'Assigned task',a.task,a.label+'-task');textField(node,'Tool',a.tool);
detailsField(node,'Live partial output - latest recorded window',a.stream_output,a.label+'-live-output');detailsField(node,'Live local reasoning - latest recorded window',a.stream_reasoning,a.label+'-live-trace');detailsField(node,'Returned local model reasoning (unverified)',a.returned_trace,a.label+'-trace');detailsField(node,'Public conclusion and next step',a.declaration,a.label+'-output');detailsField(node,'Observed result / error',a.result,a.label+'-result');detailsField(node,'Previous attempt, not current result',a.previous_result,a.label+'-previous');detailsField(node,'Source evidence / receipt',a.evidence_ref?JSON.stringify(a.evidence_ref):'No source evidence attached to this event',a.label+'-source');textField(node,'Evidence acquired',a.evidence_at);textField(node,'Event / job ID',a.event_id);textField(node,'History',a.historical?'Historical finished job, not current work':null);detailsField(node,'Journal file',a.source,a.label+'-journal');textField(node,'Last event',typeof a.updated==='number'?new Date(a.updated*1000).toLocaleString():a.updated);(readingLocations.has(node.dataset.key)?(readingLocations.get(node.dataset.key)==='agent-history'?history:grid):(a.historical?history:grid)).append(node)}
if(!grid.children.length)textField(grid,'Activity','No recent recorded agent events');
const timeline=panel('timeline-panel','Action timeline - chronological, latest 200 recorded events',agents);timeline.append($('events'));const pinned=[...$('events').children].filter(n=>n.open).map(n=>n.cloneNode(true));$('events').replaceChildren();
for(const e of [...(data.live?.events||[])]){const d=document.createElement('details');d.className='event-row';d.dataset.key=e.id||e.time+'-'+(e.request_id||'')+'-'+e.actor+'-'+e.kind;const h=document.createElement('summary');h.textContent=[e.time,e.branch,e.actor,e.kind,e.tool].filter(Boolean).join(' | ');d.append(h);textField(d,'Live delta',e.delta_text);textField(d,'Returned local reasoning (unverified)',e.returned_trace);textField(d,'Task / arguments',e.task);textField(d,'Public output / result',e.summary);textField(d,'Error / status',e.detail);if(e.usage&&Object.keys(e.usage).length)textField(d,'Tokens',JSON.stringify(e.usage));textField(d,'Seconds',e.seconds);textField(d,'Event ID',e.id);textField(d,'Source evidence / receipt',Object.keys(e.evidence_ref||{}).length?JSON.stringify(e.evidence_ref):null);textField(d,'Journal file',e.source);$('events').append(d)}
for(const n of pinned)if(![...$('events').children].some(e=>e.dataset.key===n.dataset.key))$('events').append(n);
restoreDisclosures();
}
let archiveCursor=0,archiveBusy=false;
const pendingFetch=new Set();
async function readAPI(route){const abort=new AbortController(),timer=setTimeout(()=>abort.abort(),8000);try{const response=await fetch(route,{cache:'no-store',signal:abort.signal});if(!response.ok)throw Error('HTTP '+response.status);return await response.json()}finally{clearTimeout(timer)}}
async function singleFetch(key,work){if(pendingFetch.has(key))return;pendingFetch.add(key);try{await work()}finally{pendingFetch.delete(key)}}
function archivePanel(){
if($('action-archive'))return;
const p=document.createElement('section');p.id='action-archive';const h=document.createElement('h2');h.textContent='Full recorded action archive';const note=document.createElement('p');note.textContent='Daily journals, 100 records per page. Expand an event to read its recorded arguments and result.';const select=document.createElement('select');select.id='archive-day';const button=document.createElement('button');button.id='archive-load';button.textContent='Load action history';const status=document.createElement('p');status.id='archive-status';const rows=document.createElement('div');rows.id='archive-rows';p.append(h,note,select,button,status,rows);$('timeline-panel').after(p);
}
async function loadArchive(reset=false){
if(archiveBusy)return;archiveBusy=true;
const select=document.getElementById('archive-day'),button=document.getElementById('archive-load'),status=document.getElementById('archive-status'),rows=document.getElementById('archive-rows');if(!button){archiveBusy=false;return}button.disabled=true;
try{const cursor=reset?0:archiveCursor,query=new URLSearchParams({cursor:String(cursor),limit:'100'});if(select.value)query.set('day',select.value);const result=await readAPI('/api/actions?'+query);if(reset){rows.replaceChildren();archiveCursor=0}if(!select.options.length){for(const day of result.days||[]){const option=document.createElement('option');option.value=day;option.textContent=day;select.append(option)}select.value=result.day||''}
for(const e of result.events||[]){const key=e.id||String(e.sequence||e.time);if([...rows.children].some(n=>n.dataset.key==='archive-'+key))continue;const d=document.createElement('details'),title=document.createElement('summary'),body=document.createElement('pre');d.dataset.key='archive-'+key;title.textContent=[e.time,e.branch,e.actor,e.kind].filter(Boolean).join(' | ');body.textContent=JSON.stringify({context:e.context,payload:e.payload,id:e.id},null,2);d.append(title,body);rows.append(d)}archiveCursor=result.next_cursor;status.textContent=result.days?.length?`${rows.children.length} events shown | ${archiveCursor} / ${result.archive_bytes||0} archive bytes`:'No recorded journals yet';button.textContent=result.has_more?'Load next 100 actions':'Check for new actions';
}catch(error){status.textContent='Archive unavailable: '+error.message+'. Retry using the same button; existing records retained.'}finally{button.disabled=false;archiveBusy=false}
}
function sourcesPanel(data){const p=panel('source-panel','Public source acquisition - measured work',$('plan-panel'));let body=$('source-view');if(!body){body=document.createElement('div');body.id='source-view';p.append(body)}body.replaceChildren();const s=data.report?.source_acquisition;if(!s){textField(body,'Status','No retrieval counters recorded since instrumentation');return}textField(body,'Distinct public URLs',s.distinct_urls);textField(body,'Distinct source contents',s.distinct_contents);textField(body,'Total actual retrievals',s.total_fetches);textField(body,'Unchanged rereads',s.unchanged_rereads);textField(body,'Last collection',new Date(s.updated*1000).toLocaleString());for(const r of (s.latest||[]).slice(0,5)){const node=document.createElement('article');node.className='card';node.dataset.key='source-'+r.url;textField(node,'URL',r.url);textField(node,'Retrieved',new Date(r.last_fetched*1000).toLocaleString());textField(node,'Retrieval count for identical content',r.fetch_count);detailsField(node,'Content checksum',r.sha256,'source-proof-'+r.url);body.append(node)}}
function incomePanel(data){const p=panel('income-panel','Income opportunities - evidence and next tests',$('plan-panel'));let body=$('income-view');if(!body){body=document.createElement('div');body.id='income-view';p.append(body)}body.replaceChildren();const board=data.report?.income_opportunities||{},dispatch=data.report?.income_dispatch||{};textField(body,'Research policy',board.research_policy?.capital_research?'Hypothetical capital and zero-upfront':'Zero-upfront');textField(body,'Historical comparison',data.report?.market_research?.state);textField(body,'Historical comparison blocker',data.report?.market_research?.error);textField(body,'Actual verified income','Unknown - no payment ledger connected');const work=data.report?.income_work||{},quality=data.report?.research_quality||{};textField(body,'Concrete work / blocker',work.state||'not started');textField(body,'Work source / domain',`${work.url||''} | ${work.domain||''}`);textField(body,'Fresh source dossiers / registered proposals',`${work.sources_collected??'unknown'} / ${work.proposals_registered??'unknown'}`);textField(body,'Analysis blocker',work.error||'none reported');textField(body,'Hypothesis/test audit',`${quality.state||'unknown'} | ${quality.reason||''}`);if(work.dossier)detailsField(body,'Prepared dossier and source receipt',JSON.stringify(work),'income-work-dossier');textField(body,'Research assignment',dispatch.state||'Not observed');textField(body,'Assignment blocker',dispatch.error);const candidates=board.candidates||[];if(!candidates.length)textField(body,'Evidence-backed candidates','None recorded in this snapshot');const unique=new Map();for(const c of [...candidates].reverse()){const s=c.specification||{},key=String(s.domain||'Unspecified domain').trim().toLowerCase();if(!unique.has(key))unique.set(key,{candidate:c,count:0});unique.get(key).count++}textField(body,'Domain groups',`${unique.size} domains / ${candidates.length} archived hypotheses. One representative per domain; different ideas are retained, not treated as identical.`);for(const {candidate:c,count} of [...unique.values()].slice(0,12)){const s=c.specification||{};const node=document.createElement('details');node.className='card';node.dataset.key='income-'+c.id;const title=document.createElement('summary');title.textContent=s.title||c.id;node.append(title);textField(node,'Domain / state',`${s.domain||''} | ${c.state}`);textField(node,'Proposal recorded',c.updated?new Date(c.updated*1000).toLocaleString():'Timestamp not recorded');textField(node,'Execution evidence','This is a proposal. Payment and completed-test receipts are required to treat it as a result.');textField(node,'Estimated net before personal tax',`${c.estimated_net_pln_low} to ${c.estimated_net_pln_high} PLN - unverified estimate`);textField(node,'Labor / time to first income',`${s.labor_hours} hours | ${s.first_income_days} days - estimates`);textField(node,'Mechanism',s.mechanism);textField(node,'Hypotheses in this domain',count);detailsField(node,'Eligibility and blockers',JSON.stringify({eligibility:s.eligibility,blockers:s.blockers}), 'income-eligibility-'+c.id);textField(node,'Next falsifiable test',s.next_test);textField(node,'Reject if',s.failure_condition);detailsField(node,'Evidence IDs and proposal receipt',JSON.stringify(c),'income-evidence-'+c.id);body.append(node)}if(unique.size>12)textField(body,'Additional domains',`${unique.size-12} retained in the full receipts below; this view shows the first 12.`);detailsField(body,'All archived proposal receipts in this snapshot',JSON.stringify(board,null,2),'income-all-proposals');restoreDisclosures()}
function learningPanel(data){rememberDisclosures();const p=panel('goal-learning-panel','Goal-directed ML pipeline',$('plan-panel'));let body=$('goal-learning-view');if(!body){body=document.createElement('div');body.id='goal-learning-view';p.append(body)}body.replaceChildren();const evidence=data.report?.mission_evidence||{},g=evidence.goal_learning||{};const activity=data.report?.learning_activity||{},progress=data.report?.learning_activity_status||{};const optimizer=progress.optimizer_evidence||{};const receiptAge=optimizer.receipt_age_seconds;const receiptAgeLabel=typeof receiptAge==='number'?`${Math.floor(receiptAge/3600)}h ${Math.floor(receiptAge%3600/60)}m ago`: 'age unknown';const heartbeatAge=progress.age_seconds;const heartbeatAgeLabel=typeof heartbeatAge==='number'?`${Math.floor(heartbeatAge/60)}m ${Math.floor(heartbeatAge%60)}s ago`:'age unknown';const stamp=value=>typeof value==='number'&&Number.isFinite(value)?new Date(value*1000).toLocaleString():'timestamp unknown';textField(body,'Loop heartbeat (liveness only)',progress.state||'unknown');textField(body,'Heartbeat detail',progress.detail||'No heartbeat freshness evidence');textField(body,'Last heartbeat receipt',`${stamp(activity.updated)} | ${heartbeatAgeLabel}`);textField(body,'Loop stage',activity.stage||progress.stage||'No stage receipt yet');textField(body,'Last stage entered',stamp(activity.updated));textField(body,'Cumulative optimizer evidence',`${optimizer.state||'unknown'} | ${fmt(optimizer.optimizer_updates)}`);textField(body,'Last confirmed optimizer receipt',`${optimizer.activity_state||'unknown'} | ${stamp(optimizer.receipt_updated_at)} | ${receiptAgeLabel}`);textField(body,'Optimizer receipt detail',optimizer.activity_detail||'No optimizer receipt freshness evidence');const trainingRuns=evidence.training_runs||[];const checkpoints=trainingRuns.flatMap(row=>row.checkpoints||[]);textField(body,'Verified complete checkpoints',checkpoints.length?checkpoints.map(row=>`step ${row.step}`).join(', '):'none recorded');detailsField(body,'Current-run training receipts',JSON.stringify(trainingRuns),'training-receipts');textField(body,'Training blocker / next check',activity.reason||'None recorded');textField(body,'Next scheduled check',activity.next_check?new Date(activity.next_check*1000).toLocaleString():null);textField(body,'Training frequency','Event-driven on admitted data; research loops are not optimizer updates');textField(body,'Active goal ID',g.goal_id||(data.report?.plans?.long?.id?'Goal metrics unavailable':'Goal status unavailable'));for(const [k,v] of Object.entries(g.learning_task||{}))textField(body,k,v);textField(body,'Recent forecasts',fmt(g.forecasts?.length));textField(body,'Recent resolved forecasts',g.recent_resolved);detailsField(body,'Mixed forecast archive (different policies, not a master skill score)',JSON.stringify({recent_resolved:g.recent_resolved,mixed_recent_brier:g.recent_brier}), 'mixed-score');textField(body,'Admitted training examples',g.training_admission?.admitted);textField(body,'Confirmed optimizer updates',fmt(evidence.optimizer_updates_observed));textField(body,'Candidates selected for the next serving phase',fmt(evidence.accepted_weight_updates_this_run));detailsField(body,'Training admission evidence',JSON.stringify(g.training_admission||{}),'goal-admission');textField(body,'Verified total outcomes',g.resolved_total);detailsField(body,'Independent observation pipeline',JSON.stringify(g.observer||{}),'goal-observer');const dispatch=g.owned_cpu_dispatch||{};textField(body,'Owned CPU dispatch',dispatch.state||'Not observed');textField(body,'Dispatch blocker',dispatch.reason);for(const branch of dispatch.branches||[]){textField(body,'CPU trial',`${branch.id||'Not allocated'} | ${branch.state} | ${branch.reason||branch.scope||''}`);if(branch.validation_groups!==undefined)textField(body,'Source split',`${branch.train} training / ${branch.validation} validation outcomes | ${branch.training_groups} training / ${branch.validation_groups} validation source groups`);textField(body,'Next evidence needed',branch.next_evidence)}detailsField(body,'Owned CPU provenance and receipts',JSON.stringify(dispatch),'goal-compute');restoreDisclosures()}
function svg(tag,attrs,text){let node=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [key,val] of Object.entries(attrs))node.setAttribute(key,val);if(text!==undefined)node.textContent=text;return node}
function render(data){if(latestInference&&inferenceStamp(latestInference)>=inferenceStamp(data.live))data=mergeInference(data,latestInference);latestSnapshot=data;const m=data.mission||{},r=data.report||{},ev=(m.state?.phase==='official-public-baseline'?r.mission_evidence?.official_benchmark:null)||m.evaluation||{},learn=m.learning||{},gpu=data.gpu||{};
if(data.layout?.active_sha256&&layoutVersion!==data.layout.active_sha256){pendingLayout=true;$('reload').textContent='New layout available - load when ready'}$('layout').textContent='Master HTML: /workspace/dashboard/index.html | '+(data.layout?.state||'waiting')+(data.layout?.error?' | '+data.layout.error:'');
$('fresh').textContent='Snapshot collected: '+new Date(data.collected_at*1000).toLocaleString()+' | Live status refreshes every 5 s; aggregate audit runs separately | Run: '+fmt(m.run)+(data.report_freshness?.state==='stale'?' | delayed aggregate report: '+fmt(data.report_freshness.updated_at):'')+' | Aggregate audit: '+fmt(data.report_collection?.state);
currentActivityView(data);earningActivityView(data);workView(data);learningPanel(data);incomePanel(data);sourcesPanel(data);insightsView(data);archivePanel();const currentErrors=(data.errors||[]).filter(issue=>{const value=typeof issue==='string'?issue:`${issue?.scope||''} ${issue?.detail||''}`;return !/historical|previous run|may precede current/i.test(value)});const issues=[...currentErrors.map(issue=>typeof issue==='string'?issue:[issue.scope,issue.detail].filter(Boolean).join(': ')),...(data.live?.actors||[]).filter(a=>a.state==='unavailable').map(a=>`${a.label}: unavailable | ${(a.detail||'').slice(0,160)}`)];$('warning').textContent=issues.length?`${issues.length} current collection/availability issue${issues.length===1?'':'s'}:\n`+[...new Set(issues)].slice(0,3).join('\n')+(issues.length>3?'\nAdditional current details are in Agents & resources.':''):'No current collection errors recorded. Historical errors remain in the evidence archive.';$('cards').replaceChildren();card('Mission',typeof m.running==='boolean'?(m.running?'running':'stopped'):'unknown');card('Current activity',r.current_activity?.stage||m.state?.phase);const backend=m.latest_backend_startup?.stage?m.latest_backend_startup:m.server_startup?.stage?m.server_startup:null;card('Last recorded backend event',backend?.stage?`${backend.stage}${typeof backend.updated==='number'?` (${Math.max(0,Math.round((Date.now()/1000-backend.updated)/60))} min ago)`:''}`:'unknown');card('Closed cycles - this run',learn.completed_cycles);card('Candidates selected for the next serving phase',r.accepted_weight_updates_this_run);card('Confirmed optimizer updates',r.mission_evidence?.optimizer_updates_observed);card('GPU utilization',gpu.utilization===undefined?null:gpu.utilization+'%');
$('evaluation').textContent=ev.total?`${ev.completed}/${ev.total} completed, ${ev.passed===undefined?'official grading':ev.passed+' passed'} (${ev.state})`:'No current evaluation counter';$('progress').max=ev.total||1;$('progress').value=ev.completed||0;$('case').textContent=ev.current_case?('Current case: '+ev.current_case):'';
const ledger=r.live_paper||r.paper||{};$('paper').textContent=Object.entries(ledger.branches||{}).map(([name,b])=>`${name} | Equity ${fmt(b.equity)} ${ledger.currency||''} | Net P&L ${fmt(b.net_pnl_before_personal_tax)} | Costs ${fmt(b.modeled_costs)} | Resolved fills ${fmt(b.resolved_fills)}`).join('\n\n')||(r.paper_display_status||'Paper ledger status unavailable');if(ledger.updated_at)$('paper').textContent+='\n\nSaved report: '+ledger.updated_at+' | period '+(ledger.period||'live recorded state')+' | '+(ledger.scope||'');const attempt=ledger.last_decision_attempt;if(attempt){$('paper').textContent+='\n\nLatest decision-round attempt (may be incomplete): '+attempt.status+' | '+attempt.time+' | cycle '+fmt(attempt.cycle)+' | model '+(attempt.model_version||'unknown')+' | paper decisions recorded '+fmt(attempt.decisions_recorded)+' | '+(attempt.error?attempt.error+': '+(attempt.detail||''):(attempt.branch_actions||[]).map(a=>`${a.branch} ${a.action||'no fresh quote'}`).join(', ')||'no branch decision; inspect reason receipt')+'\n'+(attempt.scope||'')}else $('paper').textContent+='\n\nLatest decision-round attempt (may be incomplete): unknown — no decision-round receipt is recorded in the canonical paper ledger.';$('blockers').textContent=(r.paper_blockers||[]).join('\n')||(Array.isArray(r.paper_blockers)?'No blockers in recorded snapshot':'Paper blockers unavailable');$('workers').textContent=JSON.stringify({drones:r.drones,external_compute:r.external_compute,desktop:r.desktop,official_benchmark:r.official_benchmark},null,2);$('log').textContent=data.controller_log||'Log unavailable';$('evidence').textContent=JSON.stringify({report_freshness:data.report_freshness,report_collection:data.report_collection,last_cycle:learn.last_cycle,evidence:r.mission_evidence,collection_errors:data.errors},null,2);
let audit=$('backtest-audit');if(!audit){audit=document.createElement('pre');audit.id='backtest-audit';const disclosure=document.createElement('details'),label=document.createElement('summary');disclosure.dataset.key='backtest-audit';label.textContent='Historical replay audit and evidence';disclosure.append(label,audit);$('chart').after(disclosure)}audit.textContent=r.backtest_audit?JSON.stringify(r.backtest_audit,null,2):'No recorded source replay audit. Drawdown is peak-to-trough loss, not standard deviation. Loss alone is not a data glitch.';
$('backtests').replaceChildren();$('chart').replaceChildren();const tests=r.recent_exploratory_backtests||[];let largest=Math.max(.01,...tests.map(t=>Math.abs(t.development_test?.net_return||0)));$('chart').append(svg('line',{x1:400,y1:0,x2:400,y2:220,stroke:'#8296b1'}));tests.forEach((test,i)=>{const result=test.development_test||{},tr=document.createElement('tr');[`${i+1}. ${test.parameters?.rule||''}`,pct(result.net_return),pct(result.max_sampled_drawdown),fmt(result.fills),test.triage?.state||'unreviewed',pct(test.double_cost_stress?.net_return)].forEach(v=>{let td=document.createElement('td');td.textContent=v;tr.append(td)});$('backtests').append(tr);if(typeof result.net_return==='number'&&Number.isFinite(result.net_return)){let width=Math.abs(result.net_return)/largest*280,y=15+i*48;$('chart').append(svg('rect',{x:result.net_return<0?400-width:400,y,width,height:26,fill:result.net_return<0?'#e68d80':'#75c5ae'}),svg('text',{x:8,y:y+19,fill:'#dce6f5','font-size':14},`${i+1}. ${pct(result.net_return)}`))}});if(!tests.length){const row=document.createElement('tr'),cell=document.createElement('td');cell.colSpan=6;cell.textContent='No stored historical test results available. Check the comparison state and blocker in Income opportunities.';row.append(cell);$('backtests').append(row)}if(!tests.length)$('chart').append(svg('text',{x:20,y:50,fill:'#adbdd1'},'No recorded results'));
$('reports').replaceChildren();for(const id of data.paper_reports||[]){const row=document.createElement('p');row.append(document.createTextNode(id+' '));for(const [name,label] of [['report.html','HTML and charts'],['report.json','JSON'],['trades.csv','CSV']]){const a=document.createElement('a');a.href='/paper/'+encodeURIComponent(id)+'/'+name;a.target='_blank';a.rel='noopener';a.textContent=label;row.append(a,document.createTextNode(' · '))}$('reports').append(row)}
}
function stablePaint(update){
if(typeof window==='undefined'||!document.body?.cloneNode){update();return}
const body=document.body,draft=body.cloneNode(true),x=window.scrollX,y=window.scrollY;
const containers=[...document.querySelectorAll('[id], [data-scroll-key]')].filter(n=>n.scrollTop||n.scrollLeft).map(n=>[n,n.scrollTop,n.scrollLeft]);
const atPoint=document.elementFromPoint?.(window.innerWidth/2,Math.min(80,window.innerHeight/2));// Anchor the actual visible field, not its whole section: rows above it can grow.
const anchor=atPoint?.nodeName==='BODY'||atPoint?.nodeName==='HTML'?null:atPoint;const reader=anchor||[...document.querySelectorAll('details[data-key][open]')].find(n=>{const r=n.getBoundingClientRect();return r.bottom>0&&r.top<window.innerHeight});const readerTop=reader?.getBoundingClientRect().top;
try{renderRoot=draft;update()}finally{renderRoot=null}
updateNode(body,draft);for(const [n,top,left] of containers){n.scrollTop=top;n.scrollLeft=left}const shift=reader?.isConnected?reader.getBoundingClientRect().top-readerTop:0;const scrollParent=containers.filter(([n])=>n!==reader&&n.contains?.(reader)).at(-1)?.[0];if(scrollParent){scrollParent.scrollTop+=shift;if(window.scrollX!==x||window.scrollY!==y)window.scrollTo(x,y)}else if(window.scrollX!==x||window.scrollY!==y+shift)window.scrollTo(x,y+shift);
}
let tradeData=null,tradeChoice='',tradeStart=0,tradeEnd=100,tradeMetric='return',tradeSelected=null;
function tradePanel(){
 if(document.getElementById('trade-explorer'))return;
 const panel=document.createElement('section');panel.id='trade-explorer';
 panel.innerHTML='<h2>Trade explorer - buys, sells and returns</h2><p class="muted">Forward paper and historical simulations are separate. Select a recorded execution for its receipt.</p><div class="grid"><label>Portfolio or test <select id="trade-series"></select></label><label>What to plot <select id="trade-metric"><option value="equity">Liquidation equity (includes future exit costs)</option><option value="spot">Spot equity (excludes future exit costs)</option><option value="frictionless">Hypothetical result WITHOUT costs (not net profit)</option><option value="return" selected>Net return %</option><option value="price">Historical close price</option></select></label></div><div class="grid"><label>From <input id="trade-start" type="range" min="0" max="99" value="0"></label><label>To <input id="trade-end" type="range" min="1" max="100" value="100"></label></div><p class="muted">Green triangles: buys/entries. Red squares: sells/exits. Equity/return curve: green above start, red below start, blue unchanged. Price and the hypothetical cost-free curve are blue. Grey dashed line: starting capital or zero return. Colored dashed line: cost comparison. Fees and slippage apply per leg. Click a marker for the cost breakdown.</p><p id="trade-explanation" class="muted"></p><p id="trade-summary"></p><p id="trade-scope" class="muted"></p><svg id="trade-plot" viewBox="0 0 960 300" style="max-height:320px;min-height:260px" role="img" aria-label="Recorded equity and execution timeline"></svg><p id="trade-hover" style="min-height:24px">Hover a point; select a buy or sell marker to inspect the execution.</p><details id="trade-receipt"><summary>Selected execution receipt</summary><pre id="trade-detail">No execution selected.</pre></details><p id="trade-errors"></p>';
 document.getElementById('paper').closest('section').before(panel);
 panel.querySelector('#trade-series').onchange=e=>{tradeChoice=e.target.value;tradeStart=0;tradeEnd=100;tradeSelected=null;drawTrades()};
 panel.querySelector('#trade-metric').onchange=e=>{tradeMetric=e.target.value;drawTrades()};
 for(const id of ['trade-start','trade-end'])panel.querySelector('#'+id).oninput=e=>{if(id==='trade-start')tradeStart=Math.min(+e.target.value,tradeEnd-1);else tradeEnd=Math.max(+e.target.value,tradeStart+1);drawTrades()};
}
function executionDescription(trade,row){
 const num=v=>typeof v==='number'&&Number.isFinite(v)?v.toFixed(6):'not recorded';
 const before=trade.spot_equity_before,after=trade.spot_equity_after;
 const change=typeof before==='number'&&before>0&&typeof after==='number'?((after/before-1)*100).toFixed(3)+'%':'not recorded';
 return [`${trade.action} at ${new Date(tradeTime(trade.time)).toLocaleString()}`,`Fill price: ${num(Number(trade.price))}`,`Same-price spot equity change: ${change}`,`Paid/modelled execution friction: ${num(Number(trade.costs))} | ${num(trade.execution_friction_percent)}% of pre-fill spot equity`,`Accounting reconciliation: ${trade.accounting_verified===true?'passed at the execution reference price':'not recorded; inspect archived receipt'}`,`Reserved future exit friction: ${num(trade.estimated_exit_cost)}`,`Fee per leg: ${num(trade.fee_bps)} bps | Slippage per leg: ${num(trade.slippage_bps)} bps`,`1 bp = 0.01%. Entry and exit are separate legs. ${row.currency}.`,`Opening execution is separate from the later candle close.`,JSON.stringify(trade,null,2)].join('\n');
}
function tradeTime(value){return typeof value==='number'?value*1000:Date.parse(value)}
function drawTrades(){
 if(!tradeData)return;
 const get=id=>document.getElementById(id), select=get('trade-series');
 const options=(tradeData.series||[]).map(row=>({id:row.id,label:row.label}));
 const signature=JSON.stringify(options);if(select.dataset.signature!==signature){select.replaceChildren();for(const row of options){const o=document.createElement('option');o.value=row.id;o.textContent=row.label;select.append(o)}select.dataset.signature=signature}
 if(!options.some(row=>row.id===tradeChoice))tradeChoice=options[0]?.id||'';select.value=tradeChoice;
 get('trade-start').value=String(tradeStart);get('trade-end').value=String(tradeEnd);get('trade-metric').value=tradeMetric;
 get('trade-errors').textContent=(tradeData.errors||[]).join('\n');
 const row=(tradeData.series||[]).find(s=>s.id===tradeChoice), plot=get('trade-plot');plot.replaceChildren();
 if(!row){get('trade-summary').textContent='No recorded trade series available.';get('trade-scope').textContent=tradeData.scope||'';return}
 get('trade-hover').textContent='Hover a valuation point or select an execution to inspect its costs.';
 get('trade-scope').textContent=(row.id.startsWith('paper-')?`Forward paper | Ledger event: ${row.ledger_updated_at||'unknown'} | Quote: ${row.quote_updated_at||'unknown'} | ${row.quote_fresh===true?'Quote fresh':row.quote_fresh===false?'STALE QUOTE':'Quote freshness unknown'} | `:'Historical replay: fixed archived data; a fresh page does not change historical prices. | ')+row.scope+(row.cost_assumptions?` Fee: ${row.cost_assumptions.fee_bps_per_leg} bps (${row.cost_assumptions.fee_bps_per_leg/100}%) per leg; slippage: ${row.cost_assumptions.slippage_bps_per_leg} bps (${row.cost_assumptions.slippage_bps_per_leg/100}%) per leg. Assumptions, not certified exchange rates.`:'');
 get('trade-explanation').textContent=tradeMetric==='frictionless'?'DIAGNOSTIC ONLY: this line removes fees and slippage from the same recorded trades. It can rise while the net result loses money. The dashed red comparison includes costs.':tradeMetric==='return'?'Net return is the change from starting capital after modeled costs, including the estimated cost of closing open positions. Zero means break-even. The dashed blue comparison removes costs.':tradeMetric==='price'?'This is the asset price, not portfolio profit. A rising price does not prove that the recorded strategy earned money.':tradeMetric==='spot'?'Cash plus open positions at the recorded market price. Future closing costs are excluded; liquidation value includes them.':'Cash plus open positions after estimated closing costs. An immediate entry drop can include both the paid entry costs and the estimated future exit costs.';
 const initial=row.initial_capital;get('trade-summary').textContent=`Net P&L: ${Number(row.net_pnl).toFixed(4)} ${row.currency} | Net return: ${(row.net_pnl/initial*100).toFixed(2)}% | Modeled costs: ${Number(row.total_costs).toFixed(4)} | Recorded executions in view archive: ${row.executions.length} | Collected ${Number.isFinite(tradeData.updated)?new Date(tradeData.updated*1000).toLocaleTimeString():'timestamp unavailable'}`;
 if(row.diagnostics)get('trade-summary').textContent+=` | Same signals without costs: ${(row.diagnostics.same_signals_without_costs_return*100).toFixed(2)}% | Cost drag (return difference): ${(row.diagnostics.cost_drag_return_difference*100).toFixed(2)} percentage points | ${row.diagnostics.state}`;
 if(row.retained_previous_snapshot)get('trade-summary').textContent+=' | STALE: selected series is absent from the latest response; previous data retained';if(row.points?.length)get('trade-summary').textContent+=' | Last recorded equity point: '+new Date(tradeTime(row.points.at(-1).time)).toLocaleString();
 const all=row.points||[],begin=Math.floor(all.length*tradeStart/100),end=Math.max(begin+1,Math.ceil(all.length*tradeEnd/100)),points=all.slice(begin,end);
 const value=p=>tradeMetric==='frictionless'?p.frictionless_equity:tradeMetric==='spot'?p.spot_equity:tradeMetric==='return'?(p.equity/initial-1)*100:tradeMetric==='price'?p.price:p.equity;
 const valid=points.filter(p=>Number.isFinite(value(p))&&Number.isFinite(tradeTime(p.time)));
 if(!valid.length){plot.append(svg('text',{x:30,y:90,fill:'#adbdd1'},tradeMetric==='price'?'No recorded close-price series for this selection. Use equity or inspect execution prices.':'This metric is not recorded for this series. Forward paper exposes liquidation equity; use that metric.'));return}
 const comparison=p=>tradeMetric==='frictionless'?p.equity:tradeMetric==='return'?(p.frictionless_equity/initial-1)*100:null;const compared=valid.filter(p=>Number.isFinite(comparison(p)));
 const times=valid.map(p=>tradeTime(p.time)),lo=Math.min(...times),hi=Math.max(...times),values=[...valid.map(value),...compared.map(comparison),...(tradeMetric==='price'?[]:[tradeMetric==='return'?0:initial]),...(tradeMetric==='price'?row.executions.filter(t=>tradeTime(t.time)>=lo&&tradeTime(t.time)<=hi&&Number.isFinite(+t.price)).map(t=>+t.price):[])],min=Math.min(...values),max=Math.max(...values),pad=Math.max((max-min)*.1,Math.abs(max)*.001,.0001);
 plot.append(svg('text',{x:65,y:17,fill:'#bacde3','font-size':12},tradeMetric==='return'?'Net return (%)':tradeMetric==='price'?'Historical price (source quote units)':(tradeMetric==='frictionless'?'Zero-cost counterfactual: ':tradeMetric==='spot'?'Spot equity: ':'Liquidation equity: ')+row.currency));
 const x=t=>65+(t-lo)/Math.max(1,hi-lo)*865,y=v=>250-(v-min+pad)/(max-min+2*pad)*215;
 for(let i=0;i<5;i++){const v=min-pad+(max-min+2*pad)*i/4;plot.append(svg('line',{x1:65,x2:930,y1:y(v),y2:y(v),stroke:'#304259'}),svg('text',{x:2,y:y(v)+4,fill:'#adbdd1','font-size':11},v.toFixed(3)))}
 const baseline=tradeMetric==='return'?0:tradeMetric==='equity'||tradeMetric==='spot'||tradeMetric==='frictionless'?initial:null;if(baseline!==null&&baseline>=min-pad&&baseline<=max+pad)plot.append(svg('line',{x1:65,x2:930,y1:y(baseline),y2:y(baseline),stroke:'#bccade','stroke-dasharray':'5 5'}));plot.append(svg('polyline',{points:valid.map(p=>`${x(tradeTime(p.time))},${y(value(p))}`).join(' '),fill:'none',stroke:tradeMetric==='price'||tradeMetric==='frictionless'?'#85baff':value(valid.at(-1))<(tradeMetric==='return'?0:initial)?'#ff8989':value(valid.at(-1))>(tradeMetric==='return'?0:initial)?'#7ce1b6':'#85baff','stroke-width':3}));
 if(compared.length){plot.append(svg('polyline',{points:compared.map(p=>`${x(tradeTime(p.time))},${y(comparison(p))}`).join(' '),fill:'none',stroke:tradeMetric==='frictionless'?'#ff8989':'#85baff','stroke-width':2,'stroke-dasharray':'6 4'}));get('trade-explanation').textContent+=' Solid line: selected view. Dashed comparison: '+(tradeMetric==='frictionless'?'net value after costs.':'hypothetical return without costs.')}
 for(const p of valid){const dot=svg('circle',{cx:x(tradeTime(p.time)),cy:y(value(p)),r:5,fill:'transparent',tabindex:0});const describe=()=>{get('trade-hover').textContent=`${new Date(tradeTime(p.time)).toLocaleString()} | ${tradeMetric}: ${value(p).toFixed(5)} | Equity: ${p.equity.toFixed(5)} ${row.currency} | ${p.valuation||'ledger valuation'}`};dot.onmouseenter=describe;dot.onfocus=describe;plot.append(dot)}
 for(const trade of row.executions||[]){if(tradeMetric==='frictionless')continue;const t=tradeTime(trade.time);if(t<lo||t>hi)continue;const nearest=valid.reduce((a,b)=>Math.abs(tradeTime(a.time)-t)<=Math.abs(tradeTime(b.time)-t)?a:b);const buy=['buy','short','bet'].includes(trade.action),color=buy?'#56d6a0':'#ff7b86';const marker=svg(buy?'polygon':'rect',{cx:x(t),cy:y(tradeMetric==='price'?+trade.price:(Number.isFinite(trade.equity_after)?value({equity:trade.equity_after,spot_equity:trade.spot_equity_after}):value(nearest))),r:7,fill:color,stroke:'#101721','stroke-width':2,tabindex:0,role:'button','aria-label':`${trade.action} ${trade.symbol||''} ${trade.time}`});if(tradeMetric==='spot'&&!Number.isFinite(trade.spot_equity_after))continue;const mx=x(t),my=y(tradeMetric==='price'?+trade.price:(Number.isFinite(trade.equity_after)?value({equity:trade.equity_after,spot_equity:trade.spot_equity_after}):value(nearest)));if(buy)marker.setAttribute('points',`${mx},${my-9} ${mx+9},${my+8} ${mx-9},${my+8}`);else{marker.setAttribute('x',mx-7);marker.setAttribute('y',my-7);marker.setAttribute('width',14);marker.setAttribute('height',14)}marker.style.cursor='pointer';marker.append(svg('title',{},`${trade.action} | price ${trade.price??'n/a'} | actual modeled fill costs ${trade.costs??'unknown'} | estimated exit costs ${trade.estimated_exit_cost??'unknown'} | net P&L ${trade.net_pnl??'not realized at entry'}`));const choose=()=>{tradeSelected=trade.receipt;get('trade-detail').textContent=JSON.stringify(trade,null,2);get('trade-hover').textContent=executionDescription(trade,row);get('trade-receipt').open=true};marker.onclick=choose;marker.onkeydown=e=>{if(e.key==='Enter'||e.key===' ')choose()};plot.append(marker)}
 plot.append(svg('text',{x:65,y:282,fill:'#adbdd1','font-size':11},new Date(lo).toLocaleString()),svg('text',{x:650,y:282,fill:'#adbdd1','font-size':11},new Date(hi).toLocaleString()));
 if(!row.executions.length)get('trade-hover').textContent='No recorded fills in this tail. A flat equity line with no fills is expected; it is not a validated strategy.';else if(valid.every(p=>value(p)===value(valid[0])))get('trade-hover').textContent='All recorded values in this range are identical. Check snapshot timestamps and execution receipts.';
 if(tradeSelected){const selected=row.executions.find(t=>t.receipt===tradeSelected);if(selected){get('trade-detail').textContent=JSON.stringify(selected,null,2);get('trade-hover').textContent=executionDescription(selected,row)}}
}
async function tradesView(){return singleFetch('trades',async()=>{try{tradePanel();dashboardNavigation();const data=await readAPI('/api/trades');const retained=tradeData?.series?.find(s=>s.id===tradeChoice);if(retained&&!(data.series||[]).some(s=>s.id===tradeChoice))data.series=[...(data.series||[]),{...retained,retained_previous_snapshot:true}];stablePaint(()=>{tradeData=data;drawTrades()})}catch(error){const node=document.getElementById('trade-errors');if(node)node.textContent='Trade data unavailable: '+error.message}})}

async function refresh(){return singleFetch('status',async()=>{try{const data=await readAPI('/api/status');stablePaint(()=>render(data));tradePanel();dashboardNavigation()}catch(error){$('warning').textContent='Refresh failed: '+error.message}})}
async function liveInference(){if(!latestSnapshot)return;return singleFetch('inference',async()=>{try{const live=await readAPI('/api/live-inference');if(latestInference&&inferenceStamp(live)<inferenceStamp(latestInference))return;latestInference=live;stablePaint(()=>agentView(mergeInference(latestSnapshot,live)));dashboardNavigation()}catch(error){$('warning').textContent='Live inference unavailable: '+error.message}})}
async function readinessView(){return singleFetch('readiness',async()=>{try{const data=await readAPI('/api/readiness');stablePaint(()=>{const p=panel('readiness-panel','Commissioning evidence',$('goal-learning-panel')||$('plan-panel'));let body=$('readiness-view');if(!body){body=document.createElement('div');body.id='readiness-view';p.append(body)}body.replaceChildren();for(const c of data.checks||[])textField(body,c.name,`${c.state} | ${c.detail} | ${c.source||''}`)});dashboardNavigation()}catch(error){$('warning').textContent='Readiness unavailable: '+error.message}})}
document.addEventListener('click',e=>{if(e.target?.id==='archive-load')loadArchive()});document.addEventListener('change',e=>{if(e.target?.id==='archive-day')loadArchive(true)});$('reload').onclick=()=>pendingLayout?location.reload():refresh();refresh();setInterval(refresh,5000);setInterval(liveInference,2000);setInterval(readinessView,5000);setInterval(refreshAgeBadge,1000);
</script></html>"""


PAGE = PAGE.replace("let renderRoot=null;", INSIGHTS_SCRIPT + "\nlet renderRoot=null;", 1)

PAGE = PAGE.replace("</script>", "tradesView();setInterval(tradesView,10000);</script>", 1)

APP_SCRIPT = PAGE.split("<script>", 1)[1].split("</script>", 1)[0]
BASE_TEMPLATE = PAGE.split("<script>", 1)[0] + "</html>"
GUEST_TEMPLATE = "/workspace/dashboard/index.html"
REQUIRED_IDS = set(re.findall(r"\$\('([a-z]+)'\)", APP_SCRIPT)) - {"actors", "goals", "events"}

HOST_STYLE = """
:root{scroll-behavior:auto!important;overflow-anchor:none;color-scheme:dark;--line:#2b4058;--muted:#a6b7cc}
body{max-width:1440px;margin:auto;padding:28px;background:linear-gradient(145deg,#0b1220,#14243b);color:#e7eef9;font:15px/1.6 system-ui}
h1{font-size:32px;letter-spacing:-1px;margin-bottom:8px}h2{font-size:21px;margin-top:0}h3{font-size:16px;overflow-wrap:anywhere}
section,details{box-shadow:0 8px 24px #0003;background:#111e30;border:1px solid var(--line);border-radius:16px;padding:22px;margin:20px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;align-items:start}
.card{background:#16263b;border:1px solid var(--line);border-radius:12px;padding:18px;margin:0;min-width:0}
.agent{border-top:3px solid #75c5ae}.agent p{font-size:13px;overflow-wrap:anywhere}.agent strong{color:#9ac5ff}
#agent-panel{border-color:#487087}#actors .card{border-top:3px solid #82aef9}.muted,small{color:var(--muted)}
.value{font-size:24px}pre{background:#0c1726;border-radius:8px;padding:14px;font-size:12px;max-height:280px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere}
summary{cursor:pointer;color:#9ac5ff}#events{overflow-anchor:none}details[open]>pre{min-height:180px}#events p{padding:10px 14px;border-left:2px solid #487087;background:#0c1726;font-size:12px;overflow-wrap:anywhere}
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
        "V100 - Centrum Dowodzenia": "Synta - mission control",
        "Monitorowanie misji w czasie rzeczywistym. Systemy operacyjne aktywne.": "Live mission evidence. Service availability and verified results are separate.",
        "Wynik netto prób historycznych. Analiza trendów i odchyleń.": "Historical net returns after assumed costs. Drawdown is peak-to-trough equity loss.",
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
        "Obsunięcie": "Maximum drawdown",
        "Odchylenie": "Maximum drawdown",
        "Wykonania": "Fills",
        'lang="pl"': 'lang="en"',
    }.items():
        content = content.replace(old, new)
    content = content.replace("Synta - mission progress", "Synta - mission control")
    script = (
        "<style>"
        + HOST_STYLE
        + INSIGHTS_STYLE
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
