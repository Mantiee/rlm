"""Host-owned navigation and evidence summaries for existing passive layouts."""

SCRIPT = r"""
let dashboardTab='overview';
const dashboardTabs=[['overview','Overview'],['trades','Trading'],['income','Other income'],['learning','Learning'],['activity','Agents & resources'],['diagnostics','Evidence archive']];
function dashboardNavigation(){
 let nav=document.getElementById('mission-navigation');
 if(!nav){nav=document.createElement('nav');nav.id='mission-navigation';nav.setAttribute('aria-label','Dashboard sections');for(const [key,label] of dashboardTabs){const b=document.createElement('button');b.textContent=label;b.dataset.tab=key;b.onclick=()=>{dashboardTab=key;dashboardNavigation()};nav.append(b)}document.getElementById('cards').before(nav)}
 const groups={diagnostics:['timeline-panel','action-archive'],trades:['trade-explorer','backtests','paper'],income:['income-panel','source-panel'],learning:['goal-learning-panel','evaluation'],activity:['resource-panel','agent-panel','plan-panel'],overview:['live-mission-board','overview-panel','current-activity-panel','model-deployment-panel']};
 for(const n of document.querySelectorAll('body > section, body > details')){let group='diagnostics';for(const [key,ids] of Object.entries(groups))if(ids.some(id=>n.id===id||n.querySelector('#'+id)))group=key;n.hidden=group!==dashboardTab;n.dataset.dashboardGroup=group}
 for(const b of nav.children){b.setAttribute('aria-pressed',String(b.dataset.tab===dashboardTab))}
 const cards=document.getElementById('cards');cards.hidden=dashboardTab!=='overview';
 // Repair older custom guest tables with four headers but six data columns.
 const table=document.getElementById('backtests')?.closest('table');if(table){let head=table.querySelector('thead');if(!head){head=document.createElement('thead');table.prepend(head)}const tr=document.createElement('tr');for(const title of ['Trial','Net return','Maximum drawdown (peak to trough)','Fills','Triage','Net return at double costs']){const th=document.createElement('th');th.scope='col';th.textContent=title;tr.append(th)}head.replaceChildren(tr)}
}
function evidenceBars(parent,title,rows,unit){
 const h=document.createElement('h3');h.textContent=title;parent.append(h);const usable=rows.filter(r=>typeof r[1]==='number'&&Number.isFinite(r[1]));if(!usable.length){textField(parent,'Status','No measured values available');return}
 textField(parent,'How to read',unit==='Brier'?'Lower is better. Bar length represents the recorded score, not income.':'Bar length is relative to the largest recorded count. These are counts, not profit.');const max=Math.max(1,...usable.map(r=>r[1]));for(const [label,value] of usable){const box=document.createElement('div');box.className='evidence-bar';const text=document.createElement('span');text.textContent=label+': '+value.toFixed(unit==='Brier'?4:0)+' '+unit;const track=document.createElement('div');track.className='bar-track';const bar=document.createElement('div');bar.style.width=(value/max*100)+'%';track.append(bar);box.append(text,track);parent.append(box)}
}
function liveMissionBoard(data){
 const report=data.report||{}, mission=data.mission||{}, learning=mission.learning||{}, live=data.live||{};
 const mount=$('cards');if(!mount)return;
 const box=panel('live-mission-board','Live mission control',mount);
 box.replaceChildren();
 const header=document.createElement('div');header.className='live-board-header';
 const h=document.createElement('h2');h.textContent='What Synta is doing';header.append(h);
 const stamp=document.createElement('small');
 const collected=Number(data.collected_at);
 const age=Number.isFinite(collected)?Math.max(0,Math.round(Date.now()/1000-collected)):null;
 stamp.textContent=age===null?'Snapshot time unknown':`Snapshot age: ${age}s${age>30?' - STALE':''}`;
 header.append(stamp);box.append(header);
 const metrics=document.createElement('div');metrics.className='live-board-metrics';box.append(metrics);
 const metric=(name,value,description)=>{const card=document.createElement('div');card.className='live-board-metric';const k=document.createElement('small');k.textContent=name;const v=document.createElement('strong');v.textContent=String(value);const d=document.createElement('small');d.textContent=description;card.append(k,v,d);metrics.append(card)};
 metric('Mission',mission.running===true?'Running':mission.running===false?'Stopped':'Unknown',mission.state?.phase||'Phase unavailable');
 metric('Learning',learning.stage||'Unknown',learning.cycle_in_progress===undefined?'Cycle not reported':`Cycle ${learning.cycle_in_progress}`);
 const rows=report.paper?.branches||report.paper_report?.branches||{};
 const a=rows.A||{},b=rows.B||{};
 const money=x=>x===undefined||x===null?'Not reported':String(x)+' PLN';
 metric('Paper A P&L',money(a.net_pnl_before_personal_tax),'Simulation, not real income');
 metric('Paper B P&L',money(b.net_pnl_before_personal_tax),'Simulation, not real income');
 const note=document.createElement('p');note.className='muted';
 note.textContent='Real income: not independently verified. HOLD is a decision, not a fill. Paper results and historical tests are not withdrawable earnings.';box.append(note);
 const blocker=document.createElement('div');blocker.className='live-board-blocker';
 const bt=document.createElement('strong');bt.textContent='Current blocker';
 const bd=document.createElement('p');bd.textContent=learning.reason||learning.last_cycle?.detail||report.income_work?.error||report.paper_fee_source?.error||'No blocker recorded in this snapshot (not a health guarantee)';
 blocker.append(bt,bd);box.append(blocker);
 const events=document.createElement('div');events.className='live-board-events';
 const eh=document.createElement('h3');eh.textContent='Recent recorded actions';events.append(eh);
 const recent=Array.isArray(live.events)?live.events.slice(-8).reverse():[];
 if(!recent.length){const p=document.createElement('p');p.textContent='No current events reported';events.append(p)}
 for(const event of recent){const line=document.createElement('div');line.className='live-board-event';const time=document.createElement('time');time.textContent=event.time||'Unknown time';const body=document.createElement('span');body.textContent=event.label||event.action||event.kind||event.type||event.summary||'Recorded event';line.append(time,body);events.append(line)}
 box.append(events);
}
function insightsView(data){
 liveMissionBoard(data);
 const r=data.report||{},g=r.mission_evidence?.goal_learning||{},policy=g.cpu_shadow_policy||{},work=r.income_work||{},sources=r.source_acquisition||{};
 const p=panel('overview-panel','What is actually happening',$('cards'));p.replaceChildren();const h=document.createElement('h2');h.textContent='Work towards the goal';p.append(h);textField(p,'Verified revenue','Unavailable: no verified payment ledger connected');textField(p,'Current model-selected work',work.state||'No work receipt in this snapshot');textField(p,'Last work recorded',work.updated?new Date(work.updated*1000).toLocaleString():'unknown');textField(p,'Work receipt age',work.updated?`${Math.max(0,Math.round((Date.now()/1000-work.updated)/60))} minutes; refreshing the page does not create a new result`:'No dated work receipt');textField(p,'Successful tool receipts in last experiment',work.successful_tool_receipts);textField(p,'Meaning','A source read or a successful process is not a validated experiment, a model update or earned money.');textField(p,'CPU shadow learning',policy.state||'No policy receipt');textField(p,'Verified outcomes used by CPU policy',policy.trained_outcomes);textField(p,'Gemma accepted updates in this run',r.accepted_weight_updates_this_run);textField(p,'Immediate blocker',work.error||r.learning_activity?.reason||r.paper_fee_source?.error||'None recorded in these receipts');
 detailsField(p,'Last experiment receipt',JSON.stringify(work,null,2),'overview-work');
 const income=$('income-view');if(income){const counts=new Map();for(const c of r.income_opportunities?.candidates||[]){const name=c.specification?.domain||'Unspecified domain';counts.set(name,(counts.get(name)||0)+1)}evidenceBars(income,'Research coverage by domain (proposals, not earnings)',[...counts],'proposals');evidenceBars(income,'Source acquisition quality',[['Distinct contents',sources.distinct_contents],['Unchanged rereads',sources.unchanged_rereads]],'retrievals');textField(income,'Outcome charts','No verified revenue or completed-test time series for these domains is recorded. Proposal counts and estimated rewards cannot supply a profit curve.');}
 const learning=$('goal-learning-view');if(learning){textField(learning,'CPU policy',policy.state||'unavailable');textField(learning,'Independent outcomes trained',policy.trained_outcomes);textField(learning,'Scored replay predictions',policy.prequential_samples);if(policy.prequential_samples>0&&typeof policy.prequential_brier==='number'){const references=[policy.uniform_baseline_brier,policy.prequential_flat_baseline_brier,policy.prequential_frequency_baseline_brier].filter(x=>typeof x==='number'&&Number.isFinite(x));textField(learning,'Compared with strongest recorded reference',references.length&&policy.prequential_brier<Math.min(...references)?'Lower Brier in this replay only; forward advantage unproven':'No improvement over the strongest reference in this replay')}evidenceBars(learning,'CPU forecast replay comparison: lower Brier is better',[['Learned shadow',policy.prequential_brier],['Uniform reference',policy.prequential_samples>0?policy.uniform_baseline_brier:null],['Always flat',policy.prequential_flat_baseline_brier],['Prior frequency',policy.prequential_frequency_baseline_brier]],'Brier');textField(learning,'Score scope','Exploratory prequential replay on the same scored samples. Not profit, not Gemma training and not proof of a forward advantage.');}
}

"""

STYLE = """
#live-mission-board{border:1px solid #355475;background:linear-gradient(130deg,#15283d,#101927)}.live-board-header{display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap}.live-board-metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:10px}.live-board-metric{display:flex;flex-direction:column;gap:7px;padding:16px;background:#101c2d;border:1px solid #37516b;border-radius:10px}.live-board-metric strong{font-size:21px;overflow-wrap:anywhere}.live-board-metric small{color:#b6c8de}.live-board-blocker{padding:12px;border-left:3px solid #f1b778;background:#241f24;margin:14px 0}.live-board-blocker p{margin-bottom:0;overflow-wrap:anywhere}.live-board-events{max-height:300px;overflow:auto}.live-board-event{padding:10px;border-bottom:1px solid #304259;display:flex;gap:12px;flex-wrap:wrap}.live-board-event time{color:#9cb6d4;font-size:12px}
[hidden]{display:none!important}body{background:#0c1422;max-width:1280px}#mission-navigation{display:flex;gap:8px;flex-wrap:wrap;position:sticky;top:0;z-index:10;padding:12px 0;background:#0c1422;border-bottom:1px solid #304259;margin-bottom:20px}#mission-navigation button{background:#17263c;border:1px solid #304259;color:#bac9de}#mission-navigation button[aria-pressed=true]{background:#29578b;color:#fff;border-color:#85baff}button:focus-visible,a:focus-visible,select:focus-visible{outline:3px solid #ffd166;outline-offset:3px}section{box-shadow:none}h1{letter-spacing:-.6px}#cards{grid-template-columns:repeat(auto-fit,minmax(170px,1fr))}#overview-panel{border-left:4px solid #80baff}table{display:block;overflow:auto;white-space:nowrap}th{font-size:12px;color:#b9cbe2}select,input{background:#14243a;color:#eef5ff;border:1px solid #486582;padding:9px;border-radius:7px;max-width:100%}label{display:grid;gap:6px;font-size:13px}.evidence-bar{margin:12px 0;font-size:13px}.bar-track{height:9px;background:#26384e;border-radius:9px;margin-top:6px}.bar-track>div{height:100%;background:#85baff;border-radius:9px}#income-view>.card{margin:8px 0}#backtest-audit{max-height:120px}#fresh,#layout{font-size:12px}#trade-plot{background:#0c1726;border-radius:10px}#trade-summary{font-weight:600}#trade-scope{font-size:12px}#trade-hover{font-size:13px}#timeline-panel{max-height:480px;overflow:auto}#income-view details.card>summary{cursor:pointer;font-weight:600}#plan-grid p{max-height:180px;overflow:auto}@media(max-width:650px){#mission-navigation{gap:6px}#mission-navigation button{padding:8px 10px;font-size:12px}}
"""
