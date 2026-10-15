"""Host-owned navigation and evidence summaries for existing passive layouts."""

SCRIPT = r"""
let dashboardTab='live';
const dashboardTabs=[['live','Na żywo'],['trades','Paper i rynki'],['income','Inne przychody'],['learning','Uczenie'],['activity','Agenty i zasoby'],['diagnostics','Archiwum dowodów']];
const dashboardGroups={diagnostics:['timeline-panel','action-archive'],trades:['trade-explorer','backtests','paper','paper-labs-panel'],income:['income-panel','source-panel'],learning:['goal-learning-panel','evaluation'],activity:['resource-panel','agent-panel','plan-panel'],live:['live-mission-board','overview-panel','current-activity-panel','model-deployment-panel','trade-explorer','backtests','paper','paper-labs-panel','income-panel','source-panel','goal-learning-panel','evaluation','resource-panel','agent-panel','plan-panel']};
function dashboardSectionGroup(n){let group=n.tagName==='DETAILS'?'diagnostics':'live';for(const [key,ids] of Object.entries(dashboardGroups))if(key!=='live'&&ids.some(id=>n.id===id||n.querySelector('#'+id)))group=key;return group}
function dashboardNavigation(){
 let nav=document.getElementById('mission-navigation');
 if(!nav){nav=document.createElement('nav');nav.id='mission-navigation';nav.setAttribute('aria-label','Dashboard sections');for(const [key,label] of dashboardTabs){const b=document.createElement('button');b.textContent=label;b.dataset.tab=key;b.onclick=()=>{dashboardTab=key;dashboardNavigation()};nav.append(b)}document.getElementById('cards').before(nav)}
 for(const n of document.querySelectorAll('body > section, body > details')){const group=dashboardSectionGroup(n);n.hidden=dashboardTab==='live'?group==='diagnostics':group!==dashboardTab;n.dataset.dashboardGroup=group}
 for(const b of nav.children){b.setAttribute('aria-pressed',String(b.dataset.tab===dashboardTab))}
 const cards=document.getElementById('cards');cards.hidden=dashboardTab!=='live';
 // Repair older custom guest tables with four headers but six data columns.
 const table=document.getElementById('backtests')?.closest('table');if(table){let head=table.querySelector('thead');if(!head){head=document.createElement('thead');table.prepend(head)}const tr=document.createElement('tr');for(const title of ['Trial','Net return','Maximum drawdown (peak to trough)','Fills','Triage','Net return at double costs']){const th=document.createElement('th');th.scope='col';th.textContent=title;tr.append(th)}head.replaceChildren(tr)}
}
function evidenceBars(parent,title,rows,unit){
 const h=document.createElement('h3');h.textContent=title;parent.append(h);const usable=rows.filter(r=>typeof r[1]==='number'&&Number.isFinite(r[1]));if(!usable.length){textField(parent,'Status','No measured values available');return}
 textField(parent,'How to read',unit==='Brier'?'Lower is better. Bar length represents the recorded score, not income.':'Bar length is relative to the largest recorded count. These are counts, not profit.');const max=Math.max(1,...usable.map(r=>r[1]));for(const [label,value] of usable){const box=document.createElement('div');box.className='evidence-bar';const text=document.createElement('span');text.textContent=label+': '+value.toFixed(unit==='Brier'?4:0)+' '+unit;const track=document.createElement('div');track.className='bar-track';const bar=document.createElement('div');bar.style.width=(value/max*100)+'%';track.append(bar);box.append(text,track);parent.append(box)}
}
function liveMissionBoard(data){
 const report=data.report||{}, mission=data.mission||{}, learning=mission.learning||{}, live=data.live||{};
 const statusFreshness=report.status_freshness||{};
 const receiptFreshness=(name,receipt)=>{
  const recorded=statusFreshness[name]?.state;
  if(['fresh','stale','unknown','unavailable'].includes(recorded))return recorded;
  const stamp=receipt?.updated;
  const observed=typeof stamp==='number'?stamp:typeof stamp==='string'?Date.parse(stamp)/1000:NaN;
  if(!Number.isFinite(observed)||observed<=0)return 'unknown';
  const age=Date.now()/1000-observed;
  return age<0?'unknown':age<=300?'fresh':'stale';
 };
 const mount=$('cards');if(!mount)return;
 const box=panel('live-mission-board','Live mission control',mount);
 box.replaceChildren();
 const header=document.createElement('div');header.className='live-board-header';
 const h=document.createElement('h2');h.textContent='What Synta is doing';header.append(h);
 const stamp=document.createElement('small');
 const collected=Number(data.collected_at);
 const age=Number.isFinite(collected)?Math.max(0,Math.round(Date.now()/1000-collected)):null;
 stamp.textContent=age===null?'Snapshot time unknown':`Status snapshot collected ${age}s ago${age>300?' - delayed':''}`;
 header.append(stamp);box.append(header);
 const metrics=document.createElement('div');metrics.className='live-board-metrics';box.append(metrics);
 const metric=(name,value,description)=>{const card=document.createElement('div');card.className='live-board-metric';const k=document.createElement('small');k.textContent=name;const v=document.createElement('strong');v.textContent=String(value);const d=document.createElement('small');d.textContent=description;card.append(k,v,d);metrics.append(card)};
 metric('Mission: paper income simulation',report.live_paper?'Observing paper A/B':mission.running===true?'Waiting for paper ledger':'Unavailable',`Actual controller: ${mission.state?.phase||'unknown'}; no real orders`);
 metric('Learning',learning.stage||'Unknown',learning.cycle_in_progress===undefined?'Cycle not reported':`Cycle ${learning.cycle_in_progress}`);
 const activity=report.current_activity||{}, evaluation=activity.evaluation||{};
 metric('Current task',activity.stage||'Not reported',activity.running===true?'Recorded as running':'No running task confirmed');
 metric('Candidate evaluation',evaluation.total?`${evaluation.completed||0}/${evaluation.total}`:'Not reported',evaluation.total?`${evaluation.passed||0} passed, ${(evaluation.completed||0)-(evaluation.passed||0)} failed; ${evaluation.state||'unknown'}; ${Math.round(evaluation.age_seconds||0)}s since last update${evaluation.stale?' (STALE)':''}`:'No evaluation receipt');
 metric('Candidate selected for next serving',activity.accepted_updates??'Unknown','A cycle receipt records candidate selection for a future serving phase; it does not prove serving loaded it or revenue. Optimizer steps are separate.');
 const rows=report.live_paper?.branches||{};
 const a=rows.A||{},b=rows.B||{};
 const money=x=>x===undefined||x===null?'Not reported':String(x)+' '+(report.live_paper?.currency||'PLN');
 metric('Paper A P&L',money(a.net_pnl_before_personal_tax),'Simulation, not real income');
 metric('Paper B P&L',money(b.net_pnl_before_personal_tax),'Simulation, not real income');
 metric('Paper fills',report.live_paper?.executed_fills??'Not verified','The live ledger alone does not certify execution count');
 const paper=report.live_paper||{};
 metric('Paper ledger sequence',paper.ledger_sequence??'Unavailable',paper.last_event?`Last verified event: ${paper.last_event.kind} at ${paper.last_event.time}`:'No verified event');
 metric('Live market inputs',`${paper.quotes_available??'unknown'} quotes / ${paper.registered_instruments??'unknown'} instruments`,paper.blockers?.length?paper.blockers.slice(0,2).join('; '):'No detected feed blocker; this is not proof of execution');
 for(const [branch,portfolio] of [['A',a],['B',b]]){
  metric(`Paper ${branch} cash / equity`,`${money(portfolio.cash)} / ${money(portfolio.equity)}`,'Mark-to-market simulation. Costs, stale quotes and unrealized changes affect equity');
  metric(`Paper ${branch} modeled costs`,money(portfolio.modeled_costs),'No verified earnings; all numbers are simulation');
 }
 const node=report.node||report.node3060||report.remote_node||{};
 const nodeFreshness=receiptFreshness('node3060',node);
 const nodeAge=statusFreshness.node3060?.age_seconds;
 const nodeAgeLabel=typeof nodeAge==='number'?` · receipt ${Math.round(nodeAge)}s old`:'';
 if(node.admitted!==undefined){metric('RTX 3060 Ti GPU',nodeFreshness==='fresh'?(node.admitted?'Admitted':'Not admitted'):nodeFreshness==='stale'?'Stale receipt':'Unknown',`${nodeFreshness==='fresh'?'Current node receipt':nodeFreshness==='stale'?'Last recorded node result':'Node receipt freshness unknown'}${nodeAgeLabel} · ${node.reason||'No qualification detail'}`);metric('RTX 3060 Ti CPU',nodeFreshness==='fresh'?(node.resources?.cpu_ready?'Available':'Not ready'):nodeFreshness==='stale'?'Stale receipt':'Unknown',`${nodeFreshness==='fresh'?'Current node receipt':nodeFreshness==='stale'?'Last recorded node result':'Node receipt freshness unknown'}${nodeAgeLabel} · ${node.resources?.cpu_active??'unknown'} jobs active, ${(node.resources?.cpu_slots??'unknown')} slots; GPU admission is independent`)}
 const dispatch=report.owned_cpu_dispatch||{};
 metric('Owned compute experiment',dispatch.state||'Not recorded',dispatch.reason||dispatch.next_evidence||'Verified source-disjoint outcomes required before GPU/CPU job');
 if(node.resources)metric('3060 GPU compute',nodeFreshness==='fresh'?(node.resources.gpu_compute_claimed?'Running experiment':node.resources.ready?'Available; no confirmed GPU task':'Not ready'):nodeFreshness==='stale'?'Stale receipt':'Unknown',`${nodeFreshness==='fresh'?'Current node receipt':nodeFreshness==='stale'?'Last recorded node result':'Node receipt freshness unknown'}${nodeAgeLabel} · Last recorded: ${node.resources.gpu_utilization_percent??'unknown'}% GPU; ${(node.resources.free_vram_gib??'unknown')} GiB VRAM free`);
 const note=document.createElement('p');note.className='muted';
 note.textContent='Real income: not independently verified. HOLD is a decision, not a fill. Paper results and historical tests are not withdrawable earnings.';box.append(note);
 const blocker=document.createElement('div');blocker.className='live-board-blocker';
 const bt=document.createElement('strong');bt.textContent='Current blocker / last recorded issue';
 const livePaperStamp=Date.parse(report.live_paper?.collected_at||'')/1000;
 const livePaperAge=Date.now()/1000-livePaperStamp;
 const livePaperFresh=Number.isFinite(livePaperStamp)&&livePaperStamp>0&&livePaperAge>=0&&livePaperAge<=300;
 const livePaperBlockers=Array.isArray(report.paper_blockers)?report.paper_blockers:[];
 const recordedIssues=[];
 if(report.paper_spot_rules?.state==='blocked')recordedIssues.push({name:'Paper spot rules',value:report.paper_spot_rules.reason,freshness:receiptFreshness('paper_spot_rules',report.paper_spot_rules)});
 if(livePaperBlockers.length)recordedIssues.push({name:'Live paper ledger',value:livePaperBlockers.join('; '),freshness:livePaperFresh?'fresh':'stale'});
 if(report.income_work?.error)recordedIssues.push({name:'Income work',value:report.income_work.error,freshness:receiptFreshness('income_work',report.income_work)});
 if(learning.reason)recordedIssues.push({name:'Learning loop',value:learning.reason,freshness:receiptFreshness('learning_activity',report.learning_activity)});
 if(report.paper_fee_source?.error)recordedIssues.push({name:'Paper fee source',value:report.paper_fee_source.error,freshness:receiptFreshness('paper_fee_source',report.paper_fee_source)});
 const currentIssue=recordedIssues.find(issue=>issue.freshness==='fresh');
 const lastIssue=recordedIssues[0];
 const bd=document.createElement('p');bd.textContent=currentIssue?`${currentIssue.name}: ${currentIssue.value}`:lastIssue?`No fresh blocker receipt. Last recorded issue (${lastIssue.freshness}): ${lastIssue.name}: ${lastIssue.value}`:'No fresh blocker receipt; trades or income can still be unavailable.';
 blocker.append(bt,bd);box.append(blocker);
 if(learning.last_cycle?.detail){const past=document.createElement('p');past.className='muted';past.textContent='Previous cycle failure (historical, not the current blocker): '+learning.last_cycle.detail;box.append(past)}
 const events=document.createElement('div');events.className='live-board-events';
 const eh=document.createElement('h3');eh.textContent='Recent recorded actions';events.append(eh);
 const raw=Array.isArray(live.events)?live.events:[];
 const actionEvents=raw.filter(e=>['tool-result','inference-finished','worker-result','development-result','paper-fill','trade-fill','compute-training'].includes(e.kind||e.type));
 const recent=(actionEvents.length?actionEvents:raw).slice(-8).reverse();
 if(!actionEvents.length){const p=document.createElement('p');p.className='muted';p.textContent='No recent completed operations; raw heartbeat and observations are not evidence of executed work.';events.append(p)}
 if(!recent.length){const p=document.createElement('p');p.textContent='No current events reported';events.append(p)}
 for(const event of recent){const line=document.createElement('div');line.className='live-board-event';const time=document.createElement('time');time.textContent=event.time||'Unknown time';const body=document.createElement('span');body.textContent=event.label||event.action||event.kind||event.type||event.summary||'Recorded event';line.append(time,body);events.append(line)}
 box.append(events);
}
function paperCapitalPreview(data){
 const anchor=$('live-mission-board');if(!anchor)return;
 const p=panel('paper-capital-preview','Co model proponuje przy kapitale 100 PLN? — symulacja',$('live-mission-board'));p.replaceChildren();
 const h=document.createElement('h2');h.textContent='Co model proponuje przy kapitale 100 PLN?';p.append(h);
 const report=data.report||{},paper=report.live_paper||{},policy=report.live_execution_policy||{},stream=paper.activity_stream||{};
 const capital=Number(policy.capital_limit_pln??100),starting=Number(paper.initial_capital),now=Date.now()/1000;
 textField(p,'Limit symulacji',`${Number.isFinite(capital)?capital.toFixed(2):'UNKNOWN'} PLN · dzienny stop ${policy.daily_loss_stop_pln??'UNKNOWN'} PLN · łączny stop ${policy.max_total_loss_pln??'UNKNOWN'} PLN · dźwignia ${policy.leverage_enabled===true?'włączona':'wyłączona'}`);
 textField(p,'Aktualizacja','Propozycje odświeżają się razem ze snapshotem dashboardu co 5 sekund. To nie są zlecenia.');
 const wallets=document.createElement('div');wallets.className='live-board-metrics';p.append(wallets);
 for(const branch of ['A','B']){
  const source=paper.branches?.[branch],card=document.createElement('article');card.className='live-board-metric';wallets.append(card);
  const title=document.createElement('strong');title.textContent=`Portfel ${branch} — równowartość na 100 PLN`;card.append(title);
  const detail=document.createElement('small');detail.style.whiteSpace='pre-wrap';
  const equity=Number(source?.equity),pnl=Number(source?.net_pnl_before_personal_tax),cash=Number(source?.cash);
  if(Number.isFinite(starting)&&starting>0&&source&&Number.isFinite(equity)){
   const money=value=>Number.isFinite(value)?`${(value*capital/starting).toFixed(2)} PLN`:'UNKNOWN';
   const positions=Object.entries(source.positions||{}).map(([symbol,position])=>{const units=Number(typeof position==='object'?position.quantity:position);return `${symbol} ${Number.isFinite(units)?(units*capital/starting).toFixed(8):'UNKNOWN'}${position?.side?' '+position.side:''}`}).join(', ')||'brak zapisanej pozycji';
   detail.textContent=`Wartość po wycenie: ${money(equity)}\nP&L księgi: ${money(pnl)}\nGotówka: ${money(cash)}\nPozycje (skalowane jednostki): ${positions}`;
  }else detail.textContent='Saldo albo pochodzenie kapitału UNKNOWN w tej migawce.';
  card.append(detail);
 }
 textField(p,'Zakres sald','To przeskalowany odczyt aktualnej księgi A/B, nie osobny rachunek 100 PLN. Wartość pozycji zależy od wyceny księgi; wynik nie uwzględnia niezweryfikowanych kosztów jako rzeczywistego wyniku. Historycznych migawek salda 100 PLN księga nie zapisuje.');
 const latest=new Map();
 for(const entry of stream.entries||[]){const proposal=entry.proposal||{};if(!entry.branch||!['A','B'].includes(entry.branch)||proposal.source!=='model-proposal')continue;latest.set(entry.branch,{entry,proposal})}
 const grid=document.createElement('div');grid.className='live-board-metrics';p.append(grid);
 for(const branch of ['A','B']){
  const item=latest.get(branch),card=document.createElement('article');card.className='live-board-metric';grid.append(card);
  const title=document.createElement('strong');title.textContent=`Strategia ${branch}`;card.append(title);
  const body=document.createElement('small');body.style.whiteSpace='pre-wrap';card.append(body);
  if(!item){body.textContent='Brak potwierdzonej decyzji modelu w zweryfikowanym oknie księgi.';continue}
  const {entry,proposal}=item,stamp=Date.parse(entry.time||'')/1000,age=Number.isFinite(stamp)?now-stamp:null;
  const action=String(proposal.action||'UNKNOWN').toUpperCase(),symbol=proposal.symbol||'instrument niepodany';
  const rawBudget=Number(proposal.budget),normalized=Number.isFinite(rawBudget)&&Number.isFinite(starting)&&starting>0&&Number.isFinite(capital)?Math.max(0,Math.min(capital,rawBudget*capital/starting)):null;
  const budget=normalized===null?'kwota nieznana':`${normalized.toFixed(2)} PLN (proporcjonalna skala z portfela ${starting.toFixed(2)} PLN)`;
  const quote=proposal.quote||{},quoteText=quote.bid!==undefined||quote.ask!==undefined?`\nBid/ask przy decyzji: ${quote.bid??'UNKNOWN'} / ${quote.ask??'UNKNOWN'}`:'';
  const blocked=proposal.model_action_blocked?`\nModel chciał: ${proposal.model_action_blocked}; bramka zamieniła to na HOLD.`:'';
  const reason=proposal.rationale?`\nPowód modelu: ${proposal.rationale}`:'';
  body.textContent=`${action} · ${symbol}\nBudżet dla 100 PLN: ${String(proposal.action||'').toLowerCase()==='hold'?'0.00 PLN — gotówka pozostaje wolna':budget}\nStatus: ${proposal.status||'zapisano'}${age===null?' · czas UNKNOWN':` · ${Math.max(0,Math.round(age))} s temu`}${quoteText}${blocked}${reason}`;
 }
 const proposals=(stream.entries||[]).filter(entry=>{
  const proposal=entry.proposal||{};
  return ['A','B'].includes(entry.branch)&&proposal.source==='model-proposal'&&Number.isFinite(Date.parse(entry.time||''));
 }).map(entry=>{
  const proposal=entry.proposal,raw=Number(proposal.budget);
  const amount=String(proposal.action||'').toLowerCase()==='hold'?0:
   Number.isFinite(raw)&&Number.isFinite(starting)&&starting>0&&Number.isFinite(capital)?Math.max(0,Math.min(capital,raw*capital/starting)):null;
  return {...entry,amount};
 }).filter(entry=>entry.amount!==null).slice(-100);
 const chartTitle=document.createElement('h3');chartTitle.textContent='Historia proponowanej ekspozycji dla dwóch osobnych portfeli po 100 PLN';p.append(chartTitle);
 const note=document.createElement('p');note.className='muted';note.textContent='Każdy punkt to zapisana decyzja modelu przeskalowana proporcjonalnie z księgi paper. HOLD = 0 PLN. Wykres pokazuje propozycję, nie wykonanie ani saldo. Oś czasu obejmuje zweryfikowane decyzje w ostatnim oknie księgi; nowe punkty dochodzą, gdy system zapisze następną decyzję. Brak prognozy przyszłości.';p.append(note);
 if(proposals.length){
  const ns='http://www.w3.org/2000/svg',chart=document.createElementNS(ns,'svg');chart.setAttribute('viewBox','0 0 800 280');chart.setAttribute('role','img');chart.setAttribute('aria-label','Historia proponowanej ekspozycji A i B, w przeliczeniu na kapitał 100 PLN');
  const add=(tag,attrs,text)=>{const node=document.createElementNS(ns,tag);for(const [key,value] of Object.entries(attrs))node.setAttribute(key,String(value));if(text!==undefined)node.textContent=text;chart.append(node);return node};
  const left=78,right=780,top=40,bottom=210,maxValue=Number.isFinite(capital)&&capital>0?capital:100,times=proposals.map(row=>Date.parse(row.time)),minTime=Math.min(...times),maxTime=Math.max(...times);
  for(let i=0;i<=4;i++){const value=maxValue*(1-i/4),y=top+(bottom-top)*i/4;add('line',{x1:left,x2:right,y1:y,y2:y,stroke:'#35445a','stroke-dasharray':'4 4'});add('text',{x:left-9,y:y+4,fill:'#bdcde1','font-size':12,'text-anchor':'end'},value.toFixed(0)+' zł')}
  for(const [branch,color,name,legendX] of [['A','#57b7ff','Strategia A',left],['B','#f9c36b','Strategia B',left+160]]){
   const rows=proposals.filter(row=>row.branch===branch),points=rows.map(row=>`${maxTime===minTime?(left+right)/2:left+(Date.parse(row.time)-minTime)/(maxTime-minTime)*(right-left)},${bottom-row.amount/maxValue*(bottom-top)}`).join(' ');
   if(rows.length>1)add('polyline',{points,fill:'none',stroke:color,'stroke-width':3});
   for(const row of rows){const x=maxTime===minTime?(left+right)/2:left+(Date.parse(row.time)-minTime)/(maxTime-minTime)*(right-left),y=bottom-row.amount/maxValue*(bottom-top);add('circle',{cx:x,cy:y,r:5,fill:color});}
   add('line',{x1:legendX,x2:legendX+24,y1:246,y2:246,stroke:color,'stroke-width':3});add('text',{x:legendX+31,y:250,fill:'#bdcde1','font-size':12},`${name} · ${rows.length} decyzji`);
  }
  const clock=value=>new Date(value).toLocaleString('pl-PL',{timeZone:'UTC',hour:'2-digit',minute:'2-digit',day:'2-digit',month:'2-digit'})+' UTC';
  add('text',{x:left,y:230,fill:'#bdcde1','font-size':11},clock(minTime));add('text',{x:right,y:230,fill:'#bdcde1','font-size':11,'text-anchor':'end'},clock(maxTime));p.append(chart);
  const last=proposals.slice(-8).reverse().map(row=>`${row.time} · ${row.branch} · ${String(row.proposal.action||'UNKNOWN').toUpperCase()} ${row.proposal.symbol||''} · ${row.amount.toFixed(2)} PLN`).join('\n');
  const history=document.createElement('pre');history.textContent=last;p.append(history);
 }else textField(p,'Historia decyzji','Brak modelowych propozycji z kompletną kwotą w zweryfikowanym oknie księgi. HOLD-y ochronne i same notowania nie są propozycją modelu.');
 const blockers=Array.isArray(report.paper_blockers)?report.paper_blockers:[];
 textField(p,'Dlaczego może nie wykonać',blockers.length?blockers.slice(0,3).join(' · '):'Brak blokad w tej migawce; nadal potrzebne jest potwierdzenie fillu i kosztów.');
 textField(p,'Jak czytać kwotę','A i B to oddzielne strategie, każda przeskalowana proporcjonalnie z zapisanej decyzji większej księgi do limitu 100 PLN. To nie osobna inferencja na koncie 100 PLN. Nie przewiduje przyszłości, fillu ani zysku. Saldo i P&L 100 PLN nie są udawane: wymagają osobnej, ciągłej księgi symulacji i potwierdzonych kosztów.');
 const comparison=(report.recent_exploratory_backtests||[]).filter(row=>row.triage?.state==='hypothesis for independent future testing');
 textField(p,'Wybór strategii','Nie ma potwierdzonej „najlepszej” strategii. Dostępne kandydatury z historycznego replay: '+(comparison.length?comparison.map(row=>row.parameters?.rule||'candidate').join(', '):'brak kandydata, który przeszedł triage')+'. Ranking historyczny nie dowodzi przewagi na przyszłość.');
}
function insightsView(data){
 liveMissionBoard(data);
 paperCapitalPreview(data);
 const r=data.report||{},g=r.mission_evidence?.goal_learning||{},policy=g.cpu_shadow_policy||{},work=r.income_work||{},sources=r.source_acquisition||{},workFreshness=r.status_freshness?.income_work||{};
 const p=panel('overview-panel','What is actually happening',$('cards'));p.replaceChildren();const h=document.createElement('h2');h.textContent='Work towards the goal';p.append(h);textField(p,'Verified revenue','Unavailable: no verified payment ledger connected');textField(p,'Current model-selected work',work.state||'No work receipt in this snapshot');textField(p,'Last work recorded',work.updated?new Date(work.updated*1000).toLocaleString():'unknown');textField(p,'Work receipt age',work.updated?`${Math.max(0,Math.round((Date.now()/1000-work.updated)/60))} minutes; refreshing the page does not create a new result`:'No dated work receipt');textField(p,'Successful tool receipts in last experiment',work.successful_tool_receipts);textField(p,'Meaning','A source read or a successful process is not a validated experiment, a model update or earned money.');textField(p,'CPU shadow learning',policy.state||'No policy receipt');textField(p,'Verified outcomes used by CPU policy',policy.trained_outcomes);textField(p,'Gemma candidates selected for next serving',r.accepted_weight_updates_this_run);const issueReceipts=[{name:'Income work',value:work.error,freshness:workFreshness.state||'unknown'},{name:'Learning loop',value:r.learning_activity?.reason,freshness:r.status_freshness?.learning_activity?.state||'unknown'},{name:'Paper fee source',value:r.paper_fee_source?.error,freshness:r.status_freshness?.paper_fee_source?.state||'unknown'}].filter(issue=>issue.value);const reportedIssue=issueReceipts.find(issue=>issue.freshness==='fresh')||issueReceipts[0];const currentWorkError=reportedIssue?`${reportedIssue.freshness==='fresh'?'Current':reportedIssue.freshness==='stale'?'Last recorded (historical)':'Last recorded (freshness unknown)'} ${reportedIssue.name} issue: ${reportedIssue.value}`:'None recorded in current receipts';textField(p,'Current blocker / last recorded error',currentWorkError);
 detailsField(p,'Last experiment receipt',JSON.stringify(work,null,2),'overview-work');
 const income=$('income-view');if(income){const counts=new Map();for(const c of r.income_opportunities?.candidates||[]){const name=c.specification?.domain||'Unspecified domain';counts.set(name,(counts.get(name)||0)+1)}evidenceBars(income,'Research coverage by domain (proposals, not earnings)',[...counts],'proposals');evidenceBars(income,'Source acquisition quality',[['Distinct contents',sources.distinct_contents],['Unchanged rereads',sources.unchanged_rereads]],'retrievals');textField(income,'Outcome charts','No verified revenue or completed-test time series for these domains is recorded. Proposal counts and estimated rewards cannot supply a profit curve.');}
 const learning=$('goal-learning-view');if(learning){textField(learning,'CPU policy',policy.state||'unavailable');textField(learning,'Independent outcomes trained',policy.trained_outcomes);textField(learning,'Scored replay predictions',policy.prequential_samples);if(policy.prequential_samples>0&&typeof policy.prequential_brier==='number'){const references=[policy.uniform_baseline_brier,policy.prequential_flat_baseline_brier,policy.prequential_frequency_baseline_brier].filter(x=>typeof x==='number'&&Number.isFinite(x));textField(learning,'Compared with strongest recorded reference',references.length&&policy.prequential_brier<Math.min(...references)?'Lower Brier in this replay only; forward advantage unproven':'No improvement over the strongest reference in this replay')}evidenceBars(learning,'CPU forecast replay comparison: lower Brier is better',[['Learned shadow',policy.prequential_brier],['Uniform reference',policy.prequential_samples>0?policy.uniform_baseline_brier:null],['Always flat',policy.prequential_flat_baseline_brier],['Prior frequency',policy.prequential_frequency_baseline_brier]],'Brier');textField(learning,'Score scope','Exploratory prequential replay on the same scored samples. Not profit, not Gemma training and not proof of a forward advantage.');}
}

"""

STYLE = """
#live-mission-board{border:1px solid #355475;background:linear-gradient(130deg,#15283d,#101927)}.live-board-header{display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap}.live-board-metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:10px}.live-board-metric{display:flex;flex-direction:column;gap:7px;padding:16px;background:#101c2d;border:1px solid #37516b;border-radius:10px}.live-board-metric strong{font-size:21px;overflow-wrap:anywhere}.live-board-metric small{color:#b6c8de}.live-board-blocker{padding:12px;border-left:3px solid #f1b778;background:#241f24;margin:14px 0}.live-board-blocker p{margin-bottom:0;overflow-wrap:anywhere}.live-board-events{max-height:300px;overflow:auto}.live-board-event{padding:10px;border-bottom:1px solid #304259;display:flex;gap:12px;flex-wrap:wrap}.live-board-event time{color:#9cb6d4;font-size:12px}
[hidden]{display:none!important}body{background:#0c1422;max-width:1280px}#mission-navigation{display:flex;gap:8px;flex-wrap:wrap;position:sticky;top:0;z-index:10;padding:12px 0;background:#0c1422;border-bottom:1px solid #304259;margin-bottom:20px}#mission-navigation button{background:#17263c;border:1px solid #304259;color:#bac9de}#mission-navigation button[aria-pressed=true]{background:#29578b;color:#fff;border-color:#85baff}button:focus-visible,a:focus-visible,select:focus-visible{outline:3px solid #ffd166;outline-offset:3px}section{box-shadow:none}h1{letter-spacing:-.6px}#cards{grid-template-columns:repeat(auto-fit,minmax(170px,1fr))}#overview-panel{border-left:4px solid #80baff}table{display:block;overflow:auto;white-space:nowrap}th{font-size:12px;color:#b9cbe2}select,input{background:#14243a;color:#eef5ff;border:1px solid #486582;padding:9px;border-radius:7px;max-width:100%}label{display:grid;gap:6px;font-size:13px}.evidence-bar{margin:12px 0;font-size:13px}.bar-track{height:9px;background:#26384e;border-radius:9px;margin-top:6px}.bar-track>div{height:100%;background:#85baff;border-radius:9px}#income-view>.card{margin:8px 0}#backtest-audit{max-height:120px}#fresh,#layout{font-size:12px}#trade-plot{background:#0c1726;border-radius:10px}#trade-summary{font-weight:600}#trade-scope{font-size:12px}#trade-hover{font-size:13px}#timeline-panel{max-height:480px;overflow:auto}#income-view details.card>summary{cursor:pointer;font-weight:600}#plan-grid p{max-height:180px;overflow:auto}@media(max-width:650px){#mission-navigation{gap:6px}#mission-navigation button{padding:8px 10px;font-size:12px}}
"""
