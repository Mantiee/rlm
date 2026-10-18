"""Host-owned navigation and evidence summaries for existing passive layouts."""

SCRIPT = r"""
let dashboardTab='live';
const dashboardTabs=[['live','Na żywo'],['trades','Paper i rynki'],['income','Inne przychody'],['learning','Uczenie'],['activity','Agenty i zasoby'],['diagnostics','Archiwum dowodów']];
const dashboardGroups={diagnostics:['timeline-panel','action-archive'],trades:['trade-explorer','backtests','paper','paper-labs-panel'],income:['income-panel','source-panel'],learning:['goal-learning-panel','parallel-backtests','evaluation'],activity:['resource-panel','agent-panel','plan-panel'],live:['live-mission-board','overview-panel','current-activity-panel','model-deployment-panel','trade-explorer','backtests','paper','paper-labs-panel','income-panel','source-panel','goal-learning-panel','evaluation','resource-panel','agent-panel','plan-panel']};
function dashboardSectionGroup(n){let group=n.tagName==='DETAILS'?'diagnostics':'live';for(const [key,ids] of Object.entries(dashboardGroups))if(key!=='live'&&ids.some(id=>n.id===id||n.querySelector('#'+id)))group=key;return group}
function dashboardNavigation(){
 let nav=document.getElementById('mission-navigation');
 if(!nav){nav=document.createElement('nav');nav.id='mission-navigation';nav.setAttribute('aria-label','Dashboard sections');for(const [key,label] of dashboardTabs){const b=document.createElement('button');b.textContent=label;b.dataset.tab=key;b.onclick=()=>{dashboardTab=key;dashboardNavigation()};nav.append(b)}document.getElementById('cards').before(nav)}
 const focused=['focus-console','paper-capital-preview'];
 for(const n of document.querySelectorAll('body > section, body > details')){const group=dashboardSectionGroup(n);n.hidden=dashboardTab==='live'?!focused.includes(n.id):dashboardTab==='diagnostics'?focused.includes(n.id):group!==dashboardTab;n.dataset.dashboardGroup=group}
 for(const b of nav.children){b.setAttribute('aria-pressed',String(b.dataset.tab===dashboardTab))}
 const cards=document.getElementById('cards');cards.hidden=true;
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
 const learningReceipt=report.learning_activity||learning,learningFreshness=receiptFreshness('learning_activity',learningReceipt),learningStage=report.learning_activity?learningReceipt.stage:learning.stage;
 metric('Learning',learningFreshness==='fresh'?(learningStage||'No stage reported'):learningFreshness==='stale'?`Stale receipt${learningStage?`: ${learningStage}`:''}`:learningFreshness==='unavailable'?'Unavailable receipt':'Unknown freshness',learningFreshness==='fresh'?((report.learning_activity?learningReceipt.cycle_in_progress:learning.cycle_in_progress)===undefined?'Cycle not reported':`Cycle ${report.learning_activity?learningReceipt.cycle_in_progress:learning.cycle_in_progress}`):`Stage is not confirmed active; receipt freshness ${learningFreshness}`);
 const activity=report.current_activity||{}, evaluation=activity.evaluation||{};
 metric('Current task',activity.stage||'Not reported',activity.running===true?'Recorded as running':'No running task confirmed');
 const evaluationAge=typeof evaluation.age_seconds==='number'&&Number.isFinite(evaluation.age_seconds)&&evaluation.age_seconds>=0?`${Math.round(evaluation.age_seconds)}s since last update`:typeof statusFreshness.evaluation?.age_seconds==='number'&&Number.isFinite(statusFreshness.evaluation.age_seconds)&&statusFreshness.evaluation.age_seconds>=0?`${Math.round(statusFreshness.evaluation.age_seconds)}s receipt age`:'update age unknown';
 metric('Candidate evaluation',evaluation.total?`${evaluation.completed??'unknown'}/${evaluation.total}`:'Not reported',evaluation.total?`${evaluation.passed??'unknown'} passed, ${evaluation.failed??(typeof evaluation.completed==='number'&&typeof evaluation.passed==='number'?evaluation.completed-evaluation.passed:'unknown')} failed; ${evaluation.state||'unknown'}; ${evaluationAge}${evaluation.stale?' (STALE)':''}`:'No evaluation receipt');
 metric('Candidate selected for next serving',activity.accepted_updates??'Unknown','A cycle receipt records candidate selection for a future serving phase; it does not prove serving loaded it or revenue. Optimizer steps are separate.');
 const rows=report.live_paper?.branches||{};
 const a=rows.A||{},b=rows.B||{};
 const money=x=>x===undefined||x===null?'Not reported':String(x)+' '+(report.live_paper?.currency||'PLN');
 metric('Paper A P&L',money(a.net_pnl_before_personal_tax),'Simulation, not real income');
 metric('Paper B P&L',money(b.net_pnl_before_personal_tax),'Simulation, not real income');
 metric('Paper fills',report.live_paper?.executed_fills??'Not verified','The live ledger alone does not certify execution count');
 const paper=report.live_paper||{};
 metric('Paper ledger sequence',paper.ledger_sequence??'Unavailable',paper.last_event?`Last verified event: ${paper.last_event.kind} at ${paper.last_event.time}`:'No verified event');
 const quoteCount=Number.isFinite(paper.quotes_available)?paper.quotes_available:null,instrumentCount=Number.isFinite(paper.registered_instruments)?paper.registered_instruments:null;
 const feedStatus=Array.isArray(paper.blockers)&&paper.blockers.length?paper.blockers.slice(0,2).join('; '):quoteCount===null||instrumentCount===null?'Feed blocker status unknown; quote or instrument counts are not reported':'No feed blocker recorded in this snapshot; counts alone do not confirm usable or executable quotes';
 metric('Live market inputs',`${quoteCount??'unknown'} quotes / ${instrumentCount??'unknown'} instruments`,feedStatus);
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
 const receiptStates=Object.entries(statusFreshness).filter(([,row])=>row&&['stale','unknown','unavailable'].includes(row.state)).map(([name,row])=>`${name}: ${row.state}`);
 const learningState=learningFreshness==='fresh'?null:`learning activity: ${learningFreshness}`;
 if(learningState&&!receiptStates.some(item=>item.startsWith('learning activity:')))receiptStates.push(learningState);
 const allReceiptsFresh=Object.keys(statusFreshness).length>0&&Object.values(statusFreshness).every(row=>row?.state==='fresh')&&learningFreshness==='fresh';
 const bd=document.createElement('p');bd.textContent=currentIssue?`${currentIssue.name}: ${currentIssue.value}`:lastIssue?`No fresh blocker receipt. Last recorded issue (${lastIssue.freshness}): ${lastIssue.name}: ${lastIssue.value}${receiptStates.length?`; required receipts not current: ${receiptStates.slice(0,5).join(', ')}`:''}`:receiptStates.length?`Collection status unverified; required receipts not current: ${receiptStates.slice(0,5).join(', ')}.`:allReceiptsFresh?'No current collection errors recorded in fresh status receipts; this does not confirm all feeds are usable or executable.':'Collection status unverified: freshness receipts are missing or incomplete.';
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
function focusConsole(data){
 const r=data.report||{},m=data.mission||{},paper=r.live_paper||{},evidence=r.mission_evidence||{},goal=evidence.goal_learning||{},policy=goal.cpu_shadow_policy||{},activity=r.current_activity||{},evaluation=activity.evaluation||m.evaluation||{};
 const box=panel('focus-console','Synta — na żywo',$('cards'));box.replaceChildren();
 const head=document.createElement('h2');head.textContent='Synta — co robi i z jakim wynikiem';box.append(head);
 const age=Number(data.collected_at)>0?Math.max(0,Math.round(Date.now()/1000-Number(data.collected_at))):null;
 textField(box,'Stan danych',age===null?'Brak czasu migawki':`${age} s temu${age>300?' · dane opóźnione':''}`);
 const grid=document.createElement('div');grid.className='live-board-metrics';box.append(grid);
 const card=(label,value,detail)=>{const item=document.createElement('article'),name=document.createElement('small'),main=document.createElement('strong'),note=document.createElement('small');item.className='live-board-metric';name.textContent=label;main.textContent=value;note.textContent=detail;item.append(name,main,note);grid.append(item)};
 card('Teraz',activity.stage||m.state?.phase||'Brak zapisanego zadania',m.running===true?'Kontroler aktywny':'Brak potwierdzenia aktywnego kontrolera');
 card('Potwierdzony przychód Synty','Brak potwierdzenia','Paper nie jest wpływem pieniędzy. Nie podłączono księgi przychodów, więc dashboard nie może potwierdzić wpływów z innych kanałów.');
 card('Test w toku',evaluation.total?`${evaluation.completed||0} / ${evaluation.total}`:'Brak bieżącego wyniku',evaluation.total?`${evaluation.passed??'—'} zaliczonych · ${evaluation.state||'stan niepodany'}`:'Nowy wynik pojawi się po zapisaniu testu');
 const training=evidence.active_training||{},modelExport=evidence.active_model_export||{},modelServer=evidence.active_model_server||{},optimizer=r.learning_activity_status?.optimizer_evidence||{},optimizerProgress=training.state==='running'&&training.progress_phase==='optimizer_updates'&&Number.isInteger(training.completed_steps)&&Number.isInteger(training.total_steps),exporting=['exporting','quantizing'].includes(modelExport.state),serverWaiting=modelServer.state==='waiting_host_io',serverPresent=modelServer.state==='process_present';
 const trainingValue=exporting?(modelExport.state==='quantizing'?'Kwantyzacja kandydata · poza treningiem':'Eksport kandydata · poza treningiem'):serverWaiting?'Serwer modelu czeka na I/O · poza treningiem':serverPresent?'Serwer modelu obecny · uczenie osobno':optimizerProgress?`${training.completed_steps}/${training.total_steps} skutecznych aktualizacji optymalizatora`:training.state==='running_without_fresh_progress'?'Proces treningu · postęp nieświeży':training.state==='running'?'Proces treningu aktywny':'Brak aktywnego treningu';
 const exportDetail=exporting?`${modelExport.state==='quantizing'?'Kandydat jest kwantyzowany':'Kandydat jest eksportowany'} przez PID ${modelExport.pid}${modelExport.output?`; plik ${modelExport.output}`:''}${Number.isInteger(modelExport.output_bytes)?`, ${modelExport.output_bytes} bajtów`:''}${Number.isInteger(modelExport.output_age_seconds)?`, zapis ${modelExport.output_age_seconds} s temu`:''}${modelExport.process_state==='D'?'; proces czeka w jądrze na I/O':''}. To przygotowanie artefaktu, nie bieżące uczenie ani wynik jakości.`:'';
 const serverDetail=serverWaiting||serverPresent?`Serwer modelu PID ${modelServer.pid}${modelServer.model?`; model ${modelServer.model}`:''}${Number.isInteger(modelServer.context_size)?`; kontekst ${modelServer.context_size}`:''}${Number.isInteger(modelServer.slots)?`; sloty ${modelServer.slots}`:''}${Number.isInteger(modelServer.gpu_layers)?`; warstwy GPU ${modelServer.gpu_layers}`:''}${serverWaiting?`; ${modelServer.wait_reason==='kernel_page_wait'?'jądro czeka na stronę pliku':'jądro czeka na I/O'}`:''}. To stan procesu modelu, nie potwierdzenie aktywnego treningu ani gotowości odpowiedzi.`:'';
 const trainingLog=typeof training.log==='string'&&training.log.length<=256?training.log.replace(/[\u0000-\u001f\u007f]/g,' '):null;
 const trainingDetail=(optimizerProgress?`Trening potwierdzony PID ${training.pid}; ${trainingLog?`receipt ${trainingLog} (${training.log_age_seconds??'—'} s temu)`:`receipt ${training.log_age_seconds??'—'} s temu`}. Próby: ${training.attempted_steps??'—'}, pominięte: ${training.skipped_steps??'—'}. To nie jest wynik jakości ani dowód zysku.`:training.state==='running_without_fresh_progress'?`Proces treningu PID ${training.pid} działa, ale receipt postępu jest nieświeży; nie pokazuję historycznego licznika jako bieżącego.`:training.state==='running'?`Proces PID ${training.pid} działa, ale faza nie potwierdza świeżych aktualizacji optymalizatora.`:`Brak potwierdzonego procesu treningu. Pętla: ${r.learning_activity_status?.state||'unknown'} — heartbeat opisuje liveness, nie trening. Ostatni optimizer receipt: ${optimizer.activity_state||'unknown'}, ${typeof optimizer.receipt_age_seconds==='number'?`${optimizer.receipt_age_seconds} s temu`:'czas nieznany'}. Historycznie: ${evidence.optimizer_updates_observed??'—'} potwierdzonych aktualizacji; ${goal.training_admission?.admitted??'—'} przykładów przyjętych do treningu, nie bieżące kroki. Liczby nie mierzą jakości ani zysku.`)+(exportDetail?` ${exportDetail}`:'')+(serverDetail?` ${serverDetail}`:'');
 card('Uczenie',trainingValue,trainingDetail);
 const reasons=document.createElement('div');box.append(reasons);const title=document.createElement('h3');title.textContent='Dlaczego tak działa — ostatnie decyzje';reasons.append(title);
 const entries=(paper.activity_stream?.entries||[]).filter(row=>row.proposal?.source==='model-proposal').slice(-6).reverse();
 if(!entries.length)textField(reasons,'Decyzje','Brak zapisanej decyzji modelu w aktualnym oknie.');
 for(const row of entries){const proposal=row.proposal,detail=document.createElement('details'),summary=document.createElement('summary');detail.dataset.key='focus-decision-'+String(row.sequence??row.time)+'-'+row.branch;summary.textContent=`${row.branch} · ${String(proposal.action||'decyzja niepodana').toUpperCase()} ${proposal.symbol||''} · ${row.time||'czas niepodany'}`;detail.append(summary);textField(detail,'Uzasadnienie modelu',proposal.rationale||'Model nie zapisał uzasadnienia.');textField(detail,'Wynik wykonania',proposal.model_action_blocked?`Zablokowano ${proposal.model_action_blocked}; pozostaje HOLD`:proposal.status||'Zapisana propozycja; wykonanie niepotwierdzone');reasons.append(detail)}
 const conclusions=document.createElement('div');box.append(conclusions);const ct=document.createElement('h3');ct.textContent='Wnioski z zapisanych eksperymentów';conclusions.append(ct);
 const tests=(r.recent_exploratory_backtests||[]).slice(0,4);
 if(!tests.length)textField(conclusions,'Wyniki','Brak zakończonego eksperymentu z wynikiem w tej migawce.');
 for(const test of tests){const item=document.createElement('details'),summary=document.createElement('summary'),result=test.development_test||{};item.dataset.key='focus-test-'+String(test.id||test.report||test.parameters?.rule);summary.textContent=`${test.parameters?.rule||'Strategia'} · ${typeof result.net_return==='number'?(100*result.net_return).toFixed(2)+'%':'wynik niepodany'} · test historyczny`;item.append(summary);textField(item,'Co wynika z testu',test.diagnostics?.state||test.triage?.state||'Brak zapisanego wniosku.');textField(item,'Weryfikacja',test.scope||'Wynik historyczny; potrzebny niezależny test na późniejszych danych.');conclusions.append(item)}
 const quality=document.createElement('h3');quality.textContent='Czy uczenie poprawia skuteczność?';box.append(quality);
 const scores=[policy.uniform_baseline_brier,policy.prequential_flat_baseline_brier,policy.prequential_frequency_baseline_brier].filter(value=>typeof value==='number'&&Number.isFinite(value));
 if(policy.prequential_samples>0&&typeof policy.prequential_brier==='number'&&scores.length){const best=Math.min(...scores);textField(box,'Porównanie na tych samych próbkach',`Model: ${policy.prequential_brier.toFixed(4)} · najlepszy punkt odniesienia: ${best.toFixed(4)} · ${policy.prequential_samples} ocenionych predykcji. Niższy Brier jest lepszy.`);textField(box,'Wniosek',policy.prequential_brier<best?'Lepszy wynik w tym replay; przewaga na nowych danych niepotwierdzona.':'Brak poprawy względem najlepszego punktu odniesienia w tym replay.');}
 else textField(box,'Skuteczność','Brak porównywalnego wyniku przed/po uczeniu. Sam trening nie potwierdza poprawy ani zysku.');
 const blockers=paper.blockers||r.paper_blockers||[];if(blockers.length){const issue=document.createElement('details'),summary=document.createElement('summary');issue.dataset.key='focus-blockers';summary.textContent=`Co blokuje transakcje (${blockers.length})`;issue.append(summary);for(const value of blockers)textField(issue,'Powód',value);box.append(issue)}
}
function paperEquityTimeline(parent,paper,starting,capital){
 const title=document.createElement('h3');title.textContent='Jak zmienia się wartość obu portfeli paper';parent.append(title);
 const series=paper.equity_series||{},rows=['A','B'].flatMap(branch=>(series[branch]||[]).map(point=>({branch,...point}))).filter(point=>Number.isFinite(Date.parse(point.time||''))&&Number.isFinite(Number(point.equity))&&Number.isFinite(starting)&&starting>0);
 const note=document.createElement('p');note.className='muted';note.textContent=rows.length?`To saldo mark-to-market z próbek wyceny zapisanych w checksum-sprawdzonym ogonie księgi. Każdy punkt przeliczono proporcjonalnie do 100 PLN; nie są to niezależne rachunki. Widok obejmuje ${rows.length} próbek, a nowa próbka pojawi się po kolejnym zdarzeniu wyceny. Nie pokazuje prognozy.`:'Brak checksum-sprawdzonych próbek wyceny w bieżącym oknie księgi. Pojawią się po zapisaniu obserwacji z wyceną.';parent.append(note);
 if(!rows.length)return;
 const ns='http://www.w3.org/2000/svg',chart=document.createElementNS(ns,'svg');chart.setAttribute('viewBox','0 0 800 300');chart.setAttribute('role','img');chart.setAttribute('aria-label','Historyczna wycena mark-to-market paper A i B przeskalowana do kapitału 100 PLN');
 const add=(tag,attrs,label)=>{const node=document.createElementNS(ns,tag);for(const [key,value] of Object.entries(attrs))node.setAttribute(key,String(value));if(label!==undefined)node.textContent=label;chart.append(node);return node};
 const vals=rows.map(point=>Number(point.equity)*capital/starting),low=Math.min(capital,...vals),high=Math.max(capital,...vals),pad=Math.max(0.5,(high-low)*.12),min=low-pad,max=high+pad,left=78,right=780,top=32,bottom=232,times=rows.map(point=>Date.parse(point.time)),minTime=Math.min(...times),maxTime=Math.max(...times);
 const y=value=>bottom-(value-min)/(max-min)*(bottom-top),x=time=>maxTime===minTime?(left+right)/2:left+(time-minTime)/(maxTime-minTime)*(right-left);
 for(let i=0;i<=4;i++){const value=max-(max-min)*i/4,yy=y(value);add('line',{x1:left,x2:right,y1:yy,y2:yy,stroke:'#35445a','stroke-dasharray':'4 4'});add('text',{x:left-9,y:yy+4,fill:'#bdcde1','font-size':12,'text-anchor':'end'},value.toFixed(2)+' zł')}
 add('line',{x1:left,x2:right,y1:y(capital),y2:y(capital),stroke:'#718096','stroke-dasharray':'5 4'});
 for(const [branch,color,name,legendX] of [['A','#57b7ff','A · saldo paper',left],['B','#f9c36b','B · saldo paper',left+190]]){const branchRows=rows.filter(point=>point.branch===branch),points=branchRows.map(point=>`${x(Date.parse(point.time))},${y(Number(point.equity)*capital/starting)}`).join(' ');if(points)add('polyline',{points,fill:'none',stroke:color,'stroke-width':3});for(const point of branchRows){const px=x(Date.parse(point.time)),value=Number(point.equity)*capital/starting,circle=add('circle',{cx:px,cy:y(value),r:4,fill:color});const tip=document.createElementNS(ns,'title');tip.textContent=`${branch} · ${point.time} · ${value.toFixed(2)} PLN · P&L ${((value-capital)).toFixed(2)} PLN`;circle.append(tip)}add('line',{x1:legendX,x2:legendX+24,y1:269,y2:269,stroke:color,'stroke-width':3});add('text',{x:legendX+31,y:273,fill:'#bdcde1','font-size':12},name)}
 const fmt=t=>new Date(t).toLocaleString('pl-PL',{timeZone:'UTC',hour:'2-digit',minute:'2-digit',day:'2-digit',month:'2-digit'})+' UTC';add('text',{x:left,y:252,fill:'#bdcde1','font-size':11},fmt(minTime));add('text',{x:right,y:252,fill:'#bdcde1','font-size':11,'text-anchor':'end'},fmt(maxTime));parent.append(chart);
}
function paperScenarioPreview(parent,paper,starting,capital){
 const title=document.createElement('h3');title.textContent='Co jeśli cena ruszy się od teraz? — scenariusze, nie prognoza';parent.append(title);
 const cases=[-.1,-.05,0,.05,.1],cards=document.createElement('div');cards.className='live-board-metrics';parent.append(cards);const usable=[];
 for(const branch of ['A','B']){const data=paper.scenario_inputs?.[branch],card=document.createElement('article');card.className='live-board-metric';cards.append(card);const head=document.createElement('strong');head.textContent=`Portfel ${branch}`;card.append(head);const body=document.createElement('small');body.style.whiteSpace='pre-wrap';card.append(body);const equity=Number(data?.equity),exposure=Number(data?.net_exposure);if(data?.state!=='recorded-current-positions'||!Number.isFinite(equity)||!Number.isFinite(exposure)||!Number.isFinite(starting)||starting<=0){body.textContent=`Scenariusz niedostępny: ${data?.reason||'brak świeżej wyceny pozycji'}`;continue}usable.push({branch,equity:equity*capital/starting,exposure:exposure*capital/starting});body.textContent=cases.map(change=>{const projected=(equity+exposure*change)*capital/starting;const delta=exposure*change*capital/starting;return `${change>0?'+':''}${(change*100).toFixed(0)}% ceny → ${projected.toFixed(2)} PLN (${delta>=0?'+':''}${delta.toFixed(2)} PLN)`}).join('\n')+`\nStan danych: ${data.as_of||'czas UNKNOWN'}`}
 if(usable.length){const ns='http://www.w3.org/2000/svg',chart=document.createElementNS(ns,'svg');chart.setAttribute('viewBox','0 0 800 280');chart.setAttribute('role','img');chart.setAttribute('aria-label','Hipotetyczna wycena obecnych pozycji przy szoku ceny od minus 10 do plus 10 procent');const add=(tag,attrs,label)=>{const node=document.createElementNS(ns,tag);for(const [key,value] of Object.entries(attrs))node.setAttribute(key,String(value));if(label!==undefined)node.textContent=label;chart.append(node);return node};const vals=usable.flatMap(row=>cases.map(change=>row.equity+row.exposure*change)),low=Math.min(...vals,capital),high=Math.max(...vals,capital),pad=Math.max(.5,(high-low)*.12),min=low-pad,max=high+pad,left=78,right=780,top=30,bottom=218,x=change=>left+(change+.1)/.2*(right-left),y=value=>bottom-(value-min)/(max-min)*(bottom-top);for(let i=0;i<=4;i++){const value=max-(max-min)*i/4,yy=y(value);add('line',{x1:left,x2:right,y1:yy,y2:yy,stroke:'#35445a','stroke-dasharray':'4 4'});add('text',{x:left-9,y:yy+4,fill:'#bdcde1','font-size':12,'text-anchor':'end'},value.toFixed(2)+' zł')}for(const [branch,color,legendX] of [['A','#57b7ff',left],['B','#f9c36b',left+150]]){const row=usable.find(item=>item.branch===branch);if(!row)continue;const points=cases.map(change=>`${x(change)},${y(row.equity+row.exposure*change)}`).join(' ');add('polyline',{points,fill:'none',stroke:color,'stroke-width':3});for(const change of cases)add('circle',{cx:x(change),cy:y(row.equity+row.exposure*change),r:4,fill:color});add('line',{x1:legendX,x2:legendX+24,y1:257,y2:257,stroke:color,'stroke-width':3});add('text',{x:legendX+31,y:261,fill:'#bdcde1','font-size':12},`Portfel ${branch}`)}for(let i=0;i<5;i++){const change=cases[i];add('text',{x:x(change),y:239,fill:'#bdcde1','font-size':11,'text-anchor':'middle'},`${change>0?'+':''}${(change*100).toFixed(0)}%`)}parent.append(chart)}
 const note=document.createElement('p');note.className='muted';note.textContent='Prosta arytmetyka dla już otwartych pozycji i aktualnej wyceny; nie przewiduje ceny ani przyszłych decyzji. Pomija opłaty, FX, poślizg, wpływ płynności i możliwość wykonania. Linie przyszłych wyników nie są obietnicą ani testem strategii.';parent.append(note);
}
function paperCapitalPreview(data){
 const anchor=$('live-mission-board');if(!anchor)return;
 const p=panel('paper-capital-preview','Co model proponuje przy kapitale 100 PLN? — symulacja',$('live-mission-board'));p.replaceChildren();
 const h=document.createElement('h2');h.textContent='Co model proponuje przy kapitale 100 PLN?';p.append(h);
 const report=data.report||{},paper=report.live_paper||{},policy=report.live_execution_policy||{},stream=paper.activity_stream||{};
 const capital=Number(policy.capital_limit_pln??100),starting=Number(paper.initial_capital),now=Date.now()/1000;
 textField(p,'Kapitał paper',`${Number.isFinite(capital)?capital.toFixed(2):'UNKNOWN'} PLN · brak stałego stopu straty; fikcyjne saldo może spaść do zera · bez pożyczek${policy.leverage_enabled===true?' · dźwignia włączona':' · dźwignia wyłączona'}`);
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
 paperEquityTimeline(p,paper,starting,capital);
 const expanded=document.createElement('details'),expandedTitle=document.createElement('summary');expanded.dataset.key='paper-strategy-detail';expandedTitle.textContent='Otwórz strategie, decyzje i scenariusze';expanded.append(expandedTitle);p.append(expanded);const compactEnd=p.children.length;
 paperScenarioPreview(p,paper,starting,capital);
 textField(p,'Zakres sald','Księga nie prowadzi osobnego rachunku 100 PLN. Aktualne saldo i historyczne próbki wyceny większej księgi są przeskalowane proporcjonalnie do 100 PLN. To mark-to-market; koszty wykonania i FX pozostają niezweryfikowane.');
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
 for(const child of Array.from(p.children).slice(compactEnd))expanded.append(child);
}
/* Synta overview: compact live summary + interactive PAPER equity chart.
   Reads only the existing /api/status snapshot passed to insightsView(data).
   Updates in place (no full rebuild), keeps the chosen range and hover. */
const syntaOverview = (() => {
  const state = {data: null, range: '6h', hoverT: null, built: false};
  const RANGES = {'1h': 3600, '6h': 21600, '24h': 86400, all: Infinity};
  const COLORS = ['#57b7ff', '#f9c36b', '#7fd6a4', '#e58fd6', '#ff8a7a', '#b9a4ff'];
  const num = v => (typeof v === 'number' && Number.isFinite(v) ? v : null);
  const pln = v => (num(v) === null ? 'brak danych' : num(v).toLocaleString('pl-PL', {minimumFractionDigits: 2, maximumFractionDigits: 2}) + ' PLN');
  const signed = v => (num(v) === null ? 'brak danych' : (v > 0 ? '+' : '') + pln(v));
  const pct = v => (num(v) === null ? 'brak danych' : Math.round(v * 100) + '%');
  const age = s => (num(s) === null ? 'nieznany' : s < 90 ? Math.round(s) + ' s' : s < 5400 ? Math.round(s / 60) + ' min' : Math.round(s / 3600) + ' h');
  const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text !== undefined) n.textContent = String(text); return n; };
  const ts = v => { const t = typeof v === 'number' ? v * (v < 1e12 ? 1000 : 1) : Date.parse(v || ''); return Number.isFinite(t) ? t : null; };
  const store = (key, value) => { try { if (value === undefined) return localStorage.getItem(key); localStorage.setItem(key, value); } catch (e) { return null; } return null; };

  function build() {
    // A div, not a section: dashboardNavigation() hides every `body > section` outside its tab groups.
    const root = el('div'); root.id = 'synta-overview';
    root.innerHTML = '<div class="ov-head"><h2>Podsumowanie</h2><span class="ov-age" data-k="age"></span><button type="button" class="ov-toggle"></button></div>'
      + '<div class="ov-grid" data-k="grid"></div>'
      + '<div class="ov-chart"><div class="ov-chart-head"><h3>Equity portfeli PAPER (100 PLN każdy)</h3><div class="ov-ranges"></div></div>'
      + '<div class="ov-plot" data-k="plot"></div><div class="ov-legend" data-k="legend"></div>'
      + '<p class="ov-note">▲/▼ = symulowany fill BUY/SELL. Symulacja PAPER z kosztami ZAŁOŻONYMI, bez brokera i realnych zleceń. Płaska linia = brak zmian wyceny, nie zawieszenie: patrz wiek ostatniego punktu.</p></div>'
      + '<div class="ov-more" data-k="more"></div>';
    const ranges = root.querySelector('.ov-ranges');
    for (const key of Object.keys(RANGES)) {
      const b = el('button', 'ov-range', key === 'all' ? 'całość' : key); b.type = 'button'; b.dataset.range = key;
      b.onclick = () => { state.range = key; store('synta-ov-range', key); draw(); };
      ranges.append(b);
    }
    const toggle = root.querySelector('.ov-toggle');
    toggle.onclick = () => { const compact = !document.body.classList.contains('synta-compact'); setCompact(compact); };
    const anchor = document.querySelector('body > h1');
    if (anchor) anchor.after(root); else document.body.prepend(root);
    const saved = store('synta-ov-range'); if (saved && RANGES[saved]) state.range = saved;
    setCompact(store('synta-ov-compact') !== '0');
    if (!state.built) setInterval(tickAge, 1000);  // one timer even if the root is rebuilt
    state.built = true;
  }

  function setCompact(compact) {
    document.body.classList.toggle('synta-compact', compact);
    store('synta-ov-compact', compact ? '1' : '0');
    const t = document.querySelector('#synta-overview .ov-toggle');
    if (t) t.textContent = compact ? 'Pokaż szczegóły ▾' : 'Ukryj szczegóły ▴';
  }

  function tickAge() {
    const node = document.querySelector('#synta-overview [data-k="age"]');
    if (!node || !state.data) return;
    const t = ts(state.data.collected_at), s = t === null ? null : (Date.now() - t) / 1000;
    node.textContent = 'Dane sprzed ' + age(s);
    node.className = 'ov-age ' + (s === null ? 'bad' : s <= 20 ? 'ok' : s <= 120 ? 'warn' : 'bad');
  }

  function card(grid, title, value, detail, tone) {
    const c = el('div', 'ov-card ' + (tone || '')); c.append(el('small', '', title), el('strong', '', value));
    if (detail) c.append(el('span', '', detail)); grid.append(c);
  }

  function cards() {
    const d = state.data || {}, r = d.report || {}, grid = document.querySelector('#synta-overview [data-k="grid"]');
    if (!grid) return; grid.replaceChildren();
    const arena = r.paper_arena || {}, forward = r.paper_forward || {}, books = arena.active_portfolios || [];
    const fresh = Object.values(r.status_freshness || {}), stale = fresh.filter(x => x && x.state !== 'fresh').length;
    card(grid, 'Misja', d.mission?.running ? 'działa' : 'zatrzymana', 'PAPER: ' + (arena.state || forward.state || 'stan nieznany'), d.mission?.running ? 'ok' : 'bad');
    const model = books.find(b => String(b.provenance?.kind || '').startsWith('actual-model')) || null;
    if (model) card(grid, 'Portfel Synty (model)', pln(model.equity), 'P&L ' + signed(model.pnl) + ' · ' + (Array.isArray(model.fills) ? model.fills.length : model.fills ?? 0) + ' filli · ' + (model.decision?.action || model.decision || 'brak decyzji'), num(model.pnl) < 0 ? 'warn' : '');
    else card(grid, 'Portfel Synty (model)', 'brak aktywnej księgi', 'nie ma portfela z inferencją modelu', 'warn');
    const controls = books.filter(b => b !== model);
    if (controls.length) card(grid, 'Kontrole (benchmark)', controls.map(b => String(b.name || b.id).replace(/^Control /, '') + ' ' + signed(b.pnl)).join(' · '), 'ten sam kapitał i koszty; punkt odniesienia', '');
    const c = forward.assumed_cost_bps_per_side || {};
    card(grid, 'Koszty', (forward.costs_status || 'ASSUMED').toUpperCase(), ['fee', 'fx', 'slippage'].filter(k => num(c[k]) !== null).map(k => k + ' ' + c[k] + ' bps').join(' · ') + ' na stronę', 'warn');
    const g = d.gpu || {};
    card(grid, 'V100', num(g.utilization) === null ? 'brak odczytu' : g.utilization + '% GPU', [num(g.temperature) !== null ? g.temperature + ' °C' : '', num(g.memory_used) !== null && num(g.memory_total) ? Math.round(g.memory_used / g.memory_total * 100) + '% VRAM' : ''].filter(Boolean).join(' · '), num(g.temperature) > 80 ? 'warn' : '');
    const u = r.node_utilization || {}, w1 = u.window_1h || {}, cur = u.current || {};
    card(grid, 'RTX 3060 Ti', u.schema ? pct(w1.busy_share) + ' zajęta (1 h)' : 'brak pomiaru', u.schema ? 'teraz: ' + (cur.state || 'nieznany') + (cur.idle_reason ? ' — ' + String(cur.idle_reason).split(';')[0].replace(/: .*/, '').slice(0, 80) : '') : '', u.schema && num(w1.busy_share) !== null && w1.busy_share < 0.3 ? 'warn' : '');
    const work = r.income_work || {}, income = num(work.actual_income_pln);
    card(grid, 'Przychód (realny)', income === null ? 'niezweryfikowany' : pln(income), 'income: ' + (work.state || r.income_dispatch?.state || 'stan nieznany') + '; brak podłączonej księgi płatności', income === null ? 'warn' : 'ok');
    const errs = (d.errors || []).length;
    card(grid, 'Problemy danych', String(stale + errs), stale + ' źródeł nieświeżych/nieznanych · ' + errs + ' błędów snapshotu', stale + errs ? 'warn' : 'ok');
  }

  // Expandable sections keep their open/closed state across refreshes and reloads.
  function section(key, title) {
    const host = document.querySelector('#synta-overview [data-k="more"]');
    if (!host) return document.createElement('div');  // root missing: render into a detached node
    let box = host.querySelector(`details[data-sec="${key}"]`);
    if (!box) {
      box = el('details', 'ov-sec'); box.dataset.sec = key;
      box.append(el('summary'), el('div', 'ov-sec-body'));
      if (store('synta-ov-open-' + key) === '1') box.open = true;
      box.addEventListener('toggle', () => store('synta-ov-open-' + key, box.open ? '1' : '0'));
      host.append(box);
    }
    box.querySelector('summary').textContent = title;
    const body = box.querySelector('.ov-sec-body'); body.replaceChildren(); return body;
  }
  function row(body, label, value, tone) { const r = el('div', 'ov-row ' + (tone || '')); r.append(el('span', 'ov-row-k', label), el('span', 'ov-row-v', value)); body.append(r); }
  const when = v => { const t = ts(v); return t === null ? 'czas nieznany' : new Date(t).toLocaleString('pl-PL') + ' (' + age((Date.now() - t) / 1000) + ' temu)'; };

  function sections() {
    const d = state.data || {}, r = d.report || {}, arena = r.paper_arena || {}, lab = arena.candidate_lab || {};
    const results = Array.isArray(lab.candidate_results) ? lab.candidate_results : [];
    const failed = results.filter(c => /fail/i.test(String(c.reason || c.status || '')));
    let body = section('research', `Badania modelu — ${lab.stage || 'stan nieznany'} · ${results.length} kandydatów, ${failed.length} z błędem`);
    row(body, 'Etap', String(lab.stage || 'nieznany'), /reject|fail/.test(String(lab.stage)) ? 'warn' : '');
    if (lab.reason) row(body, 'Powód stanu', String(lab.reason).slice(0, 220));
    row(body, 'Następny przegląd', lab.next_review_at ? when(lab.next_review_at) : 'nie zaplanowano');
    for (const c of results.slice(0, 6)) row(body, String(c.name || 'kandydat bez nazwy'), String(c.reason || c.status || 'brak opisu').slice(0, 220), /fail/i.test(String(c.reason)) ? 'warn' : '');
    const outcomes = arena.research_outcomes || {}, resolved = Array.isArray(outcomes.resolved) ? outcomes.resolved : [];
    row(body, 'Hipotezy oczekujące / rozliczone', `${outcomes.pending_count ?? 0} / ${resolved.length}`);
    for (const o of resolved.slice(-3).reverse()) row(body, 'Rozliczona: ' + String(o.action || '?').toUpperCase(), `${String(o.hypothesis || '').slice(0, 160)} → zmiana equity ${signed(num(o.actual_paper_equity_change_pln))}, ruch ceny ${num(o.price_change_pct) === null ? 'brak danych' : o.price_change_pct.toFixed(3) + '%'}`);
    row(body, 'Zakres', 'Wyniki rozwojowe PAPER; nie dowód przewagi ani zysku.');

    const fresh = Object.entries(r.status_freshness || {}), stale = fresh.filter(([, v]) => v && v.state !== 'fresh');
    const blockers = Array.isArray(r.paper_blockers) ? r.paper_blockers : [], errors = Array.isArray(d.errors) ? d.errors : [];
    body = section('issues', `Problemy i nieaktualne źródła — ${stale.length + blockers.length + errors.length}`);
    if (!stale.length && !blockers.length && !errors.length) row(body, 'Stan', 'Brak zgłoszonych problemów w tej migawce.', 'ok');
    for (const [name, v] of stale.sort((a, z) => (z[1].age_seconds || 0) - (a[1].age_seconds || 0))) row(body, name, (v.state === 'stale' ? 'NIEAKTUALNE' : String(v.state).toUpperCase()) + (num(v.age_seconds) !== null ? ' · ostatni zapis ' + age(v.age_seconds) + ' temu' : '') + (v.detail ? ' · ' + String(v.detail).slice(0, 120) : ''), 'warn');
    for (const b of blockers) row(body, 'Blokada PAPER', String(b).slice(0, 220), 'warn');
    for (const e of errors.slice(0, 10)) row(body, 'Błąd snapshotu', String(e).slice(0, 220), 'bad');

    const books = arena.active_portfolios || [];
    const fills = books.flatMap(b => (Array.isArray(b.trades) ? b.trades : []).map(f => ({...f, book: b.name || b.id}))).sort((a, z) => (ts(z.time) || 0) - (ts(a.time) || 0));
    body = section('activity', `Fille i decyzje — ${fills.length} filli w aktywnych księgach`);
    if (!fills.length) row(body, 'Fille', 'Brak symulowanych filli w aktywnych księgach. HOLD nie jest transakcją.');
    for (const f of fills.slice(0, 12)) row(body, when(f.time), `${f.book}: ${String(f.action || '?').toUpperCase()} ${f.symbol || ''} ${num(f.quantity) ?? '?'} @ ${num(f.price) ?? '?'} · koszt ${pln(num(f.cost))} (${f.execution || 'symulacja'})`);
    for (const b of books) { const dec = b.decision; if (dec) row(body, 'Ostatnia decyzja · ' + (b.name || b.id), typeof dec === 'string' ? dec : `${String(dec.action || '?').toUpperCase()} ${dec.symbol || ''} ${dec.time ? '· ' + when(dec.time) : ''}`); }
  }

  function series() {
    const books = (state.data?.report?.paper_arena?.active_portfolios) || [];
    return books.map((b, i) => ({
      name: String(b.name || b.id || 'portfel ' + (i + 1)), color: COLORS[i % COLORS.length], control: String(b.provenance?.kind || '') === 'explicit-control', equity: num(b.equity), pnl: num(b.pnl),
      points: (Array.isArray(b.history) ? b.history : []).map(p => ({t: ts(p.time), v: num(p.equity)})).filter(p => p.t !== null && p.v !== null).sort((a, z) => a.t - z.t),
      fills: (Array.isArray(b.trades) ? b.trades : []).map(f => ({t: ts(f.time), action: String(f.action || '?').toUpperCase(), price: num(f.price), qty: num(f.quantity), cost: num(f.cost), symbol: f.symbol})).filter(f => f.t !== null),
      cash: num(b.cash), position: b.position_symbol ? String(b.position_symbol) + ' ' + (num(b.units) ?? '?') + ' j. = ' + pln(b.position_value_pln) : 'brak pozycji',
    }));
  }

  function draw() {
    const plot = document.querySelector('#synta-overview [data-k="plot"]'), legend = document.querySelector('#synta-overview [data-k="legend"]');
    if (!plot || !legend) return;
    for (const b of document.querySelectorAll('#synta-overview .ov-range')) b.classList.toggle('on', b.dataset.range === state.range);
    const all = series(), now = Date.now(), span = RANGES[state.range];
    const lines = all.map(s => ({...s, shown: s.points.filter(p => span === Infinity || now - p.t <= span * 1000)}));
    const pts = lines.flatMap(s => s.shown);
    legend.replaceChildren();
    for (const s of lines) {
      const last = s.points[s.points.length - 1], lag = last ? (now - last.t) / 1000 : null;
      const item = el('span', 'ov-key' + (lag === null || lag > 300 ? ' stale' : ''));
      const dot = el('i'); dot.style.background = s.color;
      item.append(dot, el('b', '', s.name), el('span', '', ' ' + pln(s.equity) + ' · P&L ' + signed(s.pnl) + ' · gotówka ' + pln(s.cash) + ' · ' + s.position + ' · ' + s.fills.length + ' filli · ostatni punkt ' + (lag === null ? 'brak' : age(lag) + ' temu') + (lag !== null && lag > 300 ? ' (NIEAKTUALNE)' : '')));
      legend.append(item);
    }
    if (pts.length < 2) { plot.replaceChildren(el('p', 'ov-empty', 'Brak zapisanych wycen w wybranym zakresie. Nie rysuję linii bez danych.')); return; }
    const W = 900, H = 260, L = 64, R = 12, T = 14, B = 28;
    const t0 = Math.min(...pts.map(p => p.t)), t1 = Math.max(...pts.map(p => p.t));
    let v0 = Math.min(...pts.map(p => p.v)), v1 = Math.max(...pts.map(p => p.v));
    const pad = Math.max((v1 - v0) * 0.15, 0.05); v0 -= pad; v1 += pad;
    const x = t => L + (t1 === t0 ? 0.5 : (t - t0) / (t1 - t0)) * (W - L - R), y = v => T + (1 - (v - v0) / (v1 - v0)) * (H - T - B);
    const ns = 'http://www.w3.org/2000/svg', svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', 'Equity portfeli PAPER w czasie');
    const add = (tag, attrs, text) => { const n = document.createElementNS(ns, tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v)); if (text !== undefined) n.textContent = text; svg.append(n); return n; };
    for (let i = 0; i <= 4; i++) { const v = v0 + (v1 - v0) * i / 4; add('line', {x1: L, x2: W - R, y1: y(v), y2: y(v), class: 'ov-grid-line'}); add('text', {x: L - 6, y: y(v) + 4, 'text-anchor': 'end', class: 'ov-axis'}, v.toFixed(2)); }
    const clock = t => new Date(t).toLocaleTimeString('pl-PL', {hour: '2-digit', minute: '2-digit'});
    for (let i = 0; i <= 4; i++) { const t = t0 + (t1 - t0) * i / 4; add('text', {x: x(t), y: H - 8, 'text-anchor': i === 0 ? 'start' : i === 4 ? 'end' : 'middle', class: 'ov-axis'}, clock(t)); }
    // Controls dashed and thinner, drawn first: equal equity (e.g. cash vs a HOLD model) stays visible.
    for (const s of [...lines].sort((a, z) => z.control - a.control)) if (s.shown.length > 1) add('polyline', {points: s.shown.map(p => x(p.t).toFixed(1) + ',' + y(p.v).toFixed(1)).join(' '), fill: 'none', stroke: s.color, 'stroke-width': s.control ? 1.6 : 2.8, 'stroke-dasharray': s.control ? '7 5' : 'none'});
    // Simulated fills: ▲ BUY / ▼ SELL on the book's own line at the fill time.
    for (const s of lines) for (const f of s.fills) {
      if (f.t < t0 || f.t > t1) continue;
      let near = null; for (const p of s.shown) if (near === null || Math.abs(p.t - f.t) < Math.abs(near.t - f.t)) near = p;
      if (!near) continue;
      const cx = x(f.t), cy = y(near.v), up = f.action !== 'SELL';
      add('path', {d: up ? `M${cx} ${cy - 11} l6 9 h-12 z` : `M${cx} ${cy + 11} l6 -9 h-12 z`, fill: s.color, stroke: '#0c1422', 'stroke-width': 1, class: 'ov-fill'});
    }
    const cursor = add('line', {x1: 0, x2: 0, y1: T, y2: H - B, class: 'ov-cursor', visibility: 'hidden'});
    const tip = el('div', 'ov-tip'); tip.hidden = true;
    const show = t => {
      if (t === null) { cursor.setAttribute('visibility', 'hidden'); tip.hidden = true; return; }
      cursor.setAttribute('x1', x(t)); cursor.setAttribute('x2', x(t)); cursor.setAttribute('visibility', 'visible');
      const rows = lines.map(s => { let best = null; for (const p of s.shown) if (best === null || Math.abs(p.t - t) < Math.abs(best.t - t)) best = p; const fill = s.fills.find(f => Math.abs(f.t - t) <= Math.max(60000, (t1 - t0) / 150)); return best ? s.name + ': ' + best.v.toFixed(2) + ' PLN' + (fill ? ' · FILL ' + fill.action + ' ' + (fill.symbol || '') + ' @ ' + (fill.price ?? '?') + ', koszt ' + pln(fill.cost) : '') : null; }).filter(Boolean);
      tip.textContent = new Date(t).toLocaleString('pl-PL') + '\n' + rows.join('\n'); tip.hidden = false;
      const xp = x(t) / W * 100; if (xp > 50) { tip.style.left = 'auto'; tip.style.right = (100 - xp + 1) + '%'; } else { tip.style.right = 'auto'; tip.style.left = (xp + 1) + '%'; }
    };
    svg.addEventListener('pointermove', e => { const box = svg.getBoundingClientRect(); const px = (e.clientX - box.left) / box.width * W; const t = t0 + Math.max(0, Math.min(1, (px - L) / (W - L - R))) * (t1 - t0); state.hoverT = t; show(t); });
    svg.addEventListener('pointerleave', () => { state.hoverT = null; show(null); });
    plot.replaceChildren(svg, tip);
    if (state.hoverT !== null && state.hoverT >= t0 && state.hoverT <= t1) show(state.hoverT);
  }

  return function render(data) {
    state.data = data;
    // Other render steps can replace page nodes; rebuild whenever our root is gone.
    if (!document.getElementById('synta-overview')) build();
    cards(); draw(); sections(); tickAge();
  };
})();

function insightsView(data){
 try{syntaOverview(data)}catch(error){console.error('overview',error)}
 focusConsole(data);
 liveMissionBoard(data);
 paperCapitalPreview(data);
 const r=data.report||{},g=r.mission_evidence?.goal_learning||{},policy=g.cpu_shadow_policy||{},work=r.income_work||{},sources=r.source_acquisition||{},workFreshness=r.status_freshness?.income_work||{};
 const p=panel('overview-panel','What is actually happening',$('cards'));p.replaceChildren();const h=document.createElement('h2');h.textContent='Work towards the goal';p.append(h);textField(p,'Verified revenue','Unavailable: no verified payment ledger connected');textField(p,'Current model-selected work',work.state||'No work receipt in this snapshot');textField(p,'Last work recorded',work.updated?new Date(work.updated*1000).toLocaleString():'unknown');textField(p,'Work receipt age',work.updated?`${Math.max(0,Math.round((Date.now()/1000-work.updated)/60))} minutes; refreshing the page does not create a new result`:'No dated work receipt');textField(p,'Successful tool receipts in last experiment',work.successful_tool_receipts);textField(p,'Meaning','A source read or a successful process is not a validated experiment, a model update or earned money.');textField(p,'CPU shadow learning',policy.state||'No policy receipt');textField(p,'Verified outcomes used by CPU policy',policy.trained_outcomes);textField(p,'Gemma candidates selected for next serving',r.accepted_weight_updates_this_run);const issueReceipts=[{name:'Income work',value:work.error,freshness:workFreshness.state||'unknown'},{name:'Learning loop',value:r.learning_activity?.reason,freshness:r.status_freshness?.learning_activity?.state||'unknown'},{name:'Paper fee source',value:r.paper_fee_source?.error,freshness:r.status_freshness?.paper_fee_source?.state||'unknown'}].filter(issue=>issue.value);const reportedIssue=issueReceipts.find(issue=>issue.freshness==='fresh')||issueReceipts[0];const currentWorkError=reportedIssue?`${reportedIssue.freshness==='fresh'?'Current':reportedIssue.freshness==='stale'?'Last recorded (historical)':'Last recorded (freshness unknown)'} ${reportedIssue.name} issue: ${reportedIssue.value}`:'None recorded in current receipts';textField(p,'Current blocker / last recorded error',currentWorkError);
 detailsField(p,'Last experiment receipt',JSON.stringify(work,null,2),'overview-work');
 const income=$('income-view');if(income){const counts=new Map();for(const c of r.income_opportunities?.candidates||[]){const name=c.specification?.domain||'Unspecified domain';counts.set(name,(counts.get(name)||0)+1)}evidenceBars(income,'Research coverage by domain (proposals, not earnings)',[...counts],'proposals');evidenceBars(income,'Source acquisition quality',[['Distinct contents',sources.distinct_contents],['Unchanged rereads',sources.unchanged_rereads]],'retrievals');textField(income,'Outcome charts','No verified revenue or completed-test time series for these domains is recorded. Proposal counts and estimated rewards cannot supply a profit curve.');}
 const learning=$('goal-learning-view');if(learning){textField(learning,'CPU policy',policy.state||'unavailable');textField(learning,'Independent outcomes trained',policy.trained_outcomes);textField(learning,'Conflicting outcome groups excluded',policy.conflict_groups_excluded);textField(learning,'Scored replay predictions',policy.prequential_samples);if(policy.prequential_samples>0&&typeof policy.prequential_brier==='number'){const references=[policy.uniform_baseline_brier,policy.prequential_flat_baseline_brier,policy.prequential_frequency_baseline_brier].filter(x=>typeof x==='number'&&Number.isFinite(x));textField(learning,'Compared with strongest recorded reference',references.length&&policy.prequential_brier<Math.min(...references)?'Lower Brier in this replay only; forward advantage unproven':'No improvement over the strongest reference in this replay')}evidenceBars(learning,'CPU forecast replay comparison: lower Brier is better',[['Learned shadow',policy.prequential_brier],['Uniform reference',policy.prequential_samples>0?policy.uniform_baseline_brier:null],['Always flat',policy.prequential_flat_baseline_brier],['Prior frequency',policy.prequential_frequency_baseline_brier]],'Brier');textField(learning,'Score scope','Exploratory prequential replay on the same scored samples. Not profit, not Gemma training and not proof of a forward advantage.');}
}

"""

STYLE = """
#synta-overview{background:#111c2c;border:1px solid #2f4664;border-radius:14px;padding:14px 16px;margin:12px 0 16px}
#synta-overview .ov-head{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
#synta-overview h2{margin:0;font-size:1.25rem}
#synta-overview h3{margin:0;font-size:1rem}
#synta-overview .ov-age{font-size:.85rem;padding:3px 9px;border-radius:999px;border:1px solid #3c5272}
#synta-overview .ov-age.ok{color:#7fd6a4;border-color:#2f6b4c}#synta-overview .ov-age.warn{color:#f9c36b;border-color:#7a5c26}#synta-overview .ov-age.bad{color:#ff8a7a;border-color:#7a3a33}
#synta-overview .ov-toggle,#synta-overview .ov-range{background:#17263c;color:#dbe7f5;border:1px solid #3c5272;border-radius:8px;padding:5px 10px;cursor:pointer;font:inherit;font-size:.85rem}
#synta-overview .ov-toggle{margin-left:auto}
#synta-overview .ov-range.on{background:#2b4b74;border-color:#57b7ff}
#synta-overview .ov-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:12px 0}
#synta-overview .ov-card{background:#162336;border:1px solid #2a3c55;border-left:4px solid #4d6a90;border-radius:10px;padding:9px 11px;display:flex;flex-direction:column;gap:3px;min-width:0}
#synta-overview .ov-card.ok{border-left-color:#4fbf87}#synta-overview .ov-card.warn{border-left-color:#e0a83f}#synta-overview .ov-card.bad{border-left-color:#e2665a}
#synta-overview .ov-card small{color:#9fb3cc;font-size:.78rem}#synta-overview .ov-card strong{font-size:1.05rem;overflow-wrap:break-word;hyphens:auto}#synta-overview .ov-card span{color:#b9c8da;font-size:.8rem;overflow-wrap:anywhere}
#synta-overview .ov-chart-head{display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap}
#synta-overview .ov-ranges{display:flex;gap:6px}
#synta-overview .ov-plot{position:relative;margin-top:8px}
#synta-overview .ov-plot svg{width:100%;height:auto;display:block;touch-action:pan-y}
#synta-overview .ov-grid-line{stroke:#2a3a52;stroke-dasharray:3 4}
#synta-overview .ov-axis{fill:#9fb3cc;font-size:11px}
#synta-overview .ov-cursor{stroke:#dbe7f5;stroke-width:1;stroke-dasharray:2 3}
#synta-overview .ov-tip{position:absolute;top:6px;background:#0c1422;border:1px solid #3c5272;border-radius:8px;padding:6px 8px;font-size:.78rem;white-space:pre-wrap;max-width:62%;pointer-events:none}
#synta-overview .ov-legend{display:flex;flex-direction:column;gap:4px;margin-top:6px;font-size:.82rem}
#synta-overview .ov-key i{display:inline-block;width:12px;height:3px;margin-right:6px;vertical-align:middle}
#synta-overview .ov-key.stale{color:#ffbf86}
#synta-overview .ov-fill{pointer-events:none}
#synta-overview .ov-more{display:flex;flex-direction:column;gap:8px;margin-top:12px}
#synta-overview .ov-sec{background:#142033;border:1px solid #2a3c55;border-radius:10px;padding:0 12px}
#synta-overview .ov-sec summary{cursor:pointer;padding:10px 0;font-weight:600}
#synta-overview .ov-sec-body{padding:0 0 10px;display:flex;flex-direction:column;gap:6px}
#synta-overview .ov-row{display:grid;grid-template-columns:minmax(120px,30%) 1fr;gap:10px;font-size:.85rem;border-top:1px solid #213149;padding-top:6px}
#synta-overview .ov-row-k{color:#9fb3cc;overflow-wrap:anywhere}#synta-overview .ov-row-v{overflow-wrap:anywhere}
#synta-overview .ov-row.warn .ov-row-v{color:#f9c36b}#synta-overview .ov-row.bad .ov-row-v{color:#ff8a7a}#synta-overview .ov-row.ok .ov-row-v{color:#7fd6a4}
#synta-overview .ov-note,#synta-overview .ov-empty{color:#9fb3cc;font-size:.8rem;margin:8px 0 0}
body.synta-compact>:is(#layout,#mission-navigation,#cards,#focus-console,section:not(#synta-overview),details:not(#warning-details)){display:none!important}
@media (max-width:600px){#synta-overview .ov-row{grid-template-columns:1fr;gap:2px}#synta-overview{padding:10px}#synta-overview .ov-grid{grid-template-columns:1fr 1fr}#synta-overview .ov-toggle{margin-left:0}#synta-overview .ov-card strong{font-size:.95rem}}
#live-mission-board{border:1px solid #355475;background:linear-gradient(130deg,#15283d,#101927)}.live-board-header{display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap}.live-board-metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:10px}.live-board-metric{display:flex;flex-direction:column;gap:7px;padding:16px;background:#101c2d;border:1px solid #37516b;border-radius:10px}.live-board-metric strong{font-size:21px;overflow-wrap:anywhere}.live-board-metric small{color:#b6c8de}.live-board-blocker{padding:12px;border-left:3px solid #f1b778;background:#241f24;margin:14px 0}.live-board-blocker p{margin-bottom:0;overflow-wrap:anywhere}.live-board-events{max-height:300px;overflow:auto}.live-board-event{padding:10px;border-bottom:1px solid #304259;display:flex;gap:12px;flex-wrap:wrap}.live-board-event time{color:#9cb6d4;font-size:12px}
[hidden]{display:none!important}body{background:#0c1422;max-width:1280px}#mission-navigation{display:flex;gap:8px;flex-wrap:wrap;position:sticky;top:0;z-index:10;padding:12px 0;background:#0c1422;border-bottom:1px solid #304259;margin-bottom:20px}#mission-navigation button{background:#17263c;border:1px solid #304259;color:#bac9de}#mission-navigation button[aria-pressed=true]{background:#29578b;color:#fff;border-color:#85baff}button:focus-visible,a:focus-visible,select:focus-visible{outline:3px solid #ffd166;outline-offset:3px}section{box-shadow:none}h1{letter-spacing:-.6px}#cards{grid-template-columns:repeat(auto-fit,minmax(170px,1fr))}#overview-panel{border-left:4px solid #80baff}table{display:block;overflow:auto;white-space:nowrap}th{font-size:12px;color:#b9cbe2}select,input{background:#14243a;color:#eef5ff;border:1px solid #486582;padding:9px;border-radius:7px;max-width:100%}label{display:grid;gap:6px;font-size:13px}.evidence-bar{margin:12px 0;font-size:13px}.bar-track{height:9px;background:#26384e;border-radius:9px;margin-top:6px}.bar-track>div{height:100%;background:#85baff;border-radius:9px}#income-view>.card{margin:8px 0}#backtest-audit{max-height:120px}#fresh,#layout{font-size:12px}#trade-plot{background:#0c1726;border-radius:10px}#trade-summary{font-weight:600}#trade-scope{font-size:12px}#trade-hover{font-size:13px}#timeline-panel{max-height:480px;overflow:auto}#income-view details.card>summary{cursor:pointer;font-weight:600}#plan-grid p{max-height:180px;overflow:auto}@media(max-width:650px){#mission-navigation{gap:6px}#mission-navigation button{padding:8px 10px;font-size:12px}}
"""
