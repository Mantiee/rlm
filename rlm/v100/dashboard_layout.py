Warning: truncated output (original token count: 21533)
Total output lines: 452

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
for(const id of ['actors','goals','events']){if(!$(id)){const node=document.createElement(id==='goals'?'pre':'div');node.id=id;if(id==='actors')node.classN…15533 tokens truncated…'#adbdd1','font-size':11},new Date(hi).toLocaleString()));
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

