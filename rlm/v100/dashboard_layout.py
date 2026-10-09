"""Shared passive dashboard layout contract and host-owned live renderer."""

import base64
import hashlib
import json
import re
from html.parser import HTMLParser

PAGE = r"""<!doctype html><html lang="pl"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>V100 - postęp misji</title>
<style>
:root{color-scheme:dark}body{font:15px system-ui;background:#101721;color:#e9eef6;margin:0;padding:24px;max-width:1200px;margin:auto}h1{font-size:26px}h2{font-size:19px}small,.muted{color:#adbdd1}a{color:#9ac5ff}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}.card,section{background:#192433;border:1px solid #304259;border-radius:12px;padding:18px;margin:14px 0}.card{margin:0}.value{display:block;font-size:22px;margin-top:8px;overflow-wrap:anywhere}progress{width:100%;height:24px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;max-height:420px;overflow:auto}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #304259}svg{width:100%;max-height:250px}#warning{color:#ffbf86}button{padding:8px 14px;background:#294464;border:0;color:white;border-radius:6px;cursor:pointer}
</style>
<h1>V100 - postęp misji</h1><p class="muted">Podgląd odczytu. Paper P&amp;L i backtesty nie są realnym dochodem.</p>
<p id="fresh">Ładowanie danych...</p><p id="warning"></p><p id="layout" class="muted"></p><button id="reload">Odśwież widok</button>
<div id="cards" class="grid"></div>
<section><h2>Testy bieżącego przebiegu</h2><p id="evaluation"></p><progress id="progress" max="1" value="0"></progress><p id="case" class="muted"></p></section>
<section><h2>Ostatnie backtesty - eksploracyjne</h2><p class="muted">Wynik netto prób historycznych. Założenia kosztów nie są potwierdzeniem dostępnych opłat ani przewagi strategii.</p><svg id="chart" viewBox="0 0 800 220" role="img" aria-label="Wyniki ostatnich backtestów"></svg><table><thead><tr><th>Próba</th><th>Wynik netto</th><th>Obsunięcie</th><th>Wykonania</th></tr></thead><tbody id="backtests"></tbody></table></section>
<section><h2>Paper A/B</h2><pre id="paper"></pre><h3>Blokady</h3><pre id="blockers"></pre></section>
<section><h2>Raporty HTML z wykresami i pliki wyników</h2><p class="muted">Raporty są zapisanymi migawkami, mają własny czas i zakres.</p><div id="reports"></div></section>
<section><h2>Co teraz pracuje</h2><div id="actors" class="grid"></div><h3>Cele i plany</h3><pre id="goals"></pre><h3>Ostatnie zdarzenia</h3><div id="events"></div></section>
<section><h2>Pomocnicy, GUI i benchmark oficjalny</h2><pre id="workers"></pre></section>
<section><h2>Aktualny log kontrolera</h2><pre id="log"></pre></section>
<details><summary>Dowody treningu i ostatnie błędy</summary><pre id="evidence"></pre></details>
<script>
const $=id=>document.getElementById(id), fmt=x=>x===null||x===undefined?'brak potwierdzonych danych':String(x), pct=x=>typeof x==='number'&&Number.isFinite(x)?(x*100).toFixed(2)+'%':'brak danych';
const layoutVersion='__LAYOUT_SHA__';
function card(label,value){const node=document.createElement('div');node.className='card';const title=document.createElement('small');title.textContent=label;const strong=document.createElement('strong');strong.className='value';strong.textContent=fmt(value);node.append(title,strong);$('cards').append(node)}
function workView(data){
for(const id of ['actors','goals','events']){if(!$(id)){const node=document.createElement(id==='goals'?'pre':'div');node.id=id;if(id==='actors')node.className='grid';$('cards').after(node)}}
$('actors').replaceChildren();
for(const a of data.live?.actors||[]){
 const node=document.createElement('article');node.className='card';const title=document.createElement('h3');title.textContent=a.label;
 const state=document.createElement('strong');state.textContent=a.state+(a.stale?' | nieaktualne lub brak danych':'');state.style.color=a.stale?'#ffbf86':'#75c5ae';
 const detail=document.createElement('p');detail.textContent=a.detail||'';node.append(title,state,detail);
 if(a.cooldown_seconds){const p=document.createElement('p');p.textContent='Przerwa helpera: '+a.cooldown_seconds+' s';node.append(p)}
 const metrics=a.metrics||{};for(const [key,label,unit] of [['utilization','GPU','%'],['memory_used','VRAM',' MiB'],['power','Moc',' W'],['temperature','Temperatura',' °C'],['percent','CPU','%'],['available_ram_gib','Wolny RAM',' GiB']]){if(typeof metrics[key]==='number'){const p=document.createElement('p');p.textContent=label+': '+Number(metrics[key].toFixed(2))+unit;node.append(p)}}
 for(const w of a.workers||[]){const p=document.createElement('p');p.textContent=[w.name,w.phase,w.reason,w.stale?'nieaktualne':''].filter(Boolean).join(' | ');node.append(p)}
 for(const j of a.jobs||[]){const p=document.createElement('p');p.textContent=[j.kind,j.branch,j.state||j.phase,j.id,j.assignment].filter(Boolean).join(' | ');node.append(p)}
 const proof=document.createElement('details'),summary=document.createElement('summary'),raw=document.createElement('pre');summary.textContent='Dane źródłowe';raw.textContent=JSON.stringify(a,null,2);proof.append(summary,raw);node.append(proof);$('actors').append(node)
}
const plans=data.live?.goals||{};$('goals').textContent=[['long','Cel długoterminowy'],['mid','Plan średnioterminowy'],['short','Plan krótkoterminowy']].map(([key,label])=>label+': '+(plans[key]?.text||'brak planu')+(plans[key]?.needs_replanning?' (wymaga przeplanowania)':'')).join('\n\n');
$('events').replaceChildren();for(const e of [...(data.live?.events||[])].reverse().slice(0,20)){const p=document.createElement('p');p.textContent=[e.time,e.actor,e.kind,e.tool,e.detail].filter(Boolean).join(' | ');$('events').append(p)}
}
function svg(tag,attrs,text){let node=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [key,val] of Object.entries(attrs))node.setAttribute(key,val);if(text!==undefined)node.textContent=text;return node}
function render(data){const m=data.mission||{},r=data.report||{},ev=m.evaluation||{},learn=m.learning||{},gpu=data.gpu||{};
if(data.layout?.active_sha256&&layoutVersion!==data.layout.active_sha256){location.reload();return}$('layout').textContent='HTML mastera: /workspace/dashboard/index.html | '+(data.layout?.state||'oczekiwanie')+(data.layout?.error?' | '+data.layout.error:'');
$('fresh').textContent='Zebrano: '+new Date(data.collected_at*1000).toLocaleString()+' | Raport zbierany co 20 s | Przebieg: '+fmt(m.run);
workView(data);$('warning').textContent=(data.errors||[]).join('\n');$('cards').replaceChildren();card('Misja',m.running?'działa':'zatrzymana');card('Faza',m.state?.phase);card('Cykle uczenia',learn.completed_cycles);card('Zaakceptowane aktualizacje wag',r.accepted_weight_updates_this_run);card('Potwierdzone kroki optymalizatora',r.mission_evidence?.optimizer_updates_observed);card('Kontekst',m.state?.context_window);card('GPU - wykorzystanie',gpu.utilization===undefined?null:gpu.utilization+'%');card('VRAM zajęty / razem',gpu.memory_used===undefined?null:gpu.memory_used+' / '+gpu.memory_total+' MiB');card('GPU - temperatura / moc',gpu.temperature===undefined?null:gpu.temperature+' °C / '+gpu.power+' W');
$('evaluation').textContent=ev.total?`${ev.completed}/${ev.total} ukończonych, ${ev.passed} zaliczonych (${ev.state})`:'Brak bieżącego licznika ewaluacji';$('progress').max=ev.total||1;$('progress').value=ev.completed||0;$('case').textContent=ev.current_case?('Obecny przypadek: '+ev.current_case):'';
$('paper').textContent=JSON.stringify(r.paper||{},null,2);$('blockers').textContent=JSON.stringify(r.paper_blockers||[],null,2);$('workers').textContent=JSON.stringify({drones:r.drones,external_compute:r.external_compute,desktop:r.desktop,official_benchmark:r.official_benchmark},null,2);$('log').textContent=data.controller_log||'Brak logu';$('evidence').textContent=JSON.stringify({last_cycle:learn.last_cycle,evidence:r.mission_evidence,collection_errors:data.errors},null,2);
$('backtests').replaceChildren();$('chart').replaceChildren();const tests=r.recent_exploratory_backtests||[];let largest=Math.max(.01,...tests.map(t=>Math.abs(t.development_test?.net_return||0)));$('chart').append(svg('line',{x1:400,y1:0,x2:400,y2:220,stroke:'#8296b1'}));tests.forEach((test,i)=>{const result=test.development_test||{},tr=document.createElement('tr');[`${i+1}. ${test.parameters?.rule||''}`,pct(result.net_return),pct(result.max_sampled_drawdown),fmt(result.fills)].forEach(v=>{let td=document.createElement('td');td.textContent=v;tr.append(td)});$('backtests').append(tr);if(typeof result.net_return==='number'&&Number.isFinite(result.net_return)){let width=Math.abs(result.net_return)/largest*280,y=15+i*48;$('chart').append(svg('rect',{x:result.net_return<0?400-width:400,y,width,height:26,fill:result.net_return<0?'#e68d80':'#75c5ae'}),svg('text',{x:8,y:y+19,fill:'#dce6f5','font-size':14},`${i+1}. ${pct(result.net_return)}`))}});if(!tests.length)$('chart').append(svg('text',{x:20,y:50,fill:'#adbdd1'},'Brak zapisanych wyników'));
$('reports').replaceChildren();for(const id of data.paper_reports||[]){const row=document.createElement('p');row.append(document.createTextNode(id+' '));for(const [name,label] of [['report.html','HTML i wykresy'],['report.json','JSON'],['trades.csv','CSV']]){const a=document.createElement('a');a.href='/paper/'+encodeURIComponent(id)+'/'+name;a.target='_blank';a.rel='noopener';a.textContent=label;row.append(a,document.createTextNode(' · '))}$('reports').append(row)}
}
async function refresh(){try{let response=await fetch('/api/status',{cache:'no-store'});if(!response.ok)throw Error('HTTP '+response.status);render(await response.json())}catch(error){$('warning').textContent='Nie można odświeżyć: '+error.message}}
$('reload').onclick=refresh;refresh();setInterval(refresh,5000);
</script></html>"""


APP_SCRIPT = PAGE.split("<script>", 1)[1].split("</script>", 1)[0]
BASE_TEMPLATE = PAGE.split("<script>", 1)[0] + "</html>"
GUEST_TEMPLATE = "/workspace/dashboard/index.html"
REQUIRED_IDS = set(re.findall(r"\$\('([a-z]+)'\)", APP_SCRIPT)) - {"actors", "goals", "events"}


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
    script = "<script>" + APP_SCRIPT.replace("__LAYOUT_SHA__", identity) + "</script>"
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
