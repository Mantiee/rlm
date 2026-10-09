#!/usr/bin/env python3
"""Read-only LAN dashboard for an existing V100 installation; no model reload."""

import argparse
import ipaddress
import json
import math
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

REPORT_ID = re.compile(r"\d{8}-\d{6}-(?:all|daily|weekly)-[0-9a-f]{8}")
ASSETS = {
    "report.html": "text/html; charset=utf-8",
    "report.json": "application/json",
    "report.md": "text/plain; charset=utf-8",
    "trades.csv": "text/csv; charset=utf-8",
}

PAGE = r"""<!doctype html><html lang="pl"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>V100 - postęp misji</title>
<style>
:root{color-scheme:dark}body{font:15px system-ui;background:#101721;color:#e9eef6;margin:0;padding:24px;max-width:1200px;margin:auto}h1{font-size:26px}h2{font-size:19px}small,.muted{color:#adbdd1}a{color:#9ac5ff}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}.card,section{background:#192433;border:1px solid #304259;border-radius:12px;padding:18px;margin:14px 0}.card{margin:0}.value{display:block;font-size:22px;margin-top:8px;overflow-wrap:anywhere}progress{width:100%;height:24px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;max-height:420px;overflow:auto}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #304259}svg{width:100%;max-height:250px}#warning{color:#ffbf86}button{padding:8px 14px;background:#294464;border:0;color:white;border-radius:6px;cursor:pointer}
</style>
<h1>V100 - postęp misji</h1><p class="muted">Podgląd odczytu. Paper P&amp;L i backtesty nie są realnym dochodem.</p>
<p id="fresh">Ładowanie danych...</p><p id="warning"></p><button id="reload">Odśwież widok</button>
<div id="cards" class="grid"></div>
<section><h2>Testy bieżącego przebiegu</h2><p id="evaluation"></p><progress id="progress" max="1" value="0"></progress><p id="case" class="muted"></p></section>
<section><h2>Ostatnie backtesty - eksploracyjne</h2><p class="muted">Wynik netto prób historycznych. Założenia kosztów nie są potwierdzeniem dostępnych opłat ani przewagi strategii.</p><svg id="chart" viewBox="0 0 800 220" role="img" aria-label="Wyniki ostatnich backtestów"></svg><table><thead><tr><th>Próba</th><th>Wynik netto</th><th>Obsunięcie</th><th>Wykonania</th></tr></thead><tbody id="backtests"></tbody></table></section>
<section><h2>Paper A/B</h2><pre id="paper"></pre><h3>Blokady</h3><pre id="blockers"></pre></section>
<section><h2>Raporty HTML z wykresami i pliki wyników</h2><p class="muted">Raporty są zapisanymi migawkami, mają własny czas i zakres.</p><div id="reports"></div></section>
<section><h2>Pomocnicy, GUI i benchmark oficjalny</h2><pre id="workers"></pre></section>
<section><h2>Aktualny log kontrolera</h2><pre id="log"></pre></section>
<details><summary>Dowody treningu i ostatnie błędy</summary><pre id="evidence"></pre></details>
<script>
const $=id=>document.getElementById(id), fmt=x=>x===null||x===undefined?'brak potwierdzonych danych':String(x), pct=x=>typeof x==='number'&&Number.isFinite(x)?(x*100).toFixed(2)+'%':'brak danych';
function card(label,value){const node=document.createElement('div');node.className='card';const title=document.createElement('small');title.textContent=label;const strong=document.createElement('strong');strong.className='value';strong.textContent=fmt(value);node.append(title,strong);$('cards').append(node)}
function svg(tag,attrs,text){let node=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [key,val] of Object.entries(attrs))node.setAttribute(key,val);if(text!==undefined)node.textContent=text;return node}
function render(data){const m=data.mission||{},r=data.report||{},ev=m.evaluation||{},learn=m.learning||{},gpu=data.gpu||{};
$('fresh').textContent='Zebrano: '+new Date(data.collected_at*1000).toLocaleString()+' | Raport zbierany co 20 s | Przebieg: '+fmt(m.run);
$('warning').textContent=(data.errors||[]).join('\n');$('cards').replaceChildren();card('Misja',m.running?'działa':'zatrzymana');card('Faza',m.state?.phase);card('Cykle uczenia',learn.completed_cycles);card('Zaakceptowane aktualizacje wag',r.accepted_weight_updates_this_run);card('Potwierdzone kroki optymalizatora',r.mission_evidence?.optimizer_updates_observed);card('Kontekst',m.state?.context_window);card('GPU - wykorzystanie',gpu.utilization===undefined?null:gpu.utilization+'%');card('VRAM zajęty / razem',gpu.memory_used===undefined?null:gpu.memory_used+' / '+gpu.memory_total+' MiB');card('GPU - temperatura / moc',gpu.temperature===undefined?null:gpu.temperature+' °C / '+gpu.power+' W');
$('evaluation').textContent=ev.total?`${ev.completed}/${ev.total} ukończonych, ${ev.passed} zaliczonych (${ev.state})`:'Brak bieżącego licznika ewaluacji';$('progress').max=ev.total||1;$('progress').value=ev.completed||0;$('case').textContent=ev.current_case?('Obecny przypadek: '+ev.current_case):'';
$('paper').textContent=JSON.stringify(r.paper||{},null,2);$('blockers').textContent=JSON.stringify(r.paper_blockers||[],null,2);$('workers').textContent=JSON.stringify({drones:r.drones,external_compute:r.external_compute,desktop:r.desktop,official_benchmark:r.official_benchmark},null,2);$('log').textContent=data.controller_log||'Brak logu';$('evidence').textContent=JSON.stringify({last_cycle:learn.last_cycle,evidence:r.mission_evidence,collection_errors:data.errors},null,2);
$('backtests').replaceChildren();$('chart').replaceChildren();const tests=r.recent_exploratory_backtests||[];let largest=Math.max(.01,...tests.map(t=>Math.abs(t.development_test?.net_return||0)));$('chart').append(svg('line',{x1:400,y1:0,x2:400,y2:220,stroke:'#8296b1'}));tests.forEach((test,i)=>{const result=test.development_test||{},tr=document.createElement('tr');[`${i+1}. ${test.parameters?.rule||''}`,pct(result.net_return),pct(result.max_sampled_drawdown),fmt(result.fills)].forEach(v=>{let td=document.createElement('td');td.textContent=v;tr.append(td)});$('backtests').append(tr);if(typeof result.net_return==='number'&&Number.isFinite(result.net_return)){let width=Math.abs(result.net_return)/largest*280,y=15+i*48;$('chart').append(svg('rect',{x:result.net_return<0?400-width:400,y,width,height:26,fill:result.net_return<0?'#e68d80':'#75c5ae'}),svg('text',{x:8,y:y+19,fill:'#dce6f5','font-size':14},`${i+1}. ${pct(result.net_return)}`))}});if(!tests.length)$('chart').append(svg('text',{x:20,y:50,fill:'#adbdd1'},'Brak zapisanych wyników'));
$('reports').replaceChildren();for(const id of data.paper_reports||[]){const row=document.createElement('p');row.append(document.createTextNode(id+' '));for(const [name,label] of [['report.html','HTML i wykresy'],['report.json','JSON'],['trades.csv','CSV']]){const a=document.createElement('a');a.href='/paper/'+encodeURIComponent(id)+'/'+name;a.target='_blank';a.rel='noopener';a.textContent=label;row.append(a,document.createTextNode(' · '))}$('reports').append(row)}
}
async function refresh(){try{let response=await fetch('/api/status',{cache:'no-store'});if(!response.ok)throw Error('HTTP '+response.status);render(await response.json())}catch(error){$('warning').textContent='Nie można odświeżyć: '+error.message}}
$('reload').onclick=refresh;refresh();setInterval(refresh,5000);
</script></html>"""


def validate_bind(address):
    ip = ipaddress.ip_address(address)
    if (
        ip.version != 4
        or not ip.is_private
        or ip.is_unspecified
        or ip.is_multicast
        or ip.is_reserved
    ):
        raise ValueError("Bind to an explicit private IPv4 address or 127.0.0.1")
    return address


def read_json(path):
    if path.stat().st_size > 2 * 2**20:
        raise ValueError("JSON snapshot exceeds 2 MiB")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected an object snapshot")
    return value


def paper_asset(root, route):
    parts = unquote(urlsplit(route).path).split("/")
    if (
        len(parts) != 4
        or parts[1] != "paper"
        or not REPORT_ID.fullmatch(parts[2])
        or parts[3] not in ASSETS
    ):
        raise ValueError("Unknown report route")
    base = (root / "research/paper/reports").resolve()
    path = base / parts[2] / parts[3]
    if (
        path.resolve().parent != base / parts[2]
        or not path.is_file()
        or path.stat().st_size > 8 * 2**20
    ):
        raise ValueError("Report is missing, oversized or outside its directory")
    return path, ASSETS[parts[3]]


class DashboardState:
    def __init__(self, root, mission_reader):
        self.root = root.resolve()
        self.mission_reader = mission_reader
        self.data = {"collected_at": time.time(), "errors": ["Pierwsze zbieranie danych trwa"]}
        self.stop = threading.Event()

    def refresh(self):
        errors, report, gpu = [], {}, {}
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "rlm.v100.cli",
                    "--root",
                    str(self.root),
                    "mission-report",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=20,
            )
            if result.returncode:
                raise RuntimeError(result.stderr[-600:] or "mission-report failed")
            report = read_json(self.root / "research/mission/latest-report.json")
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            errors.append("Raport nie został odświeżony: " + str(error)[:600])
        mission = self.mission_reader(self.root)
        if report.get("mission_evidence", {}).get("run") != mission.get("run"):
            if report:
                errors.append(
                    "Raport dotyczy poprzedniego przebiegu; pokazano tylko bieżący status"
                )
            report = {}
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            values = result.stdout.strip().splitlines()[0].split(",")
            gpu = dict(
                zip(
                    ("utilization", "memory_used", "memory_total", "temperature", "power"),
                    (float(v.strip()) for v in values),
                    strict=True,
                )
            )
            if not all(math.isfinite(value) for value in gpu.values()):
                gpu = {}
                raise ValueError("GPU counters are nonfinite")
        except (OSError, ValueError, IndexError, subprocess.SubprocessError) as error:
            errors.append("Pomiar GPU niedostępny: " + str(error)[:200])
        log = ""
        if mission.get("run"):
            path = Path(mission["run"]) / "controller.log"
            if path.resolve().is_relative_to(self.root / "research/mission") and path.exists():
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 16000))
                    log = stream.read(16000).decode(errors="replace")
        reports = sorted(
            (self.root / "research/paper/reports").glob("*/report.html"),
            key=lambda p: p.stat().st_mtime_ns,
            reverse=True,
        )
        self.data = {
            "collected_at": time.time(),
            "mission": mission,
            "report": report,
            "gpu": gpu,
            "controller_log": log,
            "paper_reports": [p.parent.name for p in reports if REPORT_ID.fullmatch(p.parent.name)][
                :12
            ],
            "errors": errors,
        }

    def loop(self):
        while not self.stop.is_set():
            try:
                self.refresh()
            except (OSError, ValueError, RuntimeError) as error:
                self.data = {"collected_at": time.time(), "errors": [str(error)[:600]]}
            self.stop.wait(20)


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, state):
        self.state = state
        self.slots = threading.BoundedSemaphore(8)
        super().__init__(address, DashboardHandler)

    def process_request(self, request, client_address):
        request.settimeout(10)
        if not self.slots.acquire(blocking=False):
            try:
                request.sendall(b"HTTP/1.0 503 Busy\r\nContent-Length: 0\r\n\r\n")
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):  # noqa: N802 - standard HTTP callback
        route = urlsplit(self.path).path
        try:
            if route == "/":
                body, mime = PAGE.encode(), "text/html; charset=utf-8"
            elif route == "/api/status":
                body, mime = (
                    json.dumps(
                        self.server.state.data, ensure_ascii=False, allow_nan=False
                    ).encode(),
                    "application/json; charset=utf-8",
                )
            else:
                path, mime = paper_asset(self.server.state.root, self.path)
                body = path.read_bytes()
        except (ValueError, OSError):
            self.send_error(404, "Unavailable")
            return
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'"
            if route == "/"
            else "default-src 'none'; style-src 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802 - standard HTTP callback
        self.send_error(405, "Read-only dashboard")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bind", required=True, type=validate_bind)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Use an unprivileged port between 1024 and 65535")
    from rlm.v100.mission import status

    state = DashboardState(args.root, status)
    with DashboardServer((args.bind, args.port), state) as server:
        thread = threading.Thread(target=state.loop, daemon=True)
        thread.start()
        print(f"V100 dashboard: http://{args.bind}:{args.port} - read-only", flush=True)
        try:
            server.serve_forever()
        finally:
            state.stop.set()
            thread.join(timeout=1)


if __name__ == "__main__":
    main()
