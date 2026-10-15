#!/usr/bin/env python3
"""Read-only local live monitor for Synta's paper ledger. No order submission."""

import argparse
import ipaddress
import json
import sqlite3
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_LEDGER = Path.home() / "ai-v100/research/paper/ledger.sqlite3"
HTML = r"""<!doctype html><html lang="pl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Synta — Symulacja zarabiania i uczenie</title><style>
:root{color-scheme:dark;font-family:system-ui,Arial;background:#07111f;color:#e7f3ff}*{box-sizing:border-box}body{margin:0;padding:20px;max-width:1500px;margin:auto}h1{font-size:24px;margin:0 0 5px}h2{font-size:17px;color:#7fc8ff}small,.muted{color:#a2b5c9}.top{display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap}.tag{padding:8px 12px;border-radius:8px;background:#1a324a}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:10px;margin:16px 0}.card,.panel{border:1px solid #31506a;border-radius:12px;background:#102237;padding:15px}.value{font-size:26px;font-weight:750;overflow-wrap:anywhere;margin-top:8px}.warn{color:#ffbc7e}.ok{color:#8bdfad}.panels{display:grid;grid-template-columns:1.4fr 1fr;gap:12px}canvas{width:100%;height:260px;display:block}table{width:100%;border-collapse:collapse;font-size:13px}td,th{border-bottom:1px solid #2a4257;padding:9px;text-align:left;vertical-align:top}th{color:#a5cae9}td{overflow-wrap:anywhere}code{font-size:12px;white-space:pre-wrap}#error{color:#ffb4a8}details{margin:5px 0}summary{cursor:pointer;color:#8bcaff}#quotes{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:8px}#quotes>div{background:#0a1b2b;padding:10px;border-radius:8px}.scroll{max-height:440px;overflow:auto}button{background:#244d70;color:white;border:1px solid #5082a9;border-radius:8px;padding:9px;cursor:pointer}@media(max-width:850px){.panels{grid-template-columns:1fr}}
</style></head><body><div class="top"><div><h1>Synta — Symulacja zarabiania i uczenie <span class="warn">(symulacja, nie realny dochód)</span></h1><small>Na żywo: rzeczywista księga paper, decyzje A/B i postęp uczenia. Tylko odczyt; bez prawdziwych zleceń.</small></div><div><span class="tag" id="connection">Łączenie…</span> <button onclick="refresh()">Odśwież</button></div></div><div id="error"></div>
<div class="cards"><div class="card"><small>Etap kontrolera (rzeczywisty)</small><div class="value" id="backendstage">—</div></div><div class="card"><small>Ostatnia decyzja</small><div class="value" id="decisionage">—</div></div><div class="card"><small>Stan uczenia</small><div class="value" id="learning">—</div></div><div class="card"><small>GPU compute (potwierdzone)</small><div class="value" id="gpu">—</div></div><div class="card"><small>Wszystkie zdarzenia</small><div class="value" id="events">—</div></div><div class="card"><small>Obserwacje</small><div class="value" id="observations">—</div></div><div class="card"><small>Decyzje</small><div class="value" id="decisions">—</div></div><div class="card"><small>Fille (zapisane)</small><div class="value" id="fills">—</div></div><div class="card"><small>Kapitał A</small><div class="value" id="a">—</div></div><div class="card"><small>Kapitał B</small><div class="value" id="b">—</div></div><div class="card"><small>Ostatnie zdarzenie</small><div class="value" id="age">—</div></div></div>
<div class="panels"><section class="panel"><h2>Kapitał A/B — zapisane obserwacje</h2><canvas id="chart" width="950" height="280"></canvas><small>Wykres nie prognozuje zysków. Linie mogą być płaskie przy braku filli.</small></section><section class="panel"><h2>Ostatnie notowania (z księgi)</h2><div id="quotes">—</div><p class="muted">Wiek notowania liczony względem czasu komputera; źródło może mieć własne opóźnienie.</p></section></div>
<section class="panel" style="margin-top:12px"><h2>Decyzje A/B — zestawienie</h2><div id="actioncounts">—</div></section><section class="panel" style="margin-top:12px"><h2>Blokady i weryfikacja</h2><div id="diagnostics">—</div></section><section class="panel" style="margin-top:12px"><h2>Decyzje — ostatnie 30</h2><div class="scroll"><table><thead><tr><th>Czas</th><th>Gałąź</th><th>Akcja / powód</th><th>Pełny zapis</th></tr></thead><tbody id="decisionrows"></tbody></table></div></section>
<section class="panel" style="margin-top:12px"><h2>Rozkład zdarzeń i stan księgi</h2><div id="kinds"></div><details><summary>Stan księgi (fragment JSON)</summary><code id="state"></code></details><small id="updated"></small></section>
<script>
const $=id=>document.getElementById(id);const n=v=>Number(v).toLocaleString('pl-PL');const money=v=>v==null?'—':Number(v).toLocaleString('pl-PL',{maximumFractionDigits:2})+' PLN';const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function draw(series){const c=$('chart'),ctx=c.getContext('2d'),w=c.width,h=c.height;ctx.clearRect(0,0,w,h);ctx.fillStyle='#091b2c';ctx.fillRect(0,0,w,h);const all=series.flatMap(s=>s.values.map(Number)).filter(Number.isFinite);if(!all.length){ctx.fillStyle='#a2b5c9';ctx.fillText('Brak punktów equity w księdze',25,35);return;}let lo=Math.min(...all),hi=Math.max(...all);const p=Math.max((hi-lo)*.15,hi*.0005,1);lo-=p;hi+=p;const left=75,right=w-20,top=20,bottom=h-35;ctx.font='12px sans-serif';ctx.strokeStyle='#2d4761';ctx.fillStyle='#a2b5c9';for(let i=0;i<5;i++){const y=top+(bottom-top)*i/4;ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();ctx.fillText((hi-(hi-lo)*i/4).toFixed(2),5,y+4);}for(const [idx,s] of series.entries()){const a=s.values.map(Number);if(!a.length)continue;ctx.strokeStyle=idx===0?'#57b7ff':'#f9c36b';ctx.lineWidth=2;ctx.beginPath();a.forEach((v,i)=>{const x=left+(right-left)*i/Math.max(a.length-1,1),y=top+(hi-v)/(hi-lo)*(bottom-top);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();ctx.fillStyle=ctx.strokeStyle;ctx.fillText(s.branch+': '+money(a[a.length-1]),left+idx*260,bottom+23);}}
function age(ts){if(!ts)return 'brak';const s=Math.max(0,Math.round((Date.now()-Date.parse(ts))/1000));return s<60?s+' s':s<3600?Math.floor(s/60)+' min':Math.floor(s/3600)+' h';}
async function refresh(){try{const r=await fetch('/api/status',{cache:'no-store'});if(!r.ok)throw Error('HTTP '+r.status);const d=await r.json();$('backendstage').textContent=d.mission?.stage||'Brak potwierdzenia';$('learning').textContent=d.mission?.learning||'Brak potwierdzenia';$('gpu').textContent=d.gpu?.state||'Brak danych';$('decisionage').textContent=age(d.last_decision_time);$('diagnostics').textContent=(d.diagnostics||[]).join(' | ')||'Brak zgłoszonych blokad (nie oznacza zysków)';$('actioncounts').textContent=Object.entries(d.actions||{}).map(([k,v])=>k+': '+v).join(' · ')||'Brak decyzji';$('connection').textContent='SQLite: odczyt OK';$('connection').className='tag ok';$('error').textContent='';$('events').textContent=n(d.total);$('observations').textContent=n(d.kinds.observation||0);$('decisions').textContent=n(d.kinds.decision||0);$('fills').textContent=n(d.fills);$('a').textContent=money(d.latest_equity.A);$('b').textContent=money(d.latest_equity.B);$('age').textContent=age(d.last_event_time);$('kinds').textContent=Object.entries(d.kinds).map(([k,v])=>k+': '+n(v)).join(' · ');$('state').textContent=JSON.stringify(d.state,null,2).slice(0,10000);$('updated').textContent='Odczyt: '+new Date().toLocaleTimeString('pl-PL')+' · ostatnie zdarzenie: '+(d.last_event_time||'brak');$('quotes').innerHTML=d.quotes.map(q=>'<div><b>'+esc(q.symbol)+'</b><div>Bid: '+esc(q.bid)+' · Ask: '+esc(q.ask)+'</div><small>'+esc(q.time)+' · '+esc(age(q.time))+' temu</small></div>').join('')||'Brak notowań';$('decisionrows').innerHTML=d.decisions.map(x=>{const p=x.payload||{};const a=p.proposal?.action??p.action??p.decision??p.signal??'—';const reason=p.proposal?.rationale??p.reason??p.blocker??p.status??'—';return '<tr><td>'+esc(x.time)+'</td><td>'+esc(x.branch)+'</td><td><b>'+esc(typeof a==='object'?JSON.stringify(a):a)+'</b><div class="muted">'+esc(typeof reason==='object'?JSON.stringify(reason):reason)+'</div></td><td><details><summary>JSON</summary><code>'+esc(JSON.stringify(p,null,2).slice(0,4000))+'</code></details></td></tr>'}).join('');draw(d.equity_series)}catch(e){$('connection').textContent='Błąd odczytu';$('connection').className='tag warn';$('error').textContent=String(e)}}refresh();setInterval(refresh,5000);
</script></body></html>"""


def get_status(path):
    if not path.is_file():
        raise FileNotFoundError(f"Brak księgi: {path}")
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2)
    db.row_factory = sqlite3.Row
    try:
        kinds = {
            str(row["kind"]): row["n"]
            for row in db.execute("SELECT kind,COUNT(*) n FROM events GROUP BY kind")
        }
        last = db.execute("SELECT time FROM events ORDER BY sequence DESC LIMIT 1").fetchone()
        decisions = []
        for row in db.execute(
            "SELECT sequence,time,branch,payload FROM events WHERE kind='decision' ORDER BY sequence DESC LIMIT 30"
        ):
            try:
                payload = json.loads(row["payload"])
            except (TypeError, ValueError):
                payload = {"unparsed": str(row["payload"])[:1000]}
            decisions.append(
                {
                    "sequence": row["sequence"],
                    "time": row["time"],
                    "branch": row["branch"],
                    "payload": payload,
                }
            )
        last_decision = db.execute(
            "SELECT time FROM events WHERE kind='decision' ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        action_counts = {}
        for row in db.execute(
            "SELECT branch,payload FROM events WHERE kind='decision' ORDER BY sequence DESC LIMIT 1000"
        ):
            try:
                p = json.loads(row["payload"])
                action = p.get("proposal", {}).get("action", p.get("action", "unknown"))
                key = str(row["branch"]) + ": " + str(action)
                action_counts[key] = action_counts.get(key, 0) + 1
            except (ValueError, TypeError, AttributeError):
                continue
        state_row = db.execute("SELECT payload FROM state LIMIT 1").fetchone()
        try:
            state = json.loads(state_row["payload"]) if state_row else {}
        except (ValueError, TypeError):
            state = {}
        series = {"A": [], "B": []}
        latest = {"A": None, "B": None}
        quotes = {}
        # Bounded scan, no table locks or full-ledger transfer.
        for row in db.execute(
            "SELECT time,payload FROM events WHERE kind='observation' ORDER BY sequence DESC LIMIT 2000"
        ):
            try:
                p = json.loads(row["payload"])
            except (TypeError, ValueError):
                continue
            eq = p.get("equity", {})
            if isinstance(eq, dict):
                for branch in ("A", "B"):
                    try:
                        val = float(eq[branch])
                    except (ValueError, TypeError, KeyError):
                        continue
                    if latest[branch] is None:
                        latest[branch] = val
                    series[branch].append(val)
            o = p.get("observation", {})
            if isinstance(o, dict) and o.get("kind") == "quote":
                sym = str(o.get("feed_id", o.get("symbol", "?")))
                if sym not in quotes:
                    quotes[sym] = {
                        "symbol": sym,
                        "bid": o.get("bid"),
                        "ask": o.get("ask"),
                        "time": o.get("available_at") or row["time"],
                    }
        # Do not claim count of fills without explicit ledger event types.
        fills = sum(
            count
            for kind, count in kinds.items()
            if kind in ("fill", "paper-fill", "execution-fill")
        )
        return {
            "total": sum(kinds.values()),
            "kinds": kinds,
            "fills": fills,
            "last_event_time": last["time"] if last else None,
            "decisions": decisions,
            "actions": action_counts,
            "last_decision_time": last_decision["time"] if last_decision else None,
            "latest_equity": latest,
            "equity_series": [
                {"branch": k, "values": list(reversed(v))[:: max(1, len(v) // 450)]}
                for k, v in series.items()
            ],
            "quotes": list(quotes.values())[:16],
            "state": state,
        }
    finally:
        db.close()


def ledger_diagnostics(data, now):
    diagnostics = []
    decision = data.get("last_decision_time")
    try:
        if not decision:
            diagnostics.append("Brak zapisanej decyzji głównego kontrolera")
        elif now - datetime.fromisoformat(decision.replace("Z", "+00:00")).timestamp() > 21600:
            diagnostics.append("Ostatnia decyzja głównego kontrolera jest starsza niż 6 h")
    except (ValueError, TypeError, AttributeError):
        diagnostics.append("Nie można zweryfikować czasu ostatniej decyzji")
    state = data.get("state", {})
    profiles = state.get("fee_profiles", {})
    for symbol, instrument in state.get("instruments", {}).items():
        profile = profiles.get(instrument.get("fee_profile"), {})
        if (
            profile.get("execution_costs_verified") is False
            or "assumptions" in str(profile.get("cost_evidence", "")).casefold()
        ):
            diagnostics.append(
                f"{symbol}: koszty FX/slippage UNKNOWN; publikacja opłat nie weryfikuje pełnego wykonania"
            )
        try:
            expiry = datetime.fromisoformat(profile["valid_until"].replace("Z", "+00:00"))
            if expiry.tzinfo is None:
                raise ValueError("Timezone required")
            if expiry.timestamp() <= now:
                diagnostics.append(f"{symbol}: profil opłat wygasł; wykonania paper zablokowane")
        except (KeyError, ValueError, TypeError, AttributeError):
            diagnostics.append(f"{symbol}: brak poprawnego terminu ważności opłat")
    return diagnostics


def extra_status(root, data=None, now=None):
    now = time.time() if now is None else now
    result = {"mission": {}, "gpu": {}, "diagnostics": []}
    if data is not None:
        result["diagnostics"].extend(ledger_diagnostics(data, now))
    active = root / "research/mission/active.json"
    run = None
    if active.is_file():
        try:
            m = json.loads(active.read_text())
            run = Path(m.get("run", ""))
            if not run.resolve().is_relative_to((root / "research/mission").resolve()):
                raise ValueError("Run outside mission directory")
            if run.is_dir():
                phase = run / "status.json"
                heartbeat = run / "learning/heartbeat.json"
                if phase.is_file():
                    result["mission"]["stage"] = json.loads(phase.read_text()).get(
                        "phase", "unknown"
                    )
                if heartbeat.is_file():
                    h = json.loads(heartbeat.read_text())
                    result["mission"]["learning"] = h.get("stage", "unknown")
                    updated = h.get("updated")
                    if (
                        not isinstance(updated, (int, float))
                        or now - updated > 600
                        or updated > now + 30
                    ):
                        result["diagnostics"].append(
                            "Heartbeat uczenia nie potwierdza świeżego postępu"
                        )
                    if h.get("reason"):
                        result["diagnostics"].append(str(h["reason"])[:300])
                else:
                    result["diagnostics"].append("Brak heartbeat uczenia bieżącej misji")
            else:
                result["diagnostics"].append("Brak katalogu bieżącej misji")
        except (OSError, ValueError, TypeError) as exc:
            result["diagnostics"].append("Mission status unavailable: " + type(exc).__name__)
    else:
        result["diagnostics"].append("Brak potwierdzonej aktywnej misji")
    receipts = sorted((run / "learning/cpu-pilots").glob("cpu-pilot-*.json")) if run else []
    statuses = []
    for path in receipts[-8:]:
        try:
            item = json.loads(path.read_text())
            statuses.append(
                str(item.get("device", "cpu")) + ":" + str(item.get("state", "unknown"))
            )
            if item.get("state") == "blocked":
                result["diagnostics"].append(str(item.get("reason", "Compute blocked"))[:180])
        except (OSError, ValueError):
            continue
    result["gpu"]["state"] = "Brak potwierdzenia treningu CUDA"
    result["mission"]["compute_jobs"] = statuses
    result["mission"]["training_evidence"] = (
        "Brak zweryfikowanych liczników optimizer/checkpoint w tym panelu"
    )
    result["diagnostics"].append(result["mission"]["training_evidence"])
    rules = root / "research/paper/spot-rules-status.json"
    if rules.is_file():
        try:
            item = json.loads(rules.read_text())
            if item.get("state") == "blocked":
                result["diagnostics"].append(
                    "Kraken spot rules: " + str(item.get("reason", "blocked"))[:160]
                )
        except (OSError, ValueError):
            result["diagnostics"].append("Nie można odczytać statusu reguł Kraken")
    return result


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            peer = ipaddress.ip_address(self.client_address[0])
            if not (peer.is_private or peer.is_loopback):
                self.send_error(403, "LAN only")
                return
        except ValueError:
            self.send_error(403, "Invalid client address")
            return
        route = urlsplit(self.path).path
        if route == "/":
            body = HTML.encode("utf-8")
            typ = "text/html; charset=utf-8"
            status = 200
        elif route == "/api/status":
            try:
                data = get_status(self.server.ledger)
                data.update(extra_status(self.server.root, data))
                body = json.dumps(data, ensure_ascii=False, default=str).encode()
                status = 200
            except Exception as e:
                body = json.dumps({"error": str(e)}).encode()
                status = 503
            typ = "application/json; charset=utf-8"
        else:
            body = b"Not found"
            typ = "text/plain"
            status = 404
        self.send_response(status)
        self.send_header("Content-Type", typ)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--root", type=Path, default=Path.home() / "ai-v100")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    print("Ledger:", args.ledger, flush=True)
    print(f"Read-only dashboard: http://{args.host}:{args.port}/", flush=True)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.ledger = args.ledger
    httpd.root = args.root
    httpd.serve_forever()


if __name__ == "__main__":
    main()
