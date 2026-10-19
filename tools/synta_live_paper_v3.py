#!/usr/bin/env python3
"""Synta live PAPER laboratory. Separate from canonical ledger; never sends orders."""

import argparse
import datetime
import html
import json
import math
import sqlite3
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from synta_accelerator_v1 import Accelerator


def now():
    return datetime.datetime.now(datetime.UTC).isoformat()


def fetch(url, timeout=8):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Synta-Paper-Lab/3.0 (public research)",
            "Accept": "application/json, application/xml, text/xml, */*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if r.status != 200:
            raise RuntimeError("HTTP " + str(r.status))
        return r.read(2000000)


def jget(url):
    return json.loads(fetch(url))


def safe(v):
    return html.escape(str(v), quote=True)


def engine_progress(quote_age, blocker, stale_after=180):
    """Describe observable quote-loop activity without implying model learning."""
    if quote_age is None:
        return {
            "state": "waiting",
            "label": "BRAK ODEBRANEGO NOTOWANIA",
            "detail": "Brak dowodu pracy pętli notowań; odpytywanie API panelu nie oznacza aktywności silnika.",
            "class": "warn",
        }
    if not math.isfinite(float(quote_age)) or quote_age < 0 or quote_age > stale_after:
        return {
            "state": "stale",
            "label": "ZASTÓJ — NOTOWANIE NIEAKTUALNE",
            "detail": f"Ostatnie notowanie ma {quote_age} s; limit świeżości to {stale_after} s.",
            "class": "red",
        }
    if blocker:
        return {
            "state": "blocked",
            "label": "NOTOWANIA ŚWIEŻE · PAPER ZABLOKOWANY",
            "detail": f"Pętla odbiera notowania, ale nie ma autoryzacji wykonania: {blocker}",
            "class": "warn",
        }
    return {
        "state": "active",
        "label": "NOTOWANIA ŚWIEŻE · PĘTLA PAPER AKTYWNA",
        "detail": "To potwierdza tylko pracę pętli notowań i symulacji; nie potwierdza treningu modelu ani zysku.",
        "class": "green",
    }


def mission_control_url(host_header, configured=None):
    """Return the canonical dashboard URL for the old paper-app entry point."""
    if configured:
        target = urllib.parse.urlsplit(configured)
        if (
            target.scheme not in {"http", "https"}
            or not target.hostname
            or target.username
            or target.password
            or target.query
            or target.fragment
        ):
            raise ValueError("Mission Control URL must be an HTTP(S) origin or path")
        return configured.rstrip("/") + "/"
    origin = urllib.parse.urlsplit("//" + (host_header or "127.0.0.1"))
    host = origin.hostname
    if not host or any(ord(char) < 33 for char in host) or "/" in host or "\\" in host:
        host = "127.0.0.1"
    authority = f"[{host}]" if ":" in host else host
    scheme = "https" if origin.scheme == "https" else "http"
    return f"{scheme}://{authority}:8765/"


PAGE = """<!doctype html><html lang="pl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Synta — Symulacja zarabiania i uczenie</title>
<style>body{background:#07111e;color:#e5f1ff;font:15px system-ui;margin:0;padding:22px}h1{font-size:25px}small,.muted{color:#92a9c3}section{background:#112239;border:1px solid #29415b;border-radius:12px;padding:16px;margin:12px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(225px,1fr));gap:12px}.num{font-size:29px;font-weight:700}.red{color:#ff8d8d}.green{color:#7de2ae}.warn{color:#ffd086}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px;text-align:left;border-bottom:1px solid #2b4055}.chartbox{height:300px;min-width:0}.chartbox canvas{display:block;width:100%;height:100%}.legend{display:flex;flex-wrap:wrap;gap:16px;margin:8px 0 12px;color:#c1d2e4;font-size:13px}.legend span{display:flex;align-items:center;gap:7px}.swatch{width:11px;height:11px;border-radius:50%;display:inline-block}.chart-values{display:flex;flex-wrap:wrap;gap:10px;margin-top:10px}.chart-value{padding:8px 11px;border-radius:8px;background:#0b1a2a;font-variant-numeric:tabular-nums}.chart-value strong{margin-left:6px}a{color:#9bcaff}@media(max-width:600px){body{padding:12px}.chartbox{height:250px}section{padding:12px}}</style></head><body>
<h1>SY N T A · Symulacja zarabiania i uczenie</h1><p class="muted">Eksperyment papierowy — zero prawdziwych zleceń. Wyniki mogą być ujemne. Wykres pokazuje wycenę brutto mark-to-bid.</p>
<section><strong id="state">Łączenie z silnikiem…</strong><div id="state-detail" class="muted"></div></section><div class="grid"><section><small>WALLET A · research/hold</small><div class="num" id="a">—</div><div id="ap"></div><small class="muted">Hipotezy momentum są oceniane, ale A nie ma reguły wejścia.</small></section><section><small>WALLET B · benchmark</small><div class="num" id="b">—</div><div id="bp"></div></section><section><small>Decyzje / minutę</small><div class="num" id="rpm">—</div><div id="count"></div></section><section><small>Źródła i nauka</small><div id="learning">—</div></section></div>
<section><h2>Wallet A/B · aktualizacja co 2 s</h2><div class="legend" aria-label="Legenda wykresu"><span><i class="swatch" style="background:#60bdff"></i>Portfel A · research/hold (bez wejścia)</span><span><i class="swatch" style="background:#f4b966"></i>Portfel B · benchmark</span></div><div class="chartbox"><canvas id="chart" role="img" aria-label="Wycena portfeli A i B w PLN według kolejnych notowań"></canvas></div><div class="chart-values" aria-live="polite"><span class="chart-value">Ostatni punkt A<strong id="chart-a">—</strong></span><span class="chart-value">Ostatni punkt B<strong id="chart-b">—</strong></span></div><p class="muted">Oś Y: wycena brutto w PLN · oś X: kolejność notowań (od najstarszego do najnowszego).</p><p class="muted" id="market"></p></section>
<div class="grid"><section><h2>Transakcje paper</h2><div id="fills"></div></section><section><h2>Eksperymenty · walidacja poza próbką</h2><div id="tests"></div></section></div>
<section><h2>News / źródła</h2><div id="news"></div></section><section><h2>Diagnostyka</h2><pre id="debug" style="white-space:pre-wrap"></pre></section>
<script>
const E=x=>document.getElementById(x), fmt=x=>Number(x).toLocaleString('pl-PL',{minimumFractionDigits:2,maximumFractionDigits:2})+' PLN',esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function plot(data){window.lastChartData=data;let c=E('chart'),box=c.parentElement,rect=box.getBoundingClientRect(),scale=Math.min(window.devicePixelRatio||1,2),w=Math.max(320,Math.round(rect.width*scale)),h=Math.max(180,Math.round(rect.height*scale));if(c.width!==w||c.height!==h){c.width=w;c.height=h}let g=c.getContext('2d'),p={l:82,r:18,t:18,b:36};g.clearRect(0,0,w,h);g.font=(12*scale)+'px system-ui';data=(data||[]).filter(r=>r.a!=null&&r.b!=null&&Number.isFinite(Number(r.a))&&Number.isFinite(Number(r.b)));if(!data.length){g.fillStyle='#92a9c3';g.fillText('Oczekiwanie na notowania…',p.l,42*scale);E('chart-a').textContent='—';E('chart-b').textContent='—';return}let values=data.flatMap(r=>[Number(r.a),Number(r.b)]),lo=Math.min(...values),hi=Math.max(...values),pad=Math.max((hi-lo)*.12,2);lo-=pad;hi+=pad;let x=i=>p.l+(w-p.l-p.r)*i/Math.max(1,data.length-1),y=v=>p.t+(h-p.t-p.b)*(hi-v)/(hi-lo);for(let j=0;j<5;j++){let yy=p.t+(h-p.t-p.b)*j/4,val=hi-(hi-lo)*j/4;g.strokeStyle='#2b4055';g.lineWidth=scale;g.beginPath();g.moveTo(p.l,yy);g.lineTo(w-p.r,yy);g.stroke();g.fillStyle='#b7c9dc';g.textAlign='right';g.fillText(fmt(val),p.l-8*scale,yy+4*scale)}for(let [key,color] of [['a','#60bdff'],['b','#f4b966']]){g.beginPath();g.strokeStyle=color;g.lineWidth=2.5*scale;data.forEach((r,i)=>{let xx=x(i),yy=y(Number(r[key]));i?g.lineTo(xx,yy):g.moveTo(xx,yy)});g.stroke();let last=data.at(-1),lastX=x(data.length-1),lastY=y(Number(last[key]));g.fillStyle=color;g.beginPath();g.arc(lastX,lastY,4*scale,0,Math.PI*2);g.fill()}g.textAlign='left';g.fillStyle='#b7c9dc';g.fillText('1',p.l,h-9*scale);g.textAlign='center';g.fillText('Notowanie',p.l+(w-p.l-p.r)/2,h-9*scale);g.textAlign='right';g.fillText(String(data.length),w-p.r,h-9*scale);let last=data.at(-1);E('chart-a').textContent=fmt(last.a);E('chart-b').textContent=fmt(last.b)}
async function refresh(){try{let r=await fetch('/api/status',{cache:'no-store'});if(!r.ok)throw Error('HTTP '+r.status);let d=await r.json();let s=d.engine_progress||{label:d.blocker?'BLOKADA: '+d.blocker:'Stan silnika niezweryfikowany',detail:'Brak danych o świeżości notowania.',class:'warn'};E('state').textContent=s.label;E('state').className=s.class;E('state-detail').textContent=s.detail;for(let k of ['a','b']){let x=d.accounts[k];E(k).textContent=fmt(x.equity);E(k+'p').textContent='P&L '+fmt(x.pnl)+' · '+x.pnl_pct.toFixed(3)+'% · gotówka '+fmt(x.cash)+' · BTC '+x.qty.toFixed(7);E(k+'p').className=x.pnl>=0?'green':'red'}E('rpm').textContent=d.decisions_last_min;E('count').textContent='Łącznie '+d.decisions+' ocen · '+d.fills.length+' filli · '+d.ticks+' notowań';E('learning').textContent='Ocena hipotez/backtest: '+d.hypotheses_scored+'/'+d.hypotheses+' · trening wag: brak · GPU: '+d.gpu;E('market').textContent='BTC bid/ask '+(d.quote?d.quote.bid.toFixed(2)+' / '+d.quote.ask.toFixed(2)+' PLN':'brak')+' · kurs USD/PLN '+d.fx+' · ostatnie '+(d.quote?.ts||'brak')+' · wiek '+(d.quote_age??'brak')+' s';plot(d.chart);
E('fills').innerHTML=d.fills.length?'<table><tr><th>UTC</th><th>Strategia</th><th>Akcja</th><th>PLN</th></tr>'+d.fills.map(x=>'<tr><td>'+esc(x.ts)+'</td><td>'+esc(x.branch)+'</td><td>'+esc(x.side)+'</td><td>'+x.amount.toFixed(2)+'</td></tr>').join('')+'</table>':'Brak filli — brak wymuszonych zakupów';
E('tests').innerHTML=d.tests.length?'<table><tr><th>Okno</th><th>Próg</th><th>Walidacja netto</th><th>Transakcje</th></tr>'+d.tests.slice(0,12).map(x=>'<tr><td>'+x.window+'</td><td>'+x.threshold+'%</td><td>'+x.test_pct.toFixed(3)+'%</td><td>'+x.trades+'</td></tr>').join('')+'</table>':'Ładowanie historii OHLC…';
E('news').innerHTML=d.news.length?d.news.map(x=>'<p><small>'+esc(x.date)+'</small> · '+esc(x.title)+' <a target="_blank" rel="noopener" href="'+esc(x.url)+'">źródło</a></p>').join(''):'Brak potwierdzonych newsów';
E('debug').textContent=JSON.stringify({hipotezy_top10:d.hypotheses_rank,quote_age_seconds:d.quote_age,fees:d.fee_evidence,fx_date:d.fx_date,news_error:d.news_error,history_error:d.history_error,warning:d.warning},null,2)}catch(e){E('state').textContent='Błąd panelu: '+e}}
refresh();setInterval(refresh,2000);window.addEventListener('resize',()=>{if(window.lastChartData)plot(window.lastChartData)});
</script></body></html>"""


class Lab:
    def __init__(self, root):
        self.root = root
        self.lock = threading.RLock()
        self.error = "Oczekiwanie na pierwsze notowanie"
        self.fx = None
        self.fx_date = None
        self.fx_checked = 0
        self.fx_observed = 0
        self.fx_error = "No successful FX observation"
        self.news = []
        self.news_error = None
        self.news_checked = 0
        self.history_error = None
        self.hist = []
        self.hist_checked = 0
        self.quote = None
        self.last_quote = 0
        self.fee = None
        self.fee_ok = False
        self.last_fee = 0
        p = root / "research/experiments/live-paper-v3.sqlite3"
        p.parent.mkdir(parents=True, exist_ok=True)
        self.accelerator = None
        self.promoted_path = root / "research/experiments/paper-a-promotion.json"
        self.promoted = None
        self.legacy_promotion = None
        self.last_strategy_candle = None
        self.strategy_opened = 0
        try:
            # Keep the historical record, but a repeated holdout is not an
            # independently validated model promotion.
            self.legacy_promotion = json.loads(self.promoted_path.read_text())
        except Exception:
            pass
        self.db = sqlite3.connect(p, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""CREATE TABLE IF NOT EXISTS accounts(branch TEXT PRIMARY KEY,cash REAL,qty REAL,entry REAL,peak REAL);
        CREATE TABLE IF NOT EXISTS ticks(id INTEGER PRIMARY KEY,ts TEXT,bid REAL,ask REAL,a REAL,b REAL);
        CREATE TABLE IF NOT EXISTS decisions(id INTEGER PRIMARY KEY,ts REAL,branch TEXT,action TEXT,reason TEXT);
        CREATE TABLE IF NOT EXISTS fills(id INTEGER PRIMARY KEY,ts TEXT,branch TEXT,side TEXT,amount REAL,qty REAL,price REAL,fee REAL,reason TEXT);
        CREATE TABLE IF NOT EXISTS tests(id INTEGER PRIMARY KEY,ts TEXT,window INTEGER,threshold REAL,train_pct REAL,test_pct REAL,trades INTEGER);""")
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(ticks)")}
        if "valuation_basis" not in columns:
            self.db.execute(
                "ALTER TABLE ticks ADD COLUMN valuation_basis TEXT NOT NULL DEFAULT 'legacy-liquidation-estimate-v1'"
            )
        for b in "ab":
            self.db.execute(
                "INSERT OR IGNORE INTO accounts VALUES(?,?,?,?,?)", (b, 10000.0, 0.0, 0.0, 10000.0)
            )
        self.db.execute("""CREATE TABLE IF NOT EXISTS hypotheses(
 id INTEGER PRIMARY KEY, tick_id INTEGER NOT NULL, ts TEXT NOT NULL,
 variant TEXT NOT NULL, window INTEGER NOT NULL, threshold REAL NOT NULL,
 signal INTEGER NOT NULL, price REAL NOT NULL, future_price REAL,
 correct INTEGER)""")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_hyp_tick ON hypotheses(tick_id)")
        self.db.commit()
        self.last_test = 0

    def account(self, b):
        return self.db.execute(
            "SELECT cash,qty,entry,peak FROM accounts WHERE branch=?", (b,)
        ).fetchone()

    def equity(self, b, bid):
        cash, qty, _, _ = self.account(b)
        # Mark-to-market does not execute a hypothetical liquidation.
        return cash + qty * bid

    def evidence(self):
        try:
            p = json.loads((self.root / "research/paper/fee-source-status.json").read_text())
            self.fee = float(p["taker_fee_bps"])
            self.fee_ok = (
                p["state"] == "parsed"
                and math.isfinite(self.fee)
                and 0 < self.fee < 500
                and 0 <= time.time() - float(p["updated"]) < 86400
            )
        except Exception:
            self.fee_ok = False

    def fx_current(self):
        if self.fx is None or self.fx_error or not self.fx_date:
            return False
        try:
            date = datetime.date.fromisoformat(self.fx_date)
        except (TypeError, ValueError):
            return False
        age_days = (datetime.datetime.now(datetime.UTC).date() - date).days
        return (
            math.isfinite(self.fx)
            and 2 < self.fx < 10
            and 0 <= age_days <= 5
            and 0 <= time.time() - self.fx_observed < 3600
        )

    def fx_refresh(self):
        if time.time() - self.fx_checked < 3600:
            return
        self.fx_checked = time.time()
        try:
            d = jget("https://api.frankfurter.dev/v1/latest?base=USD&symbols=PLN")
            val = float(d["rates"]["PLN"])
            date = datetime.date.fromisoformat(d["date"])
            age_days = (datetime.datetime.now(datetime.UTC).date() - date).days
            if not math.isfinite(val) or not 2 < val < 10 or not 0 <= age_days <= 5:
                raise ValueError("Kurs referencyjny zbyt stary")
            self.fx = val
            self.fx_date = str(date)
            self.fx_observed = time.time()
            self.fx_error = None
        except Exception as e:
            self.fx_error = "USD/PLN: " + str(e)[:130]
            self.error = self.fx_error

    def news_refresh(self):
        if time.time() - self.news_checked < 300:
            return
        self.news_checked = time.time()
        try:
            url = "https://news.google.com/rss/search?q=bitcoin+OR+ethereum+when%3A1d&hl=en-US&gl=US&ceid=US%3Aen"
            root = ET.fromstring(fetch(url))
            items = []
            for i in root.findall(".//item")[:10]:
                title = i.findtext("title", "")
                link = i.findtext("link", "")
                date = i.findtext("pubDate", "")
                if link.startswith("https://"):
                    items.append(dict(title=title, url=link, date=date))
            self.news = items
            self.news_error = None
        except Exception as e:
            self.news_error = str(e)[:150]

    def history_refresh(self):
        if time.time() - self.hist_checked < 300:
            return
        self.hist_checked = time.time()
        try:
            d = jget("https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=60")
            if d.get("error"):
                raise ValueError(str(d["error"]))
            k = next(k for k in d["result"] if k != "last")
            # Exclude unfinished candle; OHLC close is USD. FX reference is a constant approximation.
            self.hist = [float(x[4]) for x in d["result"][k][:-1] if float(x[4]) > 0]
            self.history_error = None
        except Exception as e:
            self.history_error = str(e)[:150]

    def fill(self, b, side, price, reason):
        # Recheck at execution, including sells; never invent a fallback fee.
        self.evidence()
        if not self.fee_ok or not self.fx_current():
            self.error = "UNKNOWN/STALE: fee or FX evidence; paper fill blocked"
            return False
        if not math.isfinite(price) or price <= 0:
            raise ValueError("Invalid paper fill price")
        cash, qty, entry, peak = self.account(b)
        fee = (self.fee + 30) / 10000
        if side == "BUY" and qty == 0 and self.fee_ok:
            budget = min(500.0, cash * 0.05)
            if budget < 25:
                return False
            q = budget / (price * (1 + fee))
            self.db.execute(
                "UPDATE accounts SET cash=?,qty=?,entry=? WHERE branch=?",
                (cash - budget, q, price, b),
            )
            amount = budget
        elif side == "SELL" and qty > 0:
            amount = qty * price * (1 - fee)
            self.db.execute(
                "UPDATE accounts SET cash=?,qty=0,entry=0 WHERE branch=?", (cash + amount, b)
            )
            q = qty
        else:
            return False
        self.db.execute(
            "INSERT INTO fills(ts,branch,side,amount,qty,price,fee,reason) VALUES(?,?,?,?,?,?,?,?)",
            (now(), b, side, amount, q, price, fee, reason),
        )
        return True

    @staticmethod
    def replay(prices, window, threshold):
        cash = 10000.0
        qty = 0.0
        trades = 0
        for i in range(window, len(prices)):
            p = prices[i]
            signal = prices[i] / prices[i - window] - 1
            if not qty and signal > threshold:
                budget = min(500.0, cash * 0.05)
                qty = budget / (p * 1.011)
                cash -= budget
                trades += 1
            elif qty and (signal < -0.012 or p < prices[i - window] * 0.965):
                cash += qty * p * 0.989
                qty = 0.0
                trades += 1
        return (cash + qty * prices[-1] * 0.989 - 10000.0) / 100.0, trades

    def backtests(self):
        if time.time() - self.last_test < 60 or len(self.hist) < 100:
            return
        self.last_test = time.time()
        prices = self.hist
        split = int(len(prices) * 0.65)
        train = prices[:split]
        test = prices[split - 30 :]
        self.db.execute("DELETE FROM tests")
        for w in (5, 10, 15, 20, 30, 45):
            for t in (0.005, 0.012, 0.02, 0.026, 0.035):
                a, _ = self.replay(train, w, t)
                b, n = self.replay(test, w, t)
                self.db.execute(
                    "INSERT INTO tests(ts,window,threshold,train_pct,test_pct,trades) VALUES(?,?,?,?,?,?)",
                    (now(), w, t * 100, a, b, n),
                )
        self.db.commit()

    def step(self):
        self.fx_refresh()
        self.news_refresh()
        self.history_refresh()
        if time.time() - self.last_fee > 60:
            self.evidence()
            self.last_fee = time.time()
        if self.fx is None:
            self.error = "Brak referencyjnego USD/PLN"
            return
        d = jget("https://api.kraken.com/0/public/Ticker?pair=XBTUSD")
        if d.get("error"):
            raise ValueError(str(d["error"]))
        q = next(iter(d["result"].values()))
        bid = float(q["b"][0]) * self.fx
        ask = float(q["a"][0]) * self.fx
        if not (math.isfinite(bid) and math.isfinite(ask) and 0 < bid <= ask):
            raise ValueError("Błędny spread")
        ts = now()
        self.quote = {"ts": ts, "bid": bid, "ask": ask}
        self.last_quote = time.time()
        self.error = (
            None
            if self.fee_ok and self.fx_current()
            else ("UNKNOWN/STALE: fee or FX evidence; paper fills blocked")
        )
        history = [
            x[0] for x in self.db.execute("SELECT (bid+ask)/2 FROM ticks ORDER BY id DESC LIMIT 45")
        ][::-1]
        history.append((bid + ask) / 2)
        for b in "ab":
            cash, qty, entry, peak = self.account(b)
            eq = self.equity(b, bid)
            peak = max(eq, peak)
            self.db.execute("UPDATE accounts SET peak=? WHERE branch=?", (peak, b))
            action = "HOLD"
            reason = "Brak sygnału netto"
            if qty and (eq < peak * 0.90 or bid < entry * 0.965):
                action = "SELL"
                reason = "Limit straty"
                if not self.fill(b, action, bid, reason):
                    action, reason = "HOLD", self.error
            elif (
                b == "b"
                and qty == 0
                and self.fee_ok
                and self.db.execute("SELECT COUNT(*) FROM fills WHERE branch='b'").fetchone()[0]
                == 0
            ):
                action = "BUY"
                reason = "Benchmark 5%"
                if not self.fill(b, action, ask, reason):
                    action, reason = "HOLD", self.error
            elif b == "a":
                reason = "BRAK POTWIERDZONEJ PRZEWAGI: independent validation required — HOLD"
            self.db.execute(
                "INSERT INTO decisions(ts,branch,action,reason) VALUES(?,?,?,?)",
                (time.time(), b, action, reason),
            )
        self.db.execute(
            "INSERT INTO ticks(ts,bid,ask,a,b,valuation_basis) VALUES(?,?,?,?,?,?)",
            (ts, bid, ask, self.equity("a", bid), self.equity("b", bid), "gross-mark-to-bid-v2"),
        )
        self.db.execute("DELETE FROM ticks WHERE id < (SELECT MAX(id)-100000 FROM ticks)")
        self.db.execute("DELETE FROM decisions WHERE ts<?", (time.time() - 86400,))
        # 48 competing forecasts per tick. Each is evaluated only after 30 NEW ticks.
        tick_id = self.db.execute("SELECT MAX(id) FROM ticks").fetchone()[0]
        if len(history) >= 20:
            for w in (3, 5, 8, 12, 16, 19):
                move = history[-1] / history[-1 - w] - 1
                for threshold in (0.0005, 0.001, 0.002, 0.004):
                    for variant, direction in (("trend", 1), ("reversal", -1)):
                        signal = (
                            1
                            if move * direction > threshold
                            else (-1 if move * direction < -threshold else 0)
                        )
                        self.db.execute(
                            "INSERT INTO hypotheses(tick_id,ts,variant,window,threshold,signal,price) VALUES(?,?,?,?,?,?,?)",
                            (tick_id, ts, variant, w, threshold, signal, bid),
                        )
        self.db.execute(
            """UPDATE hypotheses SET future_price=?,
          correct=CASE WHEN signal=0 THEN NULL WHEN (?-price)*signal>0 THEN 1 ELSE 0 END
          WHERE future_price IS NULL AND tick_id<=?""",
            (bid, bid, tick_id - 30),
        )
        self.db.execute("DELETE FROM hypotheses WHERE tick_id<?", (tick_id - 10000,))
        self.db.commit()
        self.backtests()

    def status(self):
        with self.lock:
            q = self.quote
            bid = q["bid"] if q else None
            ac = {}
            for b in "ab":
                cash, qty, entry, peak = self.account(b)
                eq = self.equity(b, bid) if bid else cash
                ac[b] = {
                    "cash": round(cash, 2),
                    "qty": qty,
                    "equity": round(eq, 2),
                    "pnl": round(eq - 10000, 2),
                    "pnl_pct": (eq / 10000 - 1) * 100,
                }
            fills = [
                dict(ts=t, branch=b, side=s, amount=a)
                for t, b, s, a in self.db.execute(
                    "SELECT ts,branch,side,amount FROM fills ORDER BY id DESC LIMIT 30"
                )
            ]
            tests = [
                dict(window=w, threshold=t, train_pct=a, test_pct=b, trades=n)
                for w, t, a, b, n in self.db.execute(
                    "SELECT window,threshold,train_pct,test_pct,trades FROM tests ORDER BY test_pct DESC LIMIT 30"
                )
            ]
            chart = [
                dict(ts=t, a=a, b=b)
                for t, a, b in self.db.execute(
                    "SELECT ts,a,b FROM ticks WHERE valuation_basis='gross-mark-to-bid-v2' ORDER BY id DESC LIMIT 250"
                )
            ][::-1]
            h_total, h_scored = self.db.execute(
                "SELECT COUNT(*),COUNT(future_price) FROM hypotheses"
            ).fetchone()
            h_rank = [
                dict(
                    variant=v,
                    window=w,
                    threshold=round(t * 100, 3),
                    n=n,
                    accuracy=round(100 * hits / n, 2),
                )
                for v, w, t, n, hits in self.db.execute("""SELECT variant,window,threshold,COUNT(*),SUM(correct)
                    FROM hypotheses WHERE correct IS NOT NULL
                    GROUP BY variant,window,threshold HAVING COUNT(*)>=10
                    ORDER BY 5*1.0/4 DESC LIMIT 48""")
            ]
            h_rank.sort(key=lambda x: (-x["accuracy"], -x["n"]))
            return dict(
                time=now(),
                accounts=ac,
                quote=q,
                fx=self.fx,
                fx_date=self.fx_date,
                fx_evidence={
                    "state": "CURRENT" if self.fx_current() else "UNKNOWN/STALE",
                    "source": "https://api.frankfurter.dev/v1/latest?base=USD&symbols=PLN",
                    "observed_at": self.fx_observed or None,
                    "kind": "reference rate, not executable conversion",
                    "error": self.fx_error,
                },
                blocker=self.error,
                fee_evidence={"verified": self.fee_ok, "taker_bps": self.fee},
                decisions=self.db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0],
                decisions_last_min=self.db.execute(
                    "SELECT COUNT(*) FROM decisions WHERE ts>?", (time.time() - 60,)
                ).fetchone()[0],
                ticks=self.db.execute("SELECT COUNT(*) FROM ticks").fetchone()[0],
                fills=fills,
                tests=tests,
                chart=chart,
                price_chart=[
                    dict(ts=t, mid=round((bid + ask) / 2, 4))
                    for t, bid, ask in self.db.execute(
                        "SELECT ts,bid,ask FROM ticks ORDER BY id DESC LIMIT 500"
                    )
                ][::-1],
                news=self.news,
                news_error=self.news_error,
                history_error=self.history_error,
                quote_age=round(time.time() - self.last_quote, 1) if self.last_quote else None,
                engine_progress=engine_progress(
                    round(time.time() - self.last_quote, 1) if self.last_quote else None,
                    self.error,
                ),
                gpu="NIEPODŁĄCZONE — brak potwierdzenia CUDA",
                promoted_strategy=self.promoted,
                legacy_promotion=self.legacy_promotion,
                valuation_basis="gross mark-to-bid using reference FX; no hypothetical exit fee",
                chart_scope="gross-mark-to-bid-v2 only; legacy liquidation estimates retained separately",
                legacy_curve_points=self.db.execute(
                    "SELECT COUNT(*) FROM ticks WHERE valuation_basis='legacy-liquidation-estimate-v1'"
                ).fetchone()[0],
                execution_costs={
                    "verified": False,
                    "scope": "legacy paper scenario: sourced taker fee plus assumed 30bps; depth/slippage and executable FX costs unverified",
                    "promotion_allowed": False,
                },
                hypotheses=h_total,
                hypotheses_scored=h_scored,
                hypotheses_rank=h_rank[:10],
                warning="Izolowana księga paper v3, historyczne scenariusze. Kraken OHLC 60m; referencyjny FX. Pełne koszty niezbadane: BRAK POTWIERDZONEJ PRZEWAGI, promocja A zablokowana. News tylko do odczytu.",
            )

    def loop(self):
        while True:
            try:
                with self.lock:
                    self.step()
            except Exception as e:
                with self.lock:
                    self.error = type(e).__name__ + ": " + str(e)[:150]
            time.sleep(2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path.home() / "ai-v100")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--mission-control-url", default=None)
    args = p.parse_args()
    lab = Lab(args.root)
    accelerator = Accelerator(args.root)
    lab.accelerator = accelerator

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                try:
                    location = mission_control_url(
                        self.headers.get("Host", "127.0.0.1"), args.mission_control_url
                    )
                except ValueError as error:
                    self.send_error(500, str(error))
                    return
                self.send_response(302)
                self.send_header("Location", location)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return
            elif path == "/api/status":
                data = json.dumps(lab.status(), ensure_ascii=False).encode()
                typ = "application/json; charset=utf-8"
            elif path == "/api/accelerator":
                data = json.dumps(accelerator.snapshot(), ensure_ascii=False).encode()
                typ = "application/json; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", typ)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    threading.Thread(target=lab.loop, daemon=True).start()
    threading.Thread(target=accelerator.loop, daemon=True).start()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
