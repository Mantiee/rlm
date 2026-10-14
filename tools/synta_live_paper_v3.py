#!/usr/bin/env python3
"""Synta live PAPER laboratory. Separate from canonical ledger; never sends orders."""
import argparse, collections, datetime, hashlib, html, json, math, os, sqlite3, threading, time
import urllib.request, urllib.parse, xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from synta_accelerator_v1 import Accelerator

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def fetch(url, timeout=8):
    req=urllib.request.Request(url,headers={"User-Agent":"Synta-Paper-Lab/3.0 (public research)","Accept":"application/json, application/xml, text/xml, */*"})
    with urllib.request.urlopen(req,timeout=timeout) as r:
        if r.status!=200: raise RuntimeError("HTTP "+str(r.status))
        return r.read(2000000)
def jget(url): return json.loads(fetch(url))
def safe(v): return html.escape(str(v),quote=True)

PAGE="""<!doctype html><html lang="pl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Synta — Symulacja zarabiania i uczenie</title>
<style>body{background:#07111e;color:#e5f1ff;font:15px system-ui;margin:0;padding:22px}h1{font-size:25px}small,.muted{color:#92a9c3}section{background:#112239;border:1px solid #29415b;border-radius:12px;padding:16px;margin:12px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(225px,1fr));gap:12px}.num{font-size:29px;font-weight:700}.red{color:#ff8d8d}.green{color:#7de2ae}.warn{color:#ffd086}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px;text-align:left;border-bottom:1px solid #2b4055}canvas{width:100%;height:220px}a{color:#9bcaff}</style></head><body>
<h1>SY N T A · Symulacja zarabiania i uczenie</h1><p class="muted">Eksperyment papierowy — zero prawdziwych zleceń. Wyniki mogą być ujemne. Wycena netto po kosztach wyjścia.</p>
<section id="state">Łączenie z silnikiem…</section><div class="grid"><section><small>WALLET A · momentum</small><div class="num" id="a">—</div><div id="ap"></div></section><section><small>WALLET B · benchmark</small><div class="num" id="b">—</div><div id="bp"></div></section><section><small>Decyzje / minutę</small><div class="num" id="rpm">—</div><div id="count"></div></section><section><small>Źródła i nauka</small><div id="learning">—</div></section></div>
<section><h2>Wallet A/B · aktualizacja co 2 s</h2><canvas id="chart" width="950" height="240"></canvas><p class="muted" id="market"></p></section>
<div class="grid"><section><h2>Transakcje paper</h2><div id="fills"></div></section><section><h2>Eksperymenty · walidacja poza próbką</h2><div id="tests"></div></section></div>
<section><h2>News / źródła</h2><div id="news"></div></section><section><h2>Diagnostyka</h2><pre id="debug" style="white-space:pre-wrap"></pre></section>
<script>
const E=x=>document.getElementById(x), fmt=x=>Number(x).toLocaleString('pl-PL',{minimumFractionDigits:2,maximumFractionDigits:2})+' PLN',esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function plot(data){let c=E('chart'),g=c.getContext('2d');g.fillStyle='#07111e';g.fillRect(0,0,c.width,c.height);if(!data.length)return;let v=data.flatMap(r=>[r.a,r.b]),lo=Math.min(...v)-2,hi=Math.max(...v)+2;for(let [k,color] of [['a','#60bdff'],['b','#f4b966']]){g.strokeStyle=color;g.lineWidth=2;g.beginPath();data.forEach((r,i)=>{let x=30+(c.width-55)*i/Math.max(1,data.length-1),y=12+(c.height-35)*(hi-r[k])/(hi-lo);i?g.lineTo(x,y):g.moveTo(x,y)});g.stroke();g.fillStyle=color;g.fillText(k.toUpperCase()+': '+fmt(data.at(-1)[k]),40+(k==='b'?300:0),c.height-8)}}
async function refresh(){try{let r=await fetch('/api/status',{cache:'no-store'});if(!r.ok)throw Error('HTTP '+r.status);let d=await r.json();E('state').textContent=d.blocker?'BLOKADA: '+d.blocker:'● Odbieranie notowań · '+d.time;E('state').className=d.blocker?'warn':'green';for(let k of ['a','b']){let x=d.accounts[k];E(k).textContent=fmt(x.equity);E(k+'p').textContent='P&L '+fmt(x.pnl)+' · '+x.pnl_pct.toFixed(3)+'% · gotówka '+fmt(x.cash)+' · BTC '+x.qty.toFixed(7);E(k+'p').className=x.pnl>=0?'green':'red'}E('rpm').textContent=d.decisions_last_min;E('count').textContent='Łącznie '+d.decisions+' ocen · '+d.fills.length+' filli · '+d.ticks+' notowań';E('learning').textContent='Backtesty '+d.tests.length+' · hipotezy '+d.hypotheses+' · sprawdzone '+d.hypotheses_scored+' · GPU: '+d.gpu;E('market').textContent='BTC bid/ask '+(d.quote?d.quote.bid.toFixed(2)+' / '+d.quote.ask.toFixed(2)+' PLN':'brak')+' · kurs USD/PLN '+d.fx+' · ostatnie '+(d.quote?.ts||'brak');plot(d.chart);
E('fills').innerHTML=d.fills.length?'<table><tr><th>UTC</th><th>Strategia</th><th>Akcja</th><th>PLN</th></tr>'+d.fills.map(x=>'<tr><td>'+esc(x.ts)+'</td><td>'+esc(x.branch)+'</td><td>'+esc(x.side)+'</td><td>'+x.amount.toFixed(2)+'</td></tr>').join('')+'</table>':'Brak filli — brak wymuszonych zakupów';
E('tests').innerHTML=d.tests.length?'<table><tr><th>Okno</th><th>Próg</th><th>Walidacja netto</th><th>Transakcje</th></tr>'+d.tests.slice(0,12).map(x=>'<tr><td>'+x.window+'</td><td>'+x.threshold+'%</td><td>'+x.test_pct.toFixed(3)+'%</td><td>'+x.trades+'</td></tr>').join('')+'</table>':'Ładowanie historii OHLC…';
E('news').innerHTML=d.news.length?d.news.map(x=>'<p><small>'+esc(x.date)+'</small> · '+esc(x.title)+' <a target="_blank" rel="noopener" href="'+esc(x.url)+'">źródło</a></p>').join(''):'Brak potwierdzonych newsów';
E('debug').textContent=JSON.stringify({hipotezy_top10:d.hypotheses_rank,quote_age_seconds:d.quote_age,fees:d.fee_evidence,fx_date:d.fx_date,news_error:d.news_error,history_error:d.history_error,warning:d.warning},null,2)}catch(e){E('state').textContent='Błąd panelu: '+e}}
refresh();setInterval(refresh,2000);
</script></body></html>"""

class Lab:
    def __init__(self,root):
        self.root=root;self.lock=threading.RLock();self.error="Oczekiwanie na pierwsze notowanie";self.fx=None;self.fx_date=None;self.fx_checked=0;self.news=[];self.news_error=None;self.news_checked=0;self.history_error=None;self.hist=[];self.hist_checked=0;self.quote=None;self.last_quote=0;self.fee=None;self.fee_ok=False;self.last_fee=0
        p=root/'research/experiments/live-paper-v3.sqlite3';p.parent.mkdir(parents=True,exist_ok=True)
        self.accelerator=None;self.promoted_path=root/'research/experiments/paper-a-promotion.json';self.promoted=None;self.last_strategy_candle=None;self.strategy_opened=0
        try:self.promoted=json.loads(self.promoted_path.read_text())
        except Exception:pass
        self.db=sqlite3.connect(p,check_same_thread=False);self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""CREATE TABLE IF NOT EXISTS accounts(branch TEXT PRIMARY KEY,cash REAL,qty REAL,entry REAL,peak REAL);
        CREATE TABLE IF NOT EXISTS ticks(id INTEGER PRIMARY KEY,ts TEXT,bid REAL,ask REAL,a REAL,b REAL);
        CREATE TABLE IF NOT EXISTS decisions(id INTEGER PRIMARY KEY,ts REAL,branch TEXT,action TEXT,reason TEXT);
        CREATE TABLE IF NOT EXISTS fills(id INTEGER PRIMARY KEY,ts TEXT,branch TEXT,side TEXT,amount REAL,qty REAL,price REAL,fee REAL,reason TEXT);
        CREATE TABLE IF NOT EXISTS tests(id INTEGER PRIMARY KEY,ts TEXT,window INTEGER,threshold REAL,train_pct REAL,test_pct REAL,trades INTEGER);""")
        for b in 'ab':self.db.execute("INSERT OR IGNORE INTO accounts VALUES(?,?,?,?,?)",(b,10000.,0.,0.,10000.))
        self.db.execute("""CREATE TABLE IF NOT EXISTS hypotheses(
 id INTEGER PRIMARY KEY, tick_id INTEGER NOT NULL, ts TEXT NOT NULL,
 variant TEXT NOT NULL, window INTEGER NOT NULL, threshold REAL NOT NULL,
 signal INTEGER NOT NULL, price REAL NOT NULL, future_price REAL,
 correct INTEGER)""")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_hyp_tick ON hypotheses(tick_id)")
        self.db.commit();self.last_test=0
    def account(self,b):return self.db.execute("SELECT cash,qty,entry,peak FROM accounts WHERE branch=?",(b,)).fetchone()
    def equity(self,b,bid):
        cash,qty,_,_=self.account(b)
        return cash+qty*bid*(1-(self.fee+30)/10000 if self.fee_ok else .989)
    def evidence(self):
        try:
            p=json.loads((self.root/'research/paper/fee-source-status.json').read_text())
            self.fee=float(p['taker_fee_bps'])
            self.fee_ok=p['state']=='parsed' and 0<self.fee<500 and time.time()-float(p['updated'])<86400
        except Exception:self.fee_ok=False
    def fx_refresh(self):
        if time.time()-self.fx_checked<3600:return
        self.fx_checked=time.time()
        try:
            d=jget('https://api.frankfurter.dev/v1/latest?base=USD&symbols=PLN')
            val=float(d['rates']['PLN']);date=datetime.date.fromisoformat(d['date'])
            if not 2<val<10 or (datetime.date.today()-date).days>5:raise ValueError("Kurs referencyjny zbyt stary")
            self.fx=val;self.fx_date=str(date)
        except Exception as e:
            if self.fx is None:self.error="USD/PLN: "+str(e)[:130]
    def news_refresh(self):
        if time.time()-self.news_checked<300:return
        self.news_checked=time.time()
        try:
            url="https://news.google.com/rss/search?q=bitcoin+OR+ethereum+when%3A1d&hl=en-US&gl=US&ceid=US%3Aen"
            root=ET.fromstring(fetch(url))
            items=[]
            for i in root.findall('.//item')[:10]:
                title=i.findtext('title','');link=i.findtext('link','');date=i.findtext('pubDate','')
                if link.startswith('https://'):items.append(dict(title=title,url=link,date=date))
            self.news=items;self.news_error=None
        except Exception as e:self.news_error=str(e)[:150]
    def history_refresh(self):
        if time.time()-self.hist_checked<300:return
        self.hist_checked=time.time()
        try:
            d=jget('https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=60')
            if d.get('error'):raise ValueError(str(d['error']))
            k=next(k for k in d['result'] if k!='last')
            # Exclude unfinished candle; OHLC close is USD. FX reference is a constant approximation.
            self.hist=[float(x[4]) for x in d['result'][k][:-1] if float(x[4])>0]
            self.history_error=None
        except Exception as e:self.history_error=str(e)[:150]
    def fill(self,b,side,price,reason):
        cash,qty,entry,peak=self.account(b);fee=(self.fee+30)/10000 if self.fee_ok else .011
        if side=='BUY' and qty==0 and self.fee_ok:
            budget=min(500.,cash*.05)
            if budget<25:return
            q=budget/(price*(1+fee))
            self.db.execute("UPDATE accounts SET cash=?,qty=?,entry=? WHERE branch=?",(cash-budget,q,price,b))
            amount=budget
        elif side=='SELL' and qty>0:
            amount=qty*price*(1-fee)
            self.db.execute("UPDATE accounts SET cash=?,qty=0,entry=0 WHERE branch=?",(cash+amount,b))
            q=qty
        else:return
        self.db.execute("INSERT INTO fills(ts,branch,side,amount,qty,price,fee,reason) VALUES(?,?,?,?,?,?,?,?)",(now(),b,side,amount,q,price,fee,reason))
    @staticmethod
    def replay(prices,window,threshold):
        cash=10000.;qty=0.;trades=0
        for i in range(window,len(prices)):
            p=prices[i];signal=prices[i]/prices[i-window]-1
            if not qty and signal>threshold:
                budget=min(500.,cash*.05);qty=budget/(p*1.011);cash-=budget;trades+=1
            elif qty and (signal<-.012 or p<prices[i-window]*.965):
                cash+=qty*p*.989;qty=0.;trades+=1
        return (cash+qty*prices[-1]*.989-10000.)/100.,trades
    def backtests(self):
        if time.time()-self.last_test<60 or len(self.hist)<100:return
        self.last_test=time.time()
        prices=self.hist;split=int(len(prices)*.65);train=prices[:split];test=prices[split-30:]
        self.db.execute("DELETE FROM tests")
        for w in (5,10,15,20,30,45):
            for t in (.005,.012,.02,.026,.035):
                a,_=self.replay(train,w,t);b,n=self.replay(test,w,t)
                self.db.execute("INSERT INTO tests(ts,window,threshold,train_pct,test_pct,trades) VALUES(?,?,?,?,?,?)",(now(),w,t*100,a,b,n))
        self.db.commit()
    def step(self):
        self.fx_refresh();self.news_refresh();self.history_refresh()
        if time.time()-self.last_fee>60:self.evidence();self.last_fee=time.time()
        if self.fx is None:self.error="Brak referencyjnego USD/PLN";return
        d=jget('https://api.kraken.com/0/public/Ticker?pair=XBTUSD')
        if d.get('error'):raise ValueError(str(d['error']))
        q=next(iter(d['result'].values()));bid=float(q['b'][0])*self.fx;ask=float(q['a'][0])*self.fx
        if not (math.isfinite(bid) and math.isfinite(ask) and 0<bid<=ask):raise ValueError("Błędny spread")
        ts=now();self.quote={'ts':ts,'bid':bid,'ask':ask};self.last_quote=time.time()
        self.error=None if self.fee_ok else "Brak aktualnego, zweryfikowanego źródła opłat — zakupy zablokowane"
        history=[x[0] for x in self.db.execute("SELECT (bid+ask)/2 FROM ticks ORDER BY id DESC LIMIT 45")][::-1]
        history.append((bid+ask)/2)
        for b in 'ab':
            cash,qty,entry,peak=self.account(b);eq=self.equity(b,bid);peak=max(eq,peak)
            self.db.execute("UPDATE accounts SET peak=? WHERE branch=?",(peak,b))
            action='HOLD';reason='Brak sygnału netto'
            if qty and (eq<peak*.90 or bid<entry*.965):
                action='SELL';reason='Limit straty';self.fill(b,action,bid,reason)
            elif b=='b' and qty==0 and self.fee_ok and self.db.execute("SELECT COUNT(*) FROM fills WHERE branch='b'").fetchone()[0]==0:
                action='BUY';reason='Benchmark 5%';self.fill(b,action,ask,reason)
            elif b=='a':
                # Freeze first independently vetted historical candidate; never silently
                # switch strategies based on live outcomes or a reused holdout.
                if self.promoted is None and self.accelerator is not None:
                    snap=self.accelerator.snapshot();candidate=snap.get('candidate') or {}
                    if candidate.get('eligible') and snap.get('status')=='ready':
                        self.promoted={'params':{k:candidate[k] for k in ('window','entry','exit','stop','holding')},'selected_at':now(),'source_run':snap.get('updated'),'train':candidate['train'],'test':candidate['test']}
                        self.promoted_path.parent.mkdir(parents=True,exist_ok=True)
                        self.promoted_path.write_text(json.dumps(self.promoted,ensure_ascii=False))
                if self.promoted and len(self.hist)>65:
                    cfg=self.promoted['params'];w=cfg['window']
                    # One strategy decision per new CLOSED hourly candle, not every
                    # 2s ticker. Execution uses the live ask/bid and verified fee.
                    candle=self.hist[-1]
                    if self.last_strategy_candle!=candle and len(self.hist)>w+1:
                        self.last_strategy_candle=candle
                        mom=self.hist[-1]/self.hist[-1-w]-1
                        if qty==0 and mom>cfg['entry'] and self.fee_ok:
                            action='BUY';reason='Validated historical strategy; hourly close'
                            self.fill(b,action,ask,reason);self.strategy_opened=time.time()
                        elif qty>0 and (mom<cfg['exit'] or bid<entry*(1-cfg['stop']) or (self.strategy_opened and time.time()-self.strategy_opened>=3600*cfg['holding'])):
                            action='SELL';reason='Validated strategy exit';self.fill(b,action,bid,reason)
                else:reason='Brak strategii po niezależnej walidacji — HOLD'
            self.db.execute("INSERT INTO decisions(ts,branch,action,reason) VALUES(?,?,?,?)",(time.time(),b,action,reason))
        self.db.execute("INSERT INTO ticks(ts,bid,ask,a,b) VALUES(?,?,?,?,?)",(ts,bid,ask,self.equity('a',bid),self.equity('b',bid)))
        self.db.execute("DELETE FROM ticks WHERE id < (SELECT MAX(id)-100000 FROM ticks)")
        self.db.execute("DELETE FROM decisions WHERE ts<?",(time.time()-86400,))
        # 48 competing forecasts per tick. Each is evaluated only after 30 NEW ticks.
        tick_id=self.db.execute("SELECT MAX(id) FROM ticks").fetchone()[0]
        if len(history)>=20:
            for w in (3,5,8,12,16,19):
                move=history[-1]/history[-1-w]-1
                for threshold in (.0005,.001,.002,.004):
                    for variant,direction in (('trend',1),('reversal',-1)):
                        signal=1 if move*direction>threshold else (-1 if move*direction < -threshold else 0)
                        self.db.execute("INSERT INTO hypotheses(tick_id,ts,variant,window,threshold,signal,price) VALUES(?,?,?,?,?,?,?)",
                                        (tick_id,ts,variant,w,threshold,signal,bid))
        self.db.execute("""UPDATE hypotheses SET future_price=?,
          correct=CASE WHEN signal=0 THEN NULL WHEN (?-price)*signal>0 THEN 1 ELSE 0 END
          WHERE future_price IS NULL AND tick_id<=?""",(bid,bid,tick_id-30))
        self.db.execute("DELETE FROM hypotheses WHERE tick_id<?",(tick_id-10000,))
        self.db.commit();self.backtests()
    def status(self):
        with self.lock:
            q=self.quote;bid=q['bid'] if q else None
            ac={}
            for b in 'ab':
                cash,qty,entry,peak=self.account(b);eq=self.equity(b,bid) if bid else cash
                ac[b]={'cash':round(cash,2),'qty':qty,'equity':round(eq,2),'pnl':round(eq-10000,2),'pnl_pct':(eq/10000-1)*100}
            fills=[dict(ts=t,branch=b,side=s,amount=a) for t,b,s,a in self.db.execute("SELECT ts,branch,side,amount FROM fills ORDER BY id DESC LIMIT 30")]
            tests=[dict(window=w,threshold=t,train_pct=a,test_pct=b,trades=n) for w,t,a,b,n in self.db.execute("SELECT window,threshold,train_pct,test_pct,trades FROM tests ORDER BY test_pct DESC LIMIT 30")]
            chart=[dict(ts=t,a=a,b=b) for t,a,b in self.db.execute("SELECT ts,a,b FROM ticks ORDER BY id DESC LIMIT 250")][::-1]
            h_total,h_scored=self.db.execute("SELECT COUNT(*),COUNT(future_price) FROM hypotheses").fetchone()
            h_rank=[dict(variant=v,window=w,threshold=round(t*100,3),n=n,accuracy=round(100*hits/n,2))
                for v,w,t,n,hits in self.db.execute("""SELECT variant,window,threshold,COUNT(*),SUM(correct)
                    FROM hypotheses WHERE correct IS NOT NULL
                    GROUP BY variant,window,threshold HAVING COUNT(*)>=10
                    ORDER BY 5*1.0/4 DESC LIMIT 48""")]
            h_rank.sort(key=lambda x:(-x['accuracy'],-x['n']))
            return dict(time=now(),accounts=ac,quote=q,fx=self.fx,fx_date=self.fx_date,blocker=self.error,fee_evidence={'verified':self.fee_ok,'taker_bps':self.fee},decisions=self.db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0],decisions_last_min=self.db.execute("SELECT COUNT(*) FROM decisions WHERE ts>?",(time.time()-60,)).fetchone()[0],ticks=self.db.execute("SELECT COUNT(*) FROM ticks").fetchone()[0],fills=fills,tests=tests,chart=chart,news=self.news,news_error=self.news_error,history_error=self.history_error,quote_age=round(time.time()-self.last_quote,1) if self.last_quote else None,gpu="NIEPODŁĄCZONE — brak potwierdzenia CUDA",promoted_strategy=self.promoted,hypotheses=h_total,hypotheses_scored=h_scored,hypotheses_rank=h_rank[:10],warning="Izolowana księga paper v3. Backtest 1-min Kraken: orientacyjny, stały kurs FX; walidacja nie jest forward zyskiem. News tylko do odczytu; brak automatycznej strategii sentymentu.")
    def loop(self):
        while True:
            try:
                with self.lock:self.step()
            except Exception as e:
                with self.lock:self.error=type(e).__name__+': '+str(e)[:150]
            time.sleep(2)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.home()/'ai-v100');p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8770);args=p.parse_args()
    lab=Lab(args.root)
    accelerator=Accelerator(args.root)
    lab.accelerator=accelerator
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path=self.path.split('?',1)[0]
            if path=='/':
                dashboard=Path(__file__).with_name('synta_live_dashboard_v4.html')
                data=dashboard.read_bytes() if dashboard.is_file() else PAGE.encode()
                typ='text/html; charset=utf-8'
            elif path=='/api/status':data=json.dumps(lab.status(),ensure_ascii=False).encode();typ='application/json; charset=utf-8'
            elif path=='/api/accelerator':data=json.dumps(accelerator.snapshot(),ensure_ascii=False).encode();typ='application/json; charset=utf-8'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',typ);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def log_message(self,*args):pass
    threading.Thread(target=lab.loop,daemon=True).start()
    threading.Thread(target=accelerator.loop,daemon=True).start()
    ThreadingHTTPServer((args.host,args.port),Handler).serve_forever()
if __name__=='__main__':main()
