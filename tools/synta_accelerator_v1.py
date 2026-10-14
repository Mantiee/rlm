#!/usr/bin/env python3
"""CPU accelerated research on closed Kraken 1m candles. Never places orders.
Independent historical research, not forward validation and not GPU training."""
import datetime,json,math,random,statistics,threading,time,urllib.request
from pathlib import Path

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def fetch_ohlc():
    req=urllib.request.Request('https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=1',headers={'User-Agent':'Synta-Accelerated-Paper-Research/1'})
    with urllib.request.urlopen(req,timeout=12) as r:raw=json.load(r)
    if raw.get('error'):raise ValueError(str(raw['error']))
    key=next(k for k in raw['result'] if k!='last')
    candles=raw['result'][key][:-1] # last is the incomplete candle
    return [(int(x[0]),float(x[4])) for x in candles if math.isfinite(float(x[4])) and float(x[4])>0]

def simulate(prices,lookback,entry,exit_,stop,holding,side_cost=.011):
    """Long-only 5% portfolio, same conservative side costs as forward paper.
    Decision from previous closed candle; next candle close fill is optimistic and
    is flagged as a modeling limitation in the dashboard."""
    cash=10000.;qty=0.;entry_price=0.;opened=0;trades=0;peak=10000.;dd=0.
    for i in range(lookback+1,len(prices)):
        previous=prices[i-1];signal=previous/prices[i-1-lookback]-1
        execution=prices[i]
        if qty and (signal<exit_ or previous/entry_price-1 < -stop or i-opened>=holding):
            cash+=qty*execution*(1-side_cost);qty=0;trades+=1
        elif not qty and signal>entry:
            budget=min(500.,cash*.05)
            if budget>=25:qty=budget/(execution*(1+side_cost));cash-=budget;entry_price=execution;opened=i;trades+=1
        equity=cash+qty*execution*(1-side_cost);peak=max(peak,equity);dd=max(dd,100*(peak-equity)/peak)
    equity=cash+qty*prices[-1]*(1-side_cost)
    return {'net_pct':round((equity/10000-1)*100,5),'trades':trades,'max_dd_pct':round(dd,4)}

class Accelerator:
    def __init__(self,root):
        self.root=Path(root);self.lock=threading.RLock()
        self.state={'status':'starting','updated':None,'error':None,'trials':0,'duration_s':0,'top':[],
                    'candles':0,'source':'Kraken public OHLC 1m; last incomplete excluded',
                    'limitations':'Backtest history only, 1m OHLC close; no executable bid/ask; simulated 110bps per side; next-close execution; no lookahead for signals; not GPU training; not forward profit.',
                    'gpu':'NOT VERIFIED / NOT USED'}
        self.cache=self.root/'research/experiments/accelerator-v1.json'
        try:
            cached=json.loads(self.cache.read_text())
            if isinstance(cached,dict) and cached.get('updated'):self.state.update(cached)
        except Exception:pass
    def snapshot(self):
        with self.lock:return dict(self.state)
    def run_once(self):
        started=time.monotonic()
        with self.lock:self.state['status']='fetching'
        candles=fetch_ohlc();prices=[p for _,p in candles]
        if len(prices)<300:raise ValueError('Too few closed candles')
        cut=int(len(prices)*.65);train=prices[:cut];test=prices[cut-65:]
        rng=random.Random(20261014);configs=set()
        while len(configs)<1000:
            w=rng.choice([3,5,8,10,12,15,20,25,30,45,60])
            entry=rng.choice([.0003,.0006,.001,.0015,.002,.003,.004,.006,.009,.012,.02,.03])
            exit_=rng.choice([-.008,-.004,-.002,-.001,0,.001])
            stop=rng.choice([.01,.02,.035,.05,.08])
            hold=rng.choice([5,10,20,30,60,120,240])
            configs.add((w,entry,exit_,stop,hold))
        results=[]
        with self.lock:self.state.update(status='computing',trials=0,error=None)
        for n,cfg in enumerate(sorted(configs),1):
            a=simulate(train,*cfg);b=simulate(test,*cfg)
            results.append((cfg,a,b))
            if n%100==0:
                with self.lock:self.state.update(trials=n,duration_s=round(time.monotonic()-started,2))
        # Select by TRAIN only. Test results are reported but NEVER used for ranking.
        results.sort(key=lambda x:(x[1]['net_pct'], -x[1]['max_dd_pct']),reverse=True)
        top=[]
        for cfg,a,b in results[:15]:
            w,entry,exit_,stop,hold=cfg
            top.append({'window':w,'entry_pct':round(entry*100,3),'exit_pct':round(exit_*100,3),
                        'stop_pct':round(stop*100,3),'max_hold_min':hold,'train':a,'test':b})
        out={'status':'ready','updated':utc(),'error':None,'trials':len(results),
             'duration_s':round(time.monotonic()-started,2),'candles':len(prices),
             'first_candle':datetime.datetime.fromtimestamp(candles[0][0],datetime.timezone.utc).isoformat(),
             'last_candle':datetime.datetime.fromtimestamp(candles[-1][0],datetime.timezone.utc).isoformat(),
             'train_candles':len(train),'test_candles':len(test)-65,'top':top,
             'positive_test_of_top15':sum(1 for x in top if x['test']['net_pct']>0),
             'active_test_of_top15':sum(1 for x in top if x['test']['trades']>0),
             'source':'Kraken public OHLC 1m; last incomplete excluded',
             'limitations':'Historical simulation only. Signals use previous closed candle, fill next close; 110bps per side, no order book or executable FX. Train ranked, held-out test shown, no forward guarantee. GPU not used.',
             'gpu':'NOT VERIFIED / NOT USED'}
        with self.lock:self.state=out
        self.cache.parent.mkdir(parents=True,exist_ok=True)
        tmp=self.cache.with_suffix('.tmp');tmp.write_text(json.dumps(out,ensure_ascii=False));tmp.replace(self.cache)
    def loop(self):
        while True:
            try:self.run_once()
            except Exception as e:
                with self.lock:self.state.update(status='error',error=type(e).__name__+': '+str(e)[:160])
            time.sleep(300)
