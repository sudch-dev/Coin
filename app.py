# app.py — ERX-HYBRID BOT + DASHBOARD
# Live Trading + Web UI
# Render compatible | No WSGI | No Gunicorn

import os, time, json, math, threading, requests, hmac, hashlib
from flask import Flask, jsonify, render_template
from datetime import datetime
from pytz import timezone

app = Flask(__name__)

# ================= CONFIG =================

API_KEY = os.environ.get("API_KEY","")
API_SECRET = (os.environ.get("API_SECRET","") or "").encode()
BASE_URL = "https://api.coindcx.com"

APP_BASE_URL  = os.environ.get("APP_BASE_URL","").rstrip("/")
KEEPALIVE_SEC = int(os.environ.get("KEEPALIVE_SEC","240"))
PORT = int(os.environ.get("PORT","10000"))

IST = timezone("Asia/Kolkata")

PAIRS = [
    "BTCUSDT","ETHUSDT","XRPUSDT","SHIBUSDT","SOLUSDT",
    "DOGEUSDT","ADAUSDT","AEROUSDT","BNBUSDT","LTCUSDT"
]

PAIR_RULES = {
    "BTCUSDT": {"precision": 2, "min_qty": 0.001},
    "ETHUSDT": {"precision": 6, "min_qty": 0.0001},
    "XRPUSDT": {"precision": 4, "min_qty": 0.1},
    "SHIBUSDT": {"precision": 4, "min_qty": 10000},
    "DOGEUSDT": {"precision": 4, "min_qty": 0.01},
    "SOLUSDT": {"precision": 4, "min_qty": 0.01},
    "AEROUSDT": {"precision": 2, "min_qty": 0.01},
    "ADAUSDT": {"precision": 2, "min_qty": 2},
    "LTCUSDT": {"precision": 2, "min_qty": 0.001},
    "BNBUSDT": {"precision": 4, "min_qty": 0.001}
}

SETTINGS = {
    "candle_interval_sec": 15*60,
    "tp_pct": 0.01
}

TRADE_COOLDOWN_SEC = 300

# ================= STATE =================

tick_logs = {p:[] for p in PAIRS}
candle_logs = {p:[] for p in PAIRS}
open_positions = {}
exit_orders = []
pair_cooldown_until = {p:0 for p in PAIRS}
bot_status = {"running":True,"last_cycle":None}

# ================= UTILS =================

def ist_now():
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

def hmac_sig(payload):
    return hmac.new(API_SECRET,payload.encode(),hashlib.sha256).hexdigest()

def _pp(pair): return int(PAIR_RULES[pair]["precision"])
def _min_qty(pair): return float(PAIR_RULES[pair]["min_qty"])

def fmt_qty(pair, qty):
    step = 10**(-_pp(pair))
    q = int(float(qty)/step)*step
    return float(f"{q:.{_pp(pair)}f}")

# ================= EXCHANGE =================

def get_prices():
    try:
        r=requests.get(f"{BASE_URL}/exchange/ticker",timeout=10)
        if r.ok:
            return {it["market"]:float(it["last_price"]) for it in r.json() if it["market"] in PAIRS}
    except: pass
    return {}

def get_balances():
    payload=json.dumps({"timestamp":int(time.time()*1000)})
    headers={
        "X-AUTH-APIKEY":API_KEY,
        "X-AUTH-SIGNATURE":hmac_sig(payload),
        "Content-Type":"application/json"
    }
    try:
        r=requests.post(f"{BASE_URL}/exchange/v1/users/balances",headers=headers,data=payload,timeout=10)
        if r.ok:
            return {b["currency"]:float(b["balance"]) for b in r.json()}
    except: pass
    return {}

def place_order(pair, side, qty):
    qty=fmt_qty(pair,qty)
    if qty<=0: return None
    body={
        "market":pair,
        "side":side.lower(),
        "order_type":"market_order",
        "total_quantity":str(qty),
        "timestamp":int(time.time()*1000)
    }
    payload=json.dumps(body,separators=(',',':'))
    headers={
        "X-AUTH-APIKEY":API_KEY,
        "X-AUTH-SIGNATURE":hmac_sig(payload),
        "Content-Type":"application/json"
    }
    try:
        r=requests.post(f"{BASE_URL}/exchange/v1/orders/create",headers=headers,data=payload,timeout=10)
        if r.ok: return r.json()
    except: pass
    return None

# ================= INDICATORS =================

def ema(v,n):
    if len(v)<n: return None
    k=2/(n+1); e=sum(v[:n])/n
    for x in v[n:]: e=x*k+e*(1-k)
    return e

def rsi(v,n=14):
    if len(v)<n+1: return None
    g,l=0,0
    for i in range(-n,0):
        d=v[i]-v[i-1]
        if d>0: g+=d
        else: l+=abs(d)
    if l==0: return 100
    rs=g/l
    return 100-(100/(1+rs))

# ================= CANDLES =================

def aggregate(pair):
    t=tick_logs[pair]
    if not t: return
    candles=[]; candle=None; lastw=None
    for ts,px in sorted(t,key=lambda x:x[0]):
        w=ts-(ts%SETTINGS["candle_interval_sec"])
        if w!=lastw:
            if candle: candles.append(candle)
            candle={"open":px,"high":px,"low":px,"close":px,"start":w}
            lastw=w
        else:
            candle["high"]=max(candle["high"],px)
            candle["low"]=min(candle["low"],px)
            candle["close"]=px
    if candle: candles.append(candle)
    candle_logs[pair]=candles[-120:]

# ================= STRATEGY =================

def erx_signal(pair, price):
    c=candle_logs[pair]
    if len(c)<60: return None
    comp=c[:-1]
    closes=[x["close"] for x in comp]
    ema20=ema(closes,20); ema50=ema(closes,50)
    r=rsi(closes,14)
    if None in [ema20,ema50,r]: return None

    if ema20>ema50 and price>ema20 and 50<=r<=65:
        return "BUY"
    if ema20<ema50 and price<ema20 and 35<=r<=50:
        return "SELL"
    return None

# ================= EXIT =================

def monitor_exits(prices):
    rem=[]
    for ex in exit_orders:
        pair=ex["pair"]; side=ex["side"]; tp=ex["tp"]
        px=prices.get(pair)
        if not px: continue
        if side=="BUY" and px>=tp: rem.append(ex)
        if side=="SELL" and px<=tp: rem.append(ex)
    for r in rem:
        exit_orders.remove(r)
        pair_cooldown_until[r["pair"]]=int(time.time())+TRADE_COOLDOWN_SEC
        open_positions.pop(r["pair"],None)

# ================= MAIN LOOP =================

def trade_loop():
    while True:
        prices=get_prices()
        now=int(time.time())

        for pair,px in prices.items():
            tick_logs[pair].append((now,px))
            if len(tick_logs[pair])>5000:
                tick_logs[pair]=tick_logs[pair][-5000:]
            aggregate(pair)

        monitor_exits(prices)
        balances=get_balances()

        for pair,px in prices.items():
            if pair in open_positions: continue
            if now<pair_cooldown_until[pair]: continue

            sig=erx_signal(pair,px)
            if not sig: continue

            usdt=float(balances.get("USDT",0))
            if sig=="BUY":
                qty=(0.25*usdt)/px
                qty=max(qty,_min_qty(pair))
                if place_order(pair,"BUY",qty):
                    open_positions[pair]="BUY"
                    exit_orders.append({"pair":pair,"side":"BUY","tp":px*(1+SETTINGS["tp_pct"])})
            elif sig=="SELL":
                coin=pair[:-4]
                bal=float(balances.get(coin,0))
                if bal<=0: continue
                qty=max(bal,_min_qty(pair))
                if place_order(pair,"SELL",qty):
                    open_positions[pair]="SELL"
                    exit_orders.append({"pair":pair,"side":"SELL","tp":px*(1-SETTINGS["tp_pct"])})

        bot_status["last_cycle"]=ist_now()
        time.sleep(5)

# ================= KEEPALIVE =================

def keepalive():
    while True:
        try:
            if APP_BASE_URL:
                requests.get(f"{APP_BASE_URL}/ping",timeout=5)
        except: pass
        time.sleep(KEEPALIVE_SEC)

# ================= ROUTES =================

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/api/status")
def api_status():
    return jsonify({
        "time":ist_now(),
        "bot":bot_status,
        "positions":open_positions,
        "exit_orders":exit_orders,
        "cooldowns":pair_cooldown_until
    })

@app.route("/ping")
def ping():
    return "OK",200

# ================= BOOT =================

if __name__=="__main__":
    threading.Thread(target=trade_loop,daemon=True).start()
    threading.Thread(target=keepalive,daemon=True).start()
    app.run(host="0.0.0.0",port=PORT)