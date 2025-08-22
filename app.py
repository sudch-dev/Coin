import os, time, threading, hmac, hashlib, requests, json, traceback
from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta
from pytz import timezone
from collections import deque

# -------------------- Flask --------------------
app = Flask(__name__)

APP_BASE_URL = os.environ.get("APP_BASE_URL", "https://coin-4k37.onrender.com")
KEEPALIVE_TOKEN = os.environ.get("KEEPALIVE_TOKEN", "")

API_KEY = os.environ.get("API_KEY")
API_SECRET_RAW = os.environ.get("API_SECRET", "")
API_SECRET = API_SECRET_RAW.encode() if isinstance(API_SECRET_RAW, str) else API_SECRET_RAW
BASE_URL = "https://api.coindcx.com"

PAIRS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SHIBUSDT", "SOLUSDT",
         "DOGEUSDT", "ADAUSDT", "AEROUSDT", "BNBUSDT"]  # LTC removed

PAIR_RULES = {}
IST = timezone("Asia/Kolkata")
def ist_now(): return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

# ---- state ----
tick_logs = {p: [] for p in PAIRS}
candle_logs = {p: [] for p in PAIRS}
scan_log, trade_log, order_book = [], [], []
profit_state = {"daily": {}, "cumulative": 0.0}
running = False
status = {"msg": "Idle", "last": ""}
status_epoch = 0
error_message = ""
last_keepalive = 0

# ---- settings ----
CANDLE_INTERVAL = 5   # seconds per candle
POLL_SEC = 1.0        # tick every 1s
TP_TARGET = 0.01      # 1% profit target
EMA_FAST, EMA_SLOW = 5, 20
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
INVEST_FRAC = 0.30

# -------------------- Helpers --------------------
def hmac_signature(payload):
    return hmac.new(API_SECRET, payload.encode(), hashlib.sha256).hexdigest()

def _signed_post(url, body):
    payload = json.dumps(body, separators=(',', ':'))
    sig = hmac_signature(payload)
    headers = {"X-AUTH-APIKEY": API_KEY or "", "X-AUTH-SIGNATURE": sig, "Content-Type": "application/json"}
    try:
        r = requests.post(url, headers=headers, data=payload, timeout=10)
        return r.json() if r.ok else {}
    except Exception as e:
        scan_log.append(f"{ist_now()} | POST fail {url} | {e}")
        return {}

def fetch_pair_precisions():
    try:
        r = requests.get(f"{BASE_URL}/exchange/v1/markets_details", timeout=12)
        if not r.ok: return
        for item in r.json():
            p = item.get("pair")
            if p in PAIRS:
                PAIR_RULES[p] = {
                    "price_precision": int(item.get("target_currency_precision", 6)),
                    "qty_precision": int(item.get("base_currency_precision", 6)),
                    "min_qty": float(item.get("min_quantity", 0.0)),
                }
    except: pass

def fmt_price(pair, price):
    pp = PAIR_RULES.get(pair, {}).get("price_precision", 6)
    return float(f"{price:.{pp}f}")

def fmt_qty(pair, qty):
    qp = PAIR_RULES.get(pair, {}).get("qty_precision", 6)
    mq = PAIR_RULES.get(pair, {}).get("min_qty", 0.0)
    q = max(qty, mq)
    return float(f"{q:.{qp}f}")

def get_balances():
    payload = json.dumps({"timestamp": int(time.time()*1000)})
    sig = hmac_signature(payload)
    headers = {"X-AUTH-APIKEY": API_KEY, "X-AUTH-SIGNATURE": sig, "Content-Type": "application/json"}
    try:
        r = requests.post(f"{BASE_URL}/exchange/v1/users/balances", headers=headers, data=payload, timeout=10)
        return {b["currency"]: float(b["balance"]) for b in r.json()} if r.ok else {}
    except: return {}

def fetch_prices():
    try:
        r = requests.get(f"{BASE_URL}/exchange/ticker", timeout=10)
        return {i["market"]: float(i["last_price"]) for i in r.json() if i.get("market") in PAIRS}
    except: return {}

# ---- indicators ----
def ema(values, n):
    if len(values) < n: return None
    k = 2 / (n+1)
    e = sum(values[:n])/n
    for v in values[n:]: e = v*k + e*(1-k)
    return e

def macd(values):
    if len(values) < MACD_SLOW+MACD_SIGNAL: return None,None,None
    efast = ema(values, MACD_FAST); eslow = ema(values, MACD_SLOW)
    line = efast - eslow
    hist = []
    k = 2/(MACD_SIGNAL+1)
    sig = line
    for v in values[-MACD_SIGNAL:]:
        sig = v*k + sig*(1-k)
        hist.append(line-sig)
    return line, sig, hist[-1] if hist else 0

# ---- orders ----
def place_order(pair, side, qty):
    payload = {
        "market": pair, "side": side.lower(), "order_type": "market_order",
        "total_quantity": f"{qty}", "timestamp": int(time.time()*1000)
    }
    res = _signed_post(f"{BASE_URL}/exchange/v1/orders/create", payload)
    oid = str(res.get("id") or res.get("order_id") or "")
    order_book.append({"pair": pair, "side": side, "qty": qty, "id": oid, "res": res, "time": ist_now()})
    trade_log.append(f"{ist_now()} | {pair} {side} {qty} | res={res}")
    return res

# ---- candles ----
def update_candles(pair, ts, price):
    ticks = tick_logs[pair]; ticks.append((ts,price))
    ticks[:] = ticks[-1000:]
    candles = candle_logs[pair]
    wstart = ts - (ts % CANDLE_INTERVAL)
    if not candles or candles[-1]["start"]!=wstart:
        candles.append({"open":price,"high":price,"low":price,"close":price,"start":wstart})
    else:
        c = candles[-1]
        c["high"] = max(c["high"], price); c["low"]=min(c["low"], price); c["close"]=price
    candle_logs[pair]=candles[-500:]

# -------------------- Strategy Loop --------------------
def scan_loop():
    global running, status, status_epoch, last_keepalive
    running=True
    entry={}  # active buy entry price
    while running:
        now=int(time.time())
        if now-last_keepalive>=240:
            try: requests.get(f"{APP_BASE_URL}/ping?t={KEEPALIVE_TOKEN}",timeout=5)
            except: pass
            last_keepalive=now

        prices=fetch_prices()
        balances=get_balances()
        usdt=balances.get("USDT",0.0)

        for pair,px in prices.items():
            update_candles(pair, now, px)
            closes=[c["close"] for c in candle_logs[pair]]
            if len(closes)<30: continue
            e5,e20=ema(closes,EMA_FAST),ema(closes,EMA_SLOW)
            line,sig,hist=macd(closes)
            scan_log.append(f"{ist_now()} | {pair} px={px} e5={e5:.2f} e20={e20:.2f} macd={line:.4f} sig={sig:.4f} hist={hist:.4f}")

            # buy
            if e5 and e20 and e5>e20 and line>sig and usdt>10 and pair not in entry:
                invest=usdt*INVEST_FRAC/px
                q=fmt_qty(pair,invest)
                if q>0:
                    res=place_order(pair,"BUY",q)
                    entry[pair]=px

            # sell
            if pair in entry:
                buy_px=entry[pair]
                gain=(px-buy_px)/buy_px
                if (e5<e20 and line<sig) or gain>=TP_TARGET:
                    q=balances.get(pair[:-4],0.0)
                    q=fmt_qty(pair,q)
                    if q>0: place_order(pair,"SELL",q)
                    entry.pop(pair,None)

        status["msg"], status["last"]="Running",ist_now()
        status_epoch=int(time.time())
        time.sleep(POLL_SEC)
    status["msg"]="Idle"

# -------------------- Routes --------------------
@app.route("/")
def index(): return render_template("index.html")

@app.route("/start",methods=["POST"])
def start():
    threading.Thread(target=scan_loop,daemon=True).start()
    return jsonify({"status":"started"})

@app.route("/stop",methods=["POST"])
def stop():
    global running; running=False
    return jsonify({"status":"stopped"})

@app.route("/status")
def get_status():
    now_real=time.time(); last_age=now_real-(last_keepalive or 0)
    keepalive_info={
        "enabled":bool(KEEPALIVE_TOKEN),
        "interval_sec":240,
        "last_ping_age_sec":int(last_age) if last_keepalive else None,
        "next_due_sec":max(0,int(240-last_age)) if last_keepalive else None,
        "app_base_url":APP_BASE_URL,
    }
    return jsonify({
        "status":status["msg"],"last":status["last"],"status_epoch":status_epoch,
        "profit_today":profit_state["daily"].get(datetime.now(IST).strftime("%Y-%m-%d"),0.0),
        "pnl_cumulative":profit_state["cumulative"],"trades":trade_log[-10:],
        "scans":scan_log[-30:],"order_book":order_book[-10:],
        "keepalive":keepalive_info,"error":error_message
    })

@app.route("/ping",methods=["GET","HEAD"])
def ping():
    token=os.environ.get("KEEPALIVE_TOKEN","")
    provided=request.args.get("t") or request.headers.get("X-Keepalive-Token") or ""
    if token and provided!=token: return "forbidden",403
    return ("pong",200)

if __name__=="__main__":
    fetch_pair_precisions()
    port=int(os.environ.get("PORT","10000"))
    app.run(host="0.0.0.0",port=port)
