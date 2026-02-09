# app.py — ERX-HYBRID BOT (NO WSGI, NO GUNICORN, PURE FLASK)
# Trend + Sideways Regime Adaptive Strategy
# Render-safe, Bot-safe, Single-process architecture

import os, time, json, math, threading, requests
from flask import Flask, jsonify, request
from datetime import datetime, timedelta
from pytz import timezone

app = Flask(__name__)

# =========================
# ===== BASIC CONFIG ======
# =========================

API_KEY = os.environ.get("API_KEY", "")
API_SECRET = os.environ.get("API_SECRET", "")
BASE_URL = "https://api.coindcx.com"

APP_BASE_URL  = os.environ.get("APP_BASE_URL", "").rstrip("/")
KEEPALIVE_SEC = int(os.environ.get("KEEPALIVE_SEC", "240"))

IST = timezone('Asia/Kolkata')

def ist_now():
    return datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')

# =========================
# ===== BOT STATE =========
# =========================

running = True
status = {"msg": "Running"}
status_epoch = 0

# =========================
# ===== PAIRS =============
# =========================

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
    "candle_interval_sec": 15 * 60,
    "tp_pct": 0.01
}

TRADE_COOLDOWN_SEC = 300

tick_logs = {p: [] for p in PAIRS}
candle_logs = {p: [] for p in PAIRS}
exit_orders = []
pair_cooldown_until = {p: 0 for p in PAIRS}

# =========================
# ===== HTTP ROUTES =======
# =========================

@app.route("/")
def home():
    return jsonify({
        "service": "ERX-HYBRID BOT",
        "status": "RUNNING",
        "time": ist_now()
    })

@app.route("/status")
def status_api():
    return jsonify({
        "running": running,
        "pairs": PAIRS,
        "time": ist_now(),
        "mode": "ERX-HYBRID",
        "message": status.get("msg", "OK")
    })

@app.route("/ping")
def ping():
    return "OK", 200

# =========================
# ===== HELPERS ===========
# =========================

def _pp(pair): return int(PAIR_RULES[pair]["precision"])
def _min_qty(pair): return float(PAIR_RULES[pair]["min_qty"])

def fmt_qty_floor(pair, qty):
    qp = _pp(pair)
    step = 10 ** (-qp)
    q = int(float(qty)/step)*step
    return float(f"{q:.{qp}f}")

# =========================
# ===== CANDLES ===========
# =========================

def aggregate_candles(pair, interval_sec):
    t = tick_logs[pair]
    if not t: return
    candles, candle, lastw = [], None, None
    for ts, px in sorted(t, key=lambda x: x[0]):
        w = ts - (ts % interval_sec)
        if w != lastw:
            if candle: candles.append(candle)
            candle = {"open": px,"high": px,"low": px,"close": px,"start": w}
            lastw = w
        else:
            candle["high"] = max(candle["high"], px)
            candle["low"] = min(candle["low"], px)
            candle["close"] = px
    if candle: candles.append(candle)
    candle_logs[pair] = candles[-120:]

# =========================
# ===== INDICATORS ========
# =========================

def ema(values, n):
    if len(values) < n: return None
    k = 2/(n+1)
    e = sum(values[:n])/n
    for v in values[n:]:
        e = v*k + e*(1-k)
    return e

def rsi(values, n=14):
    if len(values) < n+1: return None
    gains, losses = 0, 0
    for i in range(-n,0):
        d = values[i]-values[i-1]
        if d>0: gains+=d
        else: losses+=abs(d)
    if losses == 0: return 100
    rs = gains/losses
    return 100-(100/(1+rs))

def atr(candles, n=14):
    if len(candles)<n+1: return None
    trs=[]
    for i in range(-n,0):
        h=candles[i]["high"]; l=candles[i]["low"]; pc=candles[i-1]["close"]
        trs.append(max(h-l,abs(h-pc),abs(l-pc)))
    return sum(trs)/n

def bb_width(values,n=20):
    if len(values)<n: return None
    v=values[-n:]
    ma=sum(v)/n
    var=sum((x-ma)**2 for x in v)/n
    std=math.sqrt(var)
    return (ma+2*std)-(ma-2*std)

# =========================
# ===== ERX-HYBRID LOGIC ==
# =========================

def erx_signal(pair, price):
    candles = candle_logs[pair]
    if len(candles)<60: return None

    completed=candles[:-1]
    closes=[c["close"] for c in completed]
    highs=[c["high"] for c in completed]
    lows=[c["low"] for c in completed]

    ema20=ema(closes,20)
    ema50=ema(closes,50)
    rsi14=rsi(closes,14)
    atr14=atr(completed,14)
    bbw=bb_width(closes,20)

    if None in [ema20,ema50,rsi14,atr14,bbw]: return None

    prev_bbw=bb_width(closes[:-1],20)
    prev_atr=atr(completed[:-1],14)

    TREND_MODE = (bbw>prev_bbw and atr14>prev_atr)
    SIDEWAYS_MODE = (bbw<prev_bbw and atr14<=prev_atr and abs(ema20-ema50)<0.002*price)

    # ===== TREND =====
    if TREND_MODE:
        if ema20>ema50 and price>ema20 and 50<=rsi14<=65 and lows[-1]<=ema20:
            return {"side":"BUY","msg":"ERX TREND BUY"}
        if ema20<ema50 and price<ema20 and 35<=rsi14<=50 and highs[-1]>=ema20:
            return {"side":"SELL","msg":"ERX TREND SELL"}

    # ===== RANGE =====
    if SIDEWAYS_MODE:
        rh=max(highs[-20:])
        rl=min(lows[-20:])
        if price<=rl*1.002 and rsi14<35:
            return {"side":"BUY","msg":"ERX RANGE BUY"}
        if price>=rh*0.998 and rsi14>65:
            return {"side":"SELL","msg":"ERX RANGE SELL"}

    return None

# =========================
# ===== EXIT ENGINE =======
# =========================

def monitor_exits(prices):
    rem=[]
    for ex in exit_orders:
        pair=ex["pair"]; side=ex["side"]; tp=ex["tp"]; qty=ex["qty"]
        px=prices.get(pair)
        if not px: continue
        if side=="BUY" and px>=tp:
            rem.append(ex)
        if side=="SELL" and px<=tp:
            rem.append(ex)
    for r in rem:
        exit_orders.remove(r)
        pair_cooldown_until[r["pair"]] = int(time.time())+TRADE_COOLDOWN_SEC

# =========================
# ===== KEEPALIVE =========
# =========================

def keepalive_loop():
    while True:
        try:
            if APP_BASE_URL:
                requests.get(f"{APP_BASE_URL}/ping",timeout=5)
        except: pass
        time.sleep(KEEPALIVE_SEC)

# =========================
# ===== BOOT ==============
# =========================

if __name__ == "__main__":
    threading.Thread(target=keepalive_loop,daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT","10000")))