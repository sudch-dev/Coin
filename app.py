from flask import Flask, jsonify, request, render_template
import threading, time, requests, hmac, hashlib, json, os
from datetime import datetime
from collections import deque
from pytz import timezone

app = Flask(__name__)

# ================== TIME ==================
IST = timezone("Asia/Kolkata")
def ist_now():
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

# ================== CONFIG ==================
API_KEY = os.getenv("API_KEY","")
API_SECRET = os.getenv("API_SECRET","")
BASE_URL = "https://api.coindcx.com"

PAIRS = ["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT"]

CAPITAL_RISK_PCT = 0.01      # 1% risk per trade
TP_PCT = 0.02                # 2% target
SL_PCT = 0.01                # 1% stoploss
TRAIL_PCT = 0.008            # 0.8% trailing

CANDLE_INTERVAL = 60         # 1 minute candles
HISTORY = 120                # 120 candles

# ================== STATE ==================
bot_running = False
bot_thread = None

STATE = {
    "status":"Idle",
    "last":None,
    "balances":{},
    "positions":{},
    "orders":[],
    "trades":[],
    "logs":[],
    "errors":[],
    "equity":0,
    "regime":"--"
}

ticks = {p:[] for p in PAIRS}
candles = {p:deque(maxlen=HISTORY) for p in PAIRS}

# ================== UTILS ==================
def log(msg):
    print(msg)
    STATE["logs"].append(f"{ist_now()} | {msg}")
    if len(STATE["logs"])>300:
        STATE["logs"]=STATE["logs"][-300:]

def hmac_signature(payload):
    return hmac.new(API_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()

def api_post(path, payload):
    body=json.dumps(payload)
    headers={
        "X-AUTH-APIKEY":API_KEY,
        "X-AUTH-SIGNATURE":hmac_signature(body),
        "Content-Type":"application/json"
    }
    return requests.post(BASE_URL+path,headers=headers,data=body,timeout=15)

# ================== EXCHANGE ==================
def get_wallet_balances():
    balances={}
    try:
        payload={"timestamp":int(time.time()*1000)}
        r=api_post("/exchange/v1/users/balances",payload)
        if r.ok:
            for b in r.json():
                balances[b["currency"]] = float(b["balance"])
    except Exception as e:
        log(f"BALANCE ERROR {e}")
    return balances

def place_order(pair, side, qty):
    STATE["orders"].append({
        "time":ist_now(),"pair":pair,"side":side,"qty":qty,"status":"SIM"
    })
    log(f"EXEC {side} {pair} {qty}")

# ================== MARKET ==================
def fetch_prices():
    r=requests.get(f"{BASE_URL}/exchange/ticker",timeout=10)
    if not r.ok: return {}
    data=r.json()
    out={}
    for x in data:
        if x["market"] in PAIRS:
            out[x["market"]] = float(x["last_price"])
    return out

def build_candles(pair, price):
    now=int(time.time())
    ticks[pair].append((now,price))
    if len(ticks[pair])>5000:
        ticks[pair]=ticks[pair][-5000:]

    bucket = now - (now % CANDLE_INTERVAL)
    if not candles[pair] or candles[pair][-1]["t"]!=bucket:
        candles[pair].append({
            "t":bucket,"o":price,"h":price,"l":price,"c":price
        })
    else:
        c=candles[pair][-1]
        c["h"]=max(c["h"],price)
        c["l"]=min(c["l"],price)
        c["c"]=price

# ================== INDICATORS ==================
def ema(values,n):
    if len(values)<n: return None
    k=2/(n+1)
    e=sum(values[:n])/n
    for v in values[n:]:
        e=v*k+e*(1-k)
    return e

# ================== REGIME ==================
def detect_regime(closes):
    if len(closes)<50: return "Sideways"
    hi=max(closes[-50:])
    lo=min(closes[-50:])
    vol=(hi-lo)/lo if lo else 0
    if vol>0.05: return "Breakout"
    ema_fast=ema(closes,9)
    ema_slow=ema(closes,21)
    if ema_fast and ema_slow and ema_fast>ema_slow:
        return "Trend"
    return "Sideways"

# ================== STRATEGY ==================
def strategy(pair):
    cs=list(candles[pair])
    if len(cs)<50: return None

    closes=[c["c"] for c in cs]

    ema9=ema(closes,9)
    ema21=ema(closes,21)
    ema50=ema(closes,50)

    regime=detect_regime(closes)
    STATE["regime"]=regime

    price=closes[-1]

    # -------- TREND STRATEGY --------
    if regime=="Trend":
        if ema9>ema21>ema50:
            return {"side":"BUY","price":price}

    # -------- BREAKOUT STRATEGY --------
    if regime=="Breakout":
        hi=max(closes[-20:])
        if price>hi:
            return {"side":"BUY","price":price}

    # -------- SIDEWAYS STRATEGY --------
    if regime=="Sideways":
        hi=max(closes[-20:])
        lo=min(closes[-20:])
        if price<=lo*1.002:
            return {"side":"BUY","price":price}
        if price>=hi*0.998:
            return {"side":"SELL","price":price}

    return None

# ================== POSITION ENGINE ==================
def manage_position(pair, price):
    pos=STATE["positions"].get(pair)
    if not pos: return

    side=pos["side"]
    entry=pos["entry"]
    qty=pos["qty"]
    sl=pos["sl"]
    tp=pos["tp"]
    trail=pos["trail"]

    # trailing
    if side=="BUY":
        new_trail = price*(1-TRAIL_PCT)
        pos["trail"]=max(trail,new_trail)

        if price<=pos["trail"]:
            place_order(pair,"SELL",qty)
            del STATE["positions"][pair]
            log(f"TRAIL EXIT {pair}")

        elif price<=sl:
            place_order(pair,"SELL",qty)
            del STATE["positions"][pair]
            log(f"SL EXIT {pair}")

        elif price>=tp:
            place_order(pair,"SELL",qty)
            del STATE["positions"][pair]
            log(f"TP EXIT {pair}")

# ================== BOT ==================
def bot_loop():
    global bot_running
    log("BOT STARTED")
    while bot_running:
        try:
            STATE["last"]=ist_now()
            balances=get_wallet_balances()
            STATE["balances"]=balances
            usdt=balances.get("USDT",0)
            STATE["equity"]=usdt

            prices=fetch_prices()

            for pair,price in prices.items():
                build_candles(pair,price)

                # manage open
                manage_position(pair,price)

                # skip if already in position
                if pair in STATE["positions"]:
                    continue

                sig=strategy(pair)
                if not sig: continue

                if sig["side"]=="BUY" and usdt>20:
                    risk_amt=usdt*CAPITAL_RISK_PCT
                    qty=round(risk_amt/price,6)

                    sl=price*(1-SL_PCT)
                    tp=price*(1+TP_PCT)
                    trail=price*(1-TRAIL_PCT)

                    place_order(pair,"BUY",qty)

                    STATE["positions"][pair]={
                        "side":"BUY","entry":price,"qty":qty,
                        "sl":sl,"tp":tp,"trail":trail
                    }

                    STATE["trades"].append({
                        "time":ist_now(),"pair":pair,
                        "side":"BUY","entry":price
                    })

                    log(f"ENTRY {pair} @ {price}")

        except Exception as e:
            STATE["errors"].append(str(e))
            log(f"BOT ERROR {e}")

        time.sleep(3)

    log("BOT STOPPED")

# ================== ROUTES ==================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/start",methods=["POST"])
def start():
    global bot_running, bot_thread
    if not bot_running:
        bot_running=True
        STATE["status"]="Running"
        bot_thread=threading.Thread(target=bot_loop,daemon=True)
        bot_thread.start()
    return jsonify({"status":"started"})

@app.route("/stop",methods=["POST"])
def stop():
    global bot_running
    bot_running=False
    STATE["status"]="Stopped"
    return jsonify({"status":"stopped"})

@app.route("/status")
def status():
    return jsonify({
        "status":STATE["status"],
        "last":STATE["last"],
        "balances":STATE["balances"],
        "equity":STATE["equity"],
        "positions":STATE["positions"],
        "orders":STATE["orders"][-20:],
        "trades":STATE["trades"][-20:],
        "regime":STATE["regime"],
        "logs":STATE["logs"][-20:],
        "errors":STATE["errors"][-5:]
    })

# ================== MAIN ==================
if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)