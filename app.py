from flask import Flask, jsonify, request, render_template
import threading, time, requests, hmac, hashlib, json, os
from datetime import datetime

app = Flask(__name__)

# ================== CONFIG ==================
API_KEY = os.getenv("API_KEY","")
API_SECRET = os.getenv("API_SECRET","")
BASE_URL = "https://api.coindcx.com"

bot_running = False
bot_thread = None

# ================== GLOBAL STATE ==================
STATE = {
    "status":"Idle",
    "last":None,
    "balances":{},
    "positions":[],
    "orders":[],
    "trades":[],
    "errors":[],
    "logs":[],
    "regime":"--",
    "strategy_mode":"Hybrid",
    "equity":0,
    "exposure":0
}

# ================== UTILS ==================
def ist_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(msg)
    STATE["logs"].append(f"{ist_now()} | {msg}")
    if len(STATE["logs"])>300:
        STATE["logs"]=STATE["logs"][-300:]

def hmac_signature(payload):
    return hmac.new(API_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()

# ================== API CORE ==================
def api_post(path, payload):
    body=json.dumps(payload)
    headers={
        "X-AUTH-APIKEY":API_KEY,
        "X-AUTH-SIGNATURE":hmac_signature(body),
        "Content-Type":"application/json"
    }
    return requests.post(BASE_URL+path,headers=headers,data=body,timeout=15)

# ================== PORTFOLIO ENGINE ==================
def get_wallet_balances():
    balances={}
    try:
        payload={"timestamp":int(time.time()*1000)}
        r=api_post("/exchange/v1/users/balances",payload)
        if r.ok:
            data=r.json()
            if isinstance(data,list):
                for b in data:
                    balances[b["currency"]]=float(b["balance"])
        else:
            log(f"BALANCE API ERROR {r.status_code}: {r.text}")
    except Exception as e:
        log(f"BALANCE EXCEPTION: {e}")
    return balances

def calc_equity(balances):
    return sum(balances.values())

# ================== MARKET ENGINE ==================
def detect_regime(prices):
    if len(prices)<20: return "Sideways"
    hi=max(prices); lo=min(prices)
    vol=(hi-lo)/lo if lo else 0
    if vol>0.03: return "Breakout"
    if prices[-1]>sum(prices)/len(prices): return "Trend"
    return "Sideways"

# ================== STRATEGY ENGINE ==================
def strategy_signal(prices, regime):
    if regime=="Breakout" and prices[-1]>max(prices[:-1]):
        return "BUY"
    if regime=="Trend" and prices[-1]>sum(prices)/len(prices):
        return "BUY"
    return "HOLD"

# ================== RISK ENGINE ==================
def risk_check(balance):
    return balance>10   # minimum capital rule

# ================== EXECUTION ENGINE ==================
def place_order(pair, side, qty):
    STATE["orders"].append({
        "time":ist_now(),
        "pair":pair,
        "side":side,
        "qty":qty,
        "status":"SIMULATED"
    })
    STATE["trades"].append({
        "time":ist_now(),
        "pair":pair,
        "side":side
    })
    log(f"ORDER {side} {pair} QTY {qty}")

# ================== BOT CORE ==================
def bot_loop():
    global bot_running
    log("BOT STARTED")
    while bot_running:
        try:
            STATE["last"]=ist_now()

            # Portfolio
            balances=get_wallet_balances()
            STATE["balances"]=balances
            usdt=balances.get("USDT",0)
            STATE["equity"]=calc_equity(balances)

            # Market sample (public ticker)
            r=requests.get(f"{BASE_URL}/exchange/ticker",timeout=10)
            if not r.ok:
                time.sleep(5); continue
            tick=r.json()[:20]
            prices=[float(x["last_price"]) for x in tick if "last_price" in x]

            # Regime
            regime=detect_regime(prices)
            STATE["regime"]=regime

            # Strategy
            signal=strategy_signal(prices,regime)

            # Risk
            if signal=="BUY" and risk_check(usdt):
                place_order("BTCUSDT","BUY",round(usdt*0.01,2))

        except Exception as e:
            STATE["errors"].append(str(e))
            log(f"BOT ERROR: {e}")

        time.sleep(10)

    log("BOT STOPPED")

# ================== ROUTES ==================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/start",methods=["POST"])
def start_bot():
    global bot_running, bot_thread
    if not bot_running:
        bot_running=True
        STATE["status"]="Running"
        bot_thread=threading.Thread(target=bot_loop,daemon=True)
        bot_thread.start()
    return jsonify({"status":"started"})

@app.route("/stop",methods=["POST"])
def stop_bot():
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
        "usdt":STATE["balances"].get("USDT",0),
        "equity":STATE["equity"],
        "positions":STATE["positions"],
        "orders":STATE["orders"],
        "trades":STATE["trades"],
        "regime":STATE["regime"],
        "strategy":STATE["strategy_mode"],
        "errors":STATE["errors"][-5:],
        "logs":STATE["logs"][-10:]
    })

# ================== DIAGNOSTICS ==================
@app.route("/api/balance_test")
def balance_test():
    b=get_wallet_balances()
    return jsonify({"time":ist_now(),"balances":b})

@app.route("/api/key_test")
def key_test():
    payload={"timestamp":int(time.time()*1000)}
    try:
        r=api_post("/exchange/v1/users/balances",payload)
        return jsonify({"status":r.status_code,"response":r.text})
    except Exception as e:
        return jsonify({"error":str(e)})

@app.route("/api/connection_test")
def conn_test():
    try:
        r=requests.get(f"{BASE_URL}/exchange/ticker",timeout=10)
        return jsonify({"ok":r.ok,"status":r.status_code})
    except Exception as e:
        return jsonify({"error":str(e)})

# ================== MAIN ==================
if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)