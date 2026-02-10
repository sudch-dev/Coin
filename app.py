from flask import Flask, jsonify, request, render_template
import threading, time, requests, hmac, hashlib, json, os, sqlite3
from datetime import datetime
from pytz import timezone

# ================== APP ==================
app = Flask(__name__)

# ================== TIME ==================
IST = timezone("Asia/Kolkata")
def ist_now():
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

# ================== CONFIG ==================
API_KEY = os.getenv("API_KEY","")
API_SECRET = os.getenv("API_SECRET","")
BASE_URL = "https://api.coindcx.com"

PAIRS = ["BTCUSDT","ETHUSDT","SOLUSDT"]

LIVE_TRADING = True     # 🔴 Set True for real orders
SPOT_ONLY = True

MAX_RISK_PER_TRADE = 0.02
MIN_USDT_BALANCE = 10
TP_PCT = 0.008     # 0.8%
SL_PCT = 0.004     # 0.4%
COOLDOWN = 60      # seconds per pair

# ================== DATABASE ==================
DB="bot.db"
def db():
    return sqlite3.connect(DB,check_same_thread=False)

def init_db():
    con=db();cur=con.cursor()
    cur.execute("create table if not exists trades(time,pair,side,qty,price)")
    cur.execute("create table if not exists logs(time,msg)")
    con.commit();con.close()

init_db()

# ================== STATE ==================
bot_running=False
bot_thread=None

STATE={
 "status":"Idle","last":None,
 "balances":{},
 "positions":{},   # pair -> {entry, qty}
 "cooldown":{},
 "logs":[],
 "errors":[],
 "health":{"api":"--","exchange":"--"}
}

# ================== UTILS ==================
def log(msg):
    line=f"{ist_now()} | {msg}"
    print(line)
    STATE["logs"].append(line)
    con=db();cur=con.cursor()
    cur.execute("insert into logs values(?,?)",(ist_now(),msg))
    con.commit();con.close()

def sign(payload):
    return hmac.new(API_SECRET.encode(),payload.encode(),hashlib.sha256).hexdigest()

def api_post(path,payload):
    body=json.dumps(payload)
    headers={
        "X-AUTH-APIKEY":API_KEY,
        "X-AUTH-SIGNATURE":sign(body),
        "Content-Type":"application/json"
    }
    return requests.post(BASE_URL+path,headers=headers,data=body,timeout=15)

# ================== EXCHANGE ==================
def balances():
    data={}
    try:
        r=api_post("/exchange/v1/users/balances",{"timestamp":int(time.time()*1000)})
        if r.ok:
            for b in r.json():
                data[b["currency"]]=float(b["balance"])
            STATE["health"]["api"]="OK"
        else:
            STATE["health"]["api"]="ERROR"
    except:
        STATE["health"]["api"]="ERROR"
    return data

def prices():
    try:
        r=requests.get(BASE_URL+"/exchange/ticker",timeout=10)
        if r.ok:
            STATE["health"]["exchange"]="OK"
            return r.json()
        else:
            STATE["health"]["exchange"]="ERROR"
    except:
        STATE["health"]["exchange"]="ERROR"
    return []

def candles(pair, interval="15m", limit=30):
    try:
        r=requests.get(
            f"{BASE_URL}/exchange/v1/markets/{pair}/candles?interval={interval}&limit={limit}",
            timeout=10
        )
        if r.ok:
            return r.json()
    except:
        pass
    return []

# ================== RISK ==================
def size(usdt, price):
    if usdt < MIN_USDT_BALANCE:
        return 0
    risk_capital = usdt * MAX_RISK_PER_TRADE
    return round(risk_capital/price,6)

# ================== LIVE ORDER ==================
def place_live_order(pair, side, qty):
    payload = {
        "market": pair,
        "side": side.lower(),
        "order_type": "market",
        "quantity": str(qty),
        "timestamp": int(time.time()*1000)
    }

    body=json.dumps(payload)
    headers={
        "X-AUTH-APIKEY":API_KEY,
        "X-AUTH-SIGNATURE":sign(body),
        "Content-Type":"application/json"
    }

    try:
        r=requests.post(BASE_URL+"/exchange/v1/orders/create",
                        headers=headers,data=body,timeout=15)
        if r.ok:
            return True,r.json()
        else:
            return False,r.text
    except Exception as e:
        return False,str(e)

# ================== STRATEGY ==================
def breakout_signal(pair, price):
    c = candles(pair,"15m",30)
    if len(c) < 20:
        return None

    highs=[float(x["high"]) for x in c[:-1]]
    vols=[float(x["volume"]) for x in c[:-1]]

    prev_high=max(highs[-10:])
    avg_vol=sum(vols[-10:])/10
    last_vol=float(c[-1]["volume"])

    # REAL breakout conditions
    if price > prev_high and last_vol > avg_vol*1.5:
        return "BUY"

    return None

# ================== EXECUTION ==================
def execute(pair, side, qty, price):
    if not LIVE_TRADING:
        log(f"[DRY] {side} {pair} {qty} @ {price}")
        STATE["positions"][pair]={"entry":price,"qty":qty}
        return

    ok,res = place_live_order(pair, side, qty)
    if ok:
        log(f"[LIVE] {side} {pair} {qty} @ {price}")
        STATE["positions"][pair]={"entry":price,"qty":qty}
    else:
        log(f"[ERROR] {res}")
        STATE["errors"].append(res)

# ================== POSITION MANAGER ==================
def manage_positions(pair, price):
    if pair not in STATE["positions"]:
        return

    pos=STATE["positions"][pair]
    entry=pos["entry"]

    tp=entry*(1+TP_PCT)
    sl=entry*(1-SL_PCT)

    if price>=tp:
        log(f"[TP HIT] {pair}")
        execute(pair,"SELL",pos["qty"],price)
        del STATE["positions"][pair]

    elif price<=sl:
        log(f"[SL HIT] {pair}")
        execute(pair,"SELL",pos["qty"],price)
        del STATE["positions"][pair]

# ================== BOT LOOP ==================
def bot_loop():
    global bot_running
    log("BOT STARTED")

    while bot_running:
        STATE["last"]=ist_now()
        STATE["balances"]=balances()
        tickers=prices()

        usdt=STATE["balances"].get("USDT",0)
        if usdt < MIN_USDT_BALANCE:
            time.sleep(10)
            continue

        for t in tickers:
            pair=t["market"]
            if pair not in PAIRS:
                continue

            price=float(t["last_price"])

            # manage open trades
            manage_positions(pair,price)

            # cooldown
            if pair in STATE["cooldown"]:
                if time.time()-STATE["cooldown"][pair] < COOLDOWN:
                    continue

            if pair in STATE["positions"]:
                continue

            sig=breakout_signal(pair,price)
            if sig=="BUY":
                qty=size(usdt,price)
                if qty>0:
                    execute(pair,"BUY",qty,price)
                    STATE["cooldown"][pair]=time.time()

        time.sleep(5)

    log("BOT STOPPED")

# ================== ROUTES ==================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/start",methods=["POST"])
def start():
    global bot_running, bot_thread
    if LIVE_TRADING and (not API_KEY or not API_SECRET):
        return jsonify({"ok":False,"error":"API keys missing"}),400

    if not bot_running:
        bot_running=True
        STATE["status"]="Running"
        bot_thread=threading.Thread(target=bot_loop,daemon=True)
        bot_thread.start()
    return jsonify({"ok":True})

@app.route("/stop",methods=["POST"])
def stop():
    global bot_running
    bot_running=False
    STATE["status"]="Stopped"
    return jsonify({"ok":True})

@app.route("/status")
def status():
    return jsonify(STATE)

@app.route("/diagnostics")
def diagnostics():
    return jsonify({
        "time":ist_now(),
        "live":LIVE_TRADING,
        "api_key_loaded":bool(API_KEY),
        "api_secret_loaded":bool(API_SECRET),
        "bot_running":bot_running,
        "health":STATE["health"]
    })

# ================== MAIN ==================
if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)