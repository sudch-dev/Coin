from flask import Flask, jsonify, request, render_template
import threading, time, requests, hmac, hashlib, json, os, sqlite3, math
from datetime import datetime
from pytz import timezone

# ================= APP =================
app = Flask(__name__)

# ================= TIME =================
IST = timezone("Asia/Kolkata")
def ist_now():
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

# ================= CONFIG =================
API_KEY = os.getenv("API_KEY","")
API_SECRET = os.getenv("API_SECRET","")
BASE_URL = "https://api.coindcx.com"

PAIRS = ["BTCUSDT","ETHUSDT","SOLUSDT"]

MODE = "PAPER"   # PAPER | LIVE (controlled by UI)

MAX_RISK_PER_TRADE = 0.1
MIN_USDT_BALANCE = 15
COOLDOWN = 120
ATR_MULT_SL = 1.2
ATR_MULT_TP = 2.5

# ================= DATABASE =================
DB="bot.db"
def db():
    return sqlite3.connect(DB,check_same_thread=False)

def init_db():
    con=db();cur=con.cursor()
    cur.execute("create table if not exists trades(time,pair,side,qty,price,mode)")
    cur.execute("create table if not exists orders(time,pair,side,qty,price,mode,order_id,status)")
    cur.execute("create table if not exists logs(time,msg)")
    con.commit();con.close()
init_db()

# ================= STATE =================
bot_running=False
bot_thread=None

STATE={
 "status":"Idle",
 "mode":"PAPER",
 "last":None,
 "balances":{},
 "positions":{},   # pair -> {entry, qty, sl, tp}
 "cooldown":{},
 "orders":[],      # visible in UI
 "trades":[],
 "logs":[],
 "errors":[],
 "health":{"api":"--","exchange":"--"}
}

# ================= UTILS =================
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

# ================= EXCHANGE =================
def balances():
    data={}
    try:
        r=api_post("/exchange/v1/users/balances",{"timestamp":int(time.time()*1000)})
        if r.ok:
            for b in r.json():
                data[b["currency"]]=float(b["balance"])
            STATE["health"]["api"]="OK"
        else: STATE["health"]["api"]="ERROR"
    except: STATE["health"]["api"]="ERROR"
    return data

def prices():
    try:
        r=requests.get(BASE_URL+"/exchange/ticker",timeout=10)
        if r.ok:
            STATE["health"]["exchange"]="OK"
            return r.json()
        else: STATE["health"]["exchange"]="ERROR"
    except:
        STATE["health"]["exchange"]="ERROR"
    return []

def candles(pair, interval="15m", limit=100):
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

# ================= INDICATORS =================
def ema(data, period):
    k = 2/(period+1)
    ema_val = data[0]
    for p in data[1:]:
        ema_val = p*k + ema_val*(1-k)
    return ema_val

def atr(candles_data, period=14):
    trs=[]
    for i in range(1,len(candles_data)):
        h=float(candles_data[i]["high"])
        l=float(candles_data[i]["low"])
        pc=float(candles_data[i-1]["close"])
        tr=max(h-l,abs(h-pc),abs(l-pc))
        trs.append(tr)
    return sum(trs[-period:])/period if len(trs)>=period else None

# ================= RISK =================
def size(usdt, entry, sl):
    risk_cap = usdt * MAX_RISK_PER_TRADE
    risk_per_unit = abs(entry-sl)
    if risk_per_unit == 0:
        return 0
    qty = risk_cap / risk_per_unit
    return round(qty,6)

# ================= EXECUTION =================
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

def record_order(pair, side, qty, price, mode, order_id, status):
    order={
        "time":ist_now(),
        "pair":pair,
        "side":side,
        "qty":qty,
        "price":price,
        "mode":mode,
        "order_id":order_id,
        "status":status
    }
    STATE["orders"].append(order)
    con=db();cur=con.cursor()
    cur.execute("insert into orders values(?,?,?,?,?,?,?,?)",
                (order["time"],pair,side,qty,price,mode,order_id,status))
    con.commit();con.close()

def execute(pair, side, qty, price):
    mode = STATE["mode"]
    order_id="SIM-"+str(int(time.time()*1000))
    status="FILLED"

    if mode=="LIVE":
        ok,res = place_live_order(pair, side, qty)
        if ok:
            order_id = res.get("id","EXCH-"+str(int(time.time()*1000)))
            status="FILLED"
            log(f"[LIVE] {side} {pair} {qty} @ {price} | ID={order_id}")
        else:
            log(f"[LIVE ERROR] {res}")
            STATE["errors"].append(res)
            status="FAILED"

    else:
        log(f"[PAPER] {side} {pair} {qty} @ {price}")

    record_order(pair,side,qty,price,mode,order_id,status)

    con=db();cur=con.cursor()
    cur.execute("insert into trades values(?,?,?,?,?,?)",
                (ist_now(),pair,side,qty,price,mode))
    con.commit();con.close()

# ================= INSTITUTIONAL ENGINE =================
def institutional_breakout(pair, price):
    c = candles(pair,"15m",100)
    if len(c) < 60:
        return None

    closes=[float(x["close"]) for x in c]
    highs=[float(x["high"]) for x in c]
    lows=[float(x["low"]) for x in c]
    vols=[float(x["volume"]) for x in c]

    ema50 = ema(closes[-60:],50)
    ema200 = ema(closes[-60:],200)
    trend_ok = ema50 > ema200

    range_high = max(highs[-20:])
    range_low  = min(lows[-20:])

    avg_vol = sum(vols[-20:])/20
    last_vol = vols[-1]

    a = atr(c,14)
    if not a:
        return None

    sweep = lows[-1] < range_low and closes[-1] > range_low
    bos = price > range_high
    vol_ok = last_vol > avg_vol*1.8

    if trend_ok and bos and vol_ok and not sweep:
        sl = price - a*ATR_MULT_SL
        tp = price + a*ATR_MULT_TP
        return {"side":"BUY","sl":sl,"tp":tp}

    return None

# ================= POSITION MANAGER =================
def manage_positions(pair, price):
    if pair not in STATE["positions"]:
        return

    pos = STATE["positions"][pair]
    if price <= pos["sl"]:
        execute(pair,"SELL",pos["qty"],price)
        log(f"[SL EXIT] {pair}")
        del STATE["positions"][pair]

    elif price >= pos["tp"]:
        execute(pair,"SELL",pos["qty"],price)
        log(f"[TP EXIT] {pair}")
        del STATE["positions"][pair]

# ================= BOT LOOP =================
def bot_loop():
    global bot_running
    log("BOT STARTED")

    while bot_running:
        STATE["last"]=ist_now()
        STATE["balances"]=balances()
        tickers=prices()

        usdt=STATE["balances"].get("USDT",0)
        if usdt < MIN_USDT_BALANCE:
            time.sleep(5)
            continue

        for t in tickers:
            pair=t["market"]
            if pair not in PAIRS:
                continue

            price=float(t["last_price"])

            manage_positions(pair,price)

            if pair in STATE["positions"]:
                continue

            if pair in STATE["cooldown"]:
                if time.time()-STATE["cooldown"][pair] < COOLDOWN:
                    continue

            sig = institutional_breakout(pair,price)
            if sig:
                qty = size(usdt,price,sig["sl"])
                if qty>0:
                    execute(pair,"BUY",qty,price)
                    STATE["positions"][pair]={
                        "entry":price,
                        "qty":qty,
                        "sl":sig["sl"],
                        "tp":sig["tp"]
                    }
                    STATE["cooldown"][pair]=time.time()

        time.sleep(5)

    log("BOT STOPPED")

# ================= ROUTES =================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/start",methods=["POST"])
def start():
    global bot_running, bot_thread
    if STATE["mode"]=="LIVE" and (not API_KEY or not API_SECRET):
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

@app.route("/set_mode",methods=["POST"])
def set_mode():
    data=request.json
    mode=data.get("mode","PAPER")
    if mode not in ["PAPER","LIVE"]:
        return jsonify({"ok":False})
    STATE["mode"]=mode
    log(f"MODE CHANGED TO {mode}")
    return jsonify({"ok":True})

@app.route("/status")
def status():
    return jsonify(STATE)

# ================= MAIN =================
if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)