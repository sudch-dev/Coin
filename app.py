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

# ================== DATABASE ==================
DB="bot.db"
def db():
    return sqlite3.connect(DB,check_same_thread=False)

def init_db():
    con=db();cur=con.cursor()
    cur.execute("create table if not exists trades(time,pair,side,qty,price,mode,order_id,status)")
    cur.execute("create table if not exists logs(time,msg)")
    con.commit();con.close()

init_db()

# ================== STATE MACHINE ==================
bot_running=False
bot_thread=None

STATE={
 "status":"Idle",
 "mode":"PAPER",          # PAPER | LIVE
 "last":None,
 "balances":{},
 "positions":{},
 "orders":[],
 "logs":[],
 "health":{"api":"--","exchange":"--"}
}

# ================== UTILS ==================
def log(msg):
    entry=f"{ist_now()} | {msg}"
    STATE["logs"].append(entry)
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

# ================== DATA ENGINE ==================
MARKET_DATA={}

def update_market(t):
    pair=t["market"]
    price=float(t["last_price"])
    vol=float(t.get("volume",0))
    if pair not in MARKET_DATA:
        MARKET_DATA[pair]={
            "high":price,
            "low":price,
            "avg_vol":vol,
            "last":price
        }
    d=MARKET_DATA[pair]
    d["high"]=max(d["high"],price)
    d["low"]=min(d["low"],price)
    d["avg_vol"]=(d["avg_vol"]*0.9)+(vol*0.1)
    d["last"]=price
    return d

# ================== INSTITUTIONAL BREAKOUT ENGINE ==================
def breakout_engine(price, high, low, vol, avg_vol):
    # Structure break + volume expansion + momentum continuation
    if price > high*1.001 and vol > avg_vol*1.7:
        return "BUY"
    if price < low*0.999 and vol > avg_vol*1.7:
        return "SELL"
    return None

# ================== RISK ENGINE ==================
def size(usdt, price):
    risk_cap = usdt * 0.02   # 2% capital risk
    if price<=0: return 0
    return round(risk_cap/price,6)

# ================== EXECUTION ROUTER ==================
def execute(pair, side, qty, price):
    mode = STATE["mode"]

    order={
        "time":ist_now(),
        "pair":pair,
        "side":side,
        "qty":qty,
        "price":price,
        "mode":mode,
        "order_id":None,
        "status":"FILLED" if mode=="PAPER" else "SENT"
    }

    # ===== LIVE ORDER ROUTING =====
    if mode=="LIVE":
        payload={
          "side": side.lower(),
          "order_type":"market",
          "market": pair,
          "quantity": qty,
          "timestamp": int(time.time()*1000)
        }
        try:
            r=api_post("/exchange/v1/orders/create",payload)
            if r.ok:
                j=r.json()
                order["order_id"]=j.get("id")
                order["status"]="LIVE"
                log(f"LIVE ORDER {side} {pair} {qty}")
            else:
                order["status"]="REJECTED"
                log(f"ORDER REJECTED {pair}")
        except:
            order["status"]="FAILED"
            log(f"ORDER FAILED {pair}")

    STATE["orders"].append(order)

    con=db();cur=con.cursor()
    cur.execute("insert into trades values(?,?,?,?,?,?,?,?)",
        (order["time"],pair,side,qty,price,mode,order["order_id"],order["status"]))
    con.commit();con.close()

    update_position(pair,side,qty,price)

# ================== POSITION ENGINE ==================
def update_position(pair, side, qty, price):
    if side=="BUY":
        STATE["positions"][pair]={
            "entry":price,
            "qty":qty,
            "sl": round(price*0.985,4),
            "tp": round(price*1.03,4)
        }
    if side=="SELL":
        STATE["positions"].pop(pair,None)

# ================== BOT LOOP ==================
def bot_loop():
    global bot_running
    log("ENGINE STARTED")
    while bot_running:
        STATE["last"]=ist_now()
        STATE["balances"]=balances()
        tickers=prices()

        usdt=STATE["balances"].get("USDT",0)

        for t in tickers:
            if t["market"] in PAIRS:
                d=update_market(t)
                price=d["last"]
                sig = breakout_engine(price,d["high"],d["low"],float(t.get("volume",0)),d["avg_vol"])

                if sig:
                    qty=size(usdt,price)
                    if qty>0:
                        execute(t["market"],sig,qty,price)

        time.sleep(5)

    log("ENGINE STOPPED")

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
    if data["mode"] in ["PAPER","LIVE"]:
        STATE["mode"]=data["mode"]
        log(f"MODE SET TO {data['mode']}")
    return jsonify({"ok":True,"mode":STATE["mode"]})

@app.route("/status")
def status():
    return jsonify(STATE)

@app.route("/diagnostics")
def diagnostics():
    return jsonify({
        "time":ist_now(),
        "api_key_loaded":bool(API_KEY),
        "api_secret_loaded":bool(API_SECRET),
        "bot_running":bot_running,
        "mode":STATE["mode"],
        "health":STATE["health"]
    })

# ================== MAIN ==================
if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000,debug=False)