# ========== IMPORTS ==========
from flask import Flask, jsonify, request, render_template
import threading, time, requests, hmac, hashlib, json, os, sqlite3
from datetime import datetime
from pytz import timezone

# ========== APP ==========
app = Flask(__name__)

# ========== TIMEZONE ==========
IST = timezone("Asia/Kolkata")
def ist_now():
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

# ========== CONFIG ==========
API_KEY = os.getenv("API_KEY","")
API_SECRET = os.getenv("API_SECRET","")
BASE_URL = "https://api.coindcx.com"
PAIRS = ["BTCUSDT","ETHUSDT","SOLUSDT"]

# ========== DATABASE ==========
DB="bot.db"
def db():
    return sqlite3.connect(DB,check_same_thread=False)

def init_db():
    con=db();cur=con.cursor()
    cur.execute("create table if not exists trades(time,pair,side,qty,price)")
    cur.execute("create table if not exists orders(time,pair,side,qty,status)")
    cur.execute("create table if not exists logs(time,msg)")
    con.commit();con.close()

init_db()

# ========== BOT STATE ==========
bot_running=False
bot_thread=None

STATE={
 "status":"Idle","last":None,
 "balances":{},"positions":{},
 "orders":[],"trades":[],
 "logs":[],"errors":[],
 "health":{"api":"--","exchange":"--"}
}

# ========== UTILS ==========
def log(msg):
    STATE["logs"].append(f"{ist_now()} | {msg}")
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

# ========== EXCHANGE ==========
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
    except: STATE["health"]["exchange"]="ERROR"
    return []

# ========== STRATEGY ==========
def strategy(price, pair):
    # simple auto strategy
    import random
    r=random.random()
    if r>0.98: return "BUY"
    if r<0.02: return "SELL"
    return None

# ========== RISK ==========
def size(usdt, price):
    risk=usdt*0.1
    return round(risk/price,6)

# ========== EXECUTION ==========
def execute(pair, side, qty, price):
    order={"time":ist_now(),"pair":pair,"side":side,"qty":qty,"price":price}
    STATE["orders"].append(order)
    STATE["trades"].append(order)
    con=db();cur=con.cursor()
    cur.execute("insert into trades values(?,?,?,?,?)",(order["time"],pair,side,qty,price))
    con.commit();con.close()
    log(f"{side} {pair} {qty} @ {price}")

# ========== BOT LOOP ==========
def bot_loop():
    global bot_running
    log("BOT STARTED")
    while bot_running:
        STATE["last"]=ist_now()
        STATE["balances"]=balances()
        tickers=prices()

        usdt=STATE["balances"].get("USDT",0)

        for t in tickers:
            if t["market"] in PAIRS:
                price=float(t["last_price"])
                sig=strategy(price,t["market"])
                if sig:
                    qty=size(usdt,price)
                    if qty>0:
                        execute(t["market"],sig,qty,price)

        time.sleep(5)

    log("BOT STOPPED")

# ========== ROUTES ==========
@app.route("/")
def home(): return render_template("index.html")

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
        "health":STATE["health"]
    })

# ========== MAIN ==========
if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)