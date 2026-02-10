from flask import Flask, jsonify, request, render_template
import threading, time, requests, hmac, hashlib, json, os, sqlite3
from datetime import datetime, date
from pytz import timezone

# ================== APP ==================
app = Flask(__name__)

# ================== TIME ==================
IST = timezone("Asia/Kolkata")
def ist_now():
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

def today():
    return date.today().isoformat()

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
    cur.execute("""create table if not exists trades(
        time,pair,side,qty,price,mode,order_id,status,pnl,day)""")
    cur.execute("create table if not exists logs(time,msg)")
    con.commit();con.close()

init_db()

# ================== GLOBAL STATE ==================
bot_running=False
bot_thread=None
KILL_SWITCH=False
KEEPALIVE=True

STATE={
 "status":"Idle",
 "mode":"PAPER",
 "last":None,
 "balances":{},
 "positions":{},
 "orders":[],
 "logs":[],
 "health":{"api":"--","exchange":"--","heartbeat":"--"},
 "risk":{
    "daily_pnl":0.0,
    "loss_streak":0,
    "trades_today":0,
    "max_dd":-0.05,
    "max_trades":10,
    "max_loss_streak":3,
    "max_exposure":0.2
 }
}

# ================== UTILS ==================
def log(msg):
    entry=f"{ist_now()} | {msg}"
    STATE["logs"].append(entry)
    try:
        con=db();cur=con.cursor()
        cur.execute("insert into logs values(?,?)",(ist_now(),msg))
        con.commit();con.close()
    except:
        pass

def sign(payload):
    return hmac.new(API_SECRET.encode(),payload.encode(),hashlib.sha256).hexdigest()

def api_post(path,payload):
    body=json.dumps(payload)
    headers={
        "X-AUTH-APIKEY":API_KEY,
        "X-AUTH-SIGNATURE":sign(body),
        "Content-Type":"application/json"
    }
    return requests.post(BASE_URL+path,headers=headers,data=body,timeout=10)

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
        r=requests.get(BASE_URL+"/exchange/ticker",timeout=8)
        if r.ok:
            STATE["health"]["exchange"]="OK"
            return r.json()
        else:
            STATE["health"]["exchange"]="ERROR"
    except:
        STATE["health"]["exchange"]="ERROR"
    return []

# ================== MARKET DATA ==================
MARKET_DATA={}

def update_market(t):
    pair=t["market"]
    price=float(t["last_price"])
    vol=float(t.get("volume",0))
    if pair not in MARKET_DATA:
        MARKET_DATA[pair]={
            "high":price,"low":price,
            "avg_vol":vol,"last":price
        }
    d=MARKET_DATA[pair]
    d["high"]=max(d["high"],price)
    d["low"]=min(d["low"],price)
    d["avg_vol"]=(d["avg_vol"]*0.9)+(vol*0.1)
    d["last"]=price
    return d

# ================== ENGINES ==================
def vol_filter(vol, avg_vol):
    return vol > avg_vol*0.8

def breakout_engine(price, high, low, vol, avg_vol):
    if price > high*1.001 and vol > avg_vol*1.7:
        return "BUY"
    if price < low*0.999 and vol > avg_vol*1.7:
        return "SELL"
    return None

def capital_allocator(usdt):
    return usdt * 0.02

def size(usdt, price):
    cap = capital_allocator(usdt)
    if price<=0: return 0
    return round(cap/price,6)

def risk_firewall():
    r=STATE["risk"]
    if r["loss_streak"] >= r["max_loss_streak"]:
        return False
    if r["trades_today"] >= r["max_trades"]:
        return False
    return True

# ================== EXECUTION ==================
def execute(pair, side, qty, price):
    mode=STATE["mode"]

    order={
        "time":ist_now(),"pair":pair,"side":side,
        "qty":qty,"price":price,"mode":mode,
        "order_id":None,"status":"FILLED" if mode=="PAPER" else "SENT"
    }

    if mode=="LIVE":
        payload={
          "side":side.lower(),
          "order_type":"market",
          "market":pair,
          "quantity":qty,
          "timestamp":int(time.time()*1000)
        }
        try:
            r=api_post("/exchange/v1/orders/create",payload)
            if r.ok:
                order["order_id"]=r.json().get("id")
                order["status"]="LIVE"
            else:
                order["status"]="REJECTED"
        except:
            order["status"]="FAILED"

    STATE["orders"].append(order)
    update_position(pair,side,qty,price)

# ================== POSITION ENGINE ==================
def update_position(pair, side, qty, price):
    if side=="BUY":
        STATE["positions"][pair]={
            "entry":price,
            "qty":qty,
            "sl":round(price*0.985,4),
            "tp":round(price*1.03,4),
            "peak":price
        }

# ================== EXIT ENGINE ==================
def exit_engine(pair, price, vol, avg_vol):
    if pair not in STATE["positions"]:
        return

    pos=STATE["positions"][pair]
    entry,sl,tp,qty=pos["entry"],pos["sl"],pos["tp"],pos["qty"]

    if price<=sl:
        exit_trade(pair,qty,price,"SL"); return

    if price>=tp:
        exit_trade(pair,qty,price,"TP"); return

    if price>=entry*1.015:
        pos["sl"]=max(pos["sl"],entry)

    if price>pos["peak"]:
        pos["peak"]=price
        pos["sl"]=max(pos["sl"],round(price*0.992,4))

    if price<entry*0.995 and vol<avg_vol*0.6:
        exit_trade(pair,qty,price,"FAKE_BREAKOUT")

def exit_trade(pair, qty, price, reason):
    mode=STATE["mode"]
    order={
        "time":ist_now(),"pair":pair,"side":"SELL",
        "qty":qty,"price":price,"mode":mode,
        "order_id":None,"status":"EXIT_"+reason
    }

    if mode=="LIVE":
        payload={
          "side":"sell","order_type":"market",
          "market":pair,"quantity":qty,
          "timestamp":int(time.time()*1000)
        }
        try:
            r=api_post("/exchange/v1/orders/create",payload)
            if r.ok:
                order["order_id"]=r.json().get("id")
                order["status"]="LIVE_EXIT_"+reason
        except:
            order["status"]="FAILED_EXIT"

    STATE["orders"].append(order)
    STATE["positions"].pop(pair,None)
    STATE["risk"]["loss_streak"]=0 if "TP" in reason else STATE["risk"]["loss_streak"]+1
    log(f"EXIT {pair} {reason}")

# ================== BOT LOOP ==================
def bot_loop():
    global bot_running
    log("ENGINE STARTED")
    while bot_running:
        if KILL_SWITCH:
            time.sleep(1); continue

        STATE["last"]=ist_now()
        STATE["balances"]=balances()
        tickers=prices()
        usdt=STATE["balances"].get("USDT",0)

        for t in tickers:
            if t["market"] in PAIRS:
                d=update_market(t)
                price=d["last"]
                vol=float(t.get("volume",0))

                exit_engine(t["market"],price,vol,d["avg_vol"])

                if t["market"] not in STATE["positions"]:
                    if not risk_firewall(): 
                        continue
                    if not vol_filter(vol,d["avg_vol"]):
                        continue

                    sig=breakout_engine(price,d["high"],d["low"],vol,d["avg_vol"])
                    if sig:
                        qty=size(usdt,price)
                        if qty>0:
                            execute(t["market"],sig,qty,price)
                            STATE["risk"]["trades_today"]+=1

        time.sleep(5)
    log("ENGINE STOPPED")

# ================== SAFE KEEPALIVE ==================
def keepalive_loop():
    while True:
        STATE["health"]["heartbeat"]=ist_now()
        time.sleep(15)

# ================== WATCHDOG ==================
def watchdog_loop():
    global bot_running, bot_thread
    while True:
        try:
            if bot_running:
                if bot_thread is None or not bot_thread.is_alive():
                    log("WATCHDOG | Engine thread died — restarting")
                    bot_thread=threading.Thread(target=bot_loop,daemon=True)
                    bot_thread.start()
        except:
            pass
        time.sleep(10)

threading.Thread(target=keepalive_loop, daemon=True).start()
threading.Thread(target=watchdog_loop, daemon=True).start()

# ================== ROUTES ==================
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

@app.route("/set_mode",methods=["POST"])
def set_mode():
    data=request.json
    if data["mode"] in ["PAPER","LIVE"]:
        STATE["mode"]=data["mode"]
        log(f"MODE SET TO {data['mode']}")
    return jsonify({"ok":True})

@app.route("/kill",methods=["POST"])
def kill():
    global KILL_SWITCH
    KILL_SWITCH=True
    log("KILL SWITCH ACTIVATED")
    return jsonify({"ok":True})

@app.route("/flatten",methods=["POST"])
def flatten():
    for pair,pos in list(STATE["positions"].items()):
        exit_trade(pair,pos["qty"],pos["entry"],"EMERGENCY")
    log("EMERGENCY FLATTEN")
    return jsonify({"ok":True})

@app.route("/ping")
def ping():
    return jsonify({
        "status":"alive",
        "time":ist_now(),
        "engine":STATE["status"],
        "mode":STATE["mode"],
        "heartbeat":STATE["health"]["heartbeat"]
    })

@app.route("/status")
def status(): 
    return jsonify(STATE)

# ================== MAIN ==================
if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000,threaded=True)