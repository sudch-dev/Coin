import os
import time
import json
import hmac
import hashlib
import threading
import requests
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
RENDER_URL = os.getenv("RENDER_URL","https://coin-4k37.onrender.com")

SYMBOLS = ["BTCINR","ETHINR","SOLINR"]

MARKET_MAP = {
"BTCINR":"B-BTC_INR",
"ETHINR":"B-ETH_INR",
"SOLINR":"B-SOL_INR"
}

bot_running = False
paper_mode = True
capital = 1000

logs=[]
positions={}

########################################
# KEEP ALIVE
########################################

def self_keepalive():
    while True:
        try:
            requests.get(f"{RENDER_URL}/ping",timeout=5)
        except:
            pass
        time.sleep(240)

threading.Thread(target=self_keepalive,daemon=True).start()

########################################
# LOG
########################################

def log(msg):
    global logs
    logs.append(f"{time.strftime('%H:%M:%S')} - {msg}")
    logs=logs[-100:]

########################################
# MARKET DATA
########################################

def get_candles(symbol):

    try:

        url=f"https://public.coindcx.com/market_data/candles?pair={symbol}&interval=1m"

        r=requests.get(url)

        data=r.json()

        df=pd.DataFrame(data)

        df["open"]=df["open"].astype(float)
        df["high"]=df["high"].astype(float)
        df["low"]=df["low"].astype(float)
        df["close"]=df["close"].astype(float)

        return df

    except:
        return None

########################################
# SMC ENGINE
########################################

def detect_bos(df):

    if len(df)<6:
        return None

    if df["high"].iloc[-1] > df["high"].iloc[-5]:
        return "BULL"

    if df["low"].iloc[-1] < df["low"].iloc[-5]:
        return "BEAR"

    return None

def detect_choch(df):

    df["ema20"]=df["close"].ewm(span=20).mean()

    if df["close"].iloc[-1] < df["ema20"].iloc[-1]:
        return "BEAR"

    if df["close"].iloc[-1] > df["ema20"].iloc[-1]:
        return "BULL"

    return None

def detect_order_block(df):

    last=df.iloc[-2]

    if last["close"]>last["open"]:
        return "BULLISH_OB"

    if last["close"]<last["open"]:
        return "BEARISH_OB"

    return None

########################################
# SIGNAL
########################################

def compute_signal(symbol):

    df=get_candles(symbol)

    if df is None:
        return None

    df["ema20"]=df["close"].ewm(span=20).mean()
    df["ema50"]=df["close"].ewm(span=50).mean()

    trend="LONG" if df["ema20"].iloc[-1] > df["ema50"].iloc[-1] else "SHORT"

    bos=detect_bos(df)
    choch=detect_choch(df)
    ob=detect_order_block(df)

    return {
    "trend":trend,
    "bos":bos,
    "choch":choch,
    "ob":ob,
    "price":df["close"].iloc[-1]
    }

########################################
# ORDER
########################################

def place_order(symbol,side,price):

    qty=capital/price

    if paper_mode:

        log(f"PAPER {side} {symbol} @ {price}")

        positions[symbol]={
        "side":side,
        "entry":price
        }

        return

    try:

        market=MARKET_MAP[symbol]

        payload={
        "side":side,
        "order_type":"market_order",
        "market":market,
        "total_quantity":qty,
        "timestamp":int(time.time()*1000)
        }

        json_payload=json.dumps(payload)

        signature=hmac.new(
        bytes(API_SECRET,'utf-8'),
        json_payload.encode(),
        hashlib.sha256
        ).hexdigest()

        headers={
        "X-AUTH-APIKEY":API_KEY,
        "X-AUTH-SIGNATURE":signature
        }

        r=requests.post(
        "https://api.coindcx.com/exchange/v1/orders/create",
        data=json_payload,
        headers=headers
        )

        log(r.text)

    except Exception as e:

        log(str(e))

########################################
# ENGINE
########################################

def trading_loop():

    global bot_running

    while True:

        if not bot_running:
            time.sleep(5)
            continue

        for symbol in SYMBOLS:

            signal=compute_signal(symbol)

            if not signal:
                continue

            price=signal["price"]

            ################################
            # ENTRY
            ################################

            if symbol not in positions:

                if signal["ob"]=="BULLISH_OB" and signal["trend"]=="LONG":

                    place_order(symbol,"buy",price)

                if signal["ob"]=="BEARISH_OB" and signal["trend"]=="SHORT":

                    place_order(symbol,"sell",price)

            ################################
            # MANAGEMENT
            ################################

            else:

                entry=positions[symbol]["entry"]

                pnl=(price-entry)/entry

                ################################
                # CONTINUATION
                ################################

                if signal["bos"]:
                    log(f"{symbol} BoS detected → continue")

                ################################
                # EXIT
                ################################

                if signal["choch"]:

                    log(f"{symbol} CHoCH exit")

                    del positions[symbol]

                elif abs(pnl)>0.02:

                    log(f"{symbol} TP/SL exit")

                    del positions[symbol]

        time.sleep(60)

threading.Thread(target=trading_loop,daemon=True).start()

########################################
# ROUTES
########################################

@app.route("/")
def home():

    return render_template(
    "index.html",
    capital=capital,
    paper_mode=paper_mode
    )

@app.route("/start")
def start():

    global bot_running
    bot_running=True

    log("BOT STARTED")

    return "started"

@app.route("/stop")
def stop():

    global bot_running
    bot_running=False

    log("BOT STOPPED")

    return "stopped"

@app.route("/toggle_mode")
def toggle_mode():

    global paper_mode
    paper_mode=not paper_mode

    log("MODE "+("PAPER" if paper_mode else "LIVE"))

    return "ok"

@app.route("/set_capital",methods=["POST"])
def set_capital():

    global capital
    capital=float(request.form["capital"])

    log(f"Capital set {capital}")

    return "ok"

@app.route("/status")
def status():

    return jsonify({
    "running":bot_running,
    "paper":paper_mode,
    "capital":capital,
    "positions":positions,
    "logs":logs
    })

@app.route("/ping")
def ping():
    return "pong"
    
    # AUTO START BOT AFTER SERVER RESTART

AUTO_START = True

def auto_start_bot():
    global bot_running

    time.sleep(10)

    if AUTO_START:
        bot_running = True
        log("AUTO RESTART BOT")

threading.Thread(target=auto_start_bot, daemon=True).start()

######################

STATE_FILE="state.json"

def save_state():
    with open(STATE_FILE,"w") as f:
        json.dump(positions,f)

def load_state():
    global positions
    try:
        with open(STATE_FILE) as f:
            positions=json.load(f)
    except:
        positions={}

########################################

if __name__=="__main__":

    port=int(os.environ.get("PORT",10000))

    app.run(host="0.0.0.0",port=port)