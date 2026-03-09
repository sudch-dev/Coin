import os
import time
import json
import hmac
import hashlib
import threading
import statistics
import requests

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# ==================================
# CONFIG
# ==================================

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

BASE_URL = "https://api.coindcx.com"
SERVER_URL = "https://coin-4k37.onrender.com"

SYMBOLS = ["BTCINR","ETHINR","SOLINR"]

TRADE_SIZE = 0.001

MODE = "paper"
BOT_RUNNING = True

trade_log = []
error_log = []


# ==================================
# SIGNATURE
# ==================================

def sign(payload):

    payload_json = json.dumps(payload,separators=(',',':'))

    signature = hmac.new(
        bytes(API_SECRET,'utf-8'),
        bytes(payload_json,'utf-8'),
        hashlib.sha256
    ).hexdigest()

    return signature,payload_json


# ==================================
# MARKET DATA
# ==================================

def get_price(symbol):

    try:

        r=requests.get(BASE_URL+"/exchange/ticker").json()

        for i in r:
            if i["market"]==symbol:
                return float(i["last_price"])

    except Exception as e:
        error_log.append(str(e))

    return 0


def get_candles(symbol):

    try:

        url=f"https://public.coindcx.com/market_data/candles?pair={symbol}&interval=1m"

        r=requests.get(url).json()

        closes=[float(x[4]) for x in r[-60:]]

        return closes

    except Exception as e:

        error_log.append(str(e))

        return []


# ==================================
# INDICATORS
# ==================================

def rsi(prices,period=14):

    if len(prices)<period+1:
        return 50

    gains=[]
    losses=[]

    for i in range(1,period+1):

        diff=prices[-i]-prices[-i-1]

        if diff>0:
            gains.append(diff)
        else:
            losses.append(abs(diff))

    avg_gain=sum(gains)/period if gains else 0.001
    avg_loss=sum(losses)/period if losses else 0.001

    rs=avg_gain/avg_loss

    return 100-(100/(1+rs))


def vwap(prices):

    return statistics.mean(prices)


# ==================================
# SIGNAL ENGINE
# ==================================

def generate_signal(symbol):

    prices=get_candles(symbol)

    if len(prices)<40:
        return "HOLD"

    r=rsi(prices)

    vw=vwap(prices)

    price=prices[-1]

    if r<35 and price>vw:
        return "BUY"

    if r>65 and price<vw:
        return "SELL"

    return "HOLD"


# ==================================
# ORDER EXECUTION
# ==================================

def place_order(symbol,side):

    global MODE

    if MODE=="paper":

        trade_log.append(f"PAPER {side} {symbol}")

        return

    try:

        payload={
            "side":side,
            "order_type":"market_order",
            "market":symbol,
            "total_quantity":TRADE_SIZE,
            "timestamp":int(time.time()*1000)
        }

        signature,payload_json=sign(payload)

        headers={
            "Content-Type":"application/json",
            "X-AUTH-APIKEY":API_KEY,
            "X-AUTH-SIGNATURE":signature
        }

        url=BASE_URL+"/exchange/v1/orders/create"

        r=requests.post(url,data=payload_json,headers=headers)

        trade_log.append(f"LIVE {side} {symbol}")

        return r.json()

    except Exception as e:

        error_log.append(str(e))


# ==================================
# TRADING ENGINE
# ==================================

def trading_engine():

    global BOT_RUNNING

    while True:

        try:

            if BOT_RUNNING:

                for symbol in SYMBOLS:

                    signal=generate_signal(symbol)

                    if signal!="HOLD":

                        place_order(symbol,signal.lower())

                    time.sleep(5)

        except Exception as e:

            error_log.append(str(e))

        time.sleep(30)


# ==================================
# KEEP ALIVE
# ==================================

def keep_alive():

    while True:

        try:
            requests.get(SERVER_URL+"/keepalive")
        except:
            pass

        time.sleep(240)


# ==================================
# ROUTES
# ==================================

@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/signals")
def signals():

    s={}

    for sym in SYMBOLS:
        s[sym]=generate_signal(sym)

    return jsonify(s)


@app.route("/price/<symbol>")
def price(symbol):

    return {"price":get_price(symbol)}


@app.route("/start")
def start_bot():

    global BOT_RUNNING
    BOT_RUNNING=True

    return {"status":"started"}


@app.route("/stop")
def stop_bot():

    global BOT_RUNNING
    BOT_RUNNING=False

    return {"status":"stopped"}


@app.route("/mode",methods=["GET","POST"])
def mode():

    global MODE

    if request.method=="POST":

        MODE=request.json["mode"]

    return {"mode":MODE}


@app.route("/trade_log")
def trade_logs():

    return jsonify(trade_log[-20:])


@app.route("/error_log")
def error_logs():

    return jsonify(error_log[-20:])


@app.route("/keepalive")
def keepalive():

    return {"alive":True}


@app.route("/health")
def health():

    return {"status":"running"}


# ==================================
# START THREADS
# ==================================

def start_threads():

    t1=threading.Thread(target=trading_engine)
    t1.daemon=True
    t1.start()

    t2=threading.Thread(target=keep_alive)
    t2.daemon=True
    t2.start()

start_threads()


# ==================================
# RUN SERVER
# ==================================

if __name__=="__main__":

    port=int(os.environ.get("PORT",10000))

    app.run(host="0.0.0.0",port=port)