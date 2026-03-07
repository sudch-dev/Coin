import os
import time
import json
import hmac
import hashlib
import threading
import requests
import pandas as pd
from flask import Flask, render_template, request, jsonify

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

RENDER_URL = "https://coin-4k37.onrender.com"
BASE_URL = "https://api.coindcx.com"

SYMBOLS = ["BTCINR","ETHINR","SOLINR"]

capital = 0
risk = 1
tp_percent = 2

positions = {}
entry_price = {}

bot_running = False

app = Flask(__name__)

# -----------------------------
# KEEP ALIVE
# -----------------------------

@app.route("/ping")
def ping():
    return "pong"

def self_keepalive():
    while True:
        try:
            requests.get(f"{RENDER_URL}/ping", timeout=5)
        except:
            pass
        time.sleep(240)

threading.Thread(target=self_keepalive, daemon=True).start()

@app.route("/test_api")
def test_api():

    try:

        data = requests.get(
            "https://api.coindcx.com/exchange/ticker",
            timeout=10
        ).json()

        return {
            "status":"connected",
            "markets":len(data)
        }

    except Exception as e:

        return {
            "status":"error",
            "message":str(e)
        }


# -----------------------------
# SIGNATURE
# -----------------------------

def sign_payload(payload):

    secret = bytes(API_SECRET,'utf-8')
    payload_json = json.dumps(payload, separators=(',',':'))

    signature = hmac.new(
        secret,
        payload_json.encode(),
        hashlib.sha256
    ).hexdigest()

    return payload_json, signature

# -----------------------------
# MARKET DATA
# -----------------------------

def get_price(symbol):

    data = requests.get(
        f"{BASE_URL}/exchange/ticker"
    ).json()

    for d in data:
        if d["market"] == symbol:
            return float(d["last_price"])

    return None

# -----------------------------
# CANDLES
# -----------------------------

def get_candles(symbol):

    url = f"https://public.coindcx.com/market_data/candles?pair={symbol}&interval=15m"

    data = requests.get(url).json()

    df = pd.DataFrame(data)

    df["close"]=df["close"].astype(float)
    df["high"]=df["high"].astype(float)
    df["low"]=df["low"].astype(float)

    return df.tail(200)

# -----------------------------
# INDICATORS
# -----------------------------

def ema(series,period):
    return series.ewm(span=period).mean()

# -----------------------------
# ORDER BLOCK
# -----------------------------

def detect_ob(df):

    candle = df.iloc[-2]

    if candle["close"] < candle["open"]:
        return "bullish"

    if candle["close"] > candle["open"]:
        return "bearish"

    return None

# -----------------------------
# BOS
# -----------------------------

def detect_bos(df):

    prev_high = df["high"].iloc[-3]
    prev_low = df["low"].iloc[-3]

    close = df["close"].iloc[-1]

    if close > prev_high:
        return "bullish"

    if close < prev_low:
        return "bearish"

    return None

# -----------------------------
# CHOCH
# -----------------------------

def detect_choch(df):

    prev_high = df["high"].iloc[-3]
    prev_low = df["low"].iloc[-3]

    close = df["close"].iloc[-1]

    if close < prev_low:
        return "bearish"

    if close > prev_high:
        return "bullish"

    return None

# -----------------------------
# ENTRY LOGIC
# -----------------------------

def entry_logic(symbol):

    df = get_candles(symbol)

    df["ema9"]=ema(df["close"],9)
    df["ema21"]=ema(df["close"],21)

    ob = detect_ob(df)

    if df["ema9"].iloc[-1] > df["ema21"].iloc[-1] and ob=="bullish":
        return "buy"

    if df["ema9"].iloc[-1] < df["ema21"].iloc[-1] and ob=="bearish":
        return "sell"

    return None

# -----------------------------
# EXIT LOGIC
# -----------------------------

def exit_logic(symbol):

    df = get_candles(symbol)

    choch = detect_choch(df)
    ob = detect_ob(df)

    side = positions.get(symbol)

    if choch:
        return True

    if side=="buy" and ob=="bearish":
        return True

    if side=="sell" and ob=="bullish":
        return True

    return False

# -----------------------------
# ORDER
# -----------------------------

def place_order(symbol,side,quantity):

    timestamp = int(time.time()*1000)

    payload={
        "side":side,
        "order_type":"market_order",
        "market":symbol,
        "total_quantity":quantity,
        "timestamp":timestamp
    }

    payload_json,signature = sign_payload(payload)

    headers={
        "Content-Type":"application/json",
        "X-AUTH-APIKEY":API_KEY,
        "X-AUTH-SIGNATURE":signature
    }

    r = requests.post(
        f"{BASE_URL}/exchange/v1/orders/create",
        data=payload_json,
        headers=headers
    )

    return r.json()

# -----------------------------
# ENGINE
# -----------------------------

def trading_engine():

    global capital,bot_running

    while True:

        if not bot_running or capital==0:
            time.sleep(5)
            continue

        for symbol in SYMBOLS:

            try:

                price = get_price(symbol)

                if symbol not in positions:

                    signal = entry_logic(symbol)

                    if signal:

                        qty = round((capital*(risk/100))/price,6)

                        place_order(symbol,signal,qty)

                        positions[symbol]=signal
                        entry_price[symbol]=price

                else:

                    entry = entry_price[symbol]

                    if price >= entry*(1+tp_percent/100) or exit_logic(symbol):

                        qty = round((capital*(risk/100))/entry,6)

                        side = "sell" if positions[symbol]=="buy" else "buy"

                        place_order(symbol,side,qty)

                        del positions[symbol]

            except Exception as e:
                print(e)

        time.sleep(30)

threading.Thread(target=trading_engine,daemon=True).start()

# -----------------------------
# ROUTES
# -----------------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/set_config",methods=["POST"])
def set_config():

    global capital,risk,tp_percent

    data=request.json

    capital=float(data["capital"])
    risk=float(data["risk"])
    tp_percent=float(data["tp"])

    return jsonify({"status":"saved"})

@app.route("/start")
def start():

    global bot_running
    bot_running=True

    return jsonify({"status":"started"})

@app.route("/stop")
def stop():

    global bot_running
    bot_running=False

    return jsonify({"status":"stopped"})

@app.route("/status")
def status():

    return jsonify({
        "capital":capital,
        "positions":positions,
        "entry":entry_price,
        "running":bot_running
    })

# -----------------------------
# MAIN
# -----------------------------

if __name__=="__main__":

    port=int(os.environ.get("PORT",10000))

    app.run(host="0.0.0.0",port=port)