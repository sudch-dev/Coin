import os
import time
import json
import hmac
import hashlib
import threading
import statistics
import requests

from flask import Flask, jsonify, render_template

app = Flask(__name__)

# =================================
# CONFIG
# =================================

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

MODE = os.getenv("MODE", "paper")

BASE_URL = "https://api.coindcx.com"

SERVER_URL = "https://coin-4k37.onrender.com"

SYMBOLS = ["BTCINR", "ETHINR", "SOLINR"]

TRADE_SIZE = 0.001


# =================================
# SIGNATURE
# =================================

def sign(payload):

    payload_json = json.dumps(payload, separators=(',', ':'))

    signature = hmac.new(
        bytes(API_SECRET, 'utf-8'),
        bytes(payload_json, 'utf-8'),
        hashlib.sha256
    ).hexdigest()

    return signature, payload_json


# =================================
# MARKET DATA
# =================================

def get_candles(symbol):

    try:

        url = f"https://public.coindcx.com/market_data/candles?pair={symbol}&interval=1m"

        r = requests.get(url).json()

        closes = [float(x[4]) for x in r[-60:]]

        return closes

    except:

        return []


# =================================
# INDICATORS
# =================================

def rsi(prices, period=14):

    if len(prices) < period + 1:
        return 50

    gains = []
    losses = []

    for i in range(1, period + 1):

        diff = prices[-i] - prices[-i - 1]

        if diff > 0:
            gains.append(diff)
        else:
            losses.append(abs(diff))

    avg_gain = sum(gains) / period if gains else 0.001
    avg_loss = sum(losses) / period if losses else 0.001

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


def vwap(prices):

    return statistics.mean(prices)


# =================================
# SMART MONEY CONCEPTS
# =================================

def liquidity_sweep(prices):

    recent_high = max(prices[-20:])
    recent_low = min(prices[-20:])
    current = prices[-1]

    if current > recent_high:
        return "HIGH_SWEEP"

    if current < recent_low:
        return "LOW_SWEEP"

    return None


def market_structure(prices):

    prev_high = max(prices[-30:-15])
    prev_low = min(prices[-30:-15])

    current = prices[-1]

    if current > prev_high:
        return "BOS_UP"

    if current < prev_low:
        return "BOS_DOWN"

    return None


# =================================
# SIGNAL ENGINE
# =================================

def generate_signal(symbol):

    prices = get_candles(symbol)

    if len(prices) < 40:
        return "HOLD"

    r = rsi(prices)
    vw = vwap(prices)

    sweep = liquidity_sweep(prices)
    structure = market_structure(prices)

    price = prices[-1]

    if structure == "BOS_UP" and sweep == "LOW_SWEEP" and r < 40 and price > vw:
        return "BUY"

    if structure == "BOS_DOWN" and sweep == "HIGH_SWEEP" and r > 60 and price < vw:
        return "SELL"

    return "HOLD"


# =================================
# ORDER EXECUTION
# =================================

def place_order(symbol, side):

    if MODE == "paper":

        print("PAPER TRADE", symbol, side)
        return

    payload = {
        "side": side,
        "order_type": "market_order",
        "market": symbol,
        "total_quantity": TRADE_SIZE,
        "timestamp": int(time.time() * 1000)
    }

    signature, payload_json = sign(payload)

    headers = {
        "Content-Type": "application/json",
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": signature
    }

    url = BASE_URL + "/exchange/v1/orders/create"

    requests.post(url, data=payload_json, headers=headers)


# =================================
# TRADING ENGINE
# =================================

def trading_engine():

    while True:

        try:

            for symbol in SYMBOLS:

                signal = generate_signal(symbol)

                print(symbol, signal)

                if signal != "HOLD":
                    place_order(symbol, signal.lower())

                time.sleep(5)

        except Exception as e:

            print("Engine error", e)

        time.sleep(60)


# =================================
# KEEPALIVE SELF PING
# =================================

def keep_alive():

    while True:

        try:
            requests.get(SERVER_URL + "/keepalive")
        except:
            pass

        time.sleep(240)


# =================================
# ROUTES
# =================================

@app.route("/")
def dashboard():

    return render_template("index.html")


@app.route("/signals")
def signals():

    s = {}

    for sym in SYMBOLS:
        s[sym] = generate_signal(sym)

    return jsonify(s)


@app.route("/health")
def health():

    return jsonify({"status": "running"})


@app.route("/keepalive")
def keepalive():

    return jsonify({"alive": True})


# =================================
# START THREADS
# =================================

def start():

    t1 = threading.Thread(target=trading_engine)
    t1.daemon = True
    t1.start()

    t2 = threading.Thread(target=keep_alive)
    t2.daemon = True
    t2.start()


start()


# =================================
# RUN
# =================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(host="0.0.0.0", port=port)