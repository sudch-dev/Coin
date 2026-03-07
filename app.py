import os
import time
import json
import hmac
import hashlib
import threading
import requests
import pandas as pd

from flask import Flask, render_template, request, jsonify

# =========================
# CONFIG
# =========================

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

RENDER_URL = "https://coin-4k37.onrender.com"

SYMBOLS = ["BTCINR","ETHINR","SOLINR"]

BASE_URL = "https://api.coindcx.com"

capital = 0
positions = {}
entry_price = {}

app = Flask(__name__)

# =========================
# KEEPALIVE
# =========================

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

# =========================
# COINDCX SIGNATURE
# =========================

def sign_payload(payload):

    secret = bytes(API_SECRET, 'utf-8')
    payload_json = json.dumps(payload, separators=(',', ':'))

    signature = hmac.new(
        secret,
        payload_json.encode(),
        hashlib.sha256
    ).hexdigest()

    return payload_json, signature


# =========================
# MARKET DATA
# =========================

def get_price(symbol):

    ticker = requests.get(
        f"{BASE_URL}/exchange/ticker"
    ).json()

    for t in ticker:
        if t["market"] == symbol:
            return float(t["last_price"])

    return None


# =========================
# ORDER EXECUTION
# =========================

def place_order(symbol, side, quantity):

    timestamp = int(time.time()*1000)

    payload = {
        "side": side,
        "order_type": "market_order",
        "market": symbol,
        "total_quantity": quantity,
        "timestamp": timestamp
    }

    payload_json, signature = sign_payload(payload)

    headers = {
        "Content-Type": "application/json",
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": signature
    }

    r = requests.post(
        f"{BASE_URL}/exchange/v1/orders/create",
        data=payload_json,
        headers=headers
    )

    return r.json()


# =========================
# HISTORICAL DATA
# =========================

def get_candles(symbol):

    url = f"https://public.coindcx.com/market_data/candles?pair={symbol}&interval=15m"

    data = requests.get(url).json()

    df = pd.DataFrame(data)

    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)

    return df.tail(200)


# =========================
# INDICATORS
# =========================

def ema(series, period):
    return series.ewm(span=period).mean()


# =========================
# ORDER BLOCK
# =========================

def detect_ob(df):

    candle = df.iloc[-2]

    if candle["close"] < candle["open"]:
        return "bullish"

    if candle["close"] > candle["open"]:
        return "bearish"

    return None


# =========================
# BOS
# =========================

def detect_bos(df):

    prev_high = df["high"].iloc[-3]
    prev_low = df["low"].iloc[-3]

    close = df["close"].iloc[-1]

    if close > prev_high:
        return "bullish"

    if close < prev_low:
        return "bearish"

    return None


# =========================
# CHOCH
# =========================

def detect_choch(df):

    prev_high = df["high"].iloc[-3]
    prev_low = df["low"].iloc[-3]

    close = df["close"].iloc[-1]

    if close < prev_low:
        return "bearish"

    if close > prev_high:
        return "bullish"

    return None


# =========================
# ENTRY LOGIC
# =========================

def entry_logic(symbol):

    df = get_candles(symbol)

    df["ema9"] = ema(df["close"],9)
    df["ema21"] = ema(df["close"],21)

    ob = detect_ob(df)

    if df["ema9"].iloc[-1] > df["ema21"].iloc[-1] and ob=="bullish":
        return "buy"

    if df["ema9"].iloc[-1] < df["ema21"].iloc[-1] and ob=="bearish":
        return "sell"

    return None


# =========================
# EXIT LOGIC
# =========================

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


# =========================
# CONTINUATION
# =========================

def continuation_logic(symbol):

    df = get_candles(symbol)

    bos = detect_bos(df)

    side = positions.get(symbol)

    if side=="buy" and bos=="bullish":
        return True

    if side=="sell" and bos=="bearish":
        return True

    return False


# =========================
# TRADING ENGINE
# =========================

def engine():

    global capital

    while True:

        if capital == 0:
            time.sleep(10)
            continue

        for symbol in SYMBOLS:

            try:

                price = get_price(symbol)

                if symbol not in positions:

                    signal = entry_logic(symbol)

                    if signal:

                        qty = round(capital/price,6)

                        place_order(symbol, signal, qty)

                        positions[symbol] = signal
                        entry_price[symbol] = price

                else:

                    if exit_logic(symbol):

                        qty = round(capital/entry_price[symbol],6)

                        side = "sell" if positions[symbol]=="buy" else "buy"

                        place_order(symbol, side, qty)

                        del positions[symbol]

                    else:
                        continuation_logic(symbol)

            except Exception as e:
                print("Error:",e)

        time.sleep(30)


threading.Thread(target=engine, daemon=True).start()

# =========================
# UI ROUTES
# =========================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/set_capital", methods=["POST"])
def set_capital():

    global capital

    capital = float(request.form["capital"])

    return jsonify({
        "status":"capital set",
        "capital":capital
    })


@app.route("/status")
def status():

    return jsonify({
        "capital":capital,
        "positions":positions
    })


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    port = int(os.environ.get("PORT",10000))

    app.run(host="0.0.0.0",port=port)