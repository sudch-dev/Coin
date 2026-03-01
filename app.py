import os
import time
import json
import hmac
import hashlib
import threading
import logging
import requests
import pandas as pd
import pandas_ta as ta

from flask import Flask, jsonify, request, render_template

# ==============================
# ENV CONFIG
# ==============================

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
KEEPALIVE_TOKEN = os.getenv("KEEPALIVE_TOKEN", "secret")
FORCE_LIQUIDATE = os.getenv("FORCE_LIQUIDATE_ON_NEED") == "1"

SYMBOL = "BTCUSDT"
INTERVAL = "1m"
BASE_URL = "https://api.coindcx.com"

PORT = int(os.environ.get("PORT", 10000))

# ==============================
# LOGGING
# ==============================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# FLASK APP
# ==============================

app = Flask(__name__)

bot_state = {
    "running": True,
    "last_signal": "NONE",
    "last_trade": "NONE"
}

# ==============================
# COINDCX AUTH
# ==============================

def sign_payload(payload):
    payload_json = json.dumps(payload, separators=(',', ':'))
    signature = hmac.new(
        API_SECRET.encode(),
        payload_json.encode(),
        hashlib.sha256
    ).hexdigest()
    return payload_json, signature

# ==============================
# PLACE ORDER
# ==============================

def place_order(side, price, quantity):
    body = {
        "side": side,
        "order_type": "limit_order",
        "price": price,
        "quantity": quantity,
        "market": SYMBOL,
        "timestamp": int(time.time() * 1000)
    }

    payload, signature = sign_payload(body)

    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": signature,
        "Content-Type": "application/json"
    }

    url = f"{BASE_URL}/exchange/v1/orders/create"

    response = requests.post(url, data=payload, headers=headers)

    return response.json()

# ==============================
# FETCH CANDLES
# ==============================

def fetch_candles():
    try:
        url = f"{BASE_URL}/exchange/v1/candles?pair={SYMBOL}&interval={INTERVAL}"
        data = requests.get(url, timeout=10).json()

        df = pd.DataFrame(data)
        df.columns = ["time", "open", "high", "low", "close", "volume"]
        df["close"] = df["close"].astype(float)

        return df.tail(100)

    except Exception as e:
        logger.error(f"Fetch Error: {e}")
        return None

# ==============================
# SIGNAL ENGINE
# ==============================

def get_signal():
    df = fetch_candles()
    if df is None or len(df) < 30:
        return "HOLD"

    df['ema_fast'] = ta.ema(df['close'], length=9)
    df['ema_slow'] = ta.ema(df['close'], length=21)
    df['rsi'] = ta.rsi(df['close'], length=14)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    if prev['ema_fast'] <= prev['ema_slow'] and last['ema_fast'] > last['ema_slow']:
        if 50 < last['rsi'] < 70:
            return "BUY"

    if prev['ema_fast'] >= prev['ema_slow'] and last['ema_fast'] < last['ema_slow']:
        return "SELL"

    return "HOLD"

# ==============================
# TRADING LOOP
# ==============================

def trading_loop():
    logger.info("Bot started")

    while True:
        if bot_state["running"]:

            signal = get_signal()
            bot_state["last_signal"] = signal

            if signal == "BUY":
                logger.info("BUY signal")
                bot_state["last_trade"] = "BUY"
                # place_order("buy", price, quantity)

            elif signal == "SELL":
                logger.info("SELL signal")
                bot_state["last_trade"] = "SELL"

        time.sleep(60)

# Start background thread
threading.Thread(target=trading_loop, daemon=True).start()

# ==============================
# ROUTES
# ==============================

@app.route("/")
def dashboard():
    return render_template("index.html", state=bot_state)

@app.route("/health")
def health():
    return jsonify({"status": "running"})

@app.route("/status")
def status():
    return jsonify(bot_state)

@app.route("/pause")
def pause():
    token = request.args.get("token")
    if token != KEEPALIVE_TOKEN:
        return "Unauthorized", 401

    bot_state["running"] = False
    return "Bot paused"

@app.route("/resume")
def resume():
    token = request.args.get("token")
    if token != KEEPALIVE_TOKEN:
        return "Unauthorized", 401

    bot_state["running"] = True
    return "Bot resumed"

@app.route("/liquidate")
def liquidate():
    token = request.args.get("token")
    if token != KEEPALIVE_TOKEN:
        return "Unauthorized", 401

    if FORCE_LIQUIDATE:
        return "Liquidation triggered"
    return "Liquidation disabled"

# ==============================
# LOCAL RUN (not used on Render)
# ==============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)