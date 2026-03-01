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

# =========================
# ENV CONFIG
# =========================

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
KEEPALIVE_TOKEN = os.getenv("KEEPALIVE_TOKEN", "secret")

SYMBOL = "BTCUSDT"
INTERVAL = "1m"
BASE_URL = "https://api.coindcx.com"

PORT = int(os.environ.get("PORT", 10000))

# =========================
# LOGGING
# =========================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# FLASK APP
# =========================

app = Flask(__name__)

bot_state = {
    "running": True,
    "last_signal": "NONE",
    "last_trade": "NONE",
    "price": 0
}

# =========================
# FETCH CANDLES (FIXED)
# =========================

def fetch_candles():
    try:
        url = f"{BASE_URL}/exchange/v1/candles?pair={SYMBOL}&interval={INTERVAL}"
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            logger.error("Bad response from API")
            return None

        data = response.json()

        # ✅ Ensure list format
        if not isinstance(data, list) or len(data) == 0:
            logger.error("Invalid candle data")
            return None

        df = pd.DataFrame(data)

        # CoinDCX returns arrays → map safely
        df = df.iloc[:, :6]
        df.columns = ["time", "open", "high", "low", "close", "volume"]

        df["close"] = pd.to_numeric(df["close"], errors="coerce")

        return df.dropna().tail(100)

    except Exception as e:
        logger.error(f"Fetch Error: {e}")
        return None

# =========================
# SIGNAL ENGINE
# =========================

def get_signal():
    df = fetch_candles()
    if df is None or len(df) < 30:
        return "HOLD"

    df["ema_fast"] = ta.ema(df["close"], length=9)
    df["ema_slow"] = ta.ema(df["close"], length=21)
    df["rsi"] = ta.rsi(df["close"], length=14)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    bot_state["price"] = round(last["close"], 2)

    if prev["ema_fast"] <= prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]:
        if 50 < last["rsi"] < 70:
            return "BUY"

    if prev["ema_fast"] >= prev["ema_slow"] and last["ema_fast"] < last["ema_slow"]:
        return "SELL"

    return "HOLD"

# =========================
# TRADING LOOP
# =========================

def trading_loop():
    logger.info("Bot started")

    while True:
        if bot_state["running"]:
            signal = get_signal()
            bot_state["last_signal"] = signal

            if signal in ["BUY", "SELL"]:
                bot_state["last_trade"] = signal
                logger.info(f"Trade Signal: {signal}")

        time.sleep(60)

threading.Thread(target=trading_loop, daemon=True).start()

# =========================
# ROUTES
# =========================

@app.route("/")
def dashboard():
    return render_template("index.html", state=bot_state)

@app.route("/status")
def status():
    return jsonify(bot_state)

@app.route("/pause")
def pause():
    if request.args.get("token") != KEEPALIVE_TOKEN:
        return "Unauthorized", 401
    bot_state["running"] = False
    return "Bot paused"

@app.route("/resume")
def resume():
    if request.args.get("token") != KEEPALIVE_TOKEN:
        return "Unauthorized", 401
    bot_state["running"] = True
    return "Bot resumed"

@app.route("/health")
def health():
    return jsonify({"status": "running"})

# =========================
# LOCAL RUN
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)