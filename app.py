import os
import time
import threading
import logging
import requests
import pandas as pd
import pandas_ta as ta
from flask import Flask, render_template, jsonify

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("coinbot")

SYMBOL = "BTCUSDT"
BASE = "https://public.coindcx.com"

latest_signal = "HOLD"


# -----------------------------
# Fetch market data
# -----------------------------
def fetch_candles():
    url = f"{BASE}/market_data/candles?pair={SYMBOL}&interval=1m"

    try:
        r = requests.get(url, timeout=10)

        if r.status_code != 200:
            raise Exception("Bad API status")

        data = r.json()

        if not isinstance(data, list) or len(data) == 0:
            raise Exception("Empty candle data")

        df = pd.DataFrame(data)

        df.rename(columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume"
        }, inplace=True)

        df["close"] = df["close"].astype(float)

        return df.tail(100)

    except Exception as e:
        logger.error(f"Fetch Error: {e}")
        return None


# -----------------------------
# Signal logic
# -----------------------------
def get_live_signals():
    global latest_signal

    df = fetch_candles()
    if df is None or len(df) < 30:
        return "HOLD"

    df["ema_fast"] = ta.ema(df["close"], length=9)
    df["ema_slow"] = ta.ema(df["close"], length=21)
    df["rsi"] = ta.rsi(df["close"], length=14)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    if prev["ema_fast"] <= prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]:
        if 50 < last["rsi"] < 70:
            latest_signal = "BUY"
            return "BUY"

    if prev["ema_fast"] >= prev["ema_slow"] and last["ema_fast"] < last["ema_slow"]:
        latest_signal = "SELL"
        return "SELL"

    latest_signal = "HOLD"
    return "HOLD"


# -----------------------------
# Background loop
# -----------------------------
def automation_loop():
    while True:
        signal = get_live_signals()

        if signal != "HOLD":
            logger.info(f"SIGNAL: {signal}")

        time.sleep(60)


# Start bot thread
threading.Thread(target=automation_loop, daemon=True).start()


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/status")
def status():
    return jsonify({
        "signal": latest_signal,
        "symbol": SYMBOL
    })


@app.route("/health")
def health():
    return "OK", 200


# Render port binding
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)