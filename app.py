import os
import time
import threading
import logging
import requests
import pandas as pd
import pandas_ta as ta
from flask import Flask, jsonify

# ==============================
# BASIC CONFIG
# ==============================

SYMBOL = "BTCUSDT"
INTERVAL = "1m"
BASE_URL = "https://api.coindcx.com"

PORT = int(os.environ.get("PORT", 10000))

# ==============================
# LOGGING
# ==============================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# ==============================
# FLASK APP (IMPORTANT FOR GUNICORN)
# ==============================

app = Flask(__name__)

# ==============================
# FETCH CANDLE DATA
# ==============================

def fetch_candles():
    try:
        url = f"{BASE_URL}/exchange/v1/candles?pair={SYMBOL}&interval={INTERVAL}"
        response = requests.get(url, timeout=10)
        data = response.json()

        if not data:
            return None

        df = pd.DataFrame(data)

        # Rename columns properly if needed
        df.columns = ["time", "open", "high", "low", "close", "volume"]

        df["close"] = df["close"].astype(float)

        return df.tail(100)

    except Exception as e:
        logger.error(f"Fetch Error: {e}")
        return None


# ==============================
# SIGNAL LOGIC
# ==============================

def get_live_signals():
    try:
        df = fetch_candles()
        if df is None or len(df) < 30:
            return "HOLD"

        df['ema_fast'] = ta.ema(df['close'], length=9)
        df['ema_slow'] = ta.ema(df['close'], length=21)
        df['rsi'] = ta.rsi(df['close'], length=14)

        last = df.iloc[-1]
        prev = df.iloc[-2]

        # BUY Condition
        if prev['ema_fast'] <= prev['ema_slow'] and last['ema_fast'] > last['ema_slow']:
            if 50 < last['rsi'] < 70:
                return "BUY"

        # SELL Condition
        if prev['ema_fast'] >= prev['ema_slow'] and last['ema_fast'] < last['ema_slow']:
            return "SELL"

    except Exception as e:
        logger.error(f"Signal Error: {e}")

    return "HOLD"


# ==============================
# MAIN BOT LOOP
# ==============================

def automation_loop():
    logger.info("Trading bot started...")
    while True:
        try:
            signal = get_live_signals()

            if signal != "HOLD":
                logger.info(f"SIGNAL DETECTED: {signal}")
                # execute_trade(signal)  # Add your order logic here

            time.sleep(60)

        except Exception as e:
            logger.error(f"Loop Error: {e}")
            time.sleep(10)


# ==============================
# BACKGROUND THREAD
# ==============================

def start_bot():
    thread = threading.Thread(target=automation_loop)
    thread.daemon = True
    thread.start()

start_bot()

# ==============================
# HEALTH CHECK ROUTES
# ==============================

@app.route("/")
def home():
    return "CoinDCX Trading Bot Running 🚀"

@app.route("/health")
def health():
    return jsonify({"status": "running"})


# ==============================
# DO NOT USE app.run() ON RENDER
# Gunicorn will serve this app
# ==============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)