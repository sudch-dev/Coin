import os
import time
import hmac
import hashlib
import json
import threading
import requests
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template
from datetime import datetime

app = Flask(__name__)

# ================= CONFIG =================

BASE_URL = "https://api.coindcx.com"

symbol = "btcusdt"      # CoinDCX pair name (lowercase)
timeframe = "15m"

ema_len = 200
rsi_len = 14
atr_len = 14

risk_pct = 0.02
max_averages = 3

API_KEY = os.getenv("COINDCX_API_KEY")
API_SECRET = os.getenv("COINDCX_SECRET_KEY")

mode = "PAPER"   # PAPER or LIVE
position = None
logs = []
start_time = datetime.now()

# ================= LOGGING =================

def log(msg):
    t = datetime.now().strftime("%H:%M:%S")
    entry = f"[{t}] {msg}"
    print(entry)
    logs.append(entry)
    if len(logs) > 50:
        logs.pop(0)

# ================= KEEP ALIVE =================

def self_ping():
    while True:
        try:
            requests.get(os.getenv("RENDER_EXTERNAL_URL"), timeout=10)
        except:
            pass
        time.sleep(240)

# ================= MARKET DATA =================

def get_market_symbol():
    r = requests.get(f"{BASE_URL}/exchange/v1/markets_details", timeout=15)
    markets = r.json()

    for m in markets:
        if m["coindcx_name"].lower() == symbol:
            return m["symbol"]

    raise Exception("Market not found")


def get_latest_data():

    market_symbol = get_market_symbol()

    url = f"{BASE_URL}/market_data/candles"
    params = {
        "pair": market_symbol,
        "interval": timeframe,
        "limit": 300
    }

    r = requests.get(url, params=params, timeout=15)

    if r.status_code != 200:
        raise Exception(f"HTTP {r.status_code}")

    data = r.json()

    if not isinstance(data, list) or len(data) == 0:
        raise Exception("Empty candle data")

    df = pd.DataFrame(data).astype(float)

    # ===== INDICATORS =====

    df["ema"] = df["close"].ewm(span=ema_len).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(rsi_len).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_len).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"] - df["close"].shift()).abs()
    df["atr"] = np.maximum(tr1, np.maximum(tr2, tr3)).rolling(atr_len).mean()

    return df.iloc[-1]

# ================= LIVE ORDER =================

def place_live_order(side, qty):

    if mode != "LIVE":
        return

    body = {
        "side": side,
        "order_type": "market_order",
        "market": symbol,
        "total_quantity": qty,
        "timestamp": int(time.time() * 1000)
    }

    payload = json.dumps(body, separators=(',', ':'))
    signature = hmac.new(
        API_SECRET.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()

    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": signature,
        "Content-Type": "application/json"
    }

    r = requests.post(
        f"{BASE_URL}/exchange/v1/orders/create",
        data=payload,
        headers=headers
    )

    log(f"LIVE ORDER → {r.text}")

# ================= STRATEGY =================

def strategy_loop():

    global position

    averages = 0

    while True:
        try:

            d = get_latest_data()

            price = d["close"]
            ema = d["ema"]
            rsi = d["rsi"]
            atr = d["atr"]

            log(f"P={price:.2f} EMA={ema:.2f} RSI={rsi:.1f}")

            # ===== ENTRY =====

            if position is None:

                if price > ema and rsi < 45:

                    position = {
                        "entry": price,
                        "qty": 1,
                        "avg_price": price
                    }
                    averages = 0

                    log("ENTER LONG")

                    place_live_order("buy", 1)

            # ===== MANAGE POSITION =====

            else:

                avg_price = position["avg_price"]

                # Averaging down
                if price < avg_price - atr and averages < max_averages:

                    new_qty = 1
                    position["qty"] += new_qty

                    position["avg_price"] = (
                        avg_price * (position["qty"] - new_qty)
                        + price * new_qty
                    ) / position["qty"]

                    averages += 1

                    log("AVERAGING BUY")

                    place_live_order("buy", new_qty)

                # Exit condition
                if price > avg_price + atr:

                    log("EXIT POSITION")

                    place_live_order("sell", position["qty"])

                    position = None

            time.sleep(60)

        except Exception as e:
            log(f"DATA ERROR → {str(e)}")
            time.sleep(30)

# ================= ROUTES =================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/status")
def status():
    uptime = datetime.now() - start_time
    return jsonify({
        "mode": mode,
        "logs": logs[::-1],
        "uptime": str(uptime),
        "position": position
    })


@app.route("/toggle")
def toggle_mode():
    global mode
    mode = "LIVE" if mode == "PAPER" else "PAPER"
    log(f"MODE → {mode}")
    return jsonify({"mode": mode})


@app.route("/ping")
def ping():
    return "alive"

# ================= START =================

if __name__ == "__main__":

    log("BOT STARTED")

    threading.Thread(target=strategy_loop, daemon=True).start()
    threading.Thread(target=self_ping, daemon=True).start()

    app.run(host="0.0.0.0", port=10000)