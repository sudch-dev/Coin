import os
import time
import hmac
import hashlib
import json
import requests
import pandas as pd
import threading
from flask import Flask, jsonify, request, render_template

# ================= CONFIG =================

API_KEY = "YOUR_API_KEY"
API_SECRET = "YOUR_API_SECRET"

BASE_URL = "https://api.coindcx.com"

SYMBOL = "ORDIUSDT"
LEVERAGE = 5

RISK_PER_TRADE = 0.01
MAX_DAILY_LOSS = 0.03
TP_MULTIPLIER = 2.0

CHECK_INTERVAL = 15

# ==========================================

app = Flask(__name__)

bot_running = False
position = None
entry_price = 0
equity = 1000.0
day_pnl = 0.0
trades = []

# ================= AUTH ====================

def sign(payload):
    return hmac.new(
        API_SECRET.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()

def private_post(endpoint, body):
    payload = json.dumps(body)
    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": sign(payload),
        "Content-Type": "application/json"
    }
    return requests.post(BASE_URL + endpoint, data=payload, headers=headers).json()

# ================= DATA ====================

def get_price():
    r = requests.get(f"{BASE_URL}/exchange/ticker")
    for s in r.json():
        if s["market"] == SYMBOL:
            return float(s["last_price"])
    return None

def get_candles():
    url = f"{BASE_URL}/market_data/candles?pair={SYMBOL}&interval=1m"
    r = requests.get(url)
    df = pd.DataFrame(r.json(),
        columns=["time","open","high","low","close","volume"])
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    return df

# ================= INDICATORS ==============

def trend_signal(df):
    return df["close"].iloc[-1] > df["close"].rolling(5).mean().iloc[-1]

def atr(df):
    tr = df["high"] - df["low"]
    return tr.rolling(14).mean().iloc[-1]

# ================= RISK ====================

def position_size(price, stop_dist):
    risk_amount = equity * RISK_PER_TRADE
    qty = risk_amount / stop_dist if stop_dist > 0 else 0
    return round(qty, 3)

def kill_switch():
    return day_pnl <= -equity * MAX_DAILY_LOSS

# ================= TRADING =================

def place_order(side, qty):
    body = {
        "side": side,
        "order_type": "market_order",
        "market": SYMBOL,
        "quantity": qty,
        "timestamp": int(time.time()*1000)
    }
    return private_post("/exchange/v1/orders/create", body)

def flatten():
    global position
    if position == "LONG":
        place_order("sell", 999999)
    elif position == "SHORT":
        place_order("buy", 999999)
    position = None

# ================= BOT LOOP =================

def run_bot():
    global position, entry_price, equity, day_pnl, bot_running

    while bot_running:
        try:

            if kill_switch():
                flatten()
                bot_running = False
                print("DAILY LOSS LIMIT HIT")
                break

            price = get_price()
            df = get_candles()

            if price is None or df.empty:
                time.sleep(CHECK_INTERVAL)
                continue

            signal_long = trend_signal(df)
            vol = atr(df)
            qty = position_size(price, vol)

            # ---- ENTRY ----
            if position is None and qty > 0:

                if signal_long:
                    place_order("buy", qty)
                    position = "LONG"
                else:
                    place_order("sell", qty)
                    position = "SHORT"

                entry_price = price

            # ---- EXIT ----
            elif position is not None:

                pnl = (price - entry_price) * qty
                if position == "SHORT":
                    pnl = -pnl

                if pnl > vol * TP_MULTIPLIER or pnl < -vol:
                    flatten()
                    day_pnl += pnl
                    equity += pnl

                    trades.append({
                        "side": position,
                        "pnl": round(pnl,2),
                        "price": price,
                        "time": time.strftime("%H:%M:%S")
                    })

        except Exception as e:
            print("ERROR:", e)

        time.sleep(CHECK_INTERVAL)

# ================= ROUTES ==================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/start", methods=["POST"])
def start_bot():
    global bot_running
    if not bot_running:
        bot_running = True
        threading.Thread(target=run_bot).start()
    return jsonify({"status": "BOT STARTED"})

@app.route("/stop", methods=["POST"])
def stop_bot():
    global bot_running
    bot_running = False
    return jsonify({"status": "BOT STOPPED"})

@app.route("/flatten", methods=["POST"])
def manual_flatten():
    flatten()
    return jsonify({"status": "POSITION CLOSED"})

@app.route("/status")
def status():
    return jsonify({
        "running": bot_running,
        "position": position,
        "entry_price": entry_price,
        "equity": round(equity,2),
        "day_pnl": round(day_pnl,2),
        "trades": trades[-10:]
    })

# ================= MAIN ====================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)