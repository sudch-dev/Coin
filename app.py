import os
import time
import threading
import requests
import numpy as np
from flask import Flask, render_template, jsonify, request
from datetime import datetime
import pytz

app = Flask(__name__)

# ================= CONFIG =================
IST = pytz.timezone("Asia/Kolkata")
PAIRS = ["BTCINR", "ETHINR", "SOLINR"]

running = False
investment_capital = 0
entry_positions = {}
trade_log = []
market_snapshot = {}
error_message = ""

price_data = {pair: [] for pair in PAIRS}

# ================= HELPERS =================

def ema(data, period):
    if len(data) < period:
        return None
    weights = np.exp(np.linspace(-1., 0., period))
    weights /= weights.sum()
    return np.convolve(data[-period:], weights, mode='valid')[0]

def detect_bos(pair):
    if len(price_data[pair]) < 12:
        return None
    window = price_data[pair][-12:]
    if window[-1] > max(window[:-1]):
        return "bullish"
    if window[-1] < min(window[:-1]):
        return "bearish"
    return None

def order_block(pair):
    if len(price_data[pair]) < 6:
        return None
    return np.mean(price_data[pair][-6:-1])

def get_price(pair):
    try:
        url = f"https://public.coindcx.com/market_data/orderbook?pair={pair}"
        r = requests.get(url, timeout=5)
        data = r.json()
        return float(data["bids"][0]["price"])
    except:
        return None

# ================= TRADE ENGINE =================

def execute_trade(pair, side, price):
    global investment_capital

    if investment_capital <= 0:
        return

    risk_amount = investment_capital * 0.01
    qty = risk_amount / price

    if side == "long":
        sl = price * 0.995
        tp = price * 1.018
    else:
        sl = price * 1.005
        tp = price * 0.982

    entry_positions[pair] = {
        "side": side,
        "qty": qty,
        "entry": price,
        "tp": tp,
        "sl": sl
    }

    trade_log.append({
        "time": datetime.now(IST).strftime("%H:%M:%S"),
        "pair": pair,
        "side": side,
        "pnl": "-"
    })

def check_exit(pair, price):
    global investment_capital

    pos = entry_positions[pair]

    if pos["side"] == "long":
        if price >= pos["tp"] or price <= pos["sl"]:
            pnl = (price - pos["entry"]) * pos["qty"]
            close_trade(pair, pnl)

    if pos["side"] == "short":
        if price <= pos["tp"] or price >= pos["sl"]:
            pnl = (pos["entry"] - price) * pos["qty"]
            close_trade(pair, pnl)

def close_trade(pair, pnl):
    global investment_capital

    investment_capital += pnl

    trade_log.append({
        "time": datetime.now(IST).strftime("%H:%M:%S"),
        "pair": pair,
        "side": "CLOSE",
        "pnl": round(pnl, 2)
    })

    del entry_positions[pair]

# ================= STRATEGY LOOP =================

def trading_loop():
    global running, error_message

    while running:
        try:
            for pair in PAIRS:

                price = get_price(pair)
                if not price:
                    continue

                price_data[pair].append(price)
                if len(price_data[pair]) > 200:
                    price_data[pair].pop(0)

                market_snapshot[pair] = {"price": price}

                if pair in entry_positions:
                    check_exit(pair, price)
                    continue

                if len(price_data[pair]) < 21:
                    continue

                ema9 = ema(price_data[pair], 9)
                ema21 = ema(price_data[pair], 21)
                bos = detect_bos(pair)
                ob = order_block(pair)

                if not ema9 or not ema21 or not bos or not ob:
                    continue

                # LONG SETUP
                if ema9 > ema21 and bos == "bullish" and price > ob:
                    execute_trade(pair, "long", price)

                # SHORT SETUP
                if ema9 < ema21 and bos == "bearish" and price < ob:
                    execute_trade(pair, "short", price)

            time.sleep(5)

        except Exception as e:
            error_message = str(e)
            time.sleep(5)

# ================= ROUTES =================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/start", methods=["POST"])
def start():
    global running
    if not running:
        running = True
        threading.Thread(target=trading_loop, daemon=True).start()
    return "Started"

@app.route("/stop", methods=["POST"])
def stop():
    global running
    running = False
    return "Stopped"

@app.route("/set_capital", methods=["POST"])
def set_capital():
    global investment_capital
    investment_capital = float(request.json["capital"])
    return "Capital Set"

@app.route("/dashboard")
def dashboard():
    return jsonify({
        "running": running,
        "capital": round(investment_capital, 2),
        "positions": entry_positions,
        "market": market_snapshot,
        "log": trade_log[-25:],
        "error": error_message
    })

@app.route("/ping")
def ping():
    return "pong"

# ================= KEEP ALIVE =================

def self_keepalive():
    while True:
        try:
            requests.get("https://your-render-url.onrender.com/ping")
        except:
            pass
        time.sleep(240)

threading.Thread(target=self_keepalive, daemon=True).start()

# ================= MAIN =================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))