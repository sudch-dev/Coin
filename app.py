import os
import time
import threading
import requests
import numpy as np
from flask import Flask, render_template, jsonify, request
from datetime import datetime
import pytz

app = Flask(__name__)

IST = pytz.timezone("Asia/Kolkata")
PAIRS = ["BTCINR", "ETHINR", "SOLINR"]

running = False
investment_capital = 0
entry_positions = {}
trade_log = []
market_snapshot = {}
error_message = ""
market_bias = {}
structure_levels = {}

price_data = {pair: [] for pair in PAIRS}
volume_data = {pair: [] for pair in PAIRS}

# ================= DATA FETCH =================

def get_market_data(pair):
    try:
        url = f"https://public.coindcx.com/market_data/orderbook?pair={pair}"
        r = requests.get(url, timeout=5)
        data = r.json()
        price = float(data["bids"][0]["price"])
        volume = float(data["bids"][0]["quantity"])
        return price, volume
    except:
        return None, None

# ================= STRUCTURE DETECTION =================

def detect_swings(pair):
    data = price_data[pair]
    if len(data) < 5:
        return None, None

    window = data[-5:]
    center = window[2]

    if center > window[0] and center > window[1] and center > window[3] and center > window[4]:
        return "swing_high", center

    if center < window[0] and center < window[1] and center < window[3] and center < window[4]:
        return "swing_low", center

    return None, None

def detect_bos(pair, price):
    level = structure_levels.get(pair)
    if not level:
        return None

    if price > level["high"]:
        return "bullish"

    if price < level["low"]:
        return "bearish"

    return None

def detect_choch(pair, price):
    bias = market_bias.get(pair)
    level = structure_levels.get(pair)

    if not bias or not level:
        return None

    if bias == "bullish" and price < level["low"]:
        return "bearish"

    if bias == "bearish" and price > level["high"]:
        return "bullish"

    return None

def liquidity_sweep(pair, price):
    level = structure_levels.get(pair)
    if not level:
        return False

    if price > level["high"] * 1.002:
        return True

    if price < level["low"] * 0.998:
        return True

    return False

def volume_spike(pair):
    if len(volume_data[pair]) < 10:
        return False
    avg = np.mean(volume_data[pair][-10:])
    return volume_data[pair][-1] > avg * 1.5

# ================= TRADE EXECUTION =================

def execute_trade(pair, side, price):
    global investment_capital

    if investment_capital <= 0:
        return

    risk = investment_capital * 0.01
    qty = risk / price

    entry_positions[pair] = {
        "side": side,
        "qty": qty,
        "entry": price
    }

    trade_log.append({
        "time": datetime.now(IST).strftime("%H:%M:%S"),
        "pair": pair,
        "side": side,
        "pnl": "-"
    })

def close_trade(pair, price):
    global investment_capital

    pos = entry_positions[pair]

    if pos["side"] == "long":
        pnl = (price - pos["entry"]) * pos["qty"]
    else:
        pnl = (pos["entry"] - price) * pos["qty"]

    investment_capital += pnl

    trade_log.append({
        "time": datetime.now(IST).strftime("%H:%M:%S"),
        "pair": pair,
        "side": "CLOSE",
        "pnl": round(pnl, 2)
    })

    del entry_positions[pair]

# ================= MAIN LOOP =================

def trading_loop():
    global running, error_message

    while running:
        try:
            for pair in PAIRS:

                price, volume = get_market_data(pair)
                if not price:
                    continue

                price_data[pair].append(price)
                volume_data[pair].append(volume)

                if len(price_data[pair]) > 300:
                    price_data[pair].pop(0)
                    volume_data[pair].pop(0)

                swing_type, swing_level = detect_swings(pair)
                if swing_type:
                    if pair not in structure_levels:
                        structure_levels[pair] = {"high": price, "low": price}

                    if swing_type == "swing_high":
                        structure_levels[pair]["high"] = swing_level
                    if swing_type == "swing_low":
                        structure_levels[pair]["low"] = swing_level

                bos = detect_bos(pair, price)
                choch = detect_choch(pair, price)

                # CHOCH → square off
                if pair in entry_positions and choch:
                    close_trade(pair, price)
                    market_bias[pair] = None
                    continue

                # BOS → maintain bias
                if bos:
                    market_bias[pair] = bos

                bias = market_bias.get(pair)

                market_snapshot[pair] = {
                    "price": price,
                    "bias": bias
                }

                if not bias:
                    continue

                if pair in entry_positions:
                    continue

                if liquidity_sweep(pair, price) and volume_spike(pair):

                    if bias == "bullish":
                        execute_trade(pair, "long", price)

                    if bias == "bearish":
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
        "log": trade_log[-30:],
        "error": error_message
    })

@app.route("/ping")
def ping():
    return "pong"

# ================= KEEP ALIVE =================

def self_keepalive():
    while True:
        try:
            requests.get("https://coin-4k37.onrender.com/ping")
        except:
            pass
        time.sleep(240)

threading.Thread(target=self_keepalive, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))