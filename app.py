import os
import time
import threading
import requests
import numpy as np
from flask import Flask, render_template, jsonify, request
from datetime import datetime
import pytz

app = Flask(__name__)

# ===== CONFIG =====
IST = pytz.timezone("Asia/Kolkata")
PAIRS = ["BTCINR", "ETHINR", "SOLINR"]
running = False
investment_capital = 0
entry_positions = {}
trade_log = []
market_snapshot = {}
error_message = ""

price_data = {pair: [] for pair in PAIRS}

# ===== HELPERS =====

def ema(data, period):
    if len(data) < period:
        return None
    return np.mean(data[-period:])

def detect_bos(pair):
    if len(price_data[pair]) < 10:
        return None
    highs = price_data[pair][-10:]
    if highs[-1] > max(highs[:-1]):
        return "bullish"
    if highs[-1] < min(highs[:-1]):
        return "bearish"
    return None

def detect_choch(pair):
    if len(price_data[pair]) < 20:
        return None
    recent = price_data[pair][-20:]
    if recent[-1] > max(recent[:10]):
        return "bullish"
    if recent[-1] < min(recent[:10]):
        return "bearish"
    return None

def order_block(pair):
    if len(price_data[pair]) < 5:
        return None
    last_candles = price_data[pair][-5:]
    return np.mean(last_candles)

def get_price(pair):
    try:
        url = f"https://public.coindcx.com/market_data/orderbook?pair={pair}"
        r = requests.get(url)
        data = r.json()
        return float(data["bids"][0]["price"])
    except:
        return None

# ===== TRADE EXECUTION (SIMULATION SAFE) =====

def execute_trade(pair, side, price):
    global investment_capital

    risk_per_trade = investment_capital * 0.01
    qty = risk_per_trade / price

    sl = price * (0.995 if side == "long" else 1.005)
    tp = price * (1.018 if side == "long" else 0.982)

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
        "entry": price
    })

# ===== STRATEGY LOOP =====

def trading_loop():
    global running, error_message

    while running:
        try:
            for pair in PAIRS:
                price = get_price(pair)
                if not price:
                    continue

                price_data[pair].append(price)
                if len(price_data[pair]) > 100:
                    price_data[pair].pop(0)

                market_snapshot[pair] = {
                    "price": price
                }

                if pair in entry_positions:
                    check_exit(pair, price)
                    continue

                if len(price_data[pair]) < 21:
                    continue

                ema9 = ema(price_data[pair], 9)
                ema21 = ema(price_data[pair], 21)

                bos = detect_bos(pair)
                choch = detect_choch(pair)

                ob = order_block(pair)

                # LONG SETUP
                if ema9 and ema21 and ema9 > ema21 and bos == "bullish" and price > ob:
                    execute_trade(pair, "long", price)

                # SHORT SETUP
                if ema9 and ema21 and ema9 < ema21 and bos == "bearish" and price < ob:
                    execute_trade(pair, "short", price)

            time.sleep(5)

        except Exception as e:
            error_message = str(e)
            time.sleep(5)

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

# ===== ROUTES =====

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
        "log": trade_log[-20:],
        "error": error_message
    })

@app.route("/ping")
def ping():
    return "pong"

# ===== KEEP ALIVE =====

def self_keepalive():
    while True:
        try:
            requests.get("https://your-render-url.onrender.com/ping")
        except:
            pass
        time.sleep(240)

threading.Thread(target=self_keepalive, daemon=True).start()

# ===== MAIN =====

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))