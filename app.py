import os
import time
import threading
import requests
import numpy as np
import hmac
import hashlib
import json
from flask import Flask, render_template, jsonify, request
from datetime import datetime
import pytz

app = Flask(__name__)

# ================= CONFIG =================
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
BASE_URL = "https://api.coindcx.com"
IST = pytz.timezone("Asia/Kolkata")
PAIRS = ["BTCINR", "ETHINR", "SOLINR"]
RENDER_URL = "https://coin-4k37.onrender.com"

# Trading State
running = False
investment_capital = 0
entry_positions = {}
trade_log = []
market_snapshot = {pair: {"price": 0, "bias": "Scanning..."} for pair in PAIRS}
error_message = ""
structure_levels = {}

# Data Buffers
price_data = {pair: [] for pair in PAIRS}
volume_data = {pair: [] for pair in PAIRS}

# ================= SIGNING & API =================
def sign_payload(payload):
    payload_json = json.dumps(payload, separators=(',', ':'))
    signature = hmac.new(API_SECRET.encode() if API_SECRET else b"", payload_json.encode(), hashlib.sha256).hexdigest()
    return signature, payload_json

def place_market_order(pair, side, quantity):
    url = f"{BASE_URL}/exchange/v1/orders/create"
    payload = {
        "side": side, "order_type": "market_order", "market": pair,
        "total_quantity": quantity, "timestamp": int(time.time() * 1000)
    }
    signature, payload_json = sign_payload(payload)
    headers = {"X-AUTH-APIKEY": API_KEY, "X-AUTH-SIGNATURE": signature, "Content-Type": "application/json"}
    try:
        r = requests.post(url, headers=headers, data=payload_json, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def get_market_data(pair):
    try:
        url = f"https://public.coindcx.com{pair}"
        r = requests.get(url, timeout=5)
        data = r.json()
        price = float(data["bids"][0]["price"]) # Fixed indexing for CoinDCX JSON
        volume = float(data["bids"][0]["quantity"])
        return price, volume
    except:
        return None, None

# ================= DYNAMIC LOGIC =================
def get_atr(pair, period=14):
    if len(price_data[pair]) < period + 1: return 0
    diffs = np.abs(np.diff(price_data[pair][-period-1:]))
    return np.mean(diffs)

def volume_spike(pair):
    v_data = volume_data[pair]
    if len(v_data) < 20: return False
    std_v = np.std(v_data[-20:])
    if std_v == 0: return False
    z_score = (v_data[-1] - np.mean(v_data[-20:])) / std_v
    return z_score > 2.0

def liquidity_sweep(pair, price):
    level = structure_levels.get(pair)
    atr = get_atr(pair)
    if not level or atr == 0: return False
    # Buffers are now volatility-dependent
    if price > level.get("high", 0) + (atr * 0.5): return True
    if price < level.get("low", float('inf')) - (atr * 0.5): return True
    return False

# ================= TRADING ENGINE =================
def execute_trade(pair, side, price):
    global investment_capital
    if investment_capital <= 0: return
    risk = investment_capital * 0.02 
    qty = round(risk / price, 6)
    order_side = "buy" if side == "long" else "sell"
    
    response = place_market_order(pair, order_side, qty)
    if "id" not in response: return

    atr = get_atr(pair)
    entry_positions[pair] = {
        "side": side, "qty": qty, "entry": price,
        "sl": price - (atr * 2) if side == "long" else price + (atr * 2),
        "peak": price
    }
    trade_log.append({"time": datetime.now(IST).strftime("%H:%M:%S"), "pair": pair, "side": side, "pnl": "OPEN"})

def trading_loop():
    global running, error_message
    while True:
        if not running:
            time.sleep(2)
            continue
        try:
            for pair in PAIRS:
                price, volume = get_market_data(pair)
                if not price: continue

                price_data[pair].append(price)
                volume_data[pair].append(volume)
                if len(price_data[pair]) > 200:
                    price_data[pair].pop(0)
                    volume_data[pair].pop(0)

                # Update Trailing Stop (Skimming Logic)
                if pair in entry_positions:
                    pos = entry_positions[pair]
                    atr = get_atr(pair)
                    if pos["side"] == "long":
                        if price > pos["peak"]: pos["peak"] = price; pos["sl"] = max(pos["sl"], price - (atr * 1.5))
                        if price < pos["sl"]: close_trade(pair, price)
                    elif pos["side"] == "short":
                        if price < pos["peak"]: pos["peak"] = price; pos["sl"] = min(pos["sl"], price + (atr * 1.5))
                        if price > pos["sl"]: close_trade(pair, price)
                    continue

                # Reactive Swing Point Tracking
                if len(price_data[pair]) > 5:
                    window = price_data[pair][-5:]
                    if window[-1] == max(window): structure_levels[pair] = {**structure_levels.get(pair, {}), "high": window[-1]}
                    if window[-1] == min(window): structure_levels[pair] = {**structure_levels.get(pair, {}), "low": window[-1]}

                # Entry Conditions
                bias = "bullish" if price > np.mean(price_data[pair][-50:]) else "bearish"
                market_snapshot[pair] = {"price": price, "bias": bias}

                if liquidity_sweep(pair, price) and volume_spike(pair):
                    execute_trade(pair, "long" if bias == "bullish" else "short", price)
            time.sleep(5)
        except Exception as e:
            error_message = str(e)
            time.sleep(5)

def close_trade(pair, price):
    global investment_capital
    pos = entry_positions[pair]
    place_market_order(pair, "sell" if pos["side"] == "long" else "buy", pos["qty"])
    pnl = (price - pos["entry"]) * pos["qty"] if pos["side"] == "long" else (pos["entry"] - price) * pos["qty"]
    investment_capital += pnl
    trade_log.append({"time": datetime.now(IST).strftime("%H:%M:%S"), "pair": pair, "side": "CLOSE", "pnl": round(pnl, 2)})
    del entry_positions[pair]

# ================= ROUTES =================
@app.route("/")
def home(): return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return jsonify({
        "running": running, "capital": round(investment_capital, 2),
        "positions": entry_positions, "market": market_snapshot,
        "log": trade_log[-20:], "error": error_message
    })

@app.route("/start", methods=["POST"])
def start():
    global running
    running = True
    return "Started"

@app.route("/stop", methods=["POST"])
def stop():
    global running
    running = False
    return "Stopped"

@app.route("/set_capital", methods=["POST"])
def set_capital():
    global investment_capital
    investment_capital = float(request.json.get("capital", 0))
    return "Capital Set"

@app.route("/ping")
def ping(): return "pong"

# ================= BACKGROUND THREADS =================
def self_keepalive():
    while True:
        try:
            # Pings the provided Render URL to prevent idling
            requests.get(f"{RENDER_URL}/ping", timeout=5)
        except:
            pass
        time.sleep(240)

# Critical: Global thread startup
threading.Thread(target=trading_loop, daemon=True).start()
threading.Thread(target=self_keepalive, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
