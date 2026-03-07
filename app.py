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

# Trading State
running = False
investment_capital = 0
entry_positions = {}
trade_log = []
market_snapshot = {}
error_message = ""
market_bias = {}
structure_levels = {}

# Data Buffers
price_data = {pair: [] for pair in PAIRS}
volume_data = {pair: [] for pair in PAIRS}

# ================= SIGNING & API =================
def sign_payload(payload):
    payload_json = json.dumps(payload, separators=(',', ':'))
    signature = hmac.new(API_SECRET.encode(), payload_json.encode(), hashlib.sha256).hexdigest()
    return signature, payload_json

def place_market_order(pair, side, quantity):
    url = f"{BASE_URL}/exchange/v1/orders/create"
    payload = {
        "side": side,
        "order_type": "market_order",
        "market": pair,
        "total_quantity": quantity,
        "timestamp": int(time.time() * 1000)
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
        # Using public orderbook for price/volume
        url = f"https://public.coindcx.com{pair}"
        r = requests.get(url, timeout=5)
        data = r.json()
        price = float(data["bids"][0]["price"])
        volume = float(data["bids"][0]["quantity"])
        return price, volume
    except:
        return None, None

# ================= DYNAMIC LOGIC =================
def get_atr(pair, period=14):
    """Calculates volatility based on recent price movements."""
    if len(price_data[pair]) < period + 1:
        return 0
    # Simplified ATR using recent price differences
    diffs = np.abs(np.diff(price_data[pair][-period-1:]))
    return np.mean(diffs)

def volume_spike(pair):
    """Uses Z-Score to detect statistically significant volume anomalies."""
    v_data = volume_data[pair]
    if len(v_data) < 20: return False
    mean_v = np.mean(v_data[-20:])
    std_v = np.std(v_data[-20:])
    if std_v == 0: return False
    z_score = (v_data[-1] - mean_v) / std_v
    return z_score > 2.0  # Significant spike filter

def liquidity_sweep(pair, price):
    """Checks if price swept liquidity beyond ATR-adjusted levels."""
    level = structure_levels.get(pair)
    atr = get_atr(pair)
    if not level or atr == 0: return False
    # Dynamic buffer: sweeps 0.5 * ATR past known high/low
    if price > level["high"] + (atr * 0.5): return True
    if price < level["low"] - (atr * 0.5): return True
    return False

# ================= POSITION MANAGEMENT =================
def execute_trade(pair, side, price):
    global investment_capital
    if investment_capital <= 0: return

    risk = investment_capital * 0.02 # 2% Risk per trade
    qty = round(risk / price, 6)
    order_side = "buy" if side == "long" else "sell"
    
    response = place_market_order(pair, order_side, qty)
    if "id" not in response: return

    atr = get_atr(pair)
    entry_positions[pair] = {
        "side": side,
        "qty": qty,
        "entry": price,
        "sl": price - (atr * 2) if side == "long" else price + (atr * 2),
        "peak": price
    }
    trade_log.append({"time": datetime.now(IST).strftime("%H:%M:%S"), "pair": pair, "side": side, "pnl": "OPEN"})

def update_trailing_stop(pair, current_price):
    """Trails the SL to lock in profit as price moves favorably."""
    pos = entry_positions[pair]
    atr = get_atr(pair)
    
    if pos["side"] == "long":
        if current_price > pos["peak"]:
            pos["peak"] = current_price
            pos["sl"] = max(pos["sl"], current_price - (atr * 1.5))
        if current_price < pos["sl"]: close_trade(pair, current_price)
            
    elif pos["side"] == "short":
        if current_price < pos["peak"]:
            pos["peak"] = current_price
            pos["sl"] = min(pos["sl"], current_price + (atr * 1.5))
        if current_price > pos["sl"]: close_trade(pair, current_price)

def close_trade(pair, price):
    global investment_capital
    pos = entry_positions[pair]
    exit_side = "sell" if pos["side"] == "long" else "buy"
    
    place_market_order(pair, exit_side, pos["qty"])
    pnl = (price - pos["entry"]) * pos["qty"] if pos["side"] == "long" else (pos["entry"] - price) * pos["qty"]
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
                if not price: continue

                price_data[pair].append(price)
                volume_data[pair].append(volume)
                if len(price_data[pair]) > 200:
                    price_data[pair].pop(0)
                    volume_data[pair].pop(0)

                # Update Trailing Stop if in position
                if pair in entry_positions:
                    update_trailing_stop(pair, price)
                    continue

                # Structure Detection (High/Low)
                if len(price_data[pair]) > 5:
                    window = price_data[pair][-5:]
                    if window[2] == max(window): structure_levels[pair] = {**structure_levels.get(pair, {}), "high": window[2]}
                    if window[2] == min(window): structure_levels[pair] = {**structure_levels.get(pair, {}), "low": window[2]}

                # Entry Logic
                bias = "bullish" if price > np.mean(price_data[pair][-50:]) else "bearish"
                market_snapshot[pair] = {"price": price, "bias": bias}

                if liquidity_sweep(pair, price) and volume_spike(pair):
                    execute_trade(pair, "long" if bias == "bullish" else "short", price)

            time.sleep(5)
        except Exception as e:
            error_message = str(e)
            time.sleep(5)

# ================= ROUTES =================
@app.route("/")
def home(): return render_template("index.html")

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
