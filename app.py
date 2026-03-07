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
# CoinDCX Public API requires the 'I-' prefix and '_' for many INR pairs
PAIRS = ["I-BTC_INR", "I-ETH_INR", "I-SOL_INR"]
RENDER_URL = "https://coin-4k37.onrender.com"

# Trading State
running = False
investment_capital = 0
entry_positions = {}
trade_log = []
market_snapshot = {pair: {"price": 0, "bias": "Scanning..."} for pair in PAIRS}
error_message = ""
structure_levels = {}
price_data = {pair: [] for pair in PAIRS}
volume_data = {pair: [] for pair in PAIRS}

# ================= DATA FETCH (DIAGNOSTIC) =================
def get_market_data(pair):
    try:
        url = f"https://public.coindcx.com{pair}"
        r = requests.get(url, timeout=5)
        data = r.json()
        
        # CoinDCX 'bids' is a dict where keys are prices: {"11570.67": "0.0008"}
        if "bids" in data and data["bids"]:
            bid_prices = [float(p) for p in data["bids"].keys()]
            highest_bid = max(bid_prices)
            volume = float(data["bids"][f"{highest_bid:.8f}"])
            return highest_bid, volume
        
        print(f"[DIAGNOSTIC] No data for {pair}: {data.get('message', 'Unknown Error')}")
        return None, None
    except Exception as e:
        print(f"[DIAGNOSTIC] Connection Error for {pair}: {e}")
        return None, None

# ================= DYNAMIC LOGIC =================
def get_atr(pair, period=14):
    if len(price_data[pair]) < period + 1: return 0
    return np.mean(np.abs(np.diff(price_data[pair][-period-1:])))

def volume_spike(pair):
    v_data = volume_data[pair]
    if len(v_data) < 20: return False
    std_v = np.std(v_data[-20:])
    return (v_data[-1] - np.mean(v_data[-20:])) / std_v > 2.0 if std_v > 0 else False

# ================= TRADING ENGINE =================
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
                if len(price_data[pair]) > 100:
                    price_data[pair].pop(0)
                    volume_data[pair].pop(0)

                # Trailing SL (Skimming)
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

                # Bias & Strategy
                if len(price_data[pair]) > 20:
                    bias = "bullish" if price > np.mean(price_data[pair][-20:]) else "bearish"
                    market_snapshot[pair] = {"price": price, "bias": bias}
                    if volume_spike(pair):
                        execute_trade(pair, "long" if bias == "bullish" else "short", price)
            time.sleep(5)
        except Exception as e:
            error_message = str(e)
            time.sleep(5)

def execute_trade(pair, side, price):
    global investment_capital
    if investment_capital <= 0: return
    qty = round((investment_capital * 0.02) / price, 6)
    entry_positions[pair] = {"side": side, "qty": qty, "entry": price, "sl": price * 0.98, "peak": price}
    trade_log.append({"time": datetime.now(IST).strftime("%H:%M:%S"), "pair": pair, "side": side, "pnl": "OPEN"})

def close_trade(pair, price):
    global investment_capital
    pos = entry_positions.pop(pair)
    pnl = (price - pos["entry"]) * pos["qty"] if pos["side"] == "long" else (pos["entry"] - price) * pos["qty"]
    investment_capital += pnl
    trade_log.append({"time": datetime.now(IST).strftime("%H:%M:%S"), "pair": pair, "side": "CLOSE", "pnl": round(pnl, 2)})

# ================= ROUTES =================
@app.route("/")
def home(): return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return jsonify({
        "running": running, "capital": round(investment_capital, 2),
        "positions": entry_positions, "market": market_snapshot,
        "log": trade_log[-15:], "error": error_message
    })

@app.route("/start", methods=["POST"])
def start(): global running; running = True; return "Started"

@app.route("/stop", methods=["POST"])
def stop(): global running; running = False; return "Stopped"

@app.route("/set_capital", methods=["POST"])
def set_capital():
    global investment_capital
    investment_capital = float(request.json.get("capital", 0))
    return "Capital Set"

@app.route("/ping")
def ping(): return "pong"

# ================= KEEP ALIVE =================
def self_keepalive():
    while True:
        try: requests.get(f"{RENDER_URL}/ping", timeout=5)
        except: pass
        time.sleep(240)

threading.Thread(target=trading_loop, daemon=True).start()
threading.Thread(target=self_keepalive, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
