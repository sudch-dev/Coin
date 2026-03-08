import os, time, threading, requests, json, hmac, hashlib
from datetime import datetime
import numpy as np
import pytz
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

# ================= CONFIG (CoinDCX Standards) =================
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
BASE_URL = "https://api.coindcx.com"
IST = pytz.timezone("Asia/Kolkata")
# Mandatory 'I-' prefix for CoinDCX INR pairs
PAIRS = ["I-BTC_INR", "I-ETH_INR", "I-SOL_INR"]
RENDER_URL = "https://coin-4k37.onrender.com"

# Global State
running = False
investment_capital = 0.0
entry_positions = {}
trade_log = []
market_snapshot = {p: {"price": 0, "bias": "Scanning...", "atr": 0} for p in PAIRS}
price_history = {p: [] for p in PAIRS}
volume_history = {p: [] for p in PAIRS}

# ================= SIGNING & ORDERING =================
def sign_payload(payload):
    payload_json = json.dumps(payload, separators=(',', ':'))
    signature = hmac.new(API_SECRET.encode(), payload_json.encode(), hashlib.sha256).hexdigest()
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
    except: return {"error": "Connection Failed"}

# ================= DATA FETCH (Dictionary Parser) =================
def fetch_market_data(pair):
    """Correctly parses CoinDCX Dictionary-based orderbook."""
    try:
        url = f"https://public.coindcx.com{pair}"
        r = requests.get(url, timeout=5)
        data = r.json()
        if "bids" in data and data["bids"]:
            # Extract prices from dictionary keys
            prices = [float(p) for p in data["bids"].keys()]
            top_price = max(prices)
            volume = float(data["bids"][f"{top_price:.8f}"])
            return top_price, volume
    except: return None, None

# ================= ADVANCED TRADING ENGINE =================
def advanced_trading_loop():
    global running, investment_capital
    while True:
        if not running:
            time.sleep(2); continue
        for pair in PAIRS:
            price, vol = fetch_market_data(pair)
            if not price: continue

            price_history[pair].append(price)
            volume_history[pair].append(vol)
            if len(price_history[pair]) > 50:
                price_history[pair].pop(0); volume_history[pair].pop(0)

            # Volatility Logic (ATR)
            diffs = np.abs(np.diff(price_history[pair][-15:])) if len(price_history[pair]) > 1 else [0]
            atr = np.mean(diffs)
            market_snapshot[pair].update({"price": round(price, 2), "atr": round(atr, 2)})

            # 1. Trailing Profit Skimming
            if pair in entry_positions:
                pos = entry_positions[pair]
                if pos["side"] == "long":
                    if price > pos["peak"]: 
                        pos["peak"] = price
                        pos["sl"] = max(pos["sl"], price - (atr * 1.5))
                    if price < pos["sl"]: close_trade(pair, price, "Trailing Stop")
                continue

            # 2. Strategy Logic
            if len(price_history[pair]) > 20:
                v_std = np.std(volume_history[pair][-20:])
                z_score = (vol - np.mean(volume_history[pair][-20:])) / v_std if v_std > 0 else 0
                bias = "Bullish" if price > np.mean(price_history[pair][-20:]) else "Bearish"
                market_snapshot[pair]["bias"] = bias

                # Trigger: High Volatility Spike (Z-Score)
                if z_score > 2.0 and investment_capital > 100:
                    execute_trade(pair, "long" if bias == "Bullish" else "short", price, atr)
        time.sleep(10)

def execute_trade(pair, side, price, atr):
    global investment_capital
    qty = round((investment_capital * 0.05) / price, 6) # 5% per trade
    entry_positions[pair] = {
        "side": side, "qty": qty, "entry": price, "peak": price,
        "sl": price - (atr * 2) if side == "long" else price + (atr * 2)
    }
    trade_log.append({"time": datetime.now(IST).strftime("%H:%M:%S"), "pair": pair, "side": side, "pnl": "OPEN"})

def close_trade(pair, price, reason):
    global investment_capital
    pos = entry_positions.pop(pair)
    pnl = (price - pos["entry"]) * pos["qty"] if pos["side"] == "long" else (pos["entry"] - price) * pos["qty"]
    investment_capital += pnl
    trade_log.append({"time": datetime.now(IST).strftime("%H:%M:%S"), "pair": pair, "side": f"CLOSE ({reason})", "pnl": round(pnl, 2)})

# ================= SERVER ROUTES =================
@app.route("/")
def home(): return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return jsonify({"running": running, "capital": round(investment_capital, 2), "positions": entry_positions, "market": market_snapshot, "log": trade_log[-15:]})

@app.route("/start", methods=["POST"])
def start(): global running; running = True; return "OK"

@app.route("/stop", methods=["POST"])
def stop(): global running; running = False; return "OK"

@app.route("/set_capital", methods=["POST"])
def set_capital():
    global investment_capital
    investment_capital = float(request.json.get("capital", 0))
    return "OK"

@app.route("/ping")
def ping(): return "pong"

# Threads
threading.Thread(target=advanced_trading_loop, daemon=True).start()
threading.Thread(target=lambda: (time.sleep(240), requests.get(f"{RENDER_URL}/ping")), daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
