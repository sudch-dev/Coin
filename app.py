import os, time, threading, requests, json, hmac, hashlib
from datetime import datetime
import numpy as np
import pytz
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

# ================= CONFIG =================
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
BASE_URL = "https://api.coindcx.com"
IST = pytz.timezone("Asia/Kolkata")
PAIRS = ["I-BTC_INR", "I-ETH_INR", "I-SOL_INR"]
# YOUR PROVIDED RENDER URL
RENDER_URL = "https://coin-4k37.onrender.com"

# Global System State
running = False
investment_capital = 0.0
entry_positions = {}
trade_log = []
market_snapshot = {p: {"price": 0, "bias": "Scanning...", "atr": 0} for p in PAIRS}

# Data Buffers
price_history = {p: [] for p in PAIRS}
volume_history = {p: [] for p in PAIRS}

# ================= ADVANCED DATA PARSER =================
def fetch_market_data(pair):
    try:
        url = f"https://public.coindcx.com{pair}"
        r = requests.get(url, timeout=5)
        data = r.json()
        if "bids" in data and data["bids"]:
            prices = [float(p) for p in data["bids"].keys()]
            top_price = max(prices)
            volume = float(data["bids"][f"{top_price:.8f}"])
            return top_price, volume
    except:
        return None, None

# ================= MATHEMATICAL ENGINE =================
def get_atr(pair, period=14):
    if len(price_history[pair]) < period + 1: return 0
    return np.mean(np.abs(np.diff(price_history[pair][-period-1:])))

def get_volume_zscore(pair):
    v_data = volume_history[pair]
    if len(v_data) < 20: return 0
    std_v = np.std(v_data[-20:])
    return (v_data[-1] - np.mean(v_data[-20:])) / std_v if std_v > 0 else 0

# ================= TRADING CORE =================
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
            if len(price_history[pair]) > 150:
                price_history[pair].pop(0); volume_history[pair].pop(0)

            atr = get_atr(pair)
            market_snapshot[pair].update({"price": round(price, 2), "atr": round(atr, 2)})

            if pair in entry_positions:
                pos = entry_positions[pair]
                if pos["side"] == "long":
                    if price > pos["peak"]: 
                        pos["peak"] = price
                        pos["sl"] = max(pos["sl"], price - (atr * 1.5))
                    if price < pos["sl"]: close_trade(pair, price, "SL/Trailing")
                continue

            if len(price_history[pair]) > 30:
                z_score = get_volume_zscore(pair)
                bias = "bullish" if price > np.mean(price_history[pair][-30:]) else "bearish"
                market_snapshot[pair]["bias"] = bias.capitalize()

                # Entry Signal: High Volume + Trend
                if z_score > 2.0 and investment_capital > 100:
                    execute_trade(pair, "long" if bias == "bullish" else "short", price, atr)
        time.sleep(10)

def execute_trade(pair, side, price, atr):
    global investment_capital
    risk = investment_capital * 0.05
    qty = round(risk / price, 6)
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

# ================= ROUTES =================
@app.route("/")
def home(): return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return jsonify({
        "running": running, "capital": round(investment_capital, 2),
        "positions": entry_positions, "market": market_snapshot,
        "log": trade_log[-15:]
    })

@app.route("/start", methods=["POST"])
def start(): global running; running = True; return "OK"

@app.route("/stop", methods=["POST"])
def stop(): global running; running = False; return "OK"

@app.route("/set_capital", methods=["POST"])
def set_capital():
    global investment_capital
    investment_capital = float(request.json.get("capital", 0))
    return "OK"

# THE MISSING PROVISION: PING ROUTE
@app.route("/ping")
def ping(): return "pong", 200

# ================= KEEP ALIVE ENGINE =================
def keep_alive_loop():
    """Pings the Render URL every 4 minutes to prevent idling."""
    while True:
        try:
            requests.get(f"{RENDER_URL}/ping", timeout=10)
        except:
            pass
        time.sleep(240)

# Start all background threads
threading.Thread(target=advanced_trading_loop, daemon=True).start()
threading.Thread(target=keep_alive_loop, daemon=True).start()

if __name__ == "__main__":
    # Render Dynamic Port
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
