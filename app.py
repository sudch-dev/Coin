import os
import time
import hmac
import hashlib
import requests
import json
import traceback
from flask import Flask, render_template, jsonify, request
from datetime import datetime
from pytz import timezone
from collections import deque

# -------------------- Flask --------------------
app = Flask(__name__)

APP_BASE_URL = os.environ.get("APP_BASE_URL", "https://coin-4k37.onrender.com")

# -------------------- API / Markets --------------------
API_KEY = os.environ.get("API_KEY")
API_SECRET_RAW = os.environ.get("API_SECRET", "")
API_SECRET = API_SECRET_RAW.encode() if isinstance(API_SECRET_RAW, str) else API_SECRET_RAW
BASE_URL = "https://api.coindcx.com"

PAIRS = [
    "BTCUSDT", "ETHUSDT", "DOGEUSDT",
    "BNBUSDT", "SOLUSDT", "XRPUSDT", "SHIBUSDT"
]  # ✅ LTC removed

# -------------------- Helpers --------------------
def ist_now():
    return datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")

def sign_request(params):
    body = json.dumps(params)
    signature = hmac.new(API_SECRET, body.encode(), hashlib.sha256).hexdigest()
    return {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": signature,
        "Content-Type": "application/json"
    }

def place_order(pair, side, qty, price=None):
    try:
        order = {
            "market": pair,
            "side": side.lower(),
            "order_type": "market",
            "total_quantity": str(qty)
        }
        if price:
            order["price_per_unit"] = str(price)

        headers = sign_request(order)
        res = requests.post(f"{BASE_URL}/exchange/v1/orders/create", headers=headers, data=json.dumps(order))
        return res.json()
    except Exception as e:
        return {"error": str(e)}

# -------------------- Trading Logic --------------------
scan_log = deque(maxlen=50)
open_trades = {}
balances = {}

TARGET_PROFIT = 0.005  # ✅ 0.5% TP only
# No Stop Loss

def log(msg):
    entry = f"{ist_now()} | {msg}"
    print(entry)
    scan_log.appendleft(entry)

def simulate_indicators(pair):
    """Dummy indicators (replace with real candle/EMA/MACD)."""
    import random
    ema_cross = random.choice(["bull", "bear", None])
    macd = random.choice(["bullish", "bearish"])
    smc_gate = random.choice(["open", "closed"])
    return ema_cross, macd, smc_gate

def scan_and_trade():
    for pair in PAIRS:
        try:
            ema_cross, macd, smc_gate = simulate_indicators(pair)
            log(f"{pair} | ema={ema_cross} macd={macd} smc={smc_gate}")

            # Check entry condition
            if pair not in open_trades:
                if ema_cross == "bull" and macd == "bullish" and smc_gate == "open":
                    qty = 1  # replace with balance logic
                    open_trades[pair] = {"entry": 100.0, "qty": qty}  # dummy entry
                    log(f"BUY {pair} qty={qty}")
                    place_order(pair, "buy", qty)

            else:
                trade = open_trades[pair]
                entry = trade["entry"]
                current_price = entry * 1.006  # dummy price rise

                if current_price >= entry * (1 + TARGET_PROFIT):
                    qty = trade["qty"]
                    log(f"SELL {pair} qty={qty} | TP HIT")
                    place_order(pair, "sell", qty)
                    del open_trades[pair]

        except Exception as e:
            log(f"{pair} ERROR {str(e)}")
            traceback.print_exc()

# -------------------- Routes --------------------
@app.route("/")
def home():
    return render_template("index.html", 
                           balances=balances, 
                           trades=open_trades,
                           scan_log=list(scan_log))

@app.route("/scan")
def manual_scan():
    scan_and_trade()
    return jsonify({"log": list(scan_log)})

@app.route("/ping")
def ping():
    token = os.environ.get("KEEPALIVE_TOKEN", "")
    provided = (request.args.get("t") or request.headers.get("X-Keepalive-Token") or "")
    if token and provided != token:
        print(f"[{ist_now()}] /ping forbidden (bad token)")
        return "forbidden", 403
    print(f"[{ist_now()}] /ping pong")
    return "pong"

# -------------------- Background Worker --------------------
def loop():
    while True:
        scan_and_trade()
        time.sleep(60)  # keep polling interval unchanged

import threading
threading.Thread(target=loop, daemon=True).start()

# -------------------- Run --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
