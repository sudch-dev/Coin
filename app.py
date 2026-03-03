import os
import time
import json
import hmac
import hashlib
import threading
import requests
from flask import Flask, jsonify, request, send_from_directory
from collections import deque

app = Flask(__name__)

# ==============================
# ENV VARIABLES
# ==============================

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

BASE_URL = "https://api.coindcx.com"
RENDER_URL = "https://coin-4k37.onrender.com"

# ==============================
# GLOBAL STATE
# ==============================

capital = 0
positions = {}
trade_log = []
trend_state = {}
order_blocks = {}
candles = {}
running = False

PAIRS = ["BTCINR", "ETHINR", "SOLINR"]

# ==============================
# KEEP ALIVE
# ==============================

def self_keepalive():
    while True:
        try:
            requests.get(f"{RENDER_URL}/ping", timeout=5)
        except:
            pass
        time.sleep(240)

@app.route("/ping")
def ping():
    return "pong"

# ==============================
# ROOT ROUTE (UI FIX)
# ==============================

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

# ==============================
# AUTH SIGNING
# ==============================

def sign_payload(payload):
    payload_json = json.dumps(payload, separators=(',', ':'))
    signature = hmac.new(
        API_SECRET.encode(),
        payload_json.encode(),
        hashlib.sha256
    ).hexdigest()
    return signature, payload_json

# ==============================
# ORDER EXECUTION
# ==============================

def place_market_order(pair, side, quantity):

    endpoint = "/exchange/v1/orders/create"
    url = BASE_URL + endpoint

    payload = {
        "side": side,
        "order_type": "market_order",
        "market": pair,
        "total_quantity": quantity,
        "timestamp": int(time.time() * 1000)
    }

    signature, payload_json = sign_payload(payload)

    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": signature,
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, data=payload_json)
    return response.json()

# ==============================
# EMA
# ==============================

def calculate_ema(pair, period):
    if len(candles[pair]) < period:
        return None

    closes = [c["close"] for c in list(candles[pair])[-period:]]
    k = 2 / (period + 1)
    ema = closes[0]

    for price in closes[1:]:
        ema = price * k + ema * (1 - k)

    return ema

# ==============================
# STRUCTURE (BoS / CHoCH / OB)
# ==============================

def process_structure(pair):

    if len(candles[pair]) < 5:
        return

    last = candles[pair][-1]
    prev = candles[pair][-2]

    high_break = last["high"] > prev["high"]
    low_break = last["low"] < prev["low"]

    if pair not in trend_state:
        trend_state[pair] = {"bias": None, "confirmed": False}

    state = trend_state[pair]

    if high_break:
        if state["bias"] == "bearish":
            state["confirmed"] = False
            trade_log.append({"event": "CHoCH", "pair": pair})

        state["bias"] = "bullish"
        state["confirmed"] = True
        trade_log.append({"event": "BoS Bullish", "pair": pair})

        order_blocks[pair] = {
            "type": "bullish",
            "low": prev["low"],
            "high": prev["high"]
        }

    elif low_break:
        if state["bias"] == "bullish":
            state["confirmed"] = False
            trade_log.append({"event": "CHoCH", "pair": pair})

        state["bias"] = "bearish"
        state["confirmed"] = True
        trade_log.append({"event": "BoS Bearish", "pair": pair})

        order_blocks[pair] = {
            "type": "bearish",
            "low": prev["low"],
            "high": prev["high"]
        }

# ==============================
# ENTRY (OB + EMA)
# ==============================

def check_entry(pair):

    if not running:
        return

    if capital <= 0:
        return

    if pair in positions:
        return

    state = trend_state.get(pair)
    if not state or not state["confirmed"]:
        return

    if pair not in order_blocks:
        return

    ob = order_blocks[pair]
    last_price = candles[pair][-1]["close"]

    ema_fast = calculate_ema(pair, 9)
    ema_slow = calculate_ema(pair, 21)

    if not ema_fast or not ema_slow:
        return

    if state["bias"] == "bullish" and ema_fast <= ema_slow:
        return

    if state["bias"] == "bearish" and ema_fast >= ema_slow:
        return

    if ob["type"] != state["bias"]:
        return

    if ob["low"] <= last_price <= ob["high"]:

        risk = capital * 0.01
        qty = round(risk / last_price, 6)

        side = "buy" if state["bias"] == "bullish" else "sell"
        order = place_market_order(pair, side, qty)

        if "id" in order:
            positions[pair] = {
                "side": side,
                "qty": qty,
                "entry": last_price,
                "tp": last_price * 1.01 if side == "buy" else last_price * 0.99,
                "continuation": False
            }
            trade_log.append({"event": "ENTRY", "pair": pair})

# ==============================
# EXIT LOGIC
# ==============================

def force_close(pair):
    pos = positions[pair]
    side = "sell" if pos["side"] == "buy" else "buy"
    place_market_order(pair, side, pos["qty"])
    del positions[pair]

def manage_position(pair):

    if pair not in positions:
        return

    pos = positions[pair]
    last_price = candles[pair][-1]["close"]
    state = trend_state.get(pair)

    # Continuation via BoS
    if state and state["confirmed"]:
        if (pos["side"] == "buy" and state["bias"] == "bullish") or \
           (pos["side"] == "sell" and state["bias"] == "bearish"):
            pos["continuation"] = True

    # CHoCH Exit
    if state and not state["confirmed"]:
        force_close(pair)
        trade_log.append({"event": "EXIT CHoCH", "pair": pair})
        return

    # Opposite OB Exit
    if pair in order_blocks:
        ob = order_blocks[pair]
        if ob["type"] != state["bias"]:
            if ob["low"] <= last_price <= ob["high"]:
                force_close(pair)
                trade_log.append({"event": "EXIT Opp OB", "pair": pair})
                return

    # TP Exit (if no continuation)
    if pos["side"] == "buy" and last_price >= pos["tp"]:
        if not pos["continuation"]:
            force_close(pair)
            trade_log.append({"event": "EXIT TP", "pair": pair})

    if pos["side"] == "sell" and last_price <= pos["tp"]:
        if not pos["continuation"]:
            force_close(pair)
            trade_log.append({"event": "EXIT TP", "pair": pair})

# ==============================
# ENGINE LOOP
# ==============================

def engine_loop():
    global candles

    for p in PAIRS:
        candles[p] = deque(maxlen=200)

    while True:
        if running:
            for pair in PAIRS:
                price = 100 + (time.time() % 50)  # placeholder live feed
                candle = {"open": price, "high": price+1, "low": price-1, "close": price}
                candles[pair].append(candle)

                process_structure(pair)
                check_entry(pair)
                manage_position(pair)

        time.sleep(5)

# ==============================
# CONTROL ROUTES
# ==============================

@app.route("/start", methods=["POST"])
def start():
    global running, capital
    data = request.json
    capital = float(data.get("capital", 0))
    running = True
    return jsonify({"status": "running", "capital": capital})

@app.route("/stop")
def stop():
    global running
    running = False
    return jsonify({"status": "stopped"})

@app.route("/status")
def status():
    return jsonify({
        "running": running,
        "capital": capital,
        "positions": positions,
        "recent_log": trade_log[-10:]
    })

# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    threading.Thread(target=self_keepalive, daemon=True).start()
    threading.Thread(target=engine_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))