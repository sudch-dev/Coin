import os
import json
import hmac
import time
import hashlib
import threading
import requests
import websocket
from flask import Flask, render_template, jsonify, request
from datetime import datetime

# ================= CONFIG =================

app = Flask(__name__)

API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("API_SECRET")

BASE_URL = "https://api.coindcx.com"
WS_URL = "wss://stream.coindcx.com"
RENDER_URL = "https://coin-4k37.onrender.com"

PAIRS = ["BTCINR", "ETHINR", "SOLINR"]

capital = 0.0
positions = {}
bias = {}
structure = {}
order_blocks = {}
trade_log = []

candles = {p: [] for p in PAIRS}
current_candle = {}

# ================= AUTH =================

def sign_payload(payload):
    secret_bytes = bytes(SECRET_KEY, encoding='utf-8')
    json_payload = json.dumps(payload, separators=(',', ':'))
    signature = hmac.new(secret_bytes, json_payload.encode(), hashlib.sha256).hexdigest()
    return json_payload, signature

# ================= ORDER =================

def place_market_order(pair, side, qty):
    payload = {
        "side": side,
        "order_type": "market_order",
        "market": pair,
        "total_quantity": qty,
        "timestamp": int(time.time() * 1000)
    }

    json_payload, signature = sign_payload(payload)

    headers = {
        "Content-Type": "application/json",
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": signature
    }

    r = requests.post(BASE_URL + "/exchange/v1/orders/create",
                      data=json_payload,
                      headers=headers)

    return r.json()

# ================= SWING DETECTION =================

def detect_swings(pair):
    if len(candles[pair]) < 5:
        return None

    window = candles[pair][-5:]
    center = window[2]

    highs = [c["high"] for c in window]
    lows = [c["low"] for c in window]

    if center["high"] == max(highs):
        return ("swing_high", center["high"])

    if center["low"] == min(lows):
        return ("swing_low", center["low"])

    return None

# ================= STRUCTURE ENGINE =================

def process_structure(pair):

    if len(candles[pair]) < 6:
        return

    swing = detect_swings(pair)
    if not swing:
        return

    swing_type, level = swing

    if pair not in structure:
        structure[pair] = {"last_high": None, "last_low": None}

    if pair not in bias:
        bias[pair] = None

    prev_high = structure[pair]["last_high"]
    prev_low = structure[pair]["last_low"]
    last_close = candles[pair][-1]["close"]

    # ----------------- BoS -----------------

    if prev_high and last_close > prev_high:

        if bias[pair] != "bullish":
            trade_log.append({"event": "BoS Bullish", "pair": pair})

        bias[pair] = "bullish"
        mark_order_block(pair, "bullish")

    elif prev_low and last_close < prev_low:

        if bias[pair] != "bearish":
            trade_log.append({"event": "BoS Bearish", "pair": pair})

        bias[pair] = "bearish"
        mark_order_block(pair, "bearish")

    # ----------------- CHoCH -----------------

    if bias[pair] == "bullish" and prev_low and last_close < prev_low:
        trade_log.append({"event": "CHoCH Bearish", "pair": pair})
        force_close(pair)
        bias[pair] = "bearish"
        order_blocks.pop(pair, None)

    elif bias[pair] == "bearish" and prev_high and last_close > prev_high:
        trade_log.append({"event": "CHoCH Bullish", "pair": pair})
        force_close(pair)
        bias[pair] = "bullish"
        order_blocks.pop(pair, None)

    # Update swings AFTER detection
    if swing_type == "swing_high":
        structure[pair]["last_high"] = level

    if swing_type == "swing_low":
        structure[pair]["last_low"] = level

# ================= ORDER BLOCK =================

def mark_order_block(pair, direction):

    if len(candles[pair]) < 2:
        return

    last_candle = candles[pair][-2]

    if direction == "bullish" and last_candle["close"] < last_candle["open"]:
        order_blocks[pair] = {
            "low": last_candle["low"],
            "high": last_candle["high"],
            "type": "bullish"
        }

    if direction == "bearish" and last_candle["close"] > last_candle["open"]:
        order_blocks[pair] = {
            "low": last_candle["low"],
            "high": last_candle["high"],
            "type": "bearish"
        }

# ================= ENTRY ENGINE =================

def check_entry(pair):

    if capital <= 0:
        return

    if pair in positions:
        return

    if pair not in order_blocks:
        return

    if bias.get(pair) is None:
        return

    ob = order_blocks[pair]
    last_price = candles[pair][-1]["close"]

    # Trade ONLY in direction of bias
    if ob["type"] != bias[pair]:
        return

    if ob["low"] <= last_price <= ob["high"]:

        risk_amount = capital * 0.01
        qty = round(risk_amount / last_price, 6)

        side = "buy" if bias[pair] == "bullish" else "sell"

        order = place_market_order(pair, side, qty)

        if "id" in order:
            positions[pair] = {
                "side": side,
                "qty": qty,
                "entry": last_price
            }

            trade_log.append({
                "event": "ENTRY",
                "pair": pair,
                "side": side,
                "price": last_price
            })

# ================= EXIT =================

def force_close(pair):

    if pair not in positions:
        return

    pos = positions[pair]
    side = "sell" if pos["side"] == "buy" else "buy"

    place_market_order(pair, side, pos["qty"])

    trade_log.append({
        "event": "EXIT",
        "pair": pair
    })

    del positions[pair]

# ================= WEBSOCKET =================

def on_message(ws, message):

    data = json.loads(message)
    if "data" not in data:
        return

    pair = data["data"]["symbol"]
    if pair not in PAIRS:
        return

    price = float(data["data"]["price"])
    timestamp = int(data["data"]["timestamp"]) // 1000
    minute = timestamp - (timestamp % 60)

    if pair not in current_candle or current_candle[pair]["time"] != minute:

        if pair in current_candle:
            candles[pair].append(current_candle[pair])
            process_structure(pair)
            check_entry(pair)

        current_candle[pair] = {
            "time": minute,
            "open": price,
            "high": price,
            "low": price,
            "close": price
        }

    else:
        c = current_candle[pair]
        c["high"] = max(c["high"], price)
        c["low"] = min(c["low"], price)
        c["close"] = price

def on_close(ws, close_status_code, close_msg):
    time.sleep(5)
    start_ws()

def start_ws():
    ws = websocket.WebSocketApp(
        WS_URL,
        on_message=on_message,
        on_close=on_close
    )
    threading.Thread(target=ws.run_forever, daemon=True).start()

# ================= KEEP ALIVE =================

def self_keepalive():
    while True:
        try:
            requests.get(f"{RENDER_URL}/ping")
        except:
            pass
        time.sleep(240)

# ================= ROUTES =================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/set_capital", methods=["POST"])
def set_capital():
    global capital
    capital = float(request.json["capital"])
    return "Capital Set"

@app.route("/dashboard")
def dashboard():
    return jsonify({
        "capital": capital,
        "positions": positions,
        "bias": bias,
        "structure": structure,
        "order_blocks": order_blocks,
        "recent_log": trade_log[-30:]
    })

@app.route("/ping")
def ping():
    return "pong"

# ================= START =================

start_ws()
threading.Thread(target=self_keepalive, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))