import os
import time
import json
import hmac
import hashlib
import requests
import threading
import numpy as np
from flask import Flask, render_template, jsonify, request
from datetime import datetime
import pytz

app = Flask(__name__)

IST = pytz.timezone("Asia/Kolkata")

API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("API_SECRET")

BASE_URL = "https://api.coindcx.com"

PAIRS = ["BTCINR", "ETHINR", "SOLINR"]

running = False
capital = 0
positions = {}
trade_log = []
market_bias = {}
structure = {}
price_data = {p: [] for p in PAIRS}

# ================= AUTH SIGN =================

def sign_payload(payload):
    secret_bytes = bytes(SECRET_KEY, encoding='utf-8')
    json_payload = json.dumps(payload, separators=(',', ':'))
    signature = hmac.new(secret_bytes, json_payload.encode(), hashlib.sha256).hexdigest()
    return json_payload, signature

# ================= BALANCE =================

def get_balance():
    payload = {
        "timestamp": int(time.time() * 1000)
    }

    json_payload, signature = sign_payload(payload)

    headers = {
        'Content-Type': 'application/json',
        'X-AUTH-APIKEY': API_KEY,
        'X-AUTH-SIGNATURE': signature
    }

    response = requests.post(
        BASE_URL + "/exchange/v1/users/balances",
        data=json_payload,
        headers=headers
    )

    return response.json()

# ================= ORDER =================

def place_market_order(pair, side, quantity):

    payload = {
        "side": side,
        "order_type": "market_order",
        "market": pair,
        "total_quantity": quantity,
        "timestamp": int(time.time() * 1000)
    }

    json_payload, signature = sign_payload(payload)

    headers = {
        'Content-Type': 'application/json',
        'X-AUTH-APIKEY': API_KEY,
        'X-AUTH-SIGNATURE': signature
    }

    response = requests.post(
        BASE_URL + "/exchange/v1/orders/create",
        data=json_payload,
        headers=headers
    )

    return response.json()

# ================= PRICE =================

def get_price(pair):
    try:
        r = requests.get(f"https://public.coindcx.com/market_data/orderbook?pair={pair}")
        data = r.json()
        return float(data["bids"][0]["price"])
    except:
        return None

# ================= STRUCTURE =================

def detect_bos(pair, price):
    level = structure.get(pair)
    if not level:
        return None

    if price > level["high"]:
        return "buy"
    if price < level["low"]:
        return "sell"
    return None

def trading_loop():
    global running, capital

    while running:

        for pair in PAIRS:

            price = get_price(pair)
            if not price:
                continue

            price_data[pair].append(price)
            if len(price_data[pair]) > 50:
                price_data[pair].pop(0)

            if len(price_data[pair]) < 10:
                continue

            structure[pair] = {
                "high": max(price_data[pair][-10:]),
                "low": min(price_data[pair][-10:])
            }

            signal = detect_bos(pair, price)

            if signal and pair not in positions:

                risk = capital * 0.01
                qty = round(risk / price, 6)

                side = "buy" if signal == "buy" else "sell"

                order = place_market_order(pair, side, qty)

                if "id" in order:
                    positions[pair] = {
                        "side": side,
                        "qty": qty,
                        "entry": price
                    }

                    trade_log.append({
                        "time": datetime.now(IST).strftime("%H:%M:%S"),
                        "pair": pair,
                        "side": side,
                        "qty": qty
                    })

        time.sleep(10)

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
    global capital
    capital = float(request.json["capital"])
    return "Capital Set"

@app.route("/dashboard")
def dashboard():
    return jsonify({
        "running": running,
        "capital": capital,
        "positions": positions,
        "log": trade_log[-20:]
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