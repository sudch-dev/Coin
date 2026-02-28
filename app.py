import os
import time
import json
import hmac
import hashlib
import requests
import threading
from flask import Flask, jsonify, render_template

app = Flask(__name__)

API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET").encode()

BASE_URL = "https://api.coindcx.com"

SYMBOL = "ORDIUSDT"  # Futures pair
LEVERAGE = 5
RISK_PER_TRADE = 0.02

running = False
status = "Idle"
last_price = 0
equity = 0
position = {}

# ---------- AUTH ----------
def sign(payload):
    return hmac.new(API_SECRET, payload.encode(), hashlib.sha256).hexdigest()

def post(endpoint, body):
    payload = json.dumps(body, separators=(',', ':'))
    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": sign(payload),
        "Content-Type": "application/json"
    }
    r = requests.post(BASE_URL + endpoint, headers=headers, data=payload)
    return r.json()

# ---------- MARKET PRICE ----------
def get_price():
    global last_price
    r = requests.get(f"{BASE_URL}/exchange/ticker")
    for item in r.json():
        if item["market"] == SYMBOL:
            last_price = float(item["last_price"])
            return last_price
    return 0

# ---------- FUTURES EQUITY ----------
def get_equity():
    global equity
    body = {"timestamp": int(time.time() * 1000)}
    data = post("/exchange/v1/users/futures/balance", body)
    equity = float(data.get("available_balance", 0))
    return equity

# ---------- OPEN POSITION ----------
def get_position():
    global position
    body = {"timestamp": int(time.time() * 1000)}
    data = post("/exchange/v1/futures/positions", body)
    for p in data:
        if p["market"] == SYMBOL:
            position = p
            return p
    position = {}
    return {}

# ---------- ORDER ----------
def place_order(side, qty):
    body = {
        "market": SYMBOL,
        "side": side,
        "order_type": "market_order",
        "total_quantity": str(qty),
        "timestamp": int(time.time() * 1000)
    }
    return post("/exchange/v1/futures/orders/create", body)

# ---------- SIMPLE PSAR-LIKE LOGIC ----------
def strategy():
    price = get_price()
    eq = get_equity()
    pos = get_position()

    if price == 0 or eq == 0:
        return

    qty = (eq * RISK_PER_TRADE * LEVERAGE) / price

    # Example logic: momentum breakout
    if not pos:
        place_order("buy", qty)

    elif float(pos.get("size", 0)) > 0:
        entry = float(pos["avg_price"])
        if price > entry * 1.01:
            place_order("sell", abs(float(pos["size"])))

# ---------- BOT LOOP ----------
def bot_loop():
    global running, status
    status = "Running"

    while running:
        try:
            strategy()
        except Exception as e:
            print("Error:", e)
        time.sleep(10)

    status = "Stopped"

# ---------- ROUTES ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start", methods=["POST"])
def start():
    global running
    if not running:
        running = True
        threading.Thread(target=bot_loop).start()
    return jsonify({"status": "started"})

@app.route("/stop", methods=["POST"])
def stop():
    global running
    running = False
    return jsonify({"status": "stopped"})

@app.route("/status")
def get_status():
    return jsonify({
        "status": status,
        "price": last_price,
        "equity": equity,
        "position": position
    })

# ---------- RENDER READY ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)