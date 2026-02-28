import os
import time
import json
import hmac
import hashlib
import requests
import threading
from flask import Flask, jsonify, render_template

app = Flask(__name__)

# ================= CONFIG =================

API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET").encode()

BASE_URL = "https://api.coindcx.com"

SYMBOL = "ORDIUSDT"
LEVERAGE = 3
RISK_PER_TRADE = 0.01  # 1% risk (safe for small accounts)
CHECK_INTERVAL = 10

running = False
status_text = "Idle"
last_price = 0
wallet_balance = 0
available_balance = 0
position_data = {}

# ================= AUTH =================

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

# ================= MARKET =================

def get_price():
    global last_price
    r = requests.get(f"{BASE_URL}/exchange/ticker")
    for item in r.json():
        if item["market"] == SYMBOL:
            last_price = float(item["last_price"])
            return last_price
    return 0

# ================= FUTURES ACCOUNT =================

def get_account():
    global wallet_balance, available_balance

    body = {"timestamp": int(time.time() * 1000)}
    data = post("/exchange/v1/derivatives/account", body)

    if "data" in data:
        acc = data["data"]
        wallet_balance = float(acc.get("wallet_balance", 0))
        available_balance = float(acc.get("available_balance", 0))

    return available_balance

# ================= POSITION =================

def get_position():
    global position_data

    body = {"timestamp": int(time.time() * 1000)}
    data = post("/exchange/v1/derivatives/positions", body)

    if "data" in data:
        for p in data["data"]:
            if p["market"] == SYMBOL:
                position_data = p
                return p

    position_data = {}
    return {}

# ================= ORDER =================

def place_order(side, qty):
    body = {
        "market": SYMBOL,
        "side": side,
        "order_type": "market_order",
        "total_quantity": str(round(qty, 3)),
        "timestamp": int(time.time() * 1000)
    }
    return post("/exchange/v1/derivatives/orders/create", body)

# ================= STRATEGY =================

def strategy():
    price = get_price()
    avail = get_account()
    pos = get_position()

    if price == 0 or avail <= 1:
        return

    qty = (wallet_balance * RISK_PER_TRADE * LEVERAGE) / price

    # Ensure minimum notional (safe for small account)
    min_notional = 5
    if qty * price < min_notional:
        qty = min_notional / price

    # ENTRY
    if not pos:
        place_order("buy", qty)

    # EXIT (1% profit target)
    elif float(pos.get("size", 0)) > 0:
        entry = float(pos["avg_price"])
        if price > entry * 1.01:
            place_order("sell", abs(float(pos["size"])))

# ================= BOT LOOP =================

def bot_loop():
    global running, status_text

    status_text = "Running"

    while running:
        try:
            strategy()
        except Exception as e:
            print("Error:", e)

        time.sleep(CHECK_INTERVAL)

    status_text = "Stopped"

# ================= ROUTES =================

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

@app.route("/flatten", methods=["POST"])
def flatten():
    pos = get_position()
    if pos:
        size = abs(float(pos.get("size", 0)))
        side = "sell" if float(pos.get("size", 0)) > 0 else "buy"
        place_order(side, size)
    return jsonify({"status": "position closed"})

@app.route("/status")
def status():
    get_price()
    get_account()
    get_position()

    return jsonify({
        "bot_status": status_text,
        "price": last_price,
        "wallet_balance": wallet_balance,
        "available_balance": available_balance,
        "position": position_data
    })

# ================= RUN =================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)