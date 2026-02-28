import os
import time
import json
import hmac
import hashlib
import requests
from flask import Flask, jsonify, send_from_directory

app = Flask(__name__)

BASE_URL = "https://api.coindcx.com"

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

SYMBOL = "ORDIUSDT"
LEVERAGE = 3
RISK_PER_TRADE = 0.01
DAILY_LOSS_LIMIT = 0.05
MIN_NOTIONAL = 5

daily_start_equity = None

# ================= SIGNED REQUEST =================

def post(endpoint, body):
    payload = json.dumps(body)

    signature = hmac.new(
        API_SECRET.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()

    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": signature,
        "Content-Type": "application/json"
    }

    r = requests.post(BASE_URL + endpoint,
                      data=payload,
                      headers=headers,
                      timeout=10)

    if r.status_code != 200:
        return {"error": "API request failed"}

    return r.json()

# ================= ACCOUNT =================

def get_equity():
    body = {"timestamp": int(time.time() * 1000)}
    data = post("/exchange/v1/derivatives/account", body)

    if "data" not in data:
        return 0, 0

    wallet = float(data["data"]["wallet_balance"])
    available = float(data["data"]["available_balance"])

    return wallet, available

# ================= PRICE =================

def get_price():
    r = requests.get(
        f"{BASE_URL}/exchange/ticker?market={SYMBOL}",
        timeout=10
    )
    return float(r.json()["last_price"])

# ================= POSITION =================

def get_position():
    body = {"timestamp": int(time.time() * 1000)}
    data = post("/exchange/v1/derivatives/positions", body)

    if "data" not in data:
        return {}

    for p in data["data"]:
        if p["market"] == SYMBOL:
            return p

    return {}

# ================= RISK ENGINE =================

def risk_checks():
    global daily_start_equity

    wallet, available = get_equity()

    if daily_start_equity is None:
        daily_start_equity = wallet

    # Too small balance
    if available < 10:
        return "Balance too low"

    # Daily loss cutoff
    if wallet < daily_start_equity * (1 - DAILY_LOSS_LIMIT):
        return "Daily loss limit reached"

    # Existing position
    if get_position():
        return "Position already open"

    return None

# ================= POSITION SIZE =================

def calculate_qty(price):
    wallet, available = get_equity()

    risk_amount = available * RISK_PER_TRADE
    position_value = risk_amount * LEVERAGE

    if position_value < MIN_NOTIONAL:
        return 0

    qty = position_value / price
    return round(qty, 2)

# ================= LEVERAGE =================

def set_leverage():
    body = {
        "timestamp": int(time.time() * 1000),
        "market": SYMBOL,
        "leverage": LEVERAGE
    }
    post("/exchange/v1/derivatives/change_leverage", body)

# ================= ORDER =================

def place_order(side):

    error = risk_checks()
    if error:
        return {"error": error}

    price = get_price()
    qty = calculate_qty(price)

    if qty <= 0:
        return {"error": "Order size too small"}

    set_leverage()

    body = {
        "timestamp": int(time.time() * 1000),
        "market": SYMBOL,
        "side": side,
        "order_type": "market_order",
        "quantity": str(qty),
        "leverage": LEVERAGE
    }

    return post("/exchange/v1/derivatives/create_order", body)

# ================= DASHBOARD API =================

@app.route("/status")
def status():

    wallet, available = get_equity()
    price = get_price()
    position = get_position()

    pnl = 0
    if position:
        pnl = float(position.get("unrealized_pnl", 0))

    return jsonify({
        "symbol": SYMBOL,
        "price": price,
        "wallet_balance": wallet,
        "available_balance": available,
        "position": position,
        "pnl": pnl
    })

@app.route("/buy")
def buy():
    return jsonify(place_order("buy"))

@app.route("/sell")
def sell():
    return jsonify(place_order("sell"))

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

# ================= RUN =================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)