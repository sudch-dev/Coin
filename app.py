import os
import time
import json
import hmac
import hashlib
import requests
from flask import Flask, jsonify, render_template

app = Flask(__name__)

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

BASE_URL = "https://api.coindcx.com"
SYMBOL = "ORDIUSDT"   # ORDI/USDT futures
LEVERAGE = 3

# ---------- SIGNED REQUEST ----------

def sign(payload):
    return hmac.new(
        API_SECRET.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()

def post(endpoint, body):
    payload = json.dumps(body, separators=(',', ':'))
    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": sign(payload),
        "Content-Type": "application/json"
    }
    r = requests.post(BASE_URL + endpoint, data=payload, headers=headers, timeout=10)
    if r.status_code == 200:
        return r.json()
    return {"error": r.text}

# ---------- MARKET PRICE ----------

def get_price():
    r = requests.get(BASE_URL + "/exchange/ticker", timeout=10)
    for x in r.json():
        if x["market"] == SYMBOL:
            return float(x["last_price"])
    return 0

# ---------- FUTURES WALLET ----------

def get_wallet():
    body = {"timestamp": int(time.time()*1000)}
    res = post("/exchange/v1/derivatives/futures/balance", body)
    return float(res.get("available_balance", 0))

# ---------- POSITION ----------

def get_position():
    body = {"timestamp": int(time.time()*1000)}
    res = post("/exchange/v1/derivatives/futures/positions", body)
    for p in res.get("data", []):
        if p["market"] == SYMBOL:
            return p
    return None

# ---------- SET LEVERAGE ----------

def set_leverage():
    body = {
        "timestamp": int(time.time()*1000),
        "market": SYMBOL,
        "leverage": LEVERAGE
    }
    return post("/exchange/v1/derivatives/change_leverage", body)

# ---------- ORDER SIZE ----------

def calc_qty(price):
    balance = get_wallet()
    risk = balance * 0.20
    position_value = risk * LEVERAGE

    if position_value < 5:
        return 0

    qty = position_value / price
    return round(qty, 2)

# ---------- PLACE ORDER ----------

def place_order(side):
    price = get_price()
    qty = calc_qty(price)

    if qty <= 0:
        return {"error": "Balance too small"}

    if get_position():
        return {"error": "Position already open"}

    body = {
        "timestamp": int(time.time()*1000),
        "market": SYMBOL,
        "side": side,
        "order_type": "market_order",
        "total_quantity": str(qty)
    }

    return post("/exchange/v1/derivatives/futures/orders/create", body)

# ---------- CLOSE POSITION ----------

def close_position():
    pos = get_position()
    if not pos:
        return {"error": "No position"}

    side = "sell" if pos["side"] == "buy" else "buy"

    body = {
        "timestamp": int(time.time()*1000),
        "market": SYMBOL,
        "side": side,
        "order_type": "market_order",
        "total_quantity": str(abs(float(pos["size"])))
    }

    return post("/exchange/v1/derivatives/futures/orders/create", body)

# ---------- ROUTES ----------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/status")
def status():
    return jsonify({
        "price": get_price(),
        "wallet": get_wallet(),
        "position": get_position()
    })

@app.route("/long", methods=["POST"])
def long_trade():
    return jsonify(place_order("buy"))

@app.route("/short", methods=["POST"])
def short_trade():
    return jsonify(place_order("sell"))

@app.route("/close", methods=["POST"])
def close_trade():
    return jsonify(close_position())

# ---------- MAIN ----------

if __name__ == "__main__":
    set_leverage()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)