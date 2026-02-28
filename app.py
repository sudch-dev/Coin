import os
import time
import hmac
import hashlib
import requests
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET")

BASE_URL = "https://api.coindcx.com"
SYMBOL = "ORDIUSDT"

# --------------------------------------------------
# Helper: Generate Signature
# --------------------------------------------------

def generate_signature(payload):
    return hmac.new(
        bytes(API_SECRET, 'utf-8'),
        bytes(payload, 'utf-8'),
        hashlib.sha256
    ).hexdigest()

# --------------------------------------------------
# Get Current Price
# --------------------------------------------------

def get_price():
    try:
        url = f"{BASE_URL}/exchange/ticker"
        res = requests.get(url).json()
        for coin in res:
            if coin["market"] == SYMBOL:
                return float(coin["last_price"])
        return 0
    except Exception as e:
        print("Price Error:", e)
        return 0

# --------------------------------------------------
# Get Futures Wallet Balance (USDT)
# --------------------------------------------------

def get_wallet():
    try:
        timestamp = int(time.time() * 1000)

        body = {
            "timestamp": timestamp
        }

        payload = str(body)
        signature = generate_signature(payload)

        headers = {
            "X-AUTH-APIKEY": API_KEY,
            "X-AUTH-SIGNATURE": signature,
            "Content-Type": "application/json"
        }

        response = requests.post(
            f"{BASE_URL}/exchange/v1/derivatives/futures/balance",
            data=payload,
            headers=headers
        )

        res = response.json()

        # CoinDCX returns LIST
        if isinstance(res, list):
            for asset in res:
                if asset.get("currency") == "USDT":
                    return float(asset.get("balance", 0))

        return 0

    except Exception as e:
        print("Wallet Error:", e)
        return 0

# --------------------------------------------------
# Get Open Position
# --------------------------------------------------

def get_position():
    try:
        timestamp = int(time.time() * 1000)

        body = {
            "timestamp": timestamp
        }

        payload = str(body)
        signature = generate_signature(payload)

        headers = {
            "X-AUTH-APIKEY": API_KEY,
            "X-AUTH-SIGNATURE": signature,
            "Content-Type": "application/json"
        }

        response = requests.post(
            f"{BASE_URL}/exchange/v1/derivatives/futures/positions",
            data=payload,
            headers=headers
        )

        res = response.json()

        # FIXED: handle list response
        if isinstance(res, list):
            for p in res:
                if p.get("market") == SYMBOL:
                    size = float(p.get("size", 0))
                    if size > 0:
                        return "LONG"
                    elif size < 0:
                        return "SHORT"

        return "NONE"

    except Exception as e:
        print("Position Error:", e)
        return "ERROR"

# --------------------------------------------------
# Place Order
# --------------------------------------------------

def place_order(side):
    try:
        price = get_price()
        wallet = get_wallet()

        if wallet <= 0:
            return {"error": "Insufficient balance"}

        quantity = round((wallet * 0.95) / price, 3)

        timestamp = int(time.time() * 1000)

        body = {
            "timestamp": timestamp,
            "market": SYMBOL,
            "side": side,
            "order_type": "market_order",
            "quantity": quantity
        }

        payload = str(body)
        signature = generate_signature(payload)

        headers = {
            "X-AUTH-APIKEY": API_KEY,
            "X-AUTH-SIGNATURE": signature,
            "Content-Type": "application/json"
        }

        response = requests.post(
            f"{BASE_URL}/exchange/v1/derivatives/futures/orders",
            data=payload,
            headers=headers
        )

        return response.json()

    except Exception as e:
        print("Order Error:", e)
        return {"error": str(e)}

# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/status")
def status():
    try:
        return jsonify({
            "price": get_price(),
            "wallet": get_wallet(),
            "position": get_position()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/long", methods=["POST"])
def long():
    return jsonify(place_order("buy"))

@app.route("/short", methods=["POST"])
def short():
    return jsonify(place_order("sell"))

@app.route("/close", methods=["POST"])
def close():
    pos = get_position()
    if pos == "LONG":
        return jsonify(place_order("sell"))
    elif pos == "SHORT":
        return jsonify(place_order("buy"))
    return jsonify({"message": "No open position"})

# --------------------------------------------------
# Render Port Binding
# --------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))