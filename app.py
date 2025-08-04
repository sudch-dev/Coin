import os
import time
import hashlib
import hmac
import json
from flask import Flask, render_template, request, redirect, url_for
import requests

app = Flask(__name__)

BASE_URL = "https://api.coindcx.com"
API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET").encode()

def generate_signature(payload):
    return hmac.new(API_SECRET, payload.encode(), hashlib.sha256).hexdigest()

def get_wallet_balance():
    payload = json.dumps({"timestamp": int(time.time() * 1000)})
    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": generate_signature(payload),
        "Content-Type": "application/json"
    }
    try:
        res = requests.post(f"{BASE_URL}/exchange/v1/users/balances", headers=headers, data=payload)
        return res.json()
    except Exception as e:
        return []

def get_current_price(symbol):
    try:
        response = requests.get(f"https://api.coindcx.com/exchange/ticker")
        data = response.json()
        for item in data:
            if item['market'] == symbol:
                return float(item['last_price'])
    except:
        return 0.0

def predict_trend(price):
    # Simple OB dummy logic — enhance later
    return "UP" if price % 2 < 1 else "DOWN"

def place_order(market, side, price, quantity):
    payload_dict = {
        "market": market,
        "total_quantity": quantity,
        "price_per_unit": price,
        "order_type": "limit_order",
        "side": side,
        "timestamp": int(time.time() * 1000)
    }
    payload = json.dumps(payload_dict)
    signature = generate_signature(payload)

    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": signature,
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(f"{BASE_URL}/exchange/v1/orders/create", headers=headers, data=payload)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

@app.route("/", methods=["GET", "POST"])
def index():
    market = "BTCINR"
    price = get_current_price(market)
    trend = predict_trend(price)
    balances = get_wallet_balance()

    if request.method == "POST":
        action = request.form.get("action")
        qty = float(request.form.get("quantity"))
        target = float(request.form.get("target"))
        sl = float(request.form.get("sl"))
        if action == "buy":
            result = place_order(market, "buy", price, qty)
        else:
            result = place_order(market, "sell", price, qty)
        return render_template("index.html", balances=balances, price=price, trend=trend, result=result)

    return render_template("index.html", balances=balances, price=price, trend=trend, result=None)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
