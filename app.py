
import os
import json
import time
import hmac
import hashlib
from flask import Flask, jsonify, render_template, request
import requests

app = Flask(__name__)

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
BASE_URL = "https://api.coindcx.com"

def get_signature(payload):
    return hmac.new(bytes(API_SECRET, 'utf-8'), msg=bytes(payload, 'utf-8'), digestmod=hashlib.sha256).hexdigest()

def get_balances():
    try:
        timestamp = int(time.time() * 1000)
        payload = json.dumps({
            "timestamp": timestamp
        })

        headers = {
            "X-AUTH-APIKEY": API_KEY,
            "X-AUTH-SIGNATURE": get_signature(payload),
            "Content-Type": "application/json"
        }

        response = requests.post(f"{BASE_URL}/exchange/v1/users/balances", data=payload, headers=headers)
        data = response.json()
        balances = {}

        for item in data:
            currency = item.get("currency")
            available = item.get("available_balance")
            if currency and available:
                balances[currency] = float(available)

        return balances
    except Exception as e:
        print("❌ Balance Fetch Error:", e)
        return {}

def get_price(symbol):
    try:
        res = requests.get(f"{BASE_URL}/exchange/ticker", params={"market": symbol})
        price = float(res.json().get("last_price", 0))
        return price
    except Exception as e:
        print("❌ Price Fetch Error:", e)
        return 0

def update_dashboard_status():
    balances = get_balances()
    btc_price = get_price("BTCINR")
    eth_price = get_price("ETHINR")

    data = [
        {
            "coin": "BTCINR",
            "price": btc_price,
            "inr_balance": balances.get("INR", 0.0),
            "coin_balance": balances.get("BTC", 0.0),
            "decision": "",
            "last_action": ""
        },
        {
            "coin": "ETHINR",
            "price": eth_price,
            "inr_balance": balances.get("INR", 0.0),
            "coin_balance": balances.get("ETH", 0.0),
            "decision": "",
            "last_action": ""
        }
    ]

    with open("trade_status.json", "w") as f:
        json.dump(data, f, indent=2)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/api/fetch-data', methods=['POST'])
def fetch_data():
    update_dashboard_status()
    return jsonify({"message": "Data fetched and updated!"})

@app.route('/api/trade-status')
def get_trade_status():
    try:
        with open("trade_status.json", "r") as f:
            data = json.load(f)
            return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # fallback to 10000 locally
    app.run(host="0.0.0.0", port=port)
    
