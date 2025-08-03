import os
import time
import json
import hmac
import hashlib
import requests
from flask import Flask, jsonify, render_template

app = Flask(__name__)

API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET")
BASE_URL = "https://api.coindcx.com"
TRADE_FILE = "trade_status.json"

def create_signature(payload):
    return hmac.new(
        bytes(API_SECRET, 'utf-8'),
        msg=bytes(payload, 'utf-8'),
        digestmod=hashlib.sha256
    ).hexdigest()

def get_balances():
    try:
        payload = str(int(time.time() * 1000))
        signature = create_signature(payload)
        headers = {
            "X-AUTH-APIKEY": API_KEY,
            "X-AUTH-SIGNATURE": signature,
            "X-AUTH-TIMESTAMP": payload
        }
        res = requests.post(f"{BASE_URL}/exchange/v1/users/balances", headers=headers)
        data = res.json()
        balances = {}
        for item in data:
            currency = item.get("currency")
            available = item.get("available") or item.get("available_balance") or item.get("balance") or 0
            if currency:
                balances[currency] = float(available)
        return balances
    except:
        return {}

def get_price(symbol):
    try:
        res = requests.get(f"{BASE_URL}/exchange/ticker", params={"market": symbol})
        return float(res.json().get("last_price", 0))
    except:
        return 0

def place_order(side, symbol, amount):
    payload = {
        "market": symbol,
        "side": side,
        "order_type": "market_order",
        "total_quantity": None,
        "total_amount": str(amount)
    }
    payload_json = json.dumps(payload, separators=(',', ':'))
    signature = create_signature(payload_json)
    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": signature,
        "X-AUTH-TIMESTAMP": str(int(time.time() * 1000)),
        "Content-Type": "application/json"
    }
    res = requests.post(f"{BASE_URL}/exchange/v1/orders/create", headers=headers, data=payload_json)
    return res.status_code == 200

def load_status():
    if os.path.exists(TRADE_FILE):
        with open(TRADE_FILE, "r") as f:
            return json.load(f)
    return []

def save_status(data):
    with open(TRADE_FILE, "w") as f:
        json.dump(data, f, indent=2)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/trade-status")
def trade_status():
    balances = get_balances()
    inr = balances.get("INR", 0)
    status = load_status()

    updated_status = []
    for coin in ["BTCINR", "ETHINR"]:
        coin_symbol = coin.replace("INR", "")
        price = get_price(coin)
        coin_bal = balances.get(coin_symbol, 0)

        record = next((x for x in status if x["coin"] == coin), {
            "coin": coin,
            "coin_balance": 0,
            "inr_balance": 0,
            "price": 0,
            "last_action": "",
            "last_buy_price": 0,
            "decision": ""
        })

        action = ""
        if coin_bal == 0 and inr > 100:
            buy_amt = round(0.3 * inr, 2)
            if place_order("buy", coin, buy_amt):
                action = "BUY"
                record["last_buy_price"] = price
        elif coin_bal > 0 and record.get("last_buy_price", 0) > 0:
            gain = (price - record["last_buy_price"]) / record["last_buy_price"]
            if gain >= 0.02:
                sell_amt = round(price * coin_bal, 2)
                if place_order("sell", coin, sell_amt):
                    action = "SELL"
                    record["last_buy_price"] = 0

        record.update({
            "coin_balance": coin_bal,
            "inr_balance": inr,
            "price": price,
            "last_action": action or record.get("last_action", ""),
            "decision": action
        })
        updated_status.append(record)

    save_status(updated_status)
    return jsonify(updated_status)

@app.route("/ping")
def ping():
    return "pong", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)