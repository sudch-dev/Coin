from flask import Flask, jsonify, render_template, request
import os
import requests
import threading
import json
import time
from pathlib import Path
TRADE_STATUS_FILE = "trade_status.json"

app = Flask(__name__)

TRADE_STATUS_FILE = 'trade_status.json'
COINS = ["BTCINR", "ETHINR"]

def get_headers():
    return {
        "X-AUTH-APIKEY": os.getenv("API_KEY"),
        "X-AUTH-SECRET": os.getenv("API_SECRET")
    }

def get_balance():
    url = "https://api.coindcx.com/exchange/v1/users/balances"
    response = requests.get(url, headers=get_headers())
    if response.status_code == 200:
        balances = response.json()
        return {item["currency"]: float(item["balance"]) for item in balances}
    return {}

def get_latest_price(coin_symbol):
    url = f"https://api.coindcx.com/market_data/current_price/{coin_symbol}"
    response = requests.get(url)
    if response.status_code == 200:
        return float(response.json().get("price", 0))
    return 0

def update_trade_status_file():
    balances = get_balance()
    data = []
    for coin in COINS:
        inr_balance = balances.get("INR", 0)
        coin_balance = balances.get(coin.replace("INR", ""), 0)
        price = get_latest_price(coin)
        data.append({
            "coin": coin,
            "coin_balance": coin_balance,
            "decision": "",
            "inr_balance": inr_balance,
            "last_action": "",
            "last_buy_price": 0,
            "price": price
        })
    with open(TRADE_STATUS_FILE, 'w') as f:
        json.dump(data, f, indent=4)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/fetch-data', methods=['GET'])
def fetch_data():
    try:
        print("API Key:", os.getenv("API_KEY"))  # Debug
        print("API Secret:", os.getenv("API_SECRET"))  # Debug

        update_trade_status_file()
        return jsonify({'status': 'success', 'message': 'Data fetched and updated'})
    except Exception as e:
        print("Exception:", str(e))  # Debug
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/trade-status', methods=['GET'])
def trade_status():
    if not Path(TRADE_STATUS_FILE).exists():
        return jsonify([])
    with open(TRADE_STATUS_FILE, 'r') as f:
        return jsonify(json.load(f))

@app.route('/ping')
def ping():
    return 'pong', 200

def self_ping():
    while True:
        try:
            requests.get("https://coin-4k37.onrender.com/ping")
        except Exception as e:
            print("Ping failed:", e)
        time.sleep(30)

if __name__ == '__main__':
    threading.Thread(target=self_ping, daemon=True).start()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
