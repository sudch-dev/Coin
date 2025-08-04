

from flask import Flask, request, jsonify, render_template
import os, time, requests
from datetime import datetime
from pathlib import Path

app = Flask(__name__)
COINS = ["BTCINR", "ETHINR"]
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
TOKEN_URL = "https://api.coindcx.com/auth/authorize"

def generate_token():
    payload = {
        "api_key": API_KEY,
        "api_secret": API_SECRET
    }
    response = requests.post(TOKEN_URL, json=payload)
    if response.status_code == 200:
        token = response.json().get("access_token")
        os.environ["API_TOKEN"] = token
        return token
    return None

def get_headers():
    return {
        "Authorization": f"Bearer {os.getenv('API_TOKEN')}",
        "Content-Type": "application/json"
    }

def get_price(coin):
    url = f"https://public.coindcx.com/market_data/price_by_symbol?symbol={coin}"
    response = requests.get(url)
    if response.status_code == 200:
        return float(response.json().get("price", 0))
    return 0

def get_balances():
    url = "https://api.coindcx.com/exchange/v1/users/balances"
    response = requests.post(url, headers=get_headers())
    if response.status_code == 200:
        data = response.json()
        return { item['currency']: float(item['balance']) for item in data }
    return {}

@app.route("/")
def index():
    return render_template("index.html", coins=COINS)

@app.route("/api/init", methods=["POST"])
def init_token():
    token = generate_token()
    return jsonify({"token": token} if token else {"error": "Token generation failed"})

@app.route("/api/fetch", methods=["GET"])
def fetch_data():
    coin = request.args.get("coin", "BTCINR")
    price = get_price(coin)
    balances = get_balances()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    trend = "UP" if price % 2 == 0 else "DOWN"  # Dummy trend logic
    return jsonify({
        "coin": coin,
        "price": price,
        "timestamp": now,
        "trend": trend,
        "inr": balances.get("INR", 0),
        "coin_balance": balances.get(coin.replace("INR", ""), 0)
    })

@app.route("/api/buy-strategy", methods=["POST"])
def buy_strategy():
    data = request.json
    return jsonify({"status": "Buy order logic placeholder", "data": data})

@app.route("/api/sell-strategy", methods=["POST"])
def sell_strategy():
    data = request.json
    return jsonify({"status": "Sell order logic placeholder", "data": data})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

