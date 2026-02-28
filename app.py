import os
import time
import json
import hmac
import hashlib
import requests
from flask import Flask, jsonify, render_template

app = Flask(__name__)

# Fetch credentials from Render Environment Variables
API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET", "")

BASE = "https://api.coindcx.com"
SYMBOL = "B-ORDI_USDT" # Correct CoinDCX Futures format

def sign(payload_str):
    return hmac.new(API_SECRET.encode(), payload_str.encode(), hashlib.sha256).hexdigest()

def signed_post(endpoint, body):
    body["timestamp"] = int(time.time() * 1000)
    payload = json.dumps(body, separators=(',', ':'))
    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": sign(payload),
        "Content-Type": "application/json"
    }
    try:
        r = requests.post(BASE + endpoint, headers=headers, data=payload, timeout=10)
        return r.json()
    except:
        return {}

@app.route("/")
def index():
    # Flask looks into the /templates folder automatically
    return render_template("index.html")

@app.route("/status")
def status():
    # Fetching live ticker price
    price = 0
    try:
        r = requests.get(f"{BASE}/exchange/ticker", timeout=5)
        for x in r.json():
            if x.get("market") == SYMBOL:
                price = float(x.get("last_price", 0))
    except: pass

    # Fetching Balance
    wallet = 0
    res = signed_post("/exchange/v1/derivatives/futures/balances", {})
    if isinstance(res, list):
        for x in res:
            if x.get("asset") == "USDT": wallet = float(x.get("balance", 0))

    return jsonify({
        "equity": wallet,
        "mark": price,
        "side": "NONE",
        "size": 0,
        "entry": 0,
        "pnl": 0,
        "roe": 0
    })

if __name__ == "__main__":
    # CRITICAL FOR RENDER: Use the port assigned by the platform
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
