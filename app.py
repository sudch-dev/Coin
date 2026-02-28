import os
import time
import json
import hmac
import hashlib
import requests
from flask import Flask, jsonify, render_template

app = Flask(__name__)

# Use environment variables for security
API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET", "")

BASE = "https://api.coindcx.com"
# Corrected symbol format for CoinDCX Futures
SYMBOL = "B-ORDI_USDT" 

# ========= SIGNATURE =========

def sign(payload_str):
    """Generates HMAC SHA256 signature using the secret key."""
    return hmac.new(
        API_SECRET.encode('utf-8'), 
        payload_str.encode('utf-8'), 
        hashlib.sha256
    ).hexdigest()

def signed_post(endpoint, body):
    """Handles signed POST requests to CoinDCX."""
    # Ensure timestamp is included in every private request
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
    except Exception as e:
        print(f"Request Error: {e}")
        return {}

# ========= DATA FETCHING =========

def get_price():
    """Fetches the latest ticker price for the symbol."""
    try:
        # Ticker endpoint often provides all markets
        r = requests.get(f"{BASE}/exchange/ticker", timeout=10)
        data = r.json()
        for x in data:
            if x.get("market") == SYMBOL:
                return float(x.get("last_price", 0))
    except:
        pass
    return 0

def get_wallet():
    """Retrieves Futures wallet balance in USDT/INR."""
    res = signed_post("/exchange/v1/derivatives/futures/balances", {})
    # Note: Extract logic depends on the specific nesting in the API response
    if isinstance(res, list):
        for x in res:
            if x.get("asset") == "USDT":
                return float(x.get("balance", 0))
    return 0.0

@app.route("/status")
def status():
    price = get_price()
    wallet = get_wallet()
    # Simplified PnL logic for status check
    return jsonify({
        "equity": wallet,
        "mark_price": price,
        "symbol": SYMBOL,
        "status": "System OK"
    })

if __name__ == "__main__":
    # Ensure you set your API_KEY and API_SECRET in your terminal/environment first
    app.run(host="0.0.0.0", port=10000)
