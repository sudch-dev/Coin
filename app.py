import os
import time
import json
import hmac
import hashlib
import requests
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Credentials from Render Environment Variables
API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET", "")

BASE = "https://api.coindcx.com"
SYMBOL = "ORDI_USDT" 

# ========= AUTHENTICATION =========

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
    except Exception as e:
        return {"error": str(e)}

# ========= PRECISION LOGIC =========

def get_market_precision(pair):
    """Fetches accepted decimal places for price and quantity."""
    try:
        r = requests.get(f"{BASE}/exchange/v1/market_details")
        for m in r.json():
            if m.get('symbol') == pair:
                return {
                    "price": m.get('base_currency_precision'), # Decimals for price
                    "qty": m.get('target_currency_precision'),  # Decimals for quantity
                    "min_qty": float(m.get('min_quantity', 0))
                }
    except: pass
    return {"price": 2, "qty": 2, "min_qty": 0.1} # Safe defaults

# ========= TRADE EXECUTION =========

@app.route("/trade", methods=["POST"])
def execute_trade():
    """Handles Market Buy/Sell with precision rounding."""
    data = request.json
    side = data.get("side") # "buy" or "sell"
    qty = float(data.get("quantity", 0))
    
    # Get live precision requirements
    prec = get_market_precision(SYMBOL)
    
    # Round quantity to the allowed target_currency_precision
    final_qty = round(qty, prec['qty'])
    
    if final_qty < prec['min_qty']:
        return jsonify({"status": "error", "message": f"Qty too low. Min: {prec['min_qty']}"})

    # Note: Market orders do not require a price_per_unit
    order_body = {
        "market": SYMBOL,
        "total_quantity": final_qty,
        "side": side,
        "order_type": "market_order"
    }
    
    res = signed_post("/exchange/v1/orders/create", order_body)
    return jsonify(res)

# ========= MONITORING =========

@app.route("/status")
def status():
    # Fetch price and balances
    price = 0
    try:
        ticker = requests.get(f"{BASE}/exchange/ticker").json()
        for x in ticker:
            if x.get("market") == SYMBOL: price = float(x.get("last_price", 0))
    except: pass

    wallet = 0.0
    res = signed_post("/exchange/v1/users/balances", {})
    if isinstance(res, list):
        for item in res:
            if item.get("currency") == "USDT": wallet = float(item.get("balance", 0))

    return jsonify({
        "equity": round(wallet, 2),
        "mark": price,
        "status": "SPOT ACTIVE"
    })

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
