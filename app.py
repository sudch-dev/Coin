import os
import time
import json
import hmac
import hashlib
import requests
from flask import Flask, jsonify, render_template

app = Flask(__name__)

API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET", "")

BASE = "https://api.coindcx.com"
# Futures symbols require the exchange prefix (e.g., 'B-' for Binance)
SYMBOL = "B-ORDI_USDT" 

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
    except: return {}

# ========= DATA FETCHING =========

@app.route("/status")
def status():
    # 1. Get Live Price
    price = 0
    try:
        r = requests.get(f"{BASE}/exchange/ticker", timeout=5)
        for x in r.json():
            if x.get("market") == SYMBOL:
                price = float(x.get("last_price", 0))
    except: pass

    # 2. Get Futures Wallet Balance
    wallet = 0
    balance_res = signed_post("/exchange/v1/derivatives/futures/balances", {})
    # CoinDCX Futures balances often return a list or a 'data' object
    balances = balance_res if isinstance(balance_res, list) else balance_res.get("data", [])
    for b in balances:
        if b.get("asset") == "USDT": 
            wallet = float(b.get("balance", 0))

    # 3. Get Active Position
    side, size, entry, pnl, roe = "NONE", 0, 0, 0, 0
    pos_res = signed_post("/exchange/v1/derivatives/futures/positions", {})
    positions = pos_res if isinstance(pos_res, list) else pos_res.get("data", [])
    
    for p in positions:
        if p.get("symbol") == SYMBOL:
            size = abs(float(p.get("quantity", 0)))
            entry = float(p.get("entry_price", 0))
            if float(p.get("quantity", 0)) > 0: side = "LONG"
            elif float(p.get("quantity", 0)) < 0: side = "SHORT"
            
            # Simple PnL Calculation
            if entry > 0:
                pnl = (price - entry) * size if side == "LONG" else (entry - price) * size
                roe = (pnl / (entry * size / 10)) * 100 # Approx 10x leverage ROE

    return jsonify({
        "equity": round(wallet, 2),
        "mark": price,
        "side": side,
        "size": size,
        "entry": entry,
        "pnl": round(pnl, 4),
        "roe": round(roe, 2)
    })

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
