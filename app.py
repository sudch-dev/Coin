import os
import time
import json
import hmac
import hashlib
import requests
from flask import Flask, jsonify, render_template

app = Flask(__name__)

# Credentials from Render Environment Variables
API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET", "")

BASE = "https://api.coindcx.com"
# FIXED: CoinDCX Futures uses 'B-' prefix and underscore for derivatives
SYMBOL = "B-ORDI_USDT" 

# ========= AUTHENTICATION =========

def sign(payload_str):
    """Generates HMAC SHA256 signature for private endpoints."""
    return hmac.new(
        API_SECRET.encode('utf-8'), 
        payload_str.encode('utf-8'), 
        hashlib.sha256
    ).hexdigest()

def signed_post(endpoint, body):
    """Handles authenticated POST requests to CoinDCX."""
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
        print(f"API Connection Error: {e}")
        return {}

# ========= DATA EXTRACTION =========

@app.route("/status")
def status():
    # 1. Fetch Live Mark Price
    price = 0
    try:
        r = requests.get(f"{BASE}/exchange/ticker", timeout=5)
        for x in r.json():
            if x.get("market") == SYMBOL:
                price = float(x.get("last_price", 0))
    except: pass

    # 2. Fetch Futures Wallet Balance (Tether/USDT)
    wallet = 0.0
    # Targeted futures balance endpoint
    res = signed_post("/exchange/v1/derivatives/futures/balances", {})
    # CoinDCX often wraps response in a 'data' key or returns a list
    balances = res if isinstance(res, list) else res.get("data", [])
    
    for b in balances:
        # Check for both "asset" and "currency" keys and both USDT/TETHER names
        asset = str(b.get("asset") or b.get("currency") or "").upper()
        if asset in ["USDT", "TETHER"]:
            wallet = float(b.get("balance") or b.get("available_balance", 0))
            break

    # 3. Fetch Active Position Data
    side, size, entry, pnl, roe = "NONE", 0, 0, 0, 0
    pos_res = signed_post("/exchange/v1/derivatives/futures/positions", {})
    positions = pos_res if isinstance(pos_res, list) else pos_res.get("data", [])
    
    for p in positions:
        if p.get("symbol") == SYMBOL:
            qty = float(p.get("quantity", 0))
            if qty != 0:
                size = abs(qty)
                entry = float(p.get("entry_price", 0))
                side = "LONG" if qty > 0 else "SHORT"
                # PnL Calculation based on current mark price
                pnl = (price - entry) * size if side == "LONG" else (entry - price) * size
                # Estimating ROE (Assuming 10x leverage for calculation)
                roe = (pnl / (entry * size / 10)) * 100 if entry > 0 else 0

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
    # Ensure Render's dynamic port is used
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
