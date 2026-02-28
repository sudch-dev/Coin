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
# MANDATORY: CoinDCX Futures uses 'B-' prefix for Binance-sourced markets
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

# ========= DEEP SCAN BALANCE =========

def get_futures_balance():
    """Scans all balance fields for USDT-M or Tether values."""
    res = signed_post("/exchange/v1/derivatives/futures/balances", {})
    
    # CoinDCX can return data as a direct list or inside a 'data' key
    balances = res if isinstance(res, list) else res.get("data", [])
    
    for item in balances:
        # Check all possible asset/currency labels
        asset = str(item.get("asset") or item.get("currency") or item.get("instrument_name", "")).upper()
        
        # Match USDT, TETHER, or USDT-M
        if any(name in asset for name in ["USDT", "TETHER"]):
            # Pull balance from any available numeric field
            val = float(item.get("balance") or item.get("available_balance") or item.get("quantity", 0))
            if val > 0: return val
            
    return 0.0

# ========= MAIN MONITOR =========

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

    # 2. Fetch Deep Scanned Wallet Balance
    wallet = get_futures_balance()

    # 3. Fetch Position Data
    side, size, entry, pnl, roe = "NONE", 0, 0, 0, 0
    pos_res = signed_post("/exchange/v1/derivatives/futures/positions", {})
    positions = pos_res if isinstance(pos_res, list) else pos_res.get("data", [])
    
    for p in positions:
        if p.get("symbol") == SYMBOL or p.get("market") == SYMBOL:
            qty = float(p.get("quantity") or p.get("size", 0))
            if qty != 0:
                size = abs(qty)
                entry = float(p.get("entry_price") or p.get("avg_price", 0))
                side = "LONG" if qty > 0 else "SHORT"
                # PnL Calculation
                pnl = (price - entry) * size if side == "LONG" else (entry - price) * size
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
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
