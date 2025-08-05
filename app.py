import os
import time
import json
import hmac
import hashlib
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

PORT = 10000
API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET").encode()
BASE_URL = "https://api.coindcx.com"
CANDLE_URL = "https://public.coindcx.com/market_data/candles"

COINS = [
    "BTCINR", "ETHINR", "XRPINR", "DOGEINR", "SHIBINR",
    "TRXINR", "LTCINR", "BCHINR", "ADAINR", "MATICINR"
]

def get_balances():
    payload = {"timestamp": int(time.time() * 1000)}
    body = json.dumps(payload)
    signature = hmac.new(API_SECRET, body.encode(), hashlib.sha256).hexdigest()
    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": signature,
        "Content-Type": "application/json"
    }
    try:
        r = requests.post(f"{BASE_URL}/exchange/v1/users/balances", headers=headers, data=body)
        data = r.json()
        return {item['currency']: float(item['balance']) for item in data}
    except:
        return {}

def fetch_candles(symbol, interval="15m", limit=40):
    url = f"{CANDLE_URL}?pair={symbol}&interval={interval}&limit={limit}"
    try:
        r = requests.get(url)
        return r.json()
    except:
        return []

def detect_ob_smc(candles):
    # SMC-style OB: Find last big bearish or bullish candle and confirm break/retest
    bullish_ob = None
    bearish_ob = None
    reason = "No OB found"
    trend = "SIDEWAYS"
    n = len(candles)
    # --- Find last bullish OB (down candle before big up move) ---
    for i in range(n-3, 3, -1):
        o = float(candles[i][1])
        h = float(candles[i][2])
        l = float(candles[i][3])
        c = float(candles[i][4])
        v = float(candles[i][5])
        next_c = float(candles[i+1][4])
        if c < o and next_c > o and (o-c)/o > 0.004:
            # Down candle, next closes above open = OB
            bullish_ob = {"idx": i, "open": o, "low": l, "high": h, "vol": v}
            break
    # --- Find last bearish OB (up candle before big drop) ---
    for i in range(n-3, 3, -1):
        o = float(candles[i][1])
        h = float(candles[i][2])
        l = float(candles[i][3])
        c = float(candles[i][4])
        v = float(candles[i][5])
        next_c = float(candles[i+1][4])
        if c > o and next_c < o and (c-o)/o > 0.004:
            bearish_ob = {"idx": i, "open": o, "high": h, "low": l, "vol": v}
            break
    last_close = float(candles[-1][4])
    last_vol = float(candles[-1][5])
    # --- Confirm mitigation and volume ---
    if bullish_ob:
        # If price touches OB zone and bounces, with above avg vol, call UP
        for i in range(bullish_ob["idx"], n):
            low = float(candles[i][3])
            close = float(candles[i][4])
            vol = float(candles[i][5])
            if bullish_ob["low"] <= low <= bullish_ob["open"]:
                if close > bullish_ob["open"] and vol > bullish_ob["vol"]:
                    trend = "UP"
                    reason = "Bullish OB zone mitigated & volume up"
                    break
    if trend == "SIDEWAYS" and bearish_ob:
        for i in range(bearish_ob["idx"], n):
            high = float(candles[i][2])
            close = float(candles[i][4])
            vol = float(candles[i][5])
            if bearish_ob["open"] <= high <= bearish_ob["high"]:
                if close < bearish_ob["open"] and vol > bearish_ob["vol"]:
                    trend = "DOWN"
                    reason = "Bearish OB zone mitigated & volume up"
                    break
    return trend, reason

@app.route("/")
def index():
    return render_template("index.html", coins=COINS)

@app.route("/api/ob_predict", methods=["POST"])
def ob_predict():
    symbol = request.json.get("symbol", "BTCINR")
    candles = fetch_candles(symbol)
    if not candles or len(candles) < 10:
        return jsonify({"trend": "N/A", "reason": "No candle data"})
    trend, reason = detect_ob_smc(candles)
    last = candles[-1]
    return jsonify({
        "trend": trend,
        "reason": reason,
        "price": float(last[4]),
        "time": time.strftime('%Y-%m-%d %H:%M', time.localtime(last[0]/1000))
    })

@app.route("/api/balance", methods=["POST"])
def api_balance():
    balances = get_balances()
    return jsonify(balances)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
