import os
import time
import json
import hmac
import hashlib
import threading
from datetime import datetime, timedelta
import requests
from flask import Flask, render_template, jsonify

app = Flask(__name__)

API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET").encode()
BASE_URL = "https://api.coindcx.com"
PUBLIC_URL = "https://public.coindcx.com"
PAIRS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "SHIBUSDT", "SOLUSDT",
    "DOGEUSDT", "ADAUSDT", "MATICUSDT", "BNBUSDT", "LTCUSDT"
]

scan_interval = 30  # seconds
trade_log = []
running = False
status = {"msg": "Idle"}

def hmac_signature(payload):
    return hmac.new(API_SECRET, payload.encode(), hashlib.sha256).hexdigest()

def get_balance():
    payload = json.dumps({"timestamp": int(time.time() * 1000)})
    sig = hmac_signature(payload)
    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": sig,
        "Content-Type": "application/json"
    }
    try:
        r = requests.post(f"{BASE_URL}/exchange/v1/users/balances", headers=headers, data=payload, timeout=10)
        if r.ok: return r.json()
    except Exception: pass
    return []

def fetch_candles(symbol, interval="5m", limit=40):
    url = f"{PUBLIC_URL}/market_data/candles?pair={symbol}&interval={interval}&limit={limit}"
    try:
        r = requests.get(url, timeout=10)
        if r.ok and r.json(): return r.json()
    except Exception: pass
    return []

def ema(vals, n):
    if len(vals) < n: return []
    alpha = 2 / (n + 1)
    result = []
    ema_val = sum(vals[:n]) / n
    result.append(ema_val)
    for price in vals[n:]:
        ema_val = (price - ema_val) * alpha + ema_val
        result.append(ema_val)
    return result

def simple_ema_signal(candles):
    closes = [float(c[4]) for c in candles]
    ema5 = ema(closes, 5)
    ema10 = ema(closes, 10)
    if not ema5 or not ema10 or len(ema5) < 2 or len(ema10) < 2:
        return None
    i = len(closes) - 1
    # Bullish crossover
    if ema5[i-1] < ema10[i-1] and ema5[i] > ema10[i]:
        return {"side": "BUY", "entry": closes[-1], "idx": i}
    # Bearish crossover
    elif ema5[i-1] > ema10[i-1] and ema5[i] < ema10[i]:
        return {"side": "SELL", "entry": closes[-1], "idx": i}
    return None

def place_order(symbol, side, qty, entry, tp, sl):
    payload = {
        "market": symbol,
        "side": "buy" if side == "BUY" else "sell",
        "order_type": "market_order",
        "total_quantity": str(qty),
        "timestamp": int(time.time() * 1000)
    }
    body = json.dumps(payload)
    sig = hmac_signature(body)
    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": sig,
        "Content-Type": "application/json"
    }
    try:
        r = requests.post(f"{BASE_URL}/exchange/v1/orders/create", headers=headers, data=body, timeout=10)
        result = r.json()
        return result
    except Exception as e:
        return {"error": str(e)}

def scan_loop():
    global running, trade_log, status
    while running:
        status["msg"] = "Scanning pairs"
        balances = get_balance()
        usdt = 0
        for b in balances:
            if b['currency'] == 'USDT': usdt = float(b['balance'])
        for symbol in PAIRS:
            if not running: break
            candles = fetch_candles(symbol, interval="5m", limit=40)
            if not candles or len(candles) < 12: continue
            sig = simple_ema_signal(candles)
            if sig and usdt > 5:
                side = sig["side"]
                entry = sig["entry"]
                qty = round((0.3 * usdt) / entry, 5)
                tp = round(entry * 1.002, 5)  # Target 0.2% above entry
                sl = round(entry * 0.99, 5)   # Stop loss 1% below entry
                result = place_order(symbol, side, qty, entry, tp, sl)
                trade_log.append({
                    "ts": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "entry": entry,
                    "tp": tp,
                    "sl": sl,
                    "result": result
                })
                status["msg"] = f"Traded {symbol} {side} {qty} at {entry}"
        status["msg"] = "Idle"
        for _ in range(int(scan_interval)):
            if not running: break
            time.sleep(1)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start", methods=["POST"])
def start():
    global running
    if not running:
        running = True
        thread = threading.Thread(target=scan_loop)
        thread.daemon = True
        thread.start()
    return jsonify({"status": "started"})

@app.route("/stop", methods=["POST"])
def stop():
    global running
    running = False
    return jsonify({"status": "stopped"})

@app.route("/status")
def get_status():
    balances = get_balance()
    usdt_bal = 0
    for b in balances:
        if b['currency'] == 'USDT': usdt_bal = b['balance']
    cutoff = datetime.now() - timedelta(hours=1)
    pnl = 0
    for trade in trade_log[-30:]:
        if datetime.strptime(trade["ts"], '%Y-%m-%d %H:%M:%S') > cutoff:
            if "result" in trade and "order_id" in trade["result"]:
                side = trade["side"]
                entry = trade["entry"]
                qty = trade["qty"]
                if side == "BUY":
                    pnl += (trade["tp"] - entry) * qty
                elif side == "SELL":
                    pnl += (entry - trade["tp"]) * qty
    return jsonify({
        "status": status["msg"],
        "usdt": usdt_bal,
        "trades": trade_log[-10:][::-1],
        "pnl": round(pnl, 2)
    })

@app.route("/ping")
def ping():
    return "pong"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
