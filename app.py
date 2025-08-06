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
PAIRS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "SHIBUSDT", "SOLUSDT",
    "DOGEUSDT", "ADAUSDT", "MATICUSDT", "BNBUSDT", "LTCUSDT"
]

scan_interval = 30  # seconds
trade_log = []
scan_log = []
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

def fetch_candles(market, interval="5m", limit=30):
    payload = json.dumps({
        "market": market,
        "interval": interval,
        "limit": limit
    })
    sig = hmac_signature(payload)
    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": sig,
        "Content-Type": "application/json"
    }
    url = f"{BASE_URL}/exchange/v1/markets/candles"
    try:
        r = requests.post(url, headers=headers, data=payload, timeout=10)
        if r.ok and r.json(): return r.json()
    except Exception as e:
        print("Candle fetch error:", e)
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

def entry_signal(candles):
    # Simple bullish cross: EMA5 crosses above EMA10, bearish cross: EMA5 crosses below EMA10
    closes = [float(c[4]) for c in candles]  # close
    ema5 = ema(closes, 5)
    ema10 = ema(closes, 10)
    if not ema5 or not ema10: return None
    # Check for last crossover
    if len(ema10) < 2: return None
    if ema5[-2] < ema10[-2] and ema5[-1] > ema10[-1]:
        return {"side": "BUY", "entry": closes[-1]}
    elif ema5[-2] > ema10[-2] and ema5[-1] < ema10[-1]:
        return {"side": "SELL", "entry": closes[-1]}
    return None

def place_order(symbol, side, qty, entry, tp, sl):
    # Demo: Simulate order, log to trade_log only.
    trade_log.append({
        "ts": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "result": "Simulated Order"
    })
    return {"order_id": "simulated", "side": side, "qty": qty}

def scan_loop():
    global running, trade_log, scan_log, status
    while running:
        status["msg"] = "Scanning pairs"
        balances = get_balance()
        usdt = 0
        for b in balances:
            if b['currency'] == 'USDT': usdt = float(b['balance'])
        for symbol in PAIRS:
            if not running: break
            candles = fetch_candles(symbol)
            logstr = ""
            if not candles or len(candles) < 12:
                logstr = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {symbol}: No candle data"
            else:
                sig = entry_signal(candles)
                if sig and usdt > 5:
                    side = sig["side"]
                    entry = sig["entry"]
                    qty = round((0.3 * usdt) / entry, 5)
                    tp = round(entry * 1.002, 4)  # 0.2% up
                    sl = round(entry * 0.99, 4)   # 1% down
                    result = place_order(symbol, side, qty, entry, tp, sl)
                    status["msg"] = f"Traded {symbol} {side} {qty} at {entry}"
                    logstr = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {symbol}: {side} at {entry}"
                else:
                    logstr = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {symbol}: No trade"
            scan_log.append(logstr)
            scan_log[:] = scan_log[-20:]
        status["msg"] = "Idle"
        for _ in range(scan_interval):
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
            if "result" in trade:
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
        "scans": scan_log[-10:][::-1],
        "pnl": round(pnl, 2)
    })

@app.route("/ping")
def ping():
    return "pong"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
