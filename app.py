import os
import time
import json
import hmac
import hashlib
import threading
from datetime import datetime, timedelta
import requests
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET").encode()
BASE_URL = "https://api.coindcx.com"
CANDLE_URL = "https://api.coindcx.com/exchange/v1/marketdata/candles"
PAIRS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "SHIBUSDT", "SOLUSDT",
    "DOGEUSDT", "ADAUSDT", "MATICUSDT", "BNBUSDT", "LTCUSDT"
]
scan_interval = 30  # seconds

trade_log = []
scan_log = {symbol: [] for symbol in PAIRS}
running = False
status = {"msg": "Idle"}

def hmac_signature(payload):
    return hmac.new(API_SECRET, payload.encode(), hashlib.sha256).hexdigest()

def get_timestamp():
    return int(time.time() * 1000)

def get_balance():
    payload = json.dumps({"timestamp": get_timestamp()})
    sig = hmac_signature(payload)
    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": sig,
        "Content-Type": "application/json"
    }
    try:
        r = requests.post(f"{BASE_URL}/exchange/v1/users/balances", headers=headers, data=payload, timeout=10)
        if r.ok: return r.json()
    except Exception as e:
        print("Balance fetch failed:", e)
    return []

def fetch_candles(symbol, interval="5m", limit=40):
    ts = get_timestamp()
    params = {
        "pair": symbol,
        "interval": interval,
        "limit": limit,
        "timestamp": ts
    }
    payload = json.dumps(params)
    sig = hmac_signature(payload)
    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": sig,
        "Content-Type": "application/json"
    }
    try:
        r = requests.post(CANDLE_URL, headers=headers, data=payload, timeout=10)
        if r.ok and r.json(): return r.json()
    except Exception as e:
        print(f"Failed to fetch candles for {symbol}:", e)
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

def ema_signal(candles):
    closes = [float(c[4]) for c in candles]
    if len(closes) < 11: return None
    ema5 = ema(closes, 5)
    ema10 = ema(closes, 10)
    if len(ema5) < 2 or len(ema10) < 2: return None
    # EMA5 crosses above EMA10: Buy. EMA5 crosses below EMA10: Sell.
    cross_up = ema5[-2] < ema10[-2] and ema5[-1] > ema10[-1]
    cross_down = ema5[-2] > ema10[-2] and ema5[-1] < ema10[-1]
    if cross_up:
        return {"side": "BUY", "entry": closes[-1]}
    elif cross_down:
        return {"side": "SELL", "entry": closes[-1]}
    return None

def place_order(symbol, side, qty):
    payload = {
        "market": symbol,
        "side": "buy" if side == "BUY" else "sell",
        "order_type": "market_order",
        "total_quantity": str(qty),
        "timestamp": get_timestamp()
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
    global running, trade_log, status, scan_log
    while running:
        status["msg"] = "Scanning pairs"
        balances = get_balance()
        usdt = 0
        for b in balances:
            if b['currency'] == 'USDT': usdt = float(b['balance'])
        for symbol in PAIRS:
            if not running: break
            candles = fetch_candles(symbol)
            scan_entry = {"time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            if not candles or len(candles) < 11:
                scan_entry["msg"] = "No candles data"
                scan_log[symbol].append(scan_entry)
                continue
            sig = ema_signal(candles)
            scan_entry["msg"] = f"Price: {candles[-1][4]}, EMA5: {ema([float(c[4]) for c in candles],5)[-1]:.2f}, EMA10: {ema([float(c[4]) for c in candles],10)[-1]:.2f}"
            if sig and usdt > 5:
                side = sig["side"]
                entry = sig["entry"]
                qty = round((0.3 * usdt) / entry, 5)
                tp = round(entry * 1.002, 2)  # Target 0.2%
                sl = round(entry * 0.99, 2)   # SL 1%
                result = place_order(symbol, side, qty)
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
                scan_entry["msg"] += f" | TRADE: {side} {qty} at {entry}"
            scan_log[symbol].append(scan_entry)
            # Keep logs to last 10 scans
            if len(scan_log[symbol]) > 10:
                scan_log[symbol] = scan_log[symbol][-10:]
        status["msg"] = "Idle"
        for _ in range(scan_interval):
            if not running: break
            time.sleep(1)

@app.route("/")
def index():
    return render_template("index.html", pairs=PAIRS)

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
        "pnl": round(pnl, 2),
        "scans": {s: scan_log[s][-10:] for s in PAIRS}
    })

@app.route("/ping")
def ping():
    return "pong"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
