import os
import time
import threading
import hmac
import hashlib
import requests
import json
from datetime import datetime
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET").encode()
BASE_URL = "https://api.coindcx.com"
PAIRS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "SHIBUSDT", "SOLUSDT",
    "DOGEUSDT", "ADAUSDT", "MATICUSDT", "BNBUSDT", "LTCUSDT"
]

tick_logs = {pair: [] for pair in PAIRS}         # list of (timestamp, price)
candle_logs = {pair: [] for pair in PAIRS}       # list of dicts: {open,high,low,close,volume,start}
scan_log = []                                   # list of strings (log lines)
trade_log = []                                  # list of dicts (trade info)
running = False
status = {"msg": "Idle", "last": ""}

def hmac_signature(payload):
    return hmac.new(API_SECRET, payload.encode(), hashlib.sha256).hexdigest()

def get_wallet_balance():
    payload = json.dumps({"timestamp": int(time.time() * 1000)})
    sig = hmac_signature(payload)
    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": sig,
        "Content-Type": "application/json"
    }
    try:
        r = requests.post(f"{BASE_URL}/exchange/v1/users/balances", headers=headers, data=payload, timeout=10)
        if r.ok:
            for b in r.json():
                if b['currency'] == "USDT":
                    return float(b['balance'])
    except Exception as e:
        pass
    return 0.0

def fetch_all_prices():
    try:
        r = requests.get(f"{BASE_URL}/exchange/ticker", timeout=10)
        if r.ok:
            data = r.json()
            price_map = {}
            now = int(time.time())
            for item in data:
                m = item["market"]
                if m in PAIRS:
                    price_map[m] = {"price": float(item["last_price"]), "ts": now}
            return price_map
    except Exception as e:
        pass
    return {}

def aggregate_candles(pair):
    """Aggregate 5m candles from ticks, keep max 50 candles"""
    ticks = tick_logs[pair]
    if not ticks:
        return
    # Group ticks by 5-min window
    candles = []
    window = 5 * 60
    ticks_sorted = sorted(ticks, key=lambda x: x[0])
    candle = None
    last_window = None
    for ts, price in ticks_sorted:
        wstart = ts - (ts % window)
        if last_window != wstart:
            if candle:
                candles.append(candle)
            candle = {
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1,
                "start": wstart
            }
            last_window = wstart
        else:
            candle["high"] = max(candle["high"], price)
            candle["low"] = min(candle["low"], price)
            candle["close"] = price
            candle["volume"] += 1
    if candle:
        candles.append(candle)
    candle_logs[pair] = candles[-50:]

def ema(vals, n):
    if len(vals) < n:
        return []
    alpha = 2 / (n + 1)
    result = []
    ema_val = sum(vals[:n]) / n
    result.append(ema_val)
    for price in vals[n:]:
        ema_val = (price - ema_val) * alpha + ema_val
        result.append(ema_val)
    return result

def ob_ema_signal(pair):
    candles = candle_logs[pair]
    if len(candles) < 12:
        return None
    closes = [c["close"] for c in candles]
    ema5 = ema(closes, 5)
    ema10 = ema(closes, 10)
    if not ema5 or not ema10 or len(ema5) < 2 or len(ema10) < 2:
        return None
    # Simple EMA cross and fake OB detection for demo
    # You can refine this block!
    # Buy: last EMA5 crosses above EMA10, last candle is green
    # Sell: last EMA5 crosses below EMA10, last candle is red
    if ema5[-2] < ema10[-2] and ema5[-1] > ema10[-1] and candles[-1]["close"] > candles[-1]["open"]:
        return {"side": "BUY", "entry": candles[-1]["close"], "msg": "EMA5 crossed above EMA10 (Bullish)"}
    if ema5[-2] > ema10[-2] and ema5[-1] < ema10[-1] and candles[-1]["close"] < candles[-1]["open"]:
        return {"side": "SELL", "entry": candles[-1]["close"], "msg": "EMA5 crossed below EMA10 (Bearish)"}
    return None

def place_order(pair, side, qty):
    # Example: you may want to wire this to /exchange/v1/orders/create endpoint
    # Omitted here to prevent accidental real trades in demo!
    return {"msg": f"Order {side} {qty} {pair} (demo mode)"}

def scan_loop():
    global running, scan_log, status
    usdt_bal = get_wallet_balance()
    scan_log.clear()
    last_candle_ts = {p: 0 for p in PAIRS}
    while running:
        prices = fetch_all_prices()
        now = int(time.time())
        log_lines = []
        for pair in PAIRS:
            if pair in prices:
                price = prices[pair]["price"]
                tick_logs[pair].append((now, price))
                # Keep max 1000 ticks per pair
                if len(tick_logs[pair]) > 1000:
                    tick_logs[pair] = tick_logs[pair][-1000:]
                log_lines.append(f"{datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S')} | {pair} | price: {price}")
                # Aggregate candles every 5 min
                aggregate_candles(pair)
                # Check for new 5-min candle
                last_candle = candle_logs[pair][-1] if candle_logs[pair] else None
                if last_candle and last_candle["start"] != last_candle_ts[pair]:
                    last_candle_ts[pair] = last_candle["start"]
                    signal = ob_ema_signal(pair)
                    if signal:
                        scan_log.append(f"{datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S')} | {pair} | SIGNAL: {signal['side']} @ {signal['entry']} ({signal['msg']})")
                        # Place demo order
                        trade = {
                            "time": datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S'),
                            "pair": pair,
                            "side": signal['side'],
                            "entry": signal['entry'],
                            "msg": signal['msg']
                        }
                        trade_log.append(trade)
                        if len(trade_log) > 20:
                            trade_log[:] = trade_log[-20:]
            else:
                log_lines.append(f"{datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S')} | {pair} | price: -")
        scan_log.extend(log_lines)
        if len(scan_log) > 100:
            scan_log[:] = scan_log[-100:]
        status["msg"] = "Running"
        status["last"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        time.sleep(5)
    status["msg"] = "Idle"

@app.route("/")
def index():
    usdt = get_wallet_balance()
    return render_template("index.html", usdt=usdt)

@app.route("/start", methods=["POST"])
def start():
    global running
    if not running:
        running = True
        t = threading.Thread(target=scan_loop)
        t.daemon = True
        t.start()
    return jsonify({"status": "started"})

@app.route("/stop", methods=["POST"])
def stop():
    global running
    running = False
    return jsonify({"status": "stopped"})

@app.route("/status")
def get_status():
    usdt = get_wallet_balance()
    trades = trade_log[-10:]
    scans = scan_log[-30:]
    return jsonify({
        "status": status["msg"],
        "last": status["last"],
        "usdt": usdt,
        "trades": trades[::-1],
        "scans": scans[::-1]
    })

@app.route("/ping")
def ping():
    return "pong"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
