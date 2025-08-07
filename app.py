import os
import time
import pytz
import threading
import hmac
import hashlib
import requests
import json
from datetime import datetime
from flask import Flask, render_template, jsonify

app = Flask(__name__)

# ENV & API Details
API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET").encode()
BASE_URL = "https://api.coindcx.com"

# Pairs and precision
PAIRS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "SHIBUSDT", "SOLUSDT",
    "DOGEUSDT", "ADAUSDT", "MATICUSDT", "BNBUSDT", "LTCUSDT"
]

PAIR_PRECISION = {
    "BTCUSDT": 6, "ETHUSDT": 5, "XRPUSDT": 2, "SHIBUSDT": 0,
    "SOLUSDT": 4, "DOGEUSDT": 2, "ADAUSDT": 2,
    "MATICUSDT": 3, "BNBUSDT": 4, "LTCUSDT": 4
}

# Runtime Storage
tick_logs = {pair: [] for pair in PAIRS}
candle_logs = {pair: [] for pair in PAIRS}
scan_log = []
trade_log = []
exit_orders = []
running = False
status = {"msg": "Idle", "last": ""}

# Signer
def hmac_signature(payload):
    return hmac.new(API_SECRET, payload.encode(), hashlib.sha256).hexdigest()

# Get all wallet balances (USDT + coins)
def get_wallet_balances():
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
            balances = r.json()
            return {b['currency']: float(b['balance']) for b in balances if float(b['balance']) > 0}
    except Exception:
        pass
    return {}

# Fetch market prices
def fetch_all_prices():
    try:
        r = requests.get(f"{BASE_URL}/exchange/ticker", timeout=10)
        if r.ok:
            data = r.json()
            now = int(time.time())
            return {
                item["market"]: {"price": float(item["last_price"]), "ts": now}
                for item in data if item["market"] in PAIRS
            }
    except Exception:
        pass
    return {}

# Candle aggregation
def aggregate_candles(pair, interval=60):
    ticks = tick_logs[pair]
    if not ticks:
        return
    window = interval
    candles = []
    ticks_sorted = sorted(ticks, key=lambda x: x[0])
    candle = None
    last_window = None
    for ts, price in ticks_sorted:
        wstart = ts - (ts % window)
        if last_window != wstart:
            if candle:
                candles.append(candle)
            candle = {
                "open": price, "high": price, "low": price,
                "close": price, "volume": 1, "start": wstart
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

# PA-based Signal
def pa_buy_sell_signal(pair):
    candles = candle_logs[pair]
    if len(candles) < 2:
        return None
    prev = candles[-2]
    curr = candles[-1]
    mid = (prev["high"] + prev["low"])

    if curr["open"] < prev["close"] and curr["high"] > mid:
        return {"side": "BUY", "entry": curr["close"], "msg": "PA BUY: open < prev close & high > mid"}
    if curr["open"] > prev["close"] and curr["low"] < mid:
        return {"side": "SELL", "entry": curr["close"], "msg": "PA SELL: open > prev close & low < mid"}
    return None

# Place Order (safe)
def place_order(pair, side, qty):
    payload = {
        "market": pair,
        "side": "buy" if side == "BUY" else "sell",
        "order_type": "market",
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
        try:
            return r.json()
        except ValueError:
            return {"error": f"Invalid JSON response: {r.status_code} {r.text.strip()}"}
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

# Exit Monitor
def monitor_exits(prices):
    to_remove = []
    balances = get_wallet_balances()
    for ex in exit_orders:
        pair, side, qty, tp, sl, entry = ex["pair"], ex["side"], ex["qty"], ex["tp"], ex["sl"], ex["entry"]
        price = prices.get(pair, {}).get("price")
        coin = pair.replace("USDT", "")
        coin_balance = balances.get(coin, 0.0)
        if price:
            now = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
            if side == "BUY" and (price >= tp or price <= sl):
                result = place_order(pair, "SELL", qty)
                scan_log.append(f"{now} | {pair} | EXIT SELL {qty} at {price} (TP/SL) | Result: {result}")
                to_remove.append(ex)
            elif side == "SELL" and (price <= tp or price >= sl):
                if coin_balance > 0:
                    precision = PAIR_PRECISION.get(pair, 6)
                    result = place_order(pair, "BUY", round(coin_balance, precision))
                    scan_log.append(f"{now} | {pair} | EXIT BUY {round(coin_balance, precision)} at {price} (TP/SL) | Result: {result}")
                    to_remove.append(ex)
    for ex in to_remove:
        exit_orders.remove(ex)

# Main Scanner Thread
def scan_loop():
    global running, scan_log, status
    scan_log.clear()
    last_candle_ts = {p: 0 for p in PAIRS}
    interval = 60
    while running:
        balances = get_wallet_balances()
        usdt = balances.get("USDT", 0.0)
        prices = fetch_all_prices()
        now = int(time.time())
        monitor_exits(prices)
        for pair in PAIRS:
            if pair in prices:
                price = prices[pair]["price"]
                tick_logs[pair].append((now, price))
                if len(tick_logs[pair]) > 1000:
                    tick_logs[pair] = tick_logs[pair][-1000:]
                aggregate_candles(pair, interval)
                last_candle = candle_logs[pair][-1] if candle_logs[pair] else None
                if last_candle and last_candle["start"] != last_candle_ts[pair]:
                    last_candle_ts[pair] = last_candle["start"]
                    signal = pa_buy_sell_signal(pair)
                    if signal:
    precision = PAIR_PRECISION.get(pair, 6)
    qty = round((0.3 * usdt) / signal['entry'], precision)
    notional = qty * signal['entry']
    
    if notional < 0.12:
        scan_log.append(f"{now_ist} | {pair} | Skipped ENTRY {signal['side']}: Qty {qty}, Notional {notional:.4f} too low")
        continue

    # 💡 [SMALL CHANGE] extra validation for BUY logic
    if signal['side'] == "BUY" and signal['entry'] > prices[pair]["price"]:
        scan_log.append(f"{now_ist} | {pair} | Skipped BUY: entry > current price (no momentum)")
        continue

    tp = round(signal['entry'] * 1.0005, 6)
    sl = round(signal['entry'] * 0.999, 6)
    result = place_order(pair, signal['side'], qty)

    scan_log.append(f"{now_ist} | {pair} | [{active_strategy['name']}] SIGNAL: {signal['side']} @ {signal['entry']} | {signal['msg']} | Result: {result}")
    
    trade_log.append({
        "time": now_ist, "pair": pair, "side": signal['side"],
        "entry": signal['entry"], "msg": signal['msg"],
        "tp": tp, "sl": sl, "qty": qty, "order_result": result
    })

    exit_orders.append({
        "pair": pair, "side": signal['side"],
        "qty": qty, "tp": tp, "sl": sl, "entry": signal['entry']
    })

    if len(trade_log) > 20:
        trade_log[:] = trade_log[-20:]
            else:
                scan_log.append(f"{datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')} | {pair} | price: -")
        if len(scan_log) > 100:
            scan_log[:] = scan_log[-100:]
        status["msg"] = "Running"
        status["last"] = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
        time.sleep(5)
    status["msg"] = "Idle"

# Flask Routes
@app.route("/")
def index():
    return render_template("index.html")

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
    balances = get_wallet_balances()
    usdt = balances.get("USDT", 0.0)
    trades = trade_log[-10:]
    scans = scan_log[-30:]
    candles_out = {p: candle_logs[p][-5:] for p in PAIRS}
    return jsonify({
        "status": status["msg"],
        "last": status["last"],
        "usdt": usdt,
        "balances": balances,
        "trades": trades[::-1],
        "scans": scans[::-1],
        "candles": candles_out
    })

@app.route("/ping")
def ping():
    return "pong"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
