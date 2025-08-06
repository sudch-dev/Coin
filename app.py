import os
import time
import threading
import hmac
import hashlib
import requests
import json
from datetime import datetime
from flask import Flask, render_template, jsonify

app = Flask(__name__)

API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET").encode()
BASE_URL = "https://api.coindcx.com"
PAIRS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "SHIBUSDT", "SOLUSDT",
    "DOGEUSDT", "ADAUSDT", "MATICUSDT", "BNBUSDT", "LTCUSDT"
]

tick_logs = {pair: [] for pair in PAIRS}
candle_logs = {pair: [] for pair in PAIRS}
scan_log = []
trade_log = []
exit_orders = []  # Each: {'pair', 'side', 'qty', 'tp', 'sl', 'entry'}
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
    except Exception:
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
    except Exception:
        pass
    return {}

def aggregate_candles(pair, interval=60):
    ticks = tick_logs[pair]
    if not ticks:
        return
    window = interval  # 1 minute = 60 seconds
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

def pa_buy_sell_signal(pair):
    candles = candle_logs[pair]
    if len(candles) < 2:
        return None
    prev = candles[-2]
    curr = candles[-1]
    mid = (prev["high"] + prev["low"]) / 2

    # BUY
    if curr["open"] < prev["close"] and curr["high"] > mid:
        return {"side": "BUY", "entry": curr["close"], "msg": "PA BUY: open < prev close & high > prev midpoint"}
    # SELL
    if curr["open"] > prev["close"] and curr["low"] < mid:
        return {"side": "SELL", "entry": curr["close"], "msg": "PA SELL: open > prev close & low < prev midpoint"}
    return None

def place_order(pair, side, qty):
    payload = {
        "market": pair,
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
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def monitor_exits(prices):
    to_remove = []
    for ex in exit_orders:
        pair, side, qty, tp, sl, entry = ex["pair"], ex["side"], ex["qty"], ex["tp"], ex["sl"], ex["entry"]
        price = prices.get(pair, {}).get("price")
        if price:
            # For spot, to exit a buy, you "sell"; to exit a sell, you "buy"
            if side == "BUY" and (price >= tp or price <= sl):
                result = place_order(pair, "SELL", qty)
                scan_log.append(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} | {pair} | EXIT SELL {qty} at {price} (TP/SL) | Result: {result}")
                to_remove.append(ex)
            elif side == "SELL" and (price <= tp or price >= sl):
                result = place_order(pair, "BUY", qty)
                scan_log.append(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} | {pair} | EXIT BUY {qty} at {price} (TP/SL) | Result: {result}")
                to_remove.append(ex)
    for ex in to_remove:
        exit_orders.remove(ex)

def scan_loop():
    global running, scan_log, status
    scan_log.clear()
    last_candle_ts = {p: 0 for p in PAIRS}
    interval = 60  # 1 min candles
    while running:
        prices = fetch_all_prices()
        now = int(time.time())
        log_lines = []
        monitor_exits(prices)
        for pair in PAIRS:
            if pair in prices:
                price = prices[pair]["price"]
                tick_logs[pair].append((now, price))
                if len(tick_logs[pair]) > 1000:
                    tick_logs[pair] = tick_logs[pair][-1000:]
                log_lines.append(f"{datetime.utcfromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S')} | {pair} | price: {price}")
                aggregate_candles(pair, interval)
                last_candle = candle_logs[pair][-1] if candle_logs[pair] else None
                if last_candle and last_candle["start"] != last_candle_ts[pair]:
                    last_candle_ts[pair] = last_candle["start"]
                    signal = pa_buy_sell_signal(pair)
                    if signal:
                        usdt = get_wallet_balance()
                        qty = round((0.3 * usdt) / signal['entry'], 6)
                        tp = round(signal['entry'] * 1.0005, 6)  # +0.05%
                        sl = round(signal['entry'] * 0.999, 6)   # -0.1%
                        result = place_order(pair, signal['side'], qty)
                        scan_log.append(f"{datetime.utcfromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S')} | {pair} | SIGNAL: {signal['side']} @ {signal['entry']} ({signal['msg']}) | ORDER: {result}")
                        trade = {
                            "time": datetime.utcfromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S'),
                            "pair": pair,
                            "side": signal['side'],
                            "entry": signal['entry'],
                            "msg": signal['msg'],
                            "tp": tp,
                            "sl": sl,
                            "qty": qty,
                            "order_result": result
                        }
                        trade_log.append(trade)
                        exit_orders.append({
                            "pair": pair,
                            "side": signal['side'],
                            "qty": qty,
                            "tp": tp,
                            "sl": sl,
                            "entry": signal['entry']
                        })
                        if len(trade_log) > 20:
                            trade_log[:] = trade_log[-20:]
            else:
                log_lines.append(f"{datetime.utcfromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S')} | {pair} | price: -")
        scan_log.extend(log_lines)
        if len(scan_log) > 100:
            scan_log[:] = scan_log[-100:]
        status["msg"] = "Running"
        status["last"] = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
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
    candles_out = {p: candle_logs[p][-5:] for p in PAIRS}
    return jsonify({
        "status": status["msg"],
        "last": status["last"],
        "usdt": usdt,
        "trades": trades[::-1],
        "scans": scans[::-1],
        "candles": candles_out
    })

@app.route("/ping")
def ping():
    return "pong"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
