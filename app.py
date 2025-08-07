
import os
import time
import threading
import hmac
import hashlib
import requests
import json
from datetime import datetime
import pytz
from flask import Flask, render_template, jsonify

app = Flask(__name__)

API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET").encode()
BASE_URL = "https://api.coindcx.com"

PAIRS = ["BTCUSDT", "ETHUSDT", "DOGEUSDT"]
PAIR_PRECISION = {"BTCUSDT": 6, "ETHUSDT": 5, "DOGEUSDT": 0}

tick_logs = {pair: [] for pair in PAIRS}
candle_logs = {pair: [] for pair in PAIRS}
ema_logs = {pair: [] for pair in PAIRS}
trade_log = []
exit_orders = []
status = {"msg": "Idle", "last": ""}
running = False
net_pnl = 0.0
coin_balances = {}

def hmac_signature(payload):
    return hmac.new(API_SECRET, payload.encode(), hashlib.sha256).hexdigest()

def get_balances():
    global coin_balances
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
            coin_balances = {b['currency']: float(b['balance']) for b in r.json()}
            return coin_balances.get("USDT", 0.0)
    except Exception:
        pass
    return 0.0

def fetch_all_prices():
    try:
        r = requests.get(f"{BASE_URL}/exchange/ticker", timeout=10)
        if r.ok:
            now = int(time.time())
            return {item["market"]: {"price": float(item["last_price"]), "ts": now} for item in r.json() if item["market"] in PAIRS}
    except Exception:
        pass
    return {}

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
            candle = {"open": price, "high": price, "low": price, "close": price, "start": wstart}
            last_window = wstart
        else:
            candle["high"] = max(candle["high"], price)
            candle["low"] = min(candle["low"], price)
            candle["close"] = price
    if candle:
        candles.append(candle)
    candle_logs[pair] = candles[-50:]
    calc_ema(pair)

def calc_ema(pair):
    closes = [c["close"] for c in candle_logs[pair]]
    if len(closes) < 10:
        return
    def ema(series, span):
        k = 2 / (span + 1)
        ema_vals = [series[0]]
        for price in series[1:]:
            ema_vals.append(price * k + ema_vals[-1] * (1 - k))
        return ema_vals
    ema5 = ema(closes, 5)
    ema10 = ema(closes, 10)
    ema_logs[pair] = [{"ema5": e5, "ema10": e10, "price": p} for e5, e10, p in zip(ema5[-5:], ema10[-5:], closes[-5:])]

def detect_crossover(pair):
    data = ema_logs[pair]
    if len(data) < 2:
        return None
    prev, curr = data[-2], data[-1]
    if prev["ema5"] < prev["ema10"] and curr["ema5"] > curr["ema10"] and curr["price"] > candle_logs[pair][-1]["open"]:
        return {"side": "BUY", "entry": curr["price"]}
    elif prev["ema5"] > prev["ema10"] and curr["ema5"] < curr["ema10"] and curr["price"] < candle_logs[pair][-1]["open"]:
        return {"side": "SELL", "entry": curr["price"]}
    return None

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
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def monitor_exits(prices):
    global net_pnl
    to_remove = []
    for ex in exit_orders:
        pair, side, qty, tp, sl, entry = ex.values()
        price = prices.get(pair, {}).get("price")
        if not price:
            continue
        if side == "BUY" and (price >= tp or price <= sl):
            result = place_order(pair, "SELL", qty)
            pnl = (price - entry) * qty
            net_pnl += pnl
            trade_log.append({"pair": pair, "exit_price": price, "side": "SELL", "qty": qty, "pnl": pnl, "exit_time": now_ist()})
            to_remove.append(ex)
        elif side == "SELL" and (price <= tp or price >= sl):
            result = place_order(pair, "BUY", qty)
            pnl = (entry - price) * qty
            net_pnl += pnl
            trade_log.append({"pair": pair, "exit_price": price, "side": "BUY", "qty": qty, "pnl": pnl, "exit_time": now_ist()})
            to_remove.append(ex)
    for ex in to_remove:
        exit_orders.remove(ex)

def now_ist():
    return datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')

def scan_loop():
    global running, status
    last_candle_ts = {p: 0 for p in PAIRS}
    while running:
        prices = fetch_all_prices()
        usdt = get_balances()
        monitor_exits(prices)
        for pair in PAIRS:
            price = prices.get(pair, {}).get("price")
            now = int(time.time())
            if price:
                tick_logs[pair].append((now, price))
                tick_logs[pair] = tick_logs[pair][-1000:]
                aggregate_candles(pair)
                last_candle = candle_logs[pair][-1]
                if last_candle["start"] != last_candle_ts[pair]:
                    last_candle_ts[pair] = last_candle["start"]
                    signal = detect_crossover(pair)
                    if signal:
                        precision = PAIR_PRECISION[pair]
                        if signal["side"] == "BUY" and usdt > 2:
                            qty = round((0.3 * usdt) / signal["entry"], precision)
                        elif signal["side"] == "SELL" and coin_balances.get(pair.replace("USDT", ""), 0.0) * signal["entry"] > 2:
                            qty = round(coin_balances.get(pair.replace("USDT", ""), 0.0), precision)
                        else:
                            continue
                        result = place_order(pair, signal["side"], qty)
                        tp = round(signal["entry"] * (1.002 if signal["side"] == "BUY" else 0.998), 6)
                        sl = round(signal["entry"] * (0.999 if signal["side"] == "BUY" else 1.001), 6)
                        exit_orders.append({"pair": pair, "side": signal["side"], "qty": qty, "tp": tp, "sl": sl, "entry": signal["entry"]})
        status["msg"] = "Running"
        status["last"] = now_ist()
        time.sleep(5)
    status["msg"] = "Idle"

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
    usdt = get_balances()
    return jsonify({
        "status": status["msg"],
        "last": status["last"],
        "usdt": usdt,
        "coins": {p.replace("USDT", ""): coin_balances.get(p.replace("USDT", ""), 0.0) for p in PAIRS},
        "net_pnl": round(net_pnl, 4),
        "trades": trade_log[-10:],
    })

@app.route("/ping")
def ping():
    return "pong"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
