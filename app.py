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
    "BTCUSDT", "ETHUSDT","SOLUSDT","DOGEUSDT"
]

PAIR_PRECISION = {
    'BTCUSDT': 4,
    'ETHUSDT': 4,
    'SOLUSDT': 3,
    'DOGEUSDT': 2
}

scan_interval = 5  # seconds, for faster candle build
trade_log = []
scan_log = []
exit_orders = []
running = False
status = {"msg": "Idle"}

tick_logs = {pair: [] for pair in PAIRS}
candle_logs = {pair: [] for pair in PAIRS}
last_candle_ts = {pair: 0 for pair in PAIRS}

def hmac_signature(payload):
    return hmac.new(API_SECRET, payload.encode(), hashlib.sha256).hexdigest()

def adjust_quantity_precision(symbol, qty):
    precision = PAIR_PRECISION.get(symbol, 4)
    return float(f"{qty:.{precision}f}")

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
    except Exception: pass
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
    candles = candle_logs.get(pair, [])
    if len(candles) < 10:
        return None

    prev = candles[-2]
    curr = candles[-1]

    # Internal OB detector
    def find_last_order_block(candles, direction="bullish"):
        for i in range(len(candles) - 3, -1, -1):
            c = candles[i]
            if direction == "bullish" and c["open"] > c["close"]:
                if candles[i+1]["close"] > c["high"] and candles[i+2]["close"] > candles[i+1]["close"]:
                    return {"low": c["low"], "high": c["high"]}
            if direction == "bearish" and c["close"] > c["open"]:
                if candles[i+1]["close"] < c["low"] and candles[i+2]["close"] < candles[i+1]["close"]:
                    return {"low": c["low"], "high": c["high"]}
        return None

    ob_bull = find_last_order_block(candles, "bullish")
    ob_bear = find_last_order_block(candles, "bearish")

    # ✅ BUY: momentum breakout + OB bounce
    if curr["close"] > curr["open"] and \
       curr["close"] > prev["high"] and \
       curr["open"] <= prev["close"]:
        if ob_bull and ob_bull["low"] <= curr["low"] <= ob_bull["high"]:
            return {
                "side": "BUY",
                "entry": curr["close"],
                "msg": f"PA BUY: Bullish breakout + OB zone {ob_bull['low']}–{ob_bull['high']}"
            }

    # ✅ SELL: breakdown + OB rejection
    if curr["close"] < curr["open"] and \
       curr["close"] < prev["low"] and \
       curr["open"] >= prev["close"]:
        if ob_bear and ob_bear["low"] <= curr["high"] <= ob_bear["high"]:
            return {
                "side": "SELL",
                "entry": curr["close"],
                "msg": f"PA SELL: Bearish breakdown + OB zone {ob_bear['low']}–{ob_bear['high']}"
            }

    return None

def place_order(symbol, side, qty):
    qty = adjust_quantity_precision(symbol, qty)
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
    interval = 60  # 1 min candles
    while running:
        prices = fetch_all_prices()
        now = int(time.time())
        log_lines = []
        monitor_exits(prices)
        balances = get_balance()
        usdt = 0
        for b in balances:
            if b['currency'] == 'USDT': usdt = float(b['balance'])
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
                    if signal and usdt > 5:
                        qty = (0.3 * usdt) / signal['entry']
                        qty = adjust_quantity_precision(pair, qty)
                        tp = round(signal['entry'] * 1.0005, 6)  # +0.05%
                        sl = round(signal['entry'] * 0.999, 6)   # -0.1%
                        result = place_order(pair, signal['side'], qty)
                        scan_log.append(f"{datetime.utcfromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S')} | {pair} | SIGNAL: {signal['side']} @ {signal['entry']} ({signal['msg']}) | ORDER: {result}")
                        trade = {
                            "ts": datetime.utcfromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S'),
                            "symbol": pair,
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
                        if len(trade_log) > 30:
                            trade_log[:] = trade_log[-30:]
            else:
                log_lines.append(f"{datetime.utcfromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S')} | {pair} | price: -")
        scan_log.extend(log_lines)
        if len(scan_log) > 120:
            scan_log[:] = scan_log[-120:]
        status["msg"] = "Running"
        status["last"] = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        time.sleep(scan_interval)
    status["msg"] = "Idle"

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
        if b['currency'] == 'USDT':
            usdt_bal = b['balance']
    now = datetime.utcnow()
    cutoff_1hr = now - timedelta(hours=1)
    cutoff_15m = now - timedelta(minutes=15)
    pnl_1hr = 0
    pnl_15m = 0

    for trade in trade_log[-50:]:
        t_time = datetime.strptime(trade["ts"], '%Y-%m-%d %H:%M:%S')
        if "side" in trade and "qty" in trade and "entry" in trade:
            exit_price = trade.get("tp", trade["entry"])  # Use TP if available, else entry
            qty = trade["qty"]
            if trade["side"] == "BUY":
                profit = (exit_price - trade["entry"]) * qty
            else:
                profit = (trade["entry"] - exit_price) * qty
            if t_time > cutoff_1hr:
                pnl_1hr += profit
            if t_time > cutoff_15m:
                pnl_15m += profit

    return jsonify({
        "status": status["msg"],
        "usdt": usdt_bal,
        "trades": trade_log[-10:][::-1],
        "scan_log": scan_log[-40:][::-1],
        "pnl": round(pnl_1hr, 2),
        "last_15m_pnl": round(pnl_15m, 2)
    })

@app.route("/ping")
def ping():
    return "pong"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
