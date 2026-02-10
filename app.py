import os, re, time, json, hmac, hashlib, threading, requests
from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta
from pytz import timezone
from collections import deque

app = Flask(__name__)

# ================== CONFIG ==================
API_KEY = os.environ.get("API_KEY", "")
API_SECRET = (os.environ.get("API_SECRET", "") or "").encode()
BASE_URL = "https://api.coindcx.com"

APP_BASE_URL  = os.environ.get("APP_BASE_URL", "").rstrip("/")
KEEPALIVE_SEC = int(os.environ.get("KEEPALIVE_SEC", "240"))

IST = timezone('Asia/Kolkata')

def ist_now(): 
    return datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')

def ist_date(): 
    return datetime.now(IST).strftime('%Y-%m-%d')

def ist_yesterday(): 
    return (datetime.now(IST) - timedelta(days=1)).strftime('%Y-%m-%d')

# ================== BOT STATE ==================
running = False
status = {"msg": "Idle", "last": ""}
status_epoch = 0
error_message = ""

# ================== PAIRS ==================
PAIRS = [
    "BTCUSDT","ETHUSDT","XRPUSDT","SHIBUSDT","SOLUSDT",
    "DOGEUSDT","ADAUSDT","AEROUSDT","BNBUSDT","LTCUSDT"
]

PAIR_RULES = {
    "BTCUSDT": {"precision": 2, "min_qty": 0.001},
    "ETHUSDT": {"precision": 6, "min_qty": 0.0001},
    "XRPUSDT": {"precision": 4, "min_qty": 0.1},
    "SHIBUSDT": {"precision": 4, "min_qty": 10000},
    "DOGEUSDT": {"precision": 4, "min_qty": 0.01},
    "SOLUSDT": {"precision": 4, "min_qty": 0.01},
    "AEROUSDT": {"precision": 2, "min_qty": 0.01},
    "ADAUSDT": {"precision": 2, "min_qty": 2},
    "LTCUSDT": {"precision": 2, "min_qty": 0.001},
    "BNBUSDT": {"precision": 4, "min_qty": 0.001}
}

SETTINGS = {
    "candle_interval_sec": 15*60,
    "tp_pct": 0.01
}

TRADE_COOLDOWN_SEC = 300

tick_logs = {p: [] for p in PAIRS}
candle_logs = {p: [] for p in PAIRS}
scan_log, trade_log, exit_orders = [], [], []
pair_cooldown_until = {p: 0 for p in PAIRS}

# ================== PROFIT ==================
profit_state = {"cumulative_pnl": 0.0, "daily": {}, "inventory": {}, "processed_orders": []}

# ================== HELPERS ==================
def hmac_signature(payload): 
    return hmac.new(API_SECRET, payload.encode(), hashlib.sha256).hexdigest()

# ================== DEBUG BALANCE ==================
def get_wallet_balances():
    payload = json.dumps({"timestamp": int(time.time() * 1000)})
    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": hmac_signature(payload),
        "Content-Type":"application/json"
    }
    balances = {}
    try:
        r = requests.post(f"{BASE_URL}/exchange/v1/users/balances",
                          headers=headers, data=payload, timeout=15)

        print("========== BALANCE DEBUG ==========")
        print("STATUS:", r.status_code)
        print("RESPONSE:", r.text)
        print("===================================")

        if r.ok:
            data = r.json()
            if isinstance(data, list):
                for b in data:
                    balances[b['currency']] = float(b['balance'])
            else:
                print("FORMAT ERROR:", data)
        else:
            print("API ERROR")

    except Exception as e:
        print("BALANCE EXCEPTION:", str(e))

    return balances

# ================== MARKET DATA ==================
def fetch_all_prices():
    try:
        r = requests.get(f"{BASE_URL}/exchange/ticker", timeout=10)
        if r.ok:
            now = int(time.time())
            return {it["market"]: {"price": float(it["last_price"]), "ts": now}
                    for it in r.json() if it.get("market") in PAIRS}
    except Exception as e:
        print("PRICE ERROR:", e)
    return {}

# ================== STRATEGY ==================
def aggregate_candles(pair, interval_sec):
    t = tick_logs[pair]
    if not t: return
    candles, candle, lastw = [], None, None
    for ts, px in sorted(t, key=lambda x: x[0]):
        w = ts - (ts % interval_sec)
        if w != lastw:
            if candle: candles.append(candle)
            candle = {"open": px, "high": px, "low": px, "close": px, "start": w}
            lastw = w
        else:
            candle["high"] = max(candle["high"], px)
            candle["low"] = min(candle["low"], px)
            candle["close"] = px
    if candle: candles.append(candle)
    candle_logs[pair] = candles[-100:]

def _compute_ema(values, n):
    if len(values) < n: return None
    sma = sum(values[:n]) / n
    k = 2 / (n + 1)
    ema = sma
    for v in values[n:]:
        ema = v * k + ema * (1 - k)
    return ema

def pa_buy_sell_signal(pair, live_price):
    candles = candle_logs[pair]
    if len(candles) < 25: return None

    closes = [c["close"] for c in candles[:-1]]
    curr = live_price

    recent = candles[-6:-1]
    don_high = max(c["high"] for c in recent)
    don_low  = min(c["low"] for c in recent)

    ema_fast = _compute_ema(closes[-30:]+[curr], 5)
    ema_slow = _compute_ema(closes[-30:]+[curr], 13)

    if not ema_fast or not ema_slow: return None

    if curr > don_high and ema_fast > ema_slow:
        return {"side":"BUY","entry":curr,"msg":"Trend Breakout BUY"}
    if curr < don_low and ema_fast < ema_slow:
        return {"side":"SELL","entry":curr,"msg":"Trend Breakdown SELL"}
    return None

# ================== ORDER ==================
def place_order(pair, side, qty):
    payload = {
        "market": pair,
        "side": side.lower(),
        "order_type": "market_order",
        "total_quantity": str(qty),
        "timestamp": int(time.time()*1000)
    }
    body = json.dumps(payload,separators=(',',':'))
    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": hmac_signature(body),
        "Content-Type":"application/json"
    }
    try:
        r = requests.post(f"{BASE_URL}/exchange/v1/orders/create",
                          headers=headers, data=body, timeout=15)
        print("ORDER:", r.text)
        return r.json()
    except Exception as e:
        return {"error":str(e)}

# ================== MAIN LOOP ==================
def scan_loop():
    global running, status_epoch, error_message
    while running:
        prices = fetch_all_prices()
        balances = get_wallet_balances()
        now = int(time.time())

        for pair in PAIRS:
            if pair not in prices: 
                continue

            price = prices[pair]["price"]
            tick_logs[pair].append((now, price))
            if len(tick_logs[pair]) > 5000:
                tick_logs[pair] = tick_logs[pair][-5000:]

            aggregate_candles(pair, SETTINGS["candle_interval_sec"])

            if time.time() < pair_cooldown_until[pair]:
                continue

            signal = pa_buy_sell_signal(pair, price)
            if not signal:
                continue

            usdt = float(balances.get("USDT",0))

            if signal["side"]=="BUY" and usdt>10:
                qty = (0.3*usdt)/price
                res = place_order(pair,"BUY",round(qty,6))
                trade_log.append({"time":ist_now(),"pair":pair,"side":"BUY","res":res})
                pair_cooldown_until[pair]=time.time()+TRADE_COOLDOWN_SEC

        status["msg"] = "Running"
        status["last"] = ist_now()
        status_epoch = int(time.time())
        time.sleep(5)

# ================== ROUTES ==================
@app.route("/")
def dashboard():
    return render_template("index.html")

@app.route("/start", methods=["POST"])
def start():
    global running
    if not running:
        running = True
        threading.Thread(target=scan_loop, daemon=True).start()
    return jsonify({"status":"started"})

@app.route("/stop", methods=["POST"])
def stop():
    global running
    running = False
    return jsonify({"status":"stopped"})

@app.route("/status")
def get_status():
    balances = get_wallet_balances()
    return jsonify({
        "status": status["msg"],
        "last": status["last"],
        "usdt": balances.get("USDT",0),
        "balances": balances,
        "trades": trade_log[-10:],
        "error": error_message
    })

# ================== DIAGNOSTICS ==================
@app.route("/api/balance_test")
def balance_test():
    balances = get_wallet_balances()
    return jsonify({
        "time": ist_now(),
        "balances": balances,
        "usdt": balances.get("USDT", 0),
        "status": "OK" if balances else "EMPTY_OR_ERROR"
    })

@app.route("/api/connection_test")
def connection_test():
    try:
        r = requests.get(f"{BASE_URL}/exchange/ticker", timeout=10)
        return jsonify({
            "status_code": r.status_code,
            "ok": r.ok,
            "sample": r.json()[:3] if r.ok else r.text
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/api/key_test")
def key_test():
    payload = json.dumps({"timestamp": int(time.time() * 1000)})
    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": hmac_signature(payload),
        "Content-Type":"application/json"
    }
    try:
        r = requests.post(f"{BASE_URL}/exchange/v1/users/balances",
                          headers=headers, data=payload, timeout=10)
        return jsonify({
            "status_code": r.status_code,
            "response": r.text,
            "ok": r.ok
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/ping")
def ping():
    return "pong",200

# ================== BOOT ==================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "10000")))