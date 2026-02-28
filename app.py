import os, time, threading, hmac, hashlib, requests, json
from flask import Flask, render_template, jsonify
from datetime import datetime
from pytz import timezone
from collections import deque

app = Flask(__name__)

API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET").encode()
BASE_URL = "https://api.coindcx.com"

PAIR = "ORDIUSDT"
LEVERAGE = 5
RISK_PER_TRADE = 0.005
MAX_DAILY_LOSS = -0.03

IST = timezone("Asia/Kolkata")
def ist_now(): return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

running = False
status = {"msg": "Idle", "last": ""}
scan_log, trade_log = [], []
tick_logs, candle_logs = [], []

# ================= AUTH =================

def hmac_signature(payload):
    return hmac.new(API_SECRET, payload.encode(), hashlib.sha256).hexdigest()

# ================= DATA =================

def fetch_price():
    try:
        r = requests.get(f"{BASE_URL}/exchange/ticker", timeout=10)
        if r.ok:
            for x in r.json():
                if x["market"] == PAIR:
                    return float(x["last_price"])
    except:
        pass
    return None

# ================= FUTURES POSITION =================

def get_position():
    payload = json.dumps({"timestamp": int(time.time()*1000)})
    sig = hmac_signature(payload)

    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": sig,
        "Content-Type": "application/json"
    }

    try:
        r = requests.post(
            f"{BASE_URL}/exchange/v1/derivatives/futures/positions",
            headers=headers, data=payload, timeout=10
        )
        if r.ok:
            for p in r.json():
                if p.get("symbol") == PAIR:
                    return p
    except:
        pass

    return None

# ================= WALLET =================

def get_balance():
    payload = json.dumps({"timestamp": int(time.time()*1000)})
    sig = hmac_signature(payload)

    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": sig,
        "Content-Type": "application/json"
    }

    try:
        r = requests.post(
            f"{BASE_URL}/exchange/v1/users/balances",
            headers=headers, data=payload, timeout=10
        )
        if r.ok:
            for b in r.json():
                if b["currency"] == "USDT":
                    return float(b["balance"])
    except:
        pass

    return 0.0

def get_equity():
    eq = get_balance()
    pos = get_position()
    if pos:
        eq += float(pos.get("unrealized_pnl", 0))
    return eq

# ================= ORDER =================

def place_order(side, qty):
    payload = {
        "symbol": PAIR,
        "side": side,
        "type": "MARKET",
        "quantity": str(qty),
        "leverage": LEVERAGE,
        "timestamp": int(time.time()*1000)
    }

    body = json.dumps(payload)
    sig = hmac_signature(body)

    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": sig,
        "Content-Type": "application/json"
    }

    try:
        r = requests.post(
            f"{BASE_URL}/exchange/v1/derivatives/futures/orders/create",
            headers=headers, data=body, timeout=10
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# ================= INDICATOR =================

def aggregate_candles(price):
    now = int(time.time())
    tick_logs.append((now, price))
    if len(tick_logs) > 500:
        tick_logs[:] = tick_logs[-500:]

    interval = 15
    candles, c, w = [], None, None

    for ts, p in sorted(tick_logs):
        s = ts - ts % interval
        if w != s:
            if c: candles.append(c)
            c = {"open": p, "high": p, "low": p, "close": p, "start": s}
            w = s
        else:
            c["high"] = max(c["high"], p)
            c["low"] = min(c["low"], p)
            c["close"] = p

    if c: candles.append(c)
    candle_logs[:] = candles[-100:]

def psar_flip():
    if len(candle_logs) < 6:
        return None

    closes = [c["close"] for c in candle_logs[-5:]]
    if closes[-1] > max(closes[:-1]):
        return "BUY"
    if closes[-1] < min(closes[:-1]):
        return "SELL"
    return None

def atr():
    if len(candle_logs) < 15:
        return None
    return sum(c["high"] - c["low"] for c in candle_logs[-14:]) / 14

# ================= POSITION SIZE =================

def compute_qty(entry, sl, equity):
    risk = equity * RISK_PER_TRADE
    rpu = abs(entry - sl)
    if rpu <= 0: return 0
    qty = risk / rpu
    max_pos = (equity * LEVERAGE) / entry
    return min(qty, max_pos)

# ================= MANAGEMENT =================

def manage_position(pos, price, atrv):
    side = pos["side"]
    entry = float(pos["entry_price"])
    qty = float(pos["size"])

    if side == "LONG":
        sl = entry - 2.5 * atrv
        tp = entry + 4 * atrv
        if price <= sl or price >= tp:
            place_order("SELL", qty)

    if side == "SHORT":
        sl = entry + 2.5 * atrv
        tp = entry - 5 * atrv
        if price >= sl or price <= tp:
            place_order("BUY", qty)

# ================= MAIN LOOP =================

def scan_loop():
    global running

    while running:
        price = fetch_price()
        if not price:
            time.sleep(2)
            continue

        aggregate_candles(price)
        atrv = atr()

        pos = get_position()

        # Manage open position
        if pos and float(pos.get("size", 0)) > 0:
            if atrv:
                manage_position(pos, price, atrv)
            time.sleep(2)
            continue

        # Entry
        signal = psar_flip()
        if signal and atrv:
            entry = price
            sl = entry - 2.5*atrv if signal=="BUY" else entry + 2.5*atrv

            equity = get_equity()
            qty = compute_qty(entry, sl, equity)

            if qty > 0:
                side = "BUY" if signal=="BUY" else "SELL"
                res = place_order(side, qty)

                trade_log.append({
                    "time": ist_now(),
                    "side": side,
                    "entry": entry,
                    "qty": qty,
                    "result": res
                })

        status["msg"], status["last"] = "Running", ist_now()
        time.sleep(3)

# ================= ROUTES =================

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
    return jsonify({"status":"started"})

@app.route("/stop", methods=["POST"])
def stop():
    global running
    running = False
    return jsonify({"status":"stopped"})

@app.route("/status")
def status_api():
    pos = get_position()
    eq = get_equity()
    bal = get_balance()

    return jsonify({
        "status": status["msg"],
        "last": status["last"],
        "equity": eq,
        "balance": bal,
        "position": pos,
        "leverage": LEVERAGE,
        "trades": trade_log[-10:][::-1],
        "scans": scan_log[-20:][::-1]
    })

@app.route("/ping")
def ping(): return "pong"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)