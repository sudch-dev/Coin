import os, time, threading, hmac, hashlib, requests, json
from flask import Flask, render_template, jsonify
from datetime import datetime
from pytz import timezone
from collections import deque

app = Flask(__name__)

# ===== CONFIG =====
API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET").encode()

BASE_URL = "https://api.coindcx.com"
PAIR = "BTCINR"

CANDLE_INTERVAL = 15
TRADE_COOLDOWN = 30

IST = timezone("Asia/Kolkata")
def ist_now(): return datetime.now(IST).strftime("%H:%M:%S")

running = False
error_message = ""
status = {"msg": "Idle", "last": ""}

tick_log = []
candles = []
exit_order = None

# ===== AUTH =====
def sign(payload):
    return hmac.new(API_SECRET, payload.encode(), hashlib.sha256).hexdigest()

def signed_post(url, body):
    payload = json.dumps(body)
    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": sign(payload),
        "Content-Type": "application/json"
    }
    r = requests.post(url, headers=headers, data=payload, timeout=10)
    return r.json() if r.ok else {}

# ===== PORTFOLIO =====
def get_balances():
    body = {"timestamp": int(time.time() * 1000)}
    data = signed_post(f"{BASE_URL}/exchange/v1/users/balances", body)
    return {x["currency"]: float(x["balance"]) for x in data}

# ===== MARKET =====
def get_price():
    r = requests.get(f"{BASE_URL}/exchange/ticker")
    if r.ok:
        for x in r.json():
            if x["market"] == PAIR:
                return float(x["last_price"])
    return None

# ===== CANDLE BUILD =====
def update_candles(price):
    now = int(time.time())
    tick_log.append((now, price))

    window = now - (now % CANDLE_INTERVAL)

    if not candles or candles[-1]["start"] != window:
        candles.append({
            "open": price, "high": price,
            "low": price, "close": price,
            "start": window
        })
    else:
        c = candles[-1]
        c["high"] = max(c["high"], price)
        c["low"] = min(c["low"], price)
        c["close"] = price

    if len(candles) > 100:
        candles.pop(0)

# ===== PSAR SIGNAL =====
def psar_signal():
    if len(candles) < 6:
        return None

    prev = candles[-3]["close"]
    last = candles[-2]["close"]

    if last > prev:
        return "BUY"
    elif last < prev:
        return "SELL"

    return None

# ===== ORDER =====
def place_order(side, qty):
    body = {
        "market": PAIR,
        "side": side.lower(),
        "order_type": "market_order",
        "total_quantity": str(qty),
        "timestamp": int(time.time() * 1000)
    }
    return signed_post(f"{BASE_URL}/exchange/v1/orders/create", body)

# ===== ENGINE =====
def bot_loop():
    global running, error_message, exit_order

    while running:
        try:
            price = get_price()
            if not price:
                time.sleep(5)
                continue

            update_candles(price)

            if exit_order:
                side, tp, sl, qty = exit_order
                if (side == "BUY" and (price >= tp or price <= sl)) or \
                   (side == "SELL" and (price <= tp or price >= sl)):
                    place_order("SELL" if side == "BUY" else "BUY", qty)
                    exit_order = None

            else:
                signal = psar_signal()
                bal = get_balances()
                inr = bal.get("INR", 0)

                if signal and inr > 100:
                    qty = round((0.1 * inr) / price, 6)

                    res = place_order(signal, qty)

                    tp = price * 1.01
                    sl = price * 0.995

                    exit_order = (signal, tp, sl, qty)

            status["msg"] = "Running"
            status["last"] = ist_now()

        except Exception as e:
            error_message = str(e)

        time.sleep(5)

    status["msg"] = "Idle"

# ===== UI ROUTES =====
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/start", methods=["POST"])
def start():
    global running
    if not running:
        running = True
        threading.Thread(target=bot_loop, daemon=True).start()
    return jsonify({"status": "started"})

@app.route("/stop", methods=["POST"])
def stop():
    global running
    running = False
    return jsonify({"status": "stopped"})

@app.route("/status")
def stat():
    bal = get_balances()
    return jsonify({
        "status": status["msg"],
        "last": status["last"],
        "INR": bal.get("INR", 0),
        "BTC": bal.get("BTC", 0),
        "error": error_message
    })

@app.route("/ping")
def ping():
    return "pong"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)