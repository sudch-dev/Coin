import os, time, threading, hmac, hashlib, requests, json
from flask import Flask, render_template, jsonify
from datetime import datetime
from pytz import timezone

app = Flask(__name__)

# ===== CONFIG =====
API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET").encode()

BASE_URL = "https://api.coindcx.com"
PAIR = "BTCINR"

CANDLE_INTERVAL = 15
COOLDOWN_SECONDS = 60

IST = timezone("Asia/Kolkata")
def ist_now(): return datetime.now(IST).strftime("%H:%M:%S")

running = False
error_message = ""
status = {"msg": "Idle", "last": ""}

tick_log = []
candles = []

# ----- Position State -----
position = None      # None or "LONG"
entry_price = None
last_trade_time = 0

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
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "start": window
        })
    else:
        c = candles[-1]
        c["high"] = max(c["high"], price)
        c["low"] = min(c["low"], price)
        c["close"] = price

    if len(candles) > 100:
        candles.pop(0)

# ===== REAL PSAR SIGNAL =====
def psar_signal():
    if len(candles) < 10:
        return None

    af = 0.02
    max_af = 0.2

    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]

    psar = lows[0]
    bull = True
    ep = highs[0]

    for i in range(1, len(candles)):
        prev_psar = psar

        if bull:
            psar = prev_psar + af * (ep - prev_psar)
            if lows[i] < psar:
                bull = False
                psar = ep
                ep = lows[i]
                af = 0.02
        else:
            psar = prev_psar + af * (ep - prev_psar)
            if highs[i] > psar:
                bull = True
                psar = ep
                ep = highs[i]
                af = 0.02

        if bull:
            if highs[i] > ep:
                ep = highs[i]
                af = min(af + 0.02, max_af)
        else:
            if lows[i] < ep:
                ep = lows[i]
                af = min(af + 0.02, max_af)

    last_close = candles[-1]["close"]

    if last_close > psar:
        return "BUY"
    elif last_close < psar:
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
    global running, error_message
    global position, entry_price, last_trade_time

    while running:
        try:
            price = get_price()
            if not price:
                time.sleep(5)
                continue

            update_candles(price)
            now = time.time()

            # ----- EXIT LOGIC -----
            if position == "LONG":
                if price >= entry_price * 1.01 or price <= entry_price * 0.995:
                    bal = get_balances()
                    btc_qty = bal.get("BTC", 0)

                    if btc_qty > 0.00001:
                        res = place_order("sell", btc_qty)

                        if res:
                            position = None
                            entry_price = None
                            last_trade_time = now
                            status["msg"] = f"Exited @ {price}"

            # ----- ENTRY LOGIC -----
            elif position is None and (now - last_trade_time > COOLDOWN_SECONDS):
                signal = psar_signal()
                bal = get_balances()
                inr = bal.get("INR", 0)

                if signal == "BUY" and inr > 200:
                    qty = round((0.1 * inr) / price, 6)
                    res = place_order("buy", qty)

                    if res:
                        position = "LONG"
                        entry_price = price
                        last_trade_time = now
                        status["msg"] = f"Entered LONG @ {price}"

            status["last"] = ist_now()

        except Exception as e:
            error_message = str(e)

        # responsive stop
        for _ in range(5):
            if not running:
                break
            time.sleep(1)

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