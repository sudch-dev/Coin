import os, time, threading, hmac, hashlib, requests, json
from flask import Flask, render_template, jsonify, request
from datetime import datetime
from pytz import timezone
import numpy as np

app = Flask(__name__)

# ========= CONFIG =========

API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET", "").encode()

BASE_URL = "https://api.coindcx.com"
IST = timezone("Asia/Kolkata")

def ist_now():
    return datetime.now(IST).strftime("%H:%M:%S")

running = False
error_message = ""
status = {"msg": "Idle", "last": ""}
investment_capital = 0

price_data = {}
entry_positions = {}
cooldown = {}

RISK_PER_TRADE = 0.15
MIN_RESERVE = 200
COOLDOWN_SEC = 120
MAX_DAILY_LOSS = 0.03

equity_start = 0
daily_loss_lock = False

# ========= AUTH =========

def sign(payload):
    return hmac.new(API_SECRET, payload.encode(), hashlib.sha256).hexdigest()

def signed_post(endpoint, body):
    global error_message
    payload = json.dumps(body)
    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": sign(payload),
        "Content-Type": "application/json"
    }
    try:
        r = requests.post(BASE_URL + endpoint, headers=headers, data=payload, timeout=20)
        if r.ok:
            error_message = ""
            return r.json()
        else:
            error_message = f"{r.status_code}: {r.text}"
    except Exception as e:
        error_message = str(e)
    return {}

# ========= BALANCES =========

def get_balances():
    body = {"timestamp": int(time.time()*1000)}
    data = signed_post("/exchange/v1/users/balances", body)
    if isinstance(data, list):
        return {x["currency"]: float(x["balance"]) for x in data}
    return {}

# ========= MARKET DISCOVERY =========

def get_inr_markets():
    try:
        r = requests.get(BASE_URL + "/exchange/ticker", timeout=10)
        if not r.ok:
            return []
        markets = []
        for x in r.json():
            if x["market"].endswith("INR"):
                markets.append(x["market"])
        return markets
    except:
        return []

def get_prices():
    try:
        r = requests.get(BASE_URL + "/exchange/ticker", timeout=10)
        if not r.ok:
            return {}
        return {x["market"]: float(x["last_price"]) for x in r.json()}
    except:
        return {}

# ========= FEATURES =========

def volatility(prices):
    if len(prices) < 20:
        return 0
    returns = np.diff(prices[-20:])
    return np.std(returns)

def detect_regime(prices):
    if len(prices) < 80:
        return "neutral"

    ma20 = np.mean(prices[-20:])
    ma50 = np.mean(prices[-50:])
    slope = ma20 - ma50
    vol = volatility(prices)

    if slope > vol:
        return "uptrend"
    elif slope < -vol:
        return "downtrend"
    else:
        return "range"

def volatility_cluster(prices):
    if len(prices) < 60:
        return False
    return volatility(prices[-30:]) > volatility(prices[-60:-30]) * 1.2

def fee_filter(price, vol):
    expected_move = vol * 3
    estimated_fee = price * 0.006
    return expected_move > estimated_fee

def opportunity_score(prices):
    if len(prices) < 80:
        return 0
    vol = volatility(prices)
    regime = detect_regime(prices)
    trend_strength = abs(np.mean(prices[-20:]) - np.mean(prices[-50:]))
    score = trend_strength * 0.5 + vol * 0.5
    if regime == "uptrend":
        score *= 1.2
    return score

# ========= ORDER =========

def place_order(pair, side, qty):
    body = {
        "market": pair,
        "side": side.lower(),
        "order_type": "market_order",
        "total_quantity": str(qty),
        "price_per_unit": "0",
        "timestamp": int(time.time()*1000)
    }
    return signed_post("/exchange/v1/orders/create", body)

# ========= BOT LOOP =========

def bot_loop():
    global running, status, investment_capital, equity_start, daily_loss_lock

    markets = get_inr_markets()

    for m in markets:
        price_data[m] = []
        cooldown[m] = 0

    bal = get_balances()
    equity_start = bal.get("INR", 0)

    while running:

        prices = get_prices()
        now = time.time()

        # update history
        for m in markets:
            if m in prices:
                price_data[m].append(prices[m])
                if len(price_data[m]) > 200:
                    price_data[m].pop(0)

        bal = get_balances()
        current_equity = bal.get("INR", 0)

        # Kill switch
        if equity_start > 0:
            dd = (equity_start - current_equity) / equity_start
            if dd > MAX_DAILY_LOSS:
                daily_loss_lock = True
                status["msg"] = "Daily Loss Limit Hit"
                break

        # Manage open trades
        for pair in list(entry_positions.keys()):
            price = prices.get(pair)
            if not price:
                continue

            side, qty, tp, sl = entry_positions[pair]

            if price >= tp or price <= sl:
                place_order(pair, "SELL", qty)
                del entry_positions[pair]
                cooldown[pair] = now + COOLDOWN_SEC

        if daily_loss_lock:
            break

        # Scan opportunities
        scored = []
        for m in markets:
            if now < cooldown[m]:
                continue

            if m not in price_data:
                continue

            score = opportunity_score(price_data[m])
            if score > 0:
                scored.append((m, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:3]

        if investment_capital > 0 and top:
            total_score = sum(x[1] for x in top)

            for m, score in top:
                if m in entry_positions:
                    continue

                price = prices.get(m)
                if not price:
                    continue

                vol = volatility(price_data[m])
                if not volatility_cluster(price_data[m]):
                    continue
                if not fee_filter(price, vol):
                    continue
                if detect_regime(price_data[m]) != "uptrend":
                    continue

                weight = score / total_score
                capital = investment_capital * weight
                qty = round((capital * RISK_PER_TRADE) / price, 6)

                if qty <= 0:
                    continue

                res = place_order(m, "BUY", qty)
                if "id" in res:
                    tp = price + vol * 4
                    sl = price - vol * 2
                    entry_positions[m] = ("BUY", qty, tp, sl)

        status["msg"] = "Running"
        status["last"] = ist_now()
        time.sleep(6)

    status["msg"] = "Stopped"

# ========= ROUTES =========

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/set_capital", methods=["POST"])
def set_capital():
    global investment_capital
    data = request.json
    investment_capital = float(data.get("capital", 0))
    return jsonify({"status": "capital_set"})

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
        "capital": investment_capital,
        "error": error_message
    })

@app.route("/ping")
def ping():
    return "pong"

# ========= KEEP ALIVE =========

def self_keepalive():
    while True:
        try:
            requests.get("https://coin-4k37.onrender.com", timeout=10)
        except:
            pass
        time.sleep(240)

threading.Thread(target=self_keepalive, daemon=True).start()

# ========= RUN =========

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)