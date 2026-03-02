import os, time, threading, hmac, hashlib, requests, json, statistics
from flask import Flask, render_template, jsonify
from datetime import datetime
from pytz import timezone
import numpy as np

app = Flask(__name__)

# ========= CONFIG =========

API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET", "").encode()

BASE_URL = "https://api.coindcx.com"

PAIRS = ["BTCINR", "ETHINR", "SOLINR"]

IST = timezone("Asia/Kolkata")

def ist_now():
    return datetime.now(IST).strftime("%H:%M:%S")

running = False
error_message = ""
status = {"msg": "Idle", "last": ""}

price_data = {p: [] for p in PAIRS}
entry = None
cooldown_until = 0

RISK_PER_TRADE = 0.08
MIN_RESERVE = 200
COOLDOWN_SEC = 120

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

# ========= MARKET =========

def get_all_prices():
    prices = {}
    try:
        r = requests.get(BASE_URL + "/exchange/ticker", timeout=10)
        if r.ok:
            for x in r.json():
                if x["market"] in PAIRS:
                    prices[x["market"]] = float(x["last_price"])
    except:
        pass
    return prices

# ========= FEATURES =========

def volatility(prices):
    if len(prices) < 20:
        return 0
    returns = np.diff(prices[-20:])
    return np.std(returns)

def build_features(prices):
    if len(prices) < 30:
        return None

    short = np.mean(prices[-5:])
    long = np.mean(prices[-20:])
    momentum = prices[-1] - prices[-10]
    vol = volatility(prices)
    breakout = prices[-1] - max(prices[-10:-1])

    return np.array([short-long, momentum, vol, breakout])

# ========= AI/ML SCORING =========

def ml_score(features):
    """
    Lightweight logistic-style scoring
    (acts like trained model)
    """
    weights = np.array([0.8, 0.6, 0.5, 0.7])
    score = np.dot(features, weights)

    prob = 1 / (1 + np.exp(-score / 100))
    return prob

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
    global running, entry, cooldown_until, error_message

    while running:
        try:
            all_prices = get_all_prices()
            if not all_prices:
                time.sleep(5)
                continue

            # Update price history
            for p, price in all_prices.items():
                price_data[p].append(price)
                if len(price_data[p]) > 200:
                    price_data[p].pop(0)

            now = time.time()

            # ===== MANAGE ACTIVE TRADE =====
            if entry:
                pair, side, qty, tp, sl = entry
                price = all_prices.get(pair)

                if price:
                    if side == "BUY":
                        if price >= tp or price <= sl:
                            place_order(pair, "SELL", qty)
                            entry = None
                            cooldown_until = now + COOLDOWN_SEC
                    else:
                        if price <= tp or price >= sl:
                            place_order(pair, "BUY", qty)
                            entry = None
                            cooldown_until = now + COOLDOWN_SEC

            # ===== NEW TRADE SEARCH =====
            else:
                if now < cooldown_until:
                    status["msg"] = "Cooldown"

                else:
                    best_pair = None
                    best_prob = 0

                    for p in PAIRS:
                        feats = build_features(price_data[p])
                        if feats is None:
                            continue

                        prob = ml_score(feats)

                        if prob > best_prob:
                            best_prob = prob
                            best_pair = p

                    if best_pair and best_prob > 0.7:
                        price = all_prices[best_pair]
                        bal = get_balances()

                        inr = bal.get("INR", 0)
                        base_asset = best_pair.replace("INR", "")
                        asset_bal = bal.get(base_asset, 0)

                        usable = max(0, inr - MIN_RESERVE)
                        capital = usable * RISK_PER_TRADE
                        qty = round(capital / price, 6)

                        res = place_order(best_pair, "BUY", qty)

                        if "id" in res:
                            vol = volatility(price_data[best_pair])
                            tp = price + vol * 2
                            sl = price - vol

                            entry = (best_pair, "BUY", qty, tp, sl)

            status["msg"] = "Running" if not entry else f"In Trade {entry[0]}"
            status["last"] = ist_now()

        except Exception as e:
            error_message = str(e)

        time.sleep(5)

    status["msg"] = "Idle"

# ========= ROUTES =========

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
        "error": error_message
    })

@app.route("/ping")
def ping():
    return "pong"

# ========= KEEP ALIVE =========

def self_keepalive():
    while True:
        try:
            requests.get("https://smc-trading.onrender.com/ping", timeout=10)
        except:
            pass
        time.sleep(240)

threading.Thread(target=self_keepalive, daemon=True).start()

# ========= RUN =========

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)