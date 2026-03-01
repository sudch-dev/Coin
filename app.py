import os, time, threading, hmac, hashlib, requests, json, statistics
from flask import Flask, render_template, jsonify
from datetime import datetime
from pytz import timezone

app = Flask(__name__)

# ========= CONFIG =========

API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET", "").encode()

BASE_URL = "https://api.coindcx.com"
PAIR = "BTCINR"

IST = timezone("Asia/Kolkata")

def ist_now():
    return datetime.now(IST).strftime("%H:%M:%S")

running = False
error_message = ""
status = {"msg": "Idle", "last": ""}

prices = []
entry = None
cooldown_until = 0

# Risk parameters
RISK_PER_TRADE = 0.08      # 8% of INR capital
MIN_RESERVE = 200          # Keep ₹200 untouched
COOLDOWN_SEC = 120         # 2-minute pause after exit

# ========= AUTH & API WRAPPERS =========

def sign(payload):
    return hmac.new(API_SECRET, payload.encode(), hashlib.sha256).hexdigest()

def signed_post(url, body, retries=3):
    """Modified with increased timeout and retry logic to prevent 'Read timed out' errors."""
    global error_message
    payload = json.dumps(body)
    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": sign(payload),
        "Content-Type": "application/json"
    }
    
    for i in range(retries):
        try:
            # Increased timeout from 10 to 30 seconds
            r = requests.post(url, headers=headers, data=payload, timeout=30)
            if r.ok:
                error_message = "" # Clear errors on success
                return r.json()
            else:
                error_message = f"API Error {r.status_code}: {r.text}"
        except requests.exceptions.RequestException as e:
            error_message = f"Retry {i+1}/{retries} - Connection Error: {str(e)}"
            time.sleep(2) # Wait before retrying
            
    return {}

# ========= PORTFOLIO =========

def get_balances():
    body = {"timestamp": int(time.time()*1000)}
    data = signed_post(f"{BASE_URL}/exchange/v1/users/balances", body)
    if isinstance(data, list):
        return {x["currency"]: float(x["balance"]) for x in data}
    return {}

# ========= MARKET =========

def get_price():
    """Fetches current market price with error handling."""
    try:
        r = requests.get(f"{BASE_URL}/exchange/ticker", timeout=15)
        if r.ok:
            for x in r.json():
                if x["market"] == PAIR:
                    return float(x["last_price"])
    except:
        pass
    return None

# ========= VOLATILITY =========

def volatility():
    if len(prices) < 20:
        return 0
    returns = [prices[i] - prices[i-1] for i in range(-20, -1)]
    return statistics.pstdev(returns)

# ========= SIGNAL ENGINE =========

def trade_signal(price):
    if len(prices) < 30:
        return None

    short = sum(prices[-5:]) / 5
    long = sum(prices[-20:]) / 20
    vol = volatility()

    # Regime filter: avoid dead markets
    if vol < 5:
        return None

    # Momentum breakout
    if short > long and price > max(prices[-10:-1]):
        return "BUY"

    if short < long and price < min(prices[-10:-1]):
        return "SELL"

    return None

# ========= ORDER =========

def place_order(side, qty):
    if qty <= 0: return {}
    body = {
        "market": PAIR,
        "side": side.lower(),
        "order_type": "market_order",
        "total_quantity": str(qty),
        "timestamp": int(time.time()*1000)
    }
    return signed_post(f"{BASE_URL}/exchange/v1/orders/create", body)

# ========= ENGINE =========

def bot_loop():
    global running, error_message, entry, cooldown_until

    while running:
        try:
            price = get_price()
            if not price:
                time.sleep(5)
                continue

            prices.append(price)
            if len(prices) > 200:
                prices.pop(0)

            now = time.time()

            # ===== ACTIVE TRADE MANAGEMENT =====
            if entry:
                side, qty, tp, sl = entry

                if side == "BUY":
                    if price >= tp or price <= sl:
                        place_order("SELL", qty)
                        entry = None
                        cooldown_until = now + COOLDOWN_SEC
                else:
                    if price <= tp or price >= sl:
                        place_order("BUY", qty)
                        entry = None
                        cooldown_until = now + COOLDOWN_SEC

            # ===== NEW TRADE SEARCH =====
            else:
                if now < cooldown_until:
                    status["msg"] = "Cooldown"
                else:
                    signal = trade_signal(price)
                    bal = get_balances()
                    inr = bal.get("INR", 0)

                    if signal and inr > MIN_RESERVE:
                        usable = inr - MIN_RESERVE
                        trade_capital = usable * RISK_PER_TRADE
                        qty = round(trade_capital / price, 6)

                        res = place_order(signal, qty)
                        
                        if res and "id" in res: # Confirm order success
                            vol = volatility()
                            tp = price + vol * 2 if signal == "BUY" else price - vol * 2
                            sl = price - vol if signal == "BUY" else price + vol
                            entry = (signal, qty, tp, sl)

            status["msg"] = "Running" if not entry else f"In Trade ({entry[0]})"
            status["last"] = ist_now()

        except Exception as e:
            error_message = f"Loop Error: {str(e)}"

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
        "BTC": bal.get("BTC", 0),
        "error": error_message
    })

@app.route("/ping")
def ping():
    return "pong"
    
def self_keepalive():
    """Prevents Render from sleeping."""
    while True:
        try:
            # Replace with your actual Render URL
            requests.get("https://coin-4k37.onrender.com", timeout=10)
        except:
            pass
        time.sleep(240)

threading.Thread(target=self_keepalive, daemon=True).start()

# ========= RUN =========

if __name__ == "__main__":
    # Ensure port is an integer for Render
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
