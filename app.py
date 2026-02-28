import time
import hmac
import hashlib
import requests
import pandas as pd
from flask import Flask, jsonify, request, render_template

# ================= CONFIG =================

API_KEY = "YOUR_API_KEY"
API_SECRET = "YOUR_API_SECRET"

BASE_URL = "https://api.coindcx.com"

SYMBOL = "ORDIUSDT"
LEVERAGE = 5

RISK_PER_TRADE = 0.01        # 1% equity risk
MAX_DAILY_LOSS = 0.03        # 3% kill switch
TP_MULTIPLIER = 2.0          # RR 1:2

CHECK_INTERVAL = 15          # seconds

# ==========================================

app = Flask(__name__)

position = None
entry_price = 0
equity = 1000          # replace with real wallet fetch if desired
day_pnl = 0
trades = []
running = True

# ================= AUTH ====================

def sign(payload):
    return hmac.new(
        API_SECRET.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()

def private_post(endpoint, body):
    payload = json.dumps(body)
    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": sign(payload),
        "Content-Type": "application/json"
    }
    return requests.post(BASE_URL + endpoint, data=payload, headers=headers).json()

# ================= DATA ====================

def get_price():
    r = requests.get(f"{BASE_URL}/exchange/ticker")
    data = r.json()
    for s in data:
        if s["market"] == SYMBOL:
            return float(s["last_price"])
    return None

def get_candles():
    url = f"{BASE_URL}/market_data/candles?pair={SYMBOL}&interval=1m"
    r = requests.get(url)
    df = pd.DataFrame(r.json(),
        columns=["time","open","high","low","close","volume"])
    df["close"] = df["close"].astype(float)
    return df

# ================= INDICATORS ==============

def psar_signal(df):
    return df["close"].iloc[-1] > df["close"].rolling(5).mean().iloc[-1]

def atr(df):
    tr = df["high"] - df["low"]
    return tr.rolling(14).mean().iloc[-1]

# ================= RISK ====================

def position_size(price, stop_dist):
    risk_amount = equity * RISK_PER_TRADE
    qty = risk_amount / stop_dist
    return round(qty, 3)

def check_kill():
    return day_pnl <= -equity * MAX_DAILY_LOSS

# ================= TRADING =================

def place_order(side, qty):
    body = {
        "side": side,
        "order_type": "market_order",
        "market": SYMBOL,
        "quantity": qty,
        "timestamp": int(time.time()*1000)
    }
    return private_post("/exchange/v1/orders/create", body)

def flatten():
    global position
    if position == "LONG":
        place_order("sell", 999999)
    elif position == "SHORT":
        place_order("buy", 999999)
    position = None

# ================= STRATEGY LOOP ===========

def run_bot():
    global position, entry_price, day_pnl, equity

    while running:
        try:

            if check_kill():
                print("DAILY LOSS LIMIT HIT")
                flatten()
                break

            price = get_price()
            df = get_candles()

            signal_long = psar_signal(df)
            vol = atr(df)

            stop_dist = vol
            qty = position_size(price, stop_dist)

            # ----- ENTRY -----
            if position is None:

                if signal_long:
                    place_order("buy", qty)
                    position = "LONG"
                    entry_price = price

                else:
                    place_order("sell", qty)
                    position = "SHORT"
                    entry_price = price

            # ----- EXIT -----
            else:
                pnl = (price - entry_price) * qty
                if position == "SHORT":
                    pnl = -pnl

                if pnl > vol * TP_MULTIPLIER or pnl < -vol:
                    flatten()
                    day_pnl += pnl
                    equity += pnl

                    trades.append({
                        "side": position,
                        "pnl": round(pnl,2),
                        "price": price
                    })

        except Exception as e:
            print("ERROR:", e)

        time.sleep(CHECK_INTERVAL)

# ================= UI ROUTES ===============

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/status")
def status():
    return jsonify({
        "position": position,
        "entry": entry_price,
        "equity": equity,
        "day_pnl": day_pnl,
        "trades": trades[-10:]
    })

@app.route("/flatten", methods=["POST"])
def manual_flatten():
    flatten()
    return jsonify({"status": "flattened"})

# ===========================================

if __name__ == "__main__":
    import threading
    threading.Thread(target=run_bot).start()
    app.run(port=5000)