import os
import time
import hmac
import json
import hashlib
import logging
import requests
from flask import Flask, jsonify

# ===============================
# CONFIG
# ===============================

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

BASE_URL = "https://api.coindcx.com"

SYMBOL = "BTCUSDT"
TRADE_AMOUNT_USDT = 10

# ===============================
# LOGGING
# ===============================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ===============================
# FLASK APP
# ===============================

app = Flask(__name__)

bot_status = {
    "running": False,
    "last_price": None,
    "last_action": "NONE",
    "pnl": 0
}

# ===============================
# AUTH HELPERS
# ===============================

def sign(payload):
    signature = hmac.new(
        API_SECRET.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    return signature

def private_post(endpoint, body):
    payload = json.dumps(body, separators=(',', ':'))

    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": sign(payload),
        "Content-Type": "application/json"
    }

    r = requests.post(BASE_URL + endpoint, data=payload, headers=headers)
    return r.json()

# ===============================
# MARKET DATA
# ===============================

def get_price():
    url = f"{BASE_URL}/exchange/ticker"
    r = requests.get(url)

    data = r.json()

    for coin in data:
        if coin["market"] == SYMBOL:
            return float(coin["last_price"])

    return None

# ===============================
# ORDER FUNCTIONS
# ===============================

def place_market_order(side):
    body = {
        "side": side,
        "order_type": "market_order",
        "market": SYMBOL,
        "total_quantity": TRADE_AMOUNT_USDT,
        "timestamp": int(time.time() * 1000)
    }

    return private_post("/exchange/v1/orders/create", body)

# ===============================
# SIMPLE STRATEGY
# ===============================

last_price = None

def strategy_loop():
    global last_price

    while True:
        try:
            price = get_price()

            if price is None:
                time.sleep(10)
                continue

            bot_status["last_price"] = price

            if last_price:
                change = (price - last_price) / last_price

                # BUY condition
                if change < -0.002:
                    place_market_order("buy")
                    bot_status["last_action"] = "BUY"

                # SELL condition
                elif change > 0.002:
                    place_market_order("sell")
                    bot_status["last_action"] = "SELL"

            last_price = price

        except Exception as e:
            logging.error(f"Bot error: {e}")

        time.sleep(15)

# ===============================
# BACKGROUND BOT START
# ===============================

from threading import Thread

def start_bot():
    if not bot_status["running"]:
        bot_status["running"] = True
        Thread(target=strategy_loop, daemon=True).start()

# ===============================
# ROUTES
# ===============================

@app.route("/")
def home():
    return f"""
    <html>
    <head>
        <title>CoinDCX Pro Bot</title>
        <style>
            body {{
                background:#0f172a;
                color:#e2e8f0;
                font-family:Arial;
                text-align:center;
                padding-top:60px;
            }}
            .card {{
                background:#1e293b;
                padding:30px;
                border-radius:12px;
                display:inline-block;
            }}
            h1 {{color:#38bdf8;}}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🚀 CoinDCX Pro Bot</h1>
            <p>Status: {"RUNNING" if bot_status["running"] else "STOPPED"}</p>
            <p>Last Price: {bot_status["last_price"]}</p>
            <p>Last Action: {bot_status["last_action"]}</p>
            <p><a href="/start" style="color:#22c55e;">Start Bot</a></p>
            <p><a href="/status" style="color:#facc15;">API Status</a></p>
        </div>
    </body>
    </html>
    """

@app.route("/start")
def start():
    start_bot()
    return jsonify({"message": "Bot started"})

@app.route("/status")
def status():
    return jsonify(bot_status)

@app.route("/health")
def health():
    return "OK", 200

# ===============================
# RENDER PORT HANDLING
# ===============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)