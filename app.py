import os
import time
import threading
import requests
import hmac
import hashlib
import json
import pandas as pd
import pandas_ta as ta
from flask import Flask, jsonify, render_template_string

# ================= ENV CONFIG =================

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

if not API_KEY or not API_SECRET:
    raise Exception("API_KEY and API_SECRET must be set in environment variables")

BASE = "https://api.coindcx.com"
SYMBOL = "BTCINR"

TRADE_INR = 1000          # INR per trade
PROFIT_TARGET = 0.01      # 1%

# ================= GLOBAL STATE =================

bot_running = False
status = "Stopped"
last_error = ""
last_action = "None"
entry_price = 0

# ================= AUTH =================

def sign(payload):
    body = json.dumps(payload, separators=(',', ':'))
    signature = hmac.new(
        API_SECRET.encode(),
        body.encode(),
        hashlib.sha256
    ).hexdigest()
    return body, signature

def private_api(endpoint, payload):
    payload["timestamp"] = int(time.time() * 1000)
    body, signature = sign(payload)

    headers = {
        "Content-Type": "application/json",
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": signature
    }

    r = requests.post(BASE + endpoint, data=body, headers=headers, timeout=10)
    return r.json()

# ================= MARKET DATA =================

def get_price():
    r = requests.get(f"{BASE}/exchange/ticker", timeout=10)
    for t in r.json():
        if t["market"] == SYMBOL:
            return float(t["last_price"])
    return None

def get_candles():
    url = f"{BASE}/exchange/v1/candles?pair={SYMBOL}&interval=1m"
    data = requests.get(url, timeout=10).json()

    df = pd.DataFrame(data)
    df.columns = ["time","open","high","low","close","volume"]
    df["close"] = df["close"].astype(float)
    return df

# ================= ACCOUNT =================

def get_balances():
    data = private_api("/exchange/v1/users/balances", {})
    bal = {b["currency"]: float(b["balance"]) for b in data}
    return bal

def get_orders():
    return private_api("/exchange/v1/orders/active_orders", {})

def get_trades():
    return private_api("/exchange/v1/orders/trade_history", {"limit": 10})

# ================= ORDER =================

def place_order(side, price):
    global last_action

    qty = round(TRADE_INR / price, 6)

    payload = {
        "side": side,
        "order_type": "market_order",
        "market": SYMBOL,
        "total_quantity": qty
    }

    res = private_api("/exchange/v1/orders/create", payload)

    if "id" in res:
        last_action = f"{side.upper()} @ ₹{price:.0f}"
    else:
        raise Exception(str(res))

# ================= STRATEGY =================

def strategy():
    global entry_price

    df = get_candles()

    df["ema_fast"] = ta.ema(df["close"], length=9)
    df["ema_slow"] = ta.ema(df["close"], length=21)
    df["rsi"] = ta.rsi(df["close"], length=14)

    last = df.iloc[-1]

    price = last["close"]
    fast = last["ema_fast"]
    slow = last["ema_slow"]
    rsi = last["rsi"]

    bal = get_balances()
    inr = bal.get("INR", 0)
    btc = bal.get("BTC", 0)

    # ===== BUY =====
    if btc < 0.0001:
        if fast > slow and 45 < rsi < 65 and inr > TRADE_INR:
            place_order("buy", price)
            entry_price = price

    # ===== SELL =====
    else:
        profit = (price - entry_price) / entry_price

        if rsi > 72 or fast < slow or profit > PROFIT_TARGET:
            place_order("sell", price)
            entry_price = 0

# ================= BOT LOOP =================

def bot_loop():
    global bot_running, status, last_error

    status = "Running"

    while bot_running:
        try:
            strategy()
            last_error = ""
        except Exception as e:
            last_error = str(e)
            status = "Error"

        time.sleep(60)

    status = "Stopped"

# ================= UI =================

app = Flask(__name__)

UI = """
<!DOCTYPE html>
<html>
<head>
<title>BTCINR Pro Bot</title>
<style>
body{background:#0f172a;color:white;font-family:Arial}
.card{background:#1e293b;padding:20px;margin:20px;border-radius:12px}
button{padding:12px 20px;font-size:16px;margin:5px;border:none;border-radius:8px}
.start{background:#16a34a;color:white}
.stop{background:#dc2626;color:white}
pre{white-space:pre-wrap}
</style>
</head>
<body>

<h1>BTCINR Trading Bot</h1>

<div class="card">
<h2>Status: <span id="status"></span></h2>
<p>Last Action: <span id="action"></span></p>
<p>Error: <span id="error"></span></p>

<button class="start" onclick="fetch('/start')">START</button>
<button class="stop" onclick="fetch('/stop')">STOP</button>
</div>

<div class="card">
<h2>Portfolio</h2>
<pre id="portfolio"></pre>
</div>

<div class="card">
<h2>Orders</h2>
<pre id="orders"></pre>
</div>

<div class="card">
<h2>Trades</h2>
<pre id="trades"></pre>
</div>

<script>
async function refresh(){
  const s = await fetch('/status').then(r=>r.json())
  document.getElementById('status').innerText = s.status
  document.getElementById('action').innerText = s.action
  document.getElementById('error').innerText = s.error

  document.getElementById('portfolio').innerText =
    JSON.stringify(await fetch('/portfolio').then(r=>r.json()),null,2)

  document.getElementById('orders').innerText =
    JSON.stringify(await fetch('/orders').then(r=>r.json()),null,2)

  document.getElementById('trades').innerText =
    JSON.stringify(await fetch('/trades').then(r=>r.json()),null,2)
}

setInterval(refresh, 5000)
</script>

</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(UI)

@app.route("/start")
def start():
    global bot_running
    if not bot_running:
        bot_running = True
        threading.Thread(target=bot_loop).start()
    return "Bot started"

@app.route("/stop")
def stop():
    global bot_running
    bot_running = False
    return "Bot stopped"

@app.route("/status")
def stat():
    return jsonify({
        "status": status,
        "action": last_action,
        "error": last_error
    })

@app.route("/portfolio")
def portfolio():
    return jsonify(get_balances())

@app.route("/orders")
def orders():
    return jsonify(get_orders())

@app.route("/trades")
def trades():
    return jsonify(get_trades())

# ================= RUN =================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)