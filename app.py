import os
import time
import json
import hmac
import hashlib
import requests
from flask import Flask, jsonify, render_template

app = Flask(__name__)

API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET", "").encode()

BASE = "https://api.coindcx.com"
SYMBOL = "ORDIUSDT"

# ========= SIGNATURE =========

def sign(payload):
    return hmac.new(API_SECRET, payload.encode(), hashlib.sha256).hexdigest()

def signed_post(endpoint, body):
    payload = json.dumps(body, separators=(',', ':'))

    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": sign(payload),
        "Content-Type": "application/json"
    }

    r = requests.post(BASE + endpoint, headers=headers, data=payload, timeout=10)

    try:
        return r.json()
    except:
        return {}

# ========= SAFE DATA EXTRACT =========

def extract_list(res):
    if isinstance(res, list):
        return res
    if isinstance(res, dict):
        if "data" in res and isinstance(res["data"], list):
            return res["data"]
    return []

# ========= PRICE =========

def get_price():
    try:
        r = requests.get(BASE + "/exchange/ticker", timeout=10)
        for x in r.json():
            if x.get("market") == SYMBOL:
                return float(x.get("last_price", 0))
    except:
        pass
    return 0

# ========= FUTURES WALLET =========

def get_wallet():
    body = {"timestamp": int(time.time() * 1000)}
    res = signed_post("/exchange/v1/derivatives/futures/balances", body)

    usdt = 0
    for x in extract_list(res):
        if x.get("currency") == "INR":
            usdt = float(x.get("balance", 0))
    return usdt

# ========= POSITION =========

def get_position():
    body = {"timestamp": int(time.time() * 1000)}
    res = signed_post("/exchange/v1/derivatives/futures/positions", body)

    side = "NONE"
    size = 0
    entry = 0

    for p in extract_list(res):
        if p.get("symbol") == SYMBOL:
            size = float(p.get("size", 0))
            entry = float(p.get("avg_price", 0))

            if size > 0:
                side = "LONG"
            elif size < 0:
                side = "SHORT"

    return side, abs(size), entry

# ========= ORDERS =========

def get_orders():
    body = {"timestamp": int(time.time() * 1000)}
    res = signed_post("/exchange/v1/derivatives/futures/orders", body)

    active = []

    for o in extract_list(res):
        if o.get("symbol") == SYMBOL and o.get("status") == "open":
            active.append(f"{o.get('side')} {o.get('price')}")

    return active

# ========= STATUS =========

@app.route("/status")
def status():

    price = get_price()
    wallet = get_wallet()
    side, size, entry = get_position()
    orders = get_orders()

    pnl = 0
    roe = 0

    if entry > 0:
        pnl = (price - entry) * size if side == "LONG" else (entry - price) * size
        roe = (pnl / wallet * 100) if wallet > 0 else 0

    return jsonify({
        "equity": wallet,
        "available": wallet,
        "margin": "--",
        "upl": round(pnl, 4),
        "rpl": 0,

        "side": side,
        "size": size,
        "entry": entry,
        "mark": price,
        "pnl": round(pnl, 4),
        "roe": round(roe, 2),
        "liq": "--",

        "orders": orders,
        "logs": ["System OK"]
    })

# ========= BOT CONTROL =========

running = False

@app.route("/start", methods=["POST"])
def start():
    global running
    running = True
    return jsonify({"status": "started"})

@app.route("/stop", methods=["POST"])
def stop():
    global running
    running = False
    return jsonify({"status": "stopped"})

# ========= UI =========

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ping")
def ping():
    return "pong"

# ========= RUN =========

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)