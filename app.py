import os, time, threading, hmac, hashlib, requests, json
from flask import Flask, render_template, jsonify, request
from datetime import datetime
from pytz import timezone
import numpy as np
import websocket

app = Flask(__name__)

# ================= CONFIG =================

API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET", "").encode()

BASE_URL = "https://api.coindcx.com"
WS_URL = "wss://stream.coindcx.com"

IST = timezone("Asia/Kolkata")

running = False
investment_capital = 0
error_message = ""

price_data = {}
entry_positions = {}
markets = []

trade_history = []
realised_pnl = 0
wins = 0
losses = 0

MAX_OPEN_TRADES = 3
RISK_PER_TRADE = 0.10
FEE_RATE = 0.006  # approx 0.6% round trip

# ================= AUTH =================

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
        r = requests.post(BASE_URL + endpoint, headers=headers, data=payload, timeout=10)
        if r.ok:
            error_message = ""
            return r.json()
        else:
            error_message = f"{r.status_code}: {r.text}"
    except Exception as e:
        error_message = str(e)
    return {}

# ================= MARKET =================

def get_inr_markets():
    try:
        r = requests.get(BASE_URL + "/exchange/ticker", timeout=10)
        if not r.ok:
            return []
        return [x["market"] for x in r.json() if x["market"].endswith("INR")]
    except:
        return []

def volatility(prices):
    if len(prices) < 20:
        return 0
    return np.std(np.diff(prices[-20:]))

def micro_breakout(prices):
    if len(prices) < 30:
        return None
    last = prices[-1]
    high = max(prices[-10:-1])
    if last > high:
        return "BUY"
    return None

def fee_filter(price, vol):
    expected_move = vol * 3
    estimated_fee = price * FEE_RATE
    return expected_move > estimated_fee

# ================= ORDER =================

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

# ================= PNL =================

def calculate_open_pnl():
    total = 0
    for pair in entry_positions:
        side, qty, entry_price, tp, sl = entry_positions[pair]
        if pair in price_data and price_data[pair]:
            current = price_data[pair][-1]
            pnl = (current - entry_price) * qty
            total += pnl
    return round(total, 2)

def win_rate():
    total = wins + losses
    if total == 0:
        return 0
    return round((wins / total) * 100, 2)

# ================= TICK =================

def process_tick(pair, price):
    global realised_pnl, wins, losses

    if pair not in price_data:
        price_data[pair] = []

    price_data[pair].append(price)
    if len(price_data[pair]) > 120:
        price_data[pair].pop(0)

    # Manage open position
    if pair in entry_positions:
        side, qty, entry_price, tp, sl = entry_positions[pair]

        if price >= tp or price <= sl:
            place_order(pair, "SELL", qty)

            pnl = (price - entry_price) * qty
            pnl -= (entry_price * qty * FEE_RATE)

            realised_pnl += pnl

            if pnl > 0:
                wins += 1
            else:
                losses += 1

            trade_history.append({
                "pair": pair,
                "entry": entry_price,
                "exit": price,
                "pnl": round(pnl, 2),
                "time": datetime.now(IST).strftime("%H:%M:%S")
            })

            del entry_positions[pair]
        return

    if len(entry_positions) >= MAX_OPEN_TRADES:
        return

    signal = micro_breakout(price_data[pair])
    if not signal:
        return

    vol = volatility(price_data[pair])
    if vol == 0:
        return

    if not fee_filter(price, vol):
        return

    capital = investment_capital * RISK_PER_TRADE
    qty = round(capital / price, 6)
    if qty <= 0:
        return

    res = place_order(pair, "BUY", qty)
    if "id" in res:
        tp = price + vol * 3
        sl = price - vol * 1.5
        entry_positions[pair] = ("BUY", qty, price, tp, sl)

# ================= WEBSOCKET =================

def start_ws():
    def on_message(ws, message):
        data = json.loads(message)
        if "data" not in data:
            return
        pair = data.get("s")
        price = float(data.get("p", 0))
        if pair in markets:
            process_tick(pair, price)

    def on_open(ws):
        subs = {
            "action": "subscribe",
            "channels": [{"name": "ticker", "symbols": markets}]
        }
        ws.send(json.dumps(subs))

    while running:
        try:
            ws = websocket.WebSocketApp(
                WS_URL,
                on_message=on_message,
                on_open=on_open
            )
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except:
            time.sleep(5)

# ================= ROUTES =================

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
    global running, markets
    if not running:
        running = True
        markets = get_inr_markets()
        threading.Thread(target=start_ws, daemon=True).start()
    return jsonify({"status": "started"})

@app.route("/stop", methods=["POST"])
def stop():
    global running
    running = False
    return jsonify({"status": "stopped"})

@app.route("/dashboard")
def dashboard():
    return jsonify({
        "running": running,
        "capital": investment_capital,
        "open_trades": entry_positions,
        "open_pnl": calculate_open_pnl(),
        "realised_pnl": round(realised_pnl, 2),
        "win_rate": win_rate(),
        "wins": wins,
        "losses": losses,
        "trade_history": trade_history[-20:],
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

# ================= RUN =================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)