import os
import time
import threading
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, jsonify

app = Flask(__name__)

# =========================
# CONFIG
# =========================

BASE_URL = "https://public.coindcx.com"

coins = ["btcusdt", "ethusdt", "xrpusdt", "solusdt"]

timeframe = "15m"

ema_len = 50
rsi_len = 14
atr_len = 14

risk_per_trade = 0.02
starting_balance = 10000

API_KEY = os.getenv("COINDCX_API_KEY")
API_SECRET = os.getenv("COINDCX_SECRET")

# =========================
# STATE
# =========================

balance = starting_balance
positions = {}
logs = []
mode = "PAPER"
start_time = datetime.utcnow()

# =========================
# LOGGING
# =========================

def log(msg):
    t = datetime.now().strftime("%H:%M:%S")
    entry = f"[{t}] {msg}"
    print(entry)
    logs.append(entry)
    if len(logs) > 200:
        logs.pop(0)

# =========================
# DATA FETCH (FIXED)
# =========================

def get_data(symbol):

    params = {
        "pair": symbol.lower(),
        "interval": timeframe,
        "limit": 300
    }

    r = requests.get(
        f"{BASE_URL}/market_data/candles",
        params=params,
        timeout=15
    )

    if r.status_code != 200:
        raise Exception(f"HTTP {r.status_code}")

    data = r.json()

    if not isinstance(data, list) or len(data) == 0:
        raise Exception(f"Bad API response: {data}")

    df = pd.DataFrame(data)

    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            raise Exception(f"Missing {c}")

    df = df.astype(float)

    # ===== INDICATORS =====

    df["ema"] = df["close"].ewm(span=ema_len).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(rsi_len).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_len).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"] - df["close"].shift()).abs()

    df["atr"] = np.maximum(tr1, np.maximum(tr2, tr3)).rolling(atr_len).mean()

    return df

# =========================
# STRATEGY
# =========================

def check_signal(df):

    last = df.iloc[-1]

    price = last["close"]
    ema = last["ema"]
    rsi = last["rsi"]
    atr = last["atr"]

    if np.isnan(atr):
        return None

    if price > ema and rsi < 65:
        return "BUY", atr

    if price < ema and rsi > 35:
        return "SELL", atr

    return None

# =========================
# EXECUTION
# =========================

def execute_trade(symbol, side, price, atr):

    global balance

    risk = balance * risk_per_trade
    qty = risk / atr

    if side == "BUY":
        sl = price - atr
        tp = price + atr * 2
    else:
        sl = price + atr
        tp = price - atr * 2

    positions[symbol] = {
        "side": side,
        "entry": price,
        "qty": qty,
        "sl": sl,
        "tp": tp
    }

    log(f"{mode} {side} {symbol.upper()} @ {price:.2f}")

    # LIVE MODE PLACE ORDER HERE (optional)

# =========================
# POSITION MANAGEMENT
# =========================

def manage_positions(symbol, price):

    global balance

    if symbol not in positions:
        return

    pos = positions[symbol]

    if pos["side"] == "BUY":

        if price <= pos["sl"]:
            pnl = (pos["sl"] - pos["entry"]) * pos["qty"]
            balance += pnl
            log(f"SL HIT {symbol} PnL={pnl:.2f}")
            del positions[symbol]

        elif price >= pos["tp"]:
            pnl = (pos["tp"] - pos["entry"]) * pos["qty"]
            balance += pnl
            log(f"TP HIT {symbol} PnL={pnl:.2f}")
            del positions[symbol]

    else:

        if price >= pos["sl"]:
            pnl = (pos["entry"] - pos["sl"]) * pos["qty"]
            balance += pnl
            log(f"SL HIT {symbol} PnL={pnl:.2f}")
            del positions[symbol]

        elif price <= pos["tp"]:
            pnl = (pos["entry"] - pos["tp"]) * pos["qty"]
            balance += pnl
            log(f"TP HIT {symbol} PnL={pnl:.2f}")
            del positions[symbol]

# =========================
# MAIN LOOP
# =========================

def trading_loop():

    log("BOT STARTED")

    while True:

        try:

            for symbol in coins:

                df = get_data(symbol)

                price = df.iloc[-1]["close"]

                manage_positions(symbol, price)

                if symbol not in positions:

                    sig = check_signal(df)

                    if sig:
                        side, atr = sig
                        execute_trade(symbol, side, price, atr)

        except Exception as e:
            log(f"DATA ERROR → {e}")

        time.sleep(30)

# =========================
# KEEP ALIVE
# =========================

def keep_alive():

    while True:
        try:
            requests.get("https://coin-4k37.onrender.com", timeout=10)
        except:
            pass
        time.sleep(300)

# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/status")
def status():

    uptime = datetime.utcnow() - start_time

    return jsonify({
        "mode": mode,
        "balance": round(balance, 2),
        "positions": positions,
        "logs": logs[-30:],
        "uptime": str(uptime)
    })

@app.route("/toggle")
def toggle():

    global mode

    mode = "LIVE" if mode == "PAPER" else "PAPER"

    log(f"MODE CHANGED → {mode}")

    return jsonify({"mode": mode})

# =========================
# START THREADS
# =========================

threading.Thread(target=trading_loop, daemon=True).start()
threading.Thread(target=keep_alive, daemon=True).start()

# =========================
# RUN
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)