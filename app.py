import os
import time
import hmac
import json
import hashlib
import requests
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template_string
from datetime import datetime

app = Flask(__name__)

# ================= CONFIG =================

API_KEY = os.getenv("COINDCX_API_KEY")
API_SECRET = os.getenv("COINDCX_API_SECRET")

coins = ["BTCINR", "ETHINR", "XRPINR", "SOLINR"]
timeframe = "5m"

ema_len = 200
rsi_len = 14
atr_len = 14

risk_pct = 0.02
paper_balance = 10000

mode = "PAPER"
positions = {}
logs = []
start_time = datetime.now()

# ================= LOGGING =================

def log(msg):
    t = datetime.now().strftime("%H:%M:%S")
    logs.insert(0, f"[{t}] {msg}")
    if len(logs) > 60:
        logs.pop()

# ================= DATA =================

def get_data(symbol):

    params = {
        "pair": symbol,
        "interval": timeframe,
        "limit": 300
    }

    r = requests.get(
        "https://public.coindcx.com/market_data/candles",
        params=params,
        timeout=15
    )

    if r.status_code != 200:
        raise Exception(f"HTTP {r.status_code}")

    data = r.json()

    if not isinstance(data, list) or len(data) == 0:
        raise Exception(f"No data for {symbol}")

    df = pd.DataFrame(data)

    df = df[["open", "high", "low", "close"]].astype(float)

    # Indicators
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

# ================= SIGNAL =================

def signal(df):

    last = df.iloc[-1]

    if last["close"] > last["ema"] and last["rsi"] < 65:
        return "BUY"

    if last["close"] < last["ema"] and last["rsi"] > 35:
        return "SELL"

    return None

# ================= LIVE ORDER =================

def place_live_order(symbol, side, qty):

    url = "https://api.coindcx.com/exchange/v1/orders/create"

    body = {
        "side": side,
        "order_type": "market_order",
        "market": symbol,
        "total_quantity": qty,
        "timestamp": int(time.time() * 1000)
    }

    payload = json.dumps(body)
    signature = hmac.new(
        API_SECRET.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()

    headers = {
        "Content-Type": "application/json",
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": signature
    }

    r = requests.post(url, data=payload, headers=headers)

    return r.json()

# ================= TRADING LOOP =================

def trade():

    global paper_balance

    for c in coins:

        try:

            df = get_data(c)
            sig = signal(df)
            price = df["close"].iloc[-1]

            if c not in positions and sig:

                qty = round((paper_balance * risk_pct) / price, 6)

                sl = price - df["atr"].iloc[-1]
                tp = price + df["atr"].iloc[-1] * 2

                positions[c] = {
                    "side": sig,
                    "entry": price,
                    "sl": sl,
                    "tp": tp,
                    "qty": qty
                }

                if mode == "LIVE":
                    place_live_order(c, sig.lower(), qty)

                log(f"{mode} {sig} {c} @ {price:.2f}")

            # ===== MANAGE POSITION =====

            if c in positions:

                pos = positions[c]

                if pos["side"] == "BUY":

                    if price <= pos["sl"] or price >= pos["tp"]:

                        pnl = (price - pos["entry"]) * pos["qty"]

                        if mode == "LIVE":
                            place_live_order(c, "sell", pos["qty"])

                        paper_balance += pnl
                        log(f"EXIT {c} PnL={pnl:.2f}")

                        del positions[c]

                if pos["side"] == "SELL":

                    if price >= pos["sl"] or price <= pos["tp"]:

                        pnl = (pos["entry"] - price) * pos["qty"]

                        if mode == "LIVE":
                            place_live_order(c, "buy", pos["qty"])

                        paper_balance += pnl
                        log(f"EXIT {c} PnL={pnl:.2f}")

                        del positions[c]

        except Exception as e:
            log(f"DATA ERROR → {str(e)}")

# ================= BACKGROUND LOOP =================

import threading

def loop():
    log("BOT STARTED")
    while True:
        trade()
        time.sleep(30)

threading.Thread(target=loop, daemon=True).start()

# ================= UI =================

HTML = """
<h1>🚀 ULTRA AI BOT</h1>
<h2>Mode: {{mode}}</h2>

<button onclick="fetch('/toggle')">
Toggle Paper / Live
</button>

<p>Uptime: {{uptime}}</p>

<h3>Active Positions</h3>
<pre>{{positions}}</pre>

<h3>Recent Logs</h3>
<pre>{{logs}}</pre>
"""

@app.route("/")
def home():
    uptime = str(datetime.now() - start_time)
    return render_template_string(
        HTML,
        mode=mode,
        uptime=uptime,
        positions=json.dumps(positions, indent=2),
        logs="\n".join(logs[:20])
    )

@app.route("/toggle")
def toggle():
    global mode
    mode = "LIVE" if mode == "PAPER" else "PAPER"
    log(f"MODE → {mode}")
    return "OK"

@app.route("/status")
def status():
    return jsonify({
        "mode": mode,
        "positions": positions,
        "logs": logs[:20],
        "uptime": str(datetime.now() - start_time)
    })

# ================= RUN =================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)