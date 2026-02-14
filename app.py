from flask import Flask, jsonify, render_template, request
import threading
import time
import datetime
import requests
import os
import hmac
import hashlib
import json
import pandas as pd
import numpy as np

app = Flask(__name__)

# =========================================================
# 🔐 CONFIG
# =========================================================

API_KEY = os.environ.get("COINDCX_API_KEY")
API_SECRET = os.environ.get("COINDCX_API_SECRET")

BASE_URL = "https://api.coindcx.com"

symbol = "btcusdt"        # lowercase for CoinDCX candles
timeframe = "15m"
order_size_usdt = 20

ema_len = 200
rsi_len = 14
atr_len = 14
rsi_long_level = 40
atr_mult = 0.8
max_adds = 4
tp_atr_mult = 1.2
sl_atr_mult = 3.0

server_start_time = datetime.datetime.utcnow()

# =========================================================
# 📊 STATE
# =========================================================

in_position = False
entry_prices = []
total_qty = 0

trade_log = []

LIVE_TRADING = False  # SAFE DEFAULT


# =========================================================
# 📝 LOGGING
# =========================================================

def log(msg):
    ts = datetime.datetime.utcnow().strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    print(entry)

    trade_log.append(entry)
    if len(trade_log) > 300:
        trade_log.pop(0)


# =========================================================
# 🪙 API SIGN
# =========================================================

def sign(payload):
    secret_bytes = bytes(API_SECRET, "utf-8")
    json_payload = json.dumps(payload, separators=(",", ":"))
    return hmac.new(secret_bytes,
                    json_payload.encode(),
                    hashlib.sha256).hexdigest()


# =========================================================
# 🛒 ORDER FUNCTIONS
# =========================================================

def place_market_buy(usdt_amount):

    if not LIVE_TRADING:
        log(f"[PAPER] BUY {usdt_amount} USDT")
        return

    payload = {
        "side": "buy",
        "order_type": "market_order",
        "market": symbol,
        "total_quantity": usdt_amount,
        "timestamp": int(time.time() * 1000)
    }

    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": sign(payload)
    }

    r = requests.post(BASE_URL + "/exchange/v1/orders/create",
                      json=payload, headers=headers)

    log(f"LIVE BUY → {r.text}")


def place_market_sell(quantity):

    if not LIVE_TRADING:
        log(f"[PAPER] SELL qty={quantity:.6f}")
        return

    payload = {
        "side": "sell",
        "order_type": "market_order",
        "market": symbol,
        "total_quantity": quantity,
        "timestamp": int(time.time() * 1000)
    }

    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": sign(payload)
    }

    r = requests.post(BASE_URL + "/exchange/v1/orders/create",
                      json=payload, headers=headers)

    log(f"LIVE SELL → {r.text}")


# =========================================================
# 📈 SAFE MARKET DATA FETCH  (FIXED)
# =========================================================

def get_latest_data():

    url = f"{BASE_URL}/market_data/candles"

    params = {
        "pair": symbol,
        "interval": timeframe,
        "limit": 300
    }

    r = requests.get(url, params=params, timeout=15)

    if r.status_code != 200:
        raise Exception(f"HTTP {r.status_code}")

    data = r.json()

    # Validate format
    if not isinstance(data, list) or len(data) == 0:
        raise Exception(f"Invalid candle data: {data}")

    df = pd.DataFrame(data)

    needed = ["open", "high", "low", "close"]
    for col in needed:
        if col not in df.columns:
            raise Exception(f"Missing column {col}")

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

    return df.iloc[-1]


# =========================================================
# 🤖 TRADING LOOP
# =========================================================

def bot_loop():
    global in_position, entry_prices, total_qty

    while True:
        try:
            d = get_latest_data()

            price = d["close"]
            ema = d["ema"]
            rsi = d["rsi"]
            atr = d["atr"]

            log(f"P={price:.2f} EMA={ema:.2f} RSI={rsi:.1f}")

            # ENTRY
            if not in_position:

                if price > ema and rsi < rsi_long_level:
                    place_market_buy(order_size_usdt)

                    entry_prices = [price]
                    total_qty = order_size_usdt / price
                    in_position = True

                    log("ENTER LONG")

            else:
                avg = sum(entry_prices) / len(entry_prices)
                grid = atr * atr_mult

                # Averaging
                if len(entry_prices) <= max_adds:
                    if price <= avg - grid:
                        place_market_buy(order_size_usdt)
                        entry_prices.append(price)
                        total_qty += order_size_usdt / price
                        log("AVERAGING BUY")

                tp = avg + atr * tp_atr_mult
                sl = avg - atr * sl_atr_mult

                if price >= tp or price <= sl:
                    place_market_sell(total_qty)
                    log("EXIT POSITION")

                    in_position = False
                    entry_prices = []
                    total_qty = 0

        except Exception as e:
            log(f"DATA ERROR → {e}")

        time.sleep(60)


# Start bot thread
threading.Thread(target=bot_loop, daemon=True).start()


# =========================================================
# 🌐 WEB ROUTES
# =========================================================

@app.route("/")
def index():
    uptime = str(datetime.datetime.utcnow() - server_start_time)

    return render_template(
        "index.html",
        uptime=uptime,
        position=in_position,
        mode="LIVE" if LIVE_TRADING else "PAPER",
        logs=trade_log[-30:]
    )


@app.route("/toggle", methods=["POST"])
def toggle():
    global LIVE_TRADING
    LIVE_TRADING = not LIVE_TRADING
    log(f"MODE → {'LIVE' if LIVE_TRADING else 'PAPER'}")
    return ("", 204)


@app.route("/health")
def health():
    return jsonify({"status": "alive"})


@app.route("/logs")
def logs():
    return jsonify(trade_log)


# =========================================================
# 🚀 RUN
# =========================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0",
            port=int(os.environ.get("PORT", 10000)))