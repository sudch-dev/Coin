import os
import time
import threading
import requests
import pandas as pd
from flask import Flask, render_template, request, jsonify
from binance.client import Client

# ==============================
# CONFIG
# ==============================

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

RENDER_URL = "https://coin-4k37.onrender.com"

SYMBOLS = [
    "BTCINR",
    "ETHINR",
    "SOLINR"
]

TIMEFRAME = Client.KLINE_INTERVAL_15MINUTE

capital = 0
in_position = {}
entry_price = {}

# ==============================
# BINANCE CLIENT
# ==============================

client = Client(API_KEY, API_SECRET)

# ==============================
# FLASK
# ==============================

app = Flask(__name__)

# ==============================
# KEEP ALIVE
# ==============================

@app.route("/ping")
def ping():
    return "pong"

def self_keepalive():
    while True:
        try:
            requests.get(f"{RENDER_URL}/ping", timeout=5)
        except:
            pass
        time.sleep(240)

threading.Thread(target=self_keepalive, daemon=True).start()

# ==============================
# DATA FETCH
# ==============================

def get_klines(symbol):

    klines = client.get_klines(
        symbol=symbol,
        interval=TIMEFRAME,
        limit=200
    )

    df = pd.DataFrame(klines, columns=[
        'time','open','high','low','close','volume',
        'ct','qav','n','tbbav','tbqav','ignore'
    ])

    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)

    return df

# ==============================
# INDICATORS
# ==============================

def ema(df, period):

    return df['close'].ewm(span=period).mean()

# ==============================
# ORDER BLOCK DETECTION
# ==============================

def detect_ob(df):

    last = df.iloc[-2]

    if last['close'] < last['open']:
        return "bullish"

    if last['close'] > last['open']:
        return "bearish"

    return None

# ==============================
# BOS DETECTION
# ==============================

def detect_bos(df):

    prev_high = df['high'].iloc[-3]
    prev_low = df['low'].iloc[-3]

    last_close = df['close'].iloc[-1]

    if last_close > prev_high:
        return "bullish"

    if last_close < prev_low:
        return "bearish"

    return None

# ==============================
# CHOCH DETECTION
# ==============================

def detect_choch(df):

    prev_low = df['low'].iloc[-3]
    prev_high = df['high'].iloc[-3]

    last_close = df['close'].iloc[-1]

    if last_close < prev_low:
        return "bearish"

    if last_close > prev_high:
        return "bullish"

    return None

# ==============================
# ENTRY LOGIC
# ==============================

def entry_logic(symbol):

    df = get_klines(symbol)

    df['ema9'] = ema(df,9)
    df['ema21'] = ema(df,21)

    ob = detect_ob(df)

    ema_cross = df['ema9'].iloc[-1] > df['ema21'].iloc[-1]

    if ob == "bullish" and ema_cross:
        return "buy"

    if ob == "bearish" and not ema_cross:
        return "sell"

    return None

# ==============================
# EXIT LOGIC
# ==============================

def exit_logic(symbol):

    df = get_klines(symbol)

    choch = detect_choch(df)
    ob = detect_ob(df)

    if choch:
        return True

    if ob == "bearish" and in_position.get(symbol) == "buy":
        return True

    if ob == "bullish" and in_position.get(symbol) == "sell":
        return True

    return False

# ==============================
# CONTINUATION LOGIC
# ==============================

def continuation_logic(symbol):

    df = get_klines(symbol)

    bos = detect_bos(df)

    if bos == "bullish" and in_position.get(symbol) == "buy":
        return True

    if bos == "bearish" and in_position.get(symbol) == "sell":
        return True

    return False

# ==============================
# EXECUTION ENGINE
# ==============================

def trade_engine():

    global capital

    while True:

        if capital == 0:
            time.sleep(10)
            continue

        for symbol in SYMBOLS:

            try:

                if symbol not in in_position:

                    signal = entry_logic(symbol)

                    if signal:

                        price = float(client.get_symbol_ticker(symbol=symbol)['price'])

                        qty = round(capital / price, 4)

                        if signal == "buy":

                            client.order_market_buy(
                                symbol=symbol,
                                quantity=qty
                            )

                        else:

                            client.order_market_sell(
                                symbol=symbol,
                                quantity=qty
                            )

                        entry_price[symbol] = price
                        in_position[symbol] = signal

                else:

                    if exit_logic(symbol):

                        qty = round(capital / entry_price[symbol],4)

                        if in_position[symbol] == "buy":

                            client.order_market_sell(
                                symbol=symbol,
                                quantity=qty
                            )

                        else:

                            client.order_market_buy(
                                symbol=symbol,
                                quantity=qty
                            )

                        del in_position[symbol]

                    else:

                        continuation_logic(symbol)

            except Exception as e:

                print("ERROR:",e)

        time.sleep(30)

threading.Thread(target=trade_engine, daemon=True).start()

# ==============================
# ROUTES
# ==============================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/set_capital", methods=["POST"])
def set_capital():

    global capital

    capital = float(request.form["capital"])

    return jsonify({
        "status":"Capital Set",
        "capital":capital
    })

@app.route("/status")
def status():

    return jsonify({
        "capital":capital,
        "positions":in_position
    })

# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    port = int(os.environ.get("PORT",10000))

    app.run(host="0.0.0.0", port=port)