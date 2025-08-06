import os
import time
import hmac
import hashlib
import requests
import json
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET", "").encode()
BASE_URL = "https://api.coindcx.com"

def hmac_signature(payload):
    return hmac.new(API_SECRET, payload.encode(), hashlib.sha256).hexdigest()

def get_headers(payload):
    signature = hmac_signature(payload)
    return {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": signature,
        "Content-Type": "application/json"
    }

def get_wallet_balances():
    payload = '{"timestamp":' + str(int(time.time() * 1000)) + '}'
    headers = get_headers(payload)
    response = requests.post(BASE_URL + "/api/v1/user/balances", data=payload, headers=headers)
    return {item['currency']: float(item['balance']) for item in response.json()}

def place_market_order(side, pair, quantity):
    payload = {
        "side": side,
        "order_type": "market_order",
        "market": pair,
        "total_quantity": quantity,
        "timestamp": int(time.time() * 1000)
    }
    payload_json = json.dumps(payload)
    headers = get_headers(payload_json)
    response = requests.post(BASE_URL + "/api/v1/orders/create", data=payload_json, headers=headers)
    return response.json()

def log_trade(entry):
    entry["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open("trade_log.json", "a") as f:
        f.write(json.dumps(entry) + "\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/buy", methods=["POST"])
def buy():
    data = request.json
    pair = data["pair"]
    price = float(data["candles"][-1]["close"])
    balances = get_wallet_balances()
    usdt = balances.get("USDT", 0.0)
    qty = round((usdt * 0.3) / price, 6)
    order = place_market_order("buy", pair, qty)
    log_trade({ "pair": pair, "side": "buy", "qty": qty, "price": price, "candles": data["candles"] })
    return jsonify(order)

@app.route("/api/sell", methods=["POST"])
def sell():
    data = request.json
    pair = data["pair"]
    coin = pair.replace("USDT", "")
    balances = get_wallet_balances()
    qty = balances.get(coin, 0.0)
    price = float(data["candles"][-1]["close"])
    order = place_market_order("sell", pair, qty)
    log_trade({ "pair": pair, "side": "sell", "qty": qty, "price": price, "candles": data["candles"] })
    return jsonify(order
def pa_buy_sell_signal(pair):
    candles = candle_logs[pair]
    if len(candles) < 2:
        return None
    prev1 = candles[-1]
    prev2 = candles[-2]
    current_price = tick_logs[pair][-1][1] if tick_logs[pair] else None

    if not current_price:
        return None

    highest_high = max(prev1["high"], prev2["high"])
    lowest_low = min(prev1["low"], prev2["low"])

    # BUY signal: price breaks above highest high
    if current_price > highest_high:
        return {"side": "BUY", "entry": current_price, "msg": "Price crossed above 2-candle high"}

    # SELL signal: price drops below lowest low
    if current_price < lowest_low:
        return {"side": "SELL", "entry": current_price, "msg": "Price crossed below 2-candle low"}

    return None



                signal = pa_buy_sell_signal(pair)
                if signal and usdt > 5:
                    # Determine quantity
                    qty = 0
                    if signal['side'] == "BUY":
                        qty = (0.3 * usdt) / signal['entry']
                    elif signal['side'] == "SELL":
                        for b in balances:
                            if b['currency'] == pair.replace("USDT", ""):
                                qty = float(b['balance'])

                    qty = adjust_quantity_precision(pair, qty)
                    tp = round(signal['entry'] * 1.0005, 6)  # +0.05%
                    sl = round(signal['entry'] * 0.999, 6)   # -0.1%
                    result = place_order(pair, signal['side'], qty)

                    scan_log.append(f"{datetime.utcfromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S')} | {pair} | SIGNAL: {signal['side']} @ {signal['entry']} ({signal['msg']}) | ORDER: {result}")

                    trade = {
                        "ts": datetime.utcfromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S'),
                        "symbol": pair,
                        "side": signal['side'],
                        "entry": signal['entry'],
                        "msg": signal['msg'],
                        "tp": tp,
                        "sl": sl,
                        "qty": qty,
                        "order_result": result
                    }

                    trade_log.append(trade)
                    exit_orders.append({
                        "pair": pair,
                        "side": signal['side'],
                        "qty": qty,
                        "tp": tp,
                        "sl": sl,
                        "entry": signal['entry']
                    })

                    )