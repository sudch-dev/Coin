import os
import time
import json
import hmac
import hashlib
import requests
from flask import Flask, jsonify, render_template

app = Flask(__name__)

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

BASE_URL = "https://api.coindcx.com"
SYMBOL = "ORDIUSDT"


# =========================
# SIGNED REQUEST
# =========================
def signed_request(endpoint, body):
    payload = json.dumps(body, separators=(',', ':'))

    signature = hmac.new(
        API_SECRET.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()

    headers = {
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": signature,
        "Content-Type": "application/json"
    }

    response = requests.post(
        BASE_URL + endpoint,
        data=payload,
        headers=headers
    )

    return response.json()


# =========================
# PRICE
# =========================
def get_price():
    try:
        r = requests.get(
            f"{BASE_URL}/exchange/ticker",
            params={"market": SYMBOL}
        )
        return float(r.json()["last_price"])
    except:
        return 0


# =========================
# FUTURES WALLET (REAL)
# =========================
def get_wallet():
    try:
        body = {"timestamp": int(time.time() * 1000)}
        res = signed_request(
            "/exchange/v1/derivatives/futures/balance",
            body
        )

        if isinstance(res, list):
            for asset in res:
                if asset.get("currency") == "USDT":
                    return float(asset.get("available_balance", 0))

        return 0

    except Exception as e:
        print("Wallet Error:", e)
        return 0


# =========================
# POSITION
# =========================
def get_position():
    try:
        body = {"timestamp": int(time.time() * 1000)}
        res = signed_request(
            "/exchange/v1/derivatives/futures/positions",
            body
        )

        if isinstance(res, list):
            for p in res:
                if p.get("symbol") == SYMBOL:
                    size = float(p.get("size", 0))

                    if size > 0:
                        return "LONG"
                    elif size < 0:
                        return "SHORT"

        return "NONE"

    except Exception as e:
        print("Position Error:", e)
        return "NONE"


# =========================
# PLACE ORDER
# =========================
def place_order(side):
    wallet = get_wallet()
    price = get_price()

    if wallet <= 0 or price <= 0:
        return {"error": "No balance or price"}

    qty = round((wallet * 0.9) / price, 3)

    body = {
        "timestamp": int(time.time() * 1000),
        "symbol": SYMBOL,
        "side": side,
        "type": "MARKET",
        "quantity": str(qty)
    }

    return signed_request(
        "/exchange/v1/derivatives/futures/orders/create",
        body
    )


# =========================
# CLOSE POSITION
# =========================
def close_position():
    pos = get_position()

    if pos == "LONG":
        return place_order("SELL")
    elif pos == "SHORT":
        return place_order("BUY")

    return {"message": "No position"}


# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/status")
def status():
    try:
        return jsonify({
            "price": get_price(),
            "wallet": get_wallet(),
            "position": get_position()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/long", methods=["POST"])
def long():
    return jsonify(place_order("BUY"))


@app.route("/short", methods=["POST"])
def short():
    return jsonify(place_order("SELL"))


@app.route("/close", methods=["POST"])
def close():
    return jsonify(close_position())


# =========================
# RUN (Render compatible)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)