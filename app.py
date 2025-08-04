from flask import Flask, render_template
import os
import time
import hashlib
import hmac
import json
import requests

app = Flask(__name__)
BASE_URL = "https://api.coindcx.com"

def get_wallet_balance():
    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")

    if not api_key or not api_secret:
        return [{"currency": "ERROR", "balance": "API key or secret not set in environment variables"}]

    payload = {
        "timestamp": int(time.time() * 1000)
    }
    json_payload = json.dumps(payload)

    signature = hmac.new(api_secret.encode(), json_payload.encode(), hashlib.sha256).hexdigest()

    headers = {
        "X-AUTH-APIKEY": api_key,
        "X-AUTH-SIGNATURE": signature,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(f"{BASE_URL}/exchange/v1/users/balances", headers=headers, data=json_payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return [{"currency": "ERROR", "balance": str(e)}]

@app.route("/")
def index():
    balances = get_wallet_balance()
    return render_template("index.html", balances=balances)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
