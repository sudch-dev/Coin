
import os
import time
import hashlib
import hmac
import json
import requests

# Base URL for authenticated API calls
BASE_URL = "https://api.coindcx.com"

# Access API key and secret from environment variables
api_key = os.environ.get("API_KEY")
api_secret = os.environ.get("API_SECRET").encode()  # Must be bytes for HMAC

def get_wallet_balance():
    payload = {
        "timestamp": int(time.time() * 1000)
    }
    json_payload = json.dumps(payload)

    # Generate signature using HMAC SHA256
    signature = hmac.new(api_secret, json_payload.encode(), hashlib.sha256).hexdigest()

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
        print(f"Error fetching wallet balance: {e}")
        return None

if __name__ == "__main__":
    if not api_key or not api_secret:
        print("Error: API_KEY or API_SECRET not set in environment variables.")
    else:
        balances = get_wallet_balance()
        if balances:
            print("Wallet Balances:")
            for balance in balances:
                print(f"  {balance['currency']}: {balance['balance']}")
