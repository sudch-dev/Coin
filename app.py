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
api_secret = os.environ.get("API_SECRET")

def get_wallet_balance():
    # Step 1: Generate timestamp
    timestamp = str(int(time.time() * 1000))

    # Step 2: Payload string for HMAC signature (must be a string, not JSON)
    payload = f"timestamp={timestamp}"

    # Step 3: Generate HMAC SHA256 signature
    signature = hmac.new(
        api_secret.encode(),  # convert secret to bytes
        payload.encode(),     # encode payload
        hashlib.sha256
    ).hexdigest()

    # Step 4: Set headers
    headers = {
        "X-AUTH-APIKEY": api_key,
        "X-AUTH-SIGNATURE": signature,
        "Content-Type": "application/json"
    }

    # Step 5: Actual request body as JSON (with timestamp only)
    body = {
        "timestamp": int(timestamp)
    }

    try:
        response = requests.post(
            f"{BASE_URL}/exchange/v1/users/me/accounts/balances",
            headers=headers,
            data=json.dumps(body)
        )
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
