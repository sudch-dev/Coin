import os
import time
import threading
import hmac
import hashlib
import requests
import json
import math
from datetime import datetime
from pytz import timezone
from flask import Flask, render_template, jsonify

# =========================
# Flask
# =========================
app = Flask(__name__)

# =========================
# Config / Constants
# =========================
API_KEY = os.environ.get("API_KEY")
API_SECRET = (os.environ.get("API_SECRET") or "").encode()
BASE_URL = "https://api.coindcx.com"

# Spot pairs to scan
PAIRS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "SHIBUSDT", "SOLUSDT",
    "DOGEUSDT", "ADAUSDT", "AEROUSDT", "BNBUSDT", "LTCUSDT"
]

# Pair precision (quantity) + min qty (coin-wise)
PAIR_RULES = {
    "BTCUSDT": {"precision": 6, "min_qty": 0.0001},
    "ETHUSDT": {"precision": 6, "min_qty": 0.0001},
    "XRPUSDT": {"precision": 4, "min_qty": 0.1},
    "SHIBUSDT": {"precision": 0, "min_qty": 10000},
    "DOGEUSDT": {"precision": 2, "min_qty": 1},
    "SOLUSDT": {"precision": 3, "min_qty": 0.01},
    "AEROUSDT": {"precision": 4, "min_qty": 1},
    "ADAUSDT": {"precision": 2, "min_qty": 1},
    "LTCUSDT": {"precision": 2, "min_qty": 0.01},
    "BNBUSDT": {"precision": 4, "min_qty": 0.001},
}

# Candle interval -> strict 5s
CANDLE_INTERVAL_SEC = 5
# Poll cadence for REST
POLL_SEC = 1.0
# Allocation per BUY from USDT
BUY_USDT_ALLOC = 0.35

# =========================
# Time helpers (IST)
# =========================
IST = timezone("Asia/Kolkata")
def ist_now(): return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

# =========================
# State
# =========================
session = requests.Session()
tick_logs   = {p: [] for p in PAIRS}   # [(ts, price)]
candle_logs = {p: [] for p in PAIRS}   # dicts: open/high/low/close/volume/start
scan_log    = []                       # rolling text logs for UI
trade_log   = []                       # entries & exits
running     = False
status      = {"msg": "Idle", "last": ""}
error_message = ""

# Track open LONG positions (spot)
# positions[pair] = {"side":"BUY","qty":float,"entry":float,"entry_bar_start":int}
positions = {p: None for p in PAIRS}

# =========================
# Utils / API
# =========================
def hmac_signature(payload: str) -> str:
    return hmac.new(API_SECRET, payload.encode(), hashlib.sha256).hexdigest()

def get_wallet_balances():
    payload = json.dumps({"timestamp": int(time.time() * 1000)})
    sig = hmac_signature(payload)
    headers = {"X-AUTH-APIKEY": API_KEY, "X-AUTH-SIGNATURE": sig, "Content-Type": "application/json"}
    balances = {}
    try:
        r = session.post(f"{BASE_URL}/exchange/v1/users/balances", headers=headers, data=payload, timeout=8)
        if r.ok:
            for b in r.json():
                balances[b["currency"]] = float(b["balance"])
        else:
            scan_log.append(f"{ist_now()} | BAL_ERR: {r.status_code} {r.text[:200]}")
    except Exception as e:
        scan_log.append(f"{ist_now()} | BAL_ERR: {e}")
    return balances

def fetch_all_prices():
    try:
        r = session.get(f"{BASE_URL}/exchange/ticker", timeout=8)
        if r.ok:
            now = int(time.time())
            return {
                item["market"]: {"price": float(item["last_price"]), "ts": now}
                for item in r.json() if item.get("market") in PAIRS
            }
        else:
            scan_log.append(f"{ist_now()} | PRICE_ERR: {r.status_code} {r.text[:200]}")
    except Exception as e:
        scan_log.append(f"{ist_now()} | PRICE_ERR: {e}")
    return {}

def clamp_buy_qty(pair, qty):
    rule = PAIR_RULES.get(pair, {"precision": 6, "min_qty": 0.0})
    p = int(rule["precision"]); minq = float(rule["min_qty"])
    step = (10 ** (-p)) if p > 0 else 1.0
    qty = (int(max(qty, 0.0) / step)) * step  # floor
    if qty < minq: return 0.0
    return round(qty, p)

def precise_sell_qty(pair, balance_qty):
    rule = PAIR_RULES.get(pair, {"precision": 6, "min_qty": 0.0})
    p = int(rule["precision"]); minq = float(rule["min_qty"])
    step = (10 ** (-p)) if p > 0 else 1.0
    qty = (int(max(balance_qty, 0.0) / step)) * step  # floor to avoid insufficient balance
    if qty < minq: return 0.0
    return round(qty, p)

def place_order(pair, side, qty):
    payload = {
        "market": pair,
        "side": side.lower(),             # "buy" / "sell"
        "order_type": "market_order",
        "total_quantity": str(qty),
        "timestamp": int(time.time() * 1000),
    }
    body = json.dumps(payload)
    sig = hmac_signature(body)
    headers = {"X-AUTH-APIKEY": API_KEY, "X-AUTH-SIGNATURE": sig, "Content-Type": "application/json"}
    try:
        r = session.post(f"{BASE_URL}/exchange/v1/orders/create", headers=headers, data=body, timeout=8)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# =========================
# Indicators
# =========================
def ema_series(values, period):
    if len(values) < period:
        return []
    k = 2 / (period + 1)
    out = [float(values[0])]
    for v in values[1:]:
        out.append(float(v) * k + out[-1] * (1 - k))
    return out

def ema20_angle_deg_from_candles(candles):
    """
    Compute EMA20 on closes and return the angle (deg) between last two EMA points.
    Angle is atan( (ΔEMA) / (0.01 * EMA_prev) ) in degrees, so a +30° means ~+0.577% step.
    """
    closes = [c["close"] for c in candles]
    ema20 = ema_series(closes, 20)
    if len(ema20) < 2:
        return None
    prev = ema20[-2]
    curr = ema20[-1]
    if prev <= 0:
        return None
    slope_ratio = (curr - prev) / (0.01 * prev)  # ΔEMA per 1% of price
    return math.degrees(math.atan(slope_ratio))

# =========================
# Aggregation (5s candles)
# =========================
def aggregate_candles(pair, interval=CANDLE_INTERVAL_SEC):
    ticks = tick_logs[pair]
    if not ticks: return
    candles, candle, last_window = [], None, None
    for ts, price in sorted(ticks, key=lambda x: x[0]):
        wstart = ts - (ts % interval)
        if last_window != wstart:
            if candle: candles.append(candle)
            candle = {"open": price, "high": price, "low": price, "close": price, "volume": 1, "start": wstart}
            last_window = wstart
        else:
            candle["high"] = max(candle["high"], price)
            candle["low"]  = min(candle["low"], price)
            candle["close"]= price
            candle["volume"] += 1
    if candle: candles.append(candle)
    candle_logs[pair] = candles[-1500:]  # enough history

# =========================
# Signals per your rules
# =========================
def buy_signal(candles):
    """
    Conditions on last CLOSED candle (index -1), referencing previous (-2):
      1) EMA20 slope > +30°
      2) open < prev_close AND open >= prev_close * 0.999  (within -0.1%)
      3) last candle index is ODD (0-based)
    """
    if len(candles) < 22:
        return False
    idx = len(candles) - 1
    last = candles[-1]
    prev = candles[-2]

    # (1) slope
    ang = ema20_angle_deg_from_candles(candles)
    if ang is None or ang <= 30.0:
        return False

    # (2) open within -0.1% of prev close and below it
    if not (last["open"] < prev["close"] and last["open"] >= prev["close"] * 0.999):
        return False

    # (3) odd candle eligibility
    if (idx % 2) != 1:
        return False

    return True

def sell_signal(candles):
    """
    Reverse of buy:
      1) EMA20 slope < -30°
      2) open > prev_close AND open <= prev_close * 1.001  (within +0.1%)
      3) last candle index is EVEN (reverse of buy)
    """
    if len(candles) < 22:
        return False
    idx = len(candles) - 1
    last = candles[-1]
    prev = candles[-2]

    ang = ema20_angle_deg_from_candles(candles)
    if ang is None or ang >= -30.0:
        return False

    if not (last["open"] > prev["close"] and last["open"] <= prev["close"] * 1.001):
        return False

    if (idx % 2) != 0:
        return False

    return True

# =========================
# Trading Loop (5s candles)
# =========================
def scan_loop():
    global running, error_message
    scan_log.clear()
    last_candle_ts = {p: 0 for p in PAIRS}

    while running:
        prices = fetch_all_prices()
        now = int(time.time())
        balances = get_wallet_balances()

        for pair in PAIRS:
            info = prices.get(pair)
            if not info:
                continue

            price = info["price"]
            # Tick ingest
            tick_logs[pair].append((now, price))
            if len(tick_logs[pair]) > 20000:
                tick_logs[pair] = tick_logs[pair][-20000:]

            # Candle aggregation
            aggregate_candles(pair, CANDLE_INTERVAL_SEC)
            C = candle_logs[pair]
            if not C:
                continue

            # Act only when a *new* candle has closed
            if C[-1]["start"] == last_candle_ts[pair]:
                continue
            last_candle_ts[pair] = C[-1]["start"]

            # --- Exit rule for existing BUY: exit at the close of the next candle ---
            pos = positions.get(pair)
            if pos and pos["side"] == "BUY":
                # If we have moved to a candle whose start > entry_bar_start, that candle just closed
                if C[-1]["start"] > pos["entry_bar_start"]:
                    qty = precise_sell_qty(pair, pos["qty"])
                    if qty > 0:
                        res = place_order(pair, "SELL", qty)
                        scan_log.append(f"{ist_now()} | {pair} | EXIT (close of next bar): SELL {qty} @ {C[-1]['close']} | {res}")
                        trade_log.append({
                            "time": ist_now(), "pair": pair, "side": "SELL",
                            "entry": C[-1]["close"], "msg": "Exit at close of next candle",
                            "qty": qty, "order_result": res
                        })
                        positions[pair] = None
                        if "error" in res: error_message = res["error"]
                    # After exit, continue to evaluate fresh signals on this bar as well
            # -------------------------------------------------------------

            # Signals based on last closed candle
            will_buy = buy_signal(C)
            will_sell = sell_signal(C)

            # SELL (reverse): liquidate wallet if signal
            if will_sell:
                coin = pair[:-4]
                bal_qty = balances.get(coin, 0.0)
                qty = precise_sell_qty(pair, bal_qty)
                if qty > 0:
                    res = place_order(pair, "SELL", qty)
                    scan_log.append(f"{ist_now()} | {pair} | SELL signal: liquidate {qty} @ {C[-1]['close']} | {res}")
                    trade_log.append({
                        "time": ist_now(), "pair": pair, "side": "SELL",
                        "entry": C[-1]["close"], "msg": "SELL signal (reverse rules)",
                        "qty": qty, "order_result": res
                    })
                    positions[pair] = None
                    if "error" in res: error_message = res["error"]
                else:
                    scan_log.append(f"{ist_now()} | {pair} | SELL signal but no {coin} balance ≥ min qty")

            # BUY if signal (and not already long)
            if will_buy:
                pos = positions.get(pair)
                if pos and pos["side"] == "BUY" and pos["qty"] > 0:
                    scan_log.append(f"{ist_now()} | {pair} | BUY signal but already long; skip")
                else:
                    usdt = balances.get("USDT", 0.0)
                    close_px = C[-1]["close"]
                    if close_px > 0 and usdt > 0:
                        raw_qty = (BUY_USDT_ALLOC * usdt) / close_px
                        qty = clamp_buy_qty(pair, raw_qty)
                        if qty > 0:
                            res = place_order(pair, "BUY", qty)
                            # We exit at close of next candle -> remember the entry bar start
                            positions[pair] = {
                                "side": "BUY", "qty": qty, "entry": close_px,
                                "entry_bar_start": C[-1]["start"]
                            }
                            scan_log.append(f"{ist_now()} | {pair} | BUY {qty} @ {close_px} | Exit next bar close | {res}")
                            trade_log.append({
                                "time": ist_now(), "pair": pair, "side": "BUY",
                                "entry": close_px, "msg": "BUY signal (odd bar + slope + open condition)",
                                "qty": qty, "order_result": res
                            })
                            if "error" in res: error_message = res["error"]
                        else:
                            scan_log.append(f"{ist_now()} | {pair} | BUY signal but qty < min after clamp")
                    else:
                        scan_log.append(f"{ist_now()} | {pair} | BUY sig but no USDT or bad price")

            if not will_buy and not will_sell and not positions.get(pair):
                scan_log.append(f"{ist_now()} | {pair} | No Signal")

        # Trim logs to keep memory bounded
        if len(scan_log) > 600: scan_log[:] = scan_log[-600:]
        if len(trade_log) > 200: trade_log[:] = trade_log[-200:]

        status["msg"], status["last"] = "Running", ist_now()
        time.sleep(POLL_SEC)

    status["msg"] = "Idle"

# =========================
# Routes
# =========================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start", methods=["POST"])
def start():
    global running
    if not running:
        running = True
        t = threading.Thread(target=scan_loop, daemon=True)
        t.start()
    return jsonify({"status": "started"})

@app.route("/stop", methods=["POST"])
def stop():
    global running
    running = False
    return jsonify({"status": "stopped"})

@app.route("/status")
def get_status():
    balances = get_wallet_balances()
    coins = {pair[:-4]: balances.get(pair[:-4], 0.0) for pair in PAIRS}
    pos_view = {p: positions[p] for p in PAIRS if positions[p]}
    return jsonify({
        "status": status["msg"],
        "last": status["last"],
        "usdt": balances.get("USDT", 0.0),
        "coins": coins,
        "positions": pos_view,
        "trades": trade_log[-12:][::-1],
        "scans": scan_log[-40:][::-1],
        "error": error_message
    })

@app.route("/ping")
def ping():
    return "pong"

# =========================
# Main
# =========================
if __name__ == "__main__":
    # Program ID: ABCD
    app.run(host="0.0.0.0", port=10000)
