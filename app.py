import os
import time
import threading
import hmac
import hashlib
import requests
import json
from datetime import datetime, timedelta
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

# Pairs to scan
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

# Candle interval (seconds). 60 = 1m candles.
CANDLE_INTERVAL_SEC = 60

# PSAR parameters
PSAR_STEP = 0.02
PSAR_MAX  = 0.20

# Speed & risk knobs (tune these)
CONFIG = {
    "POLL_SEC": 1.0,           # fetch ticker every 1s
    "USE_TREND_FILTER": True,  # require slow EMA up/down alignment
    "SLOW_EMA_PERIOD": 50,     # slow EMA on 1m closes
    "RSI_LONG_MIN": 55.0,      # >55 for longs (avoid chop)
    "RSI_SHORT_MAX": 45.0,     # <45 for shorts
    "MIN_BODY_RATIO": 0.35,    # body / true_range must exceed this
    "BUY_USDT_ALLOC": 0.30,    # 30% of USDT per BUY
}

# =========================
# Time helpers (IST)
# =========================
IST = timezone("Asia/Kolkata")
def ist_now(): return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
def ist_date(): return datetime.now(IST).strftime("%Y-%m-%d")

# =========================
# State
# =========================
# Price ingest + candles
tick_logs   = {p: [] for p in PAIRS}      # [(ts, price)] (kept tiny now)
candle_logs = {p: [] for p in PAIRS}      # list of closed candles
candle_live = {p: None for p in PAIRS}    # current building candle
last_bar_ts = {p: 0 for p in PAIRS}       # last closed candle start

# UI logs
scan_log    = []                          # rolling text logs for UI
trade_log   = []                          # entries & reverse-exits

# runtime
running     = False
status      = {"msg": "Idle", "last": ""}
error_message = ""

# Positions (spot longs only)
# positions[pair] = {"side":"BUY","qty":float,"entry":float}
positions = {p: None for p in PAIRS}

# HTTP session + cached balances (for speed)
session = requests.Session()
balances_cache = {"ts": 0, "data": {}}
BAL_TTL_SEC = 12  # refresh wallet balances at most every 12s unless forced

# =========================
# Utils / API
# =========================
def hmac_signature(payload: str) -> str:
    return hmac.new(API_SECRET, payload.encode(), hashlib.sha256).hexdigest()

def get_wallet_balances(force: bool = False):
    now = time.time()
    if not force and (now - balances_cache["ts"] < BAL_TTL_SEC):
        return dict(balances_cache["data"])

    payload = json.dumps({"timestamp": int(now * 1000)})
    sig = hmac_signature(payload)
    headers = {"X-AUTH-APIKEY": API_KEY, "X-AUTH-SIGNATURE": sig, "Content-Type": "application/json"}
    balances = {}
    try:
        r = session.post(f"{BASE_URL}/exchange/v1/users/balances", headers=headers, data=payload, timeout=8)
        if r.ok:
            for b in r.json():
                balances[b["currency"]] = float(b["balance"])
            # cache
            balances_cache["data"] = balances
            balances_cache["ts"] = now
        else:
            scan_log.append(f"{ist_now()} | BAL_ERR: {r.status_code} {r.text[:120]}")
    except Exception as e:
        scan_log.append(f"{ist_now()} | BAL_ERR: {e}")
    return balances

def fetch_all_prices():
    try:
        # /exchange/ticker returns a big list; we filter our pairs
        r = session.get(f"{BASE_URL}/exchange/ticker", timeout=8)
        if r.ok:
            now = int(time.time())
            out = {}
            for item in r.json():
                m = item.get("market")
                if m in PAIRS:
                    out[m] = {"price": float(item["last_price"]), "ts": now}
            return out
        else:
            scan_log.append(f"{ist_now()} | PRICE_ERR: {r.status_code} {r.text[:120]}")
    except Exception as e:
        scan_log.append(f"{ist_now()} | PRICE_ERR: {e}")
    return {}

def clamp_buy_qty(pair, qty):
    """Round down to step, ensure >= min for BUY sizing."""
    rule = PAIR_RULES.get(pair, {"precision": 6, "min_qty": 0.0})
    precision = int(rule["precision"]); min_qty = float(rule["min_qty"])
    step = (10 ** (-precision)) if precision > 0 else 1.0
    qty = (int(max(qty, 0.0) / step)) * step
    if qty < min_qty: return 0.0
    return round(qty, precision)

def precise_sell_qty(pair, balance_qty):
    """Round down so qty <= balance, enforce min, respect precision."""
    rule = PAIR_RULES.get(pair, {"precision": 6, "min_qty": 0.0})
    precision = int(rule["precision"]); min_qty = float(rule["min_qty"])
    step = (10 ** (-precision)) if precision > 0 else 1.0
    qty = (int(max(balance_qty, 0.0) / step)) * step
    if qty < min_qty: return 0.0
    return round(qty, precision)

def place_order(pair, side, qty):
    """Order protocol unchanged: market order with total_quantity."""
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
def ema(values, period):
    if len(values) < period: return None
    k = 2 / (period + 1)
    e = float(values[0])
    for v in values[1:]:
        e = float(v) * k + e * (1 - k)
    return e

def rsi_wilder_from_closes(closes, period=14):
    if len(closes) < period + 1: return None
    gains, losses = [], []
    for i in range(1, period + 1):
        ch = closes[i] - closes[i-1]
        gains.append(max(0.0, ch)); losses.append(max(0.0, -ch))
    avg_gain = sum(gains)/period; avg_loss = sum(losses)/period
    for i in range(period + 1, len(closes)):
        ch = closes[i] - closes[i-1]
        gain = max(0.0, ch); loss = max(0.0, -ch)
        avg_gain = (avg_gain*(period-1) + gain)/period
        avg_loss = (avg_loss*(period-1) + loss)/period
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def psar_series(highs, lows, step=0.02, max_step=0.2):
    """
    Classic PSAR (Wilder). Returns:
      sar_list: list of SAR values
      bull: list of bools where True = uptrend (SAR below price)
    """
    n = len(highs)
    if n < 2:
        return [], []
    sar = [0.0] * n
    bull = [False] * n
    uptrend = highs[1] >= highs[0]
    bull[1] = uptrend
    ep = highs[1] if uptrend else lows[1]
    sar[1] = lows[0] if uptrend else highs[0]
    af = step
    for i in range(2, n):
        prev_sar = sar[i-1]; prev_up  = bull[i-1]
        sar_i = prev_sar + af * (ep - prev_sar)
        if prev_up:
            sar_i = min(sar_i, lows[i-1], lows[i-2] if i-2 >= 0 else lows[i-1])
        else:
            sar_i = max(sar_i, highs[i-1], highs[i-2] if i-2 >= 0 else highs[i-1])
        if prev_up:
            if lows[i] < sar_i:
                bull[i] = False; sar[i]  = ep; af = step; ep = lows[i]
            else:
                bull[i] = True; sar[i]  = sar_i
                if highs[i] > ep:
                    ep = highs[i]; af = min(max_step, af + step)
        else:
            if highs[i] > sar_i:
                bull[i] = True; sar[i]  = ep; af = step; ep = highs[i]
            else:
                bull[i] = False; sar[i]  = sar_i
                if lows[i] < ep:
                    ep = lows[i]; af = min(max_step, af + step)
    sar[0] = sar[1]; bull[0] = bull[1]
    return sar, bull

# =========================
# Signal (PSAR flip + EMA14 + RSI14 + filters)
# =========================
def psar_ema_rsi_signal(pair):
    """
    BUY  when PSAR flips below price AND close > EMA14 AND RSI>RSI_LONG_MIN (+ filters)
    SELL when PSAR flips above price AND close < EMA14 AND RSI<RSI_SHORT_MAX (+ filters)
    """
    candles = candle_logs[pair]
    if len(candles) < 20:   # need enough bars for EMA/RSI/PSAR
        return None

    highs  = [c["high"]  for c in candles]
    lows   = [c["low"]   for c in candles]
    closes = [c["close"] for c in candles]
    curr   = candles[-1]

    # Indicators
    sar, bull = psar_series(highs, lows, step=PSAR_STEP, max_step=PSAR_MAX)
    if len(sar) < 2: return None

    ema14 = ema(closes, 14)
    rsi14 = rsi_wilder_from_closes(closes[-15:], 14)  # last 15 closes for RSI
    if ema14 is None or rsi14 is None:
        return None

    # Optional slow trend filter
    if CONFIG["USE_TREND_FILTER"]:
        ema_slow = ema(closes, CONFIG["SLOW_EMA_PERIOD"])
    else:
        ema_slow = None

    # PSAR flip detection
    flipped_up   = (bull[-2] is False) and (bull[-1] is True)
    flipped_down = (bull[-2] is True)  and (bull[-1] is False)

    # Body strength filter
    tr = max(curr["high"] - curr["low"], 1e-9)
    body = abs(curr["close"] - curr["open"])
    if (body / tr) < CONFIG["MIN_BODY_RATIO"]:
        return None

    # BUY trigger
    if flipped_up and curr["close"] > ema14 and rsi14 > CONFIG["RSI_LONG_MIN"]:
        if (ema_slow is None) or (curr["close"] > ema_slow):
            return {"side": "BUY", "entry": curr["close"],
                    "msg": f"BUY: PSAR up | close>{round(ema14,6)} | RSI={round(rsi14,2)}"
                           + (f" | slowEMA={round(ema_slow,6)}" if ema_slow else "")}

    # SELL trigger
    if flipped_down and curr["close"] < ema14 and rsi14 < CONFIG["RSI_SHORT_MAX"]:
        if (ema_slow is None) or (curr["close"] < ema_slow):
            return {"side": "SELL", "entry": curr["close"],
                    "msg": f"SELL: PSAR down | close<{round(ema14,6)} | RSI={round(rsi14,2)}"
                           + (f" | slowEMA={round(ema_slow,6)}" if ema_slow else "")}

    return None

# =========================
# Incremental candle builder (fast)
# =========================
def on_price(pair, ts, price):
    """
    Update the live candle with a new tick.
    If a candle closes (new minute starts), finalize it and return True.
    """
    wstart = ts - (ts % CANDLE_INTERVAL_SEC)
    c = candle_live[pair]

    if (c is None) or (c["start"] != wstart):
        # finalize previous candle if exists
        if c is not None:
            candle_logs[pair].append(c)
            if len(candle_logs[pair]) > 600:
                candle_logs[pair] = candle_logs[pair][-600:]
        # start new live candle
        candle_live[pair] = {
            "open": price, "high": price, "low": price, "close": price,
            "volume": 1, "start": wstart
        }
        # a new candle started -> the previous candle just closed
        return True

    # update current live candle
    c["high"] = max(c["high"], price)
    c["low"]  = min(c["low"], price)
    c["close"] = price
    c["volume"] += 1
    return False

# =========================
# Trading Loop (entry + reverse-exit)
# =========================
def scan_loop():
    global running, error_message
    scan_log.clear()

    while running:
        prices = fetch_all_prices()
        now = int(time.time())

        # lightweight balance refresh (cached)
        balances = get_wallet_balances(force=False)

        for pair in PAIRS:
            info = prices.get(pair)
            if not info:
                continue

            price = info["price"]
            # tiny tick buffer (we no longer sort or rebuild candles)
            tl = tick_logs[pair]
            tl.append((now, price))
            if len(tl) > 50:
                tick_logs[pair] = tl[-50:]

            # update incremental candle; if previous just closed, evaluate
            candle_closed = on_price(pair, now, price)
            if candle_closed:
                # only act on *closed* candle
                signal = psar_ema_rsi_signal(pair)
                if signal:
                    side  = signal["side"]
                    entry = signal["entry"]
                    coin  = pair[:-4]

                    if side == "SELL":
                        # Reverse-exit: close any open long (sell wallet balance) with precise pair precision
                        had_pos = positions.get(pair) is not None
                        raw_qty = balances.get(coin, 0.0)
                        qty = precise_sell_qty(pair, raw_qty)
                        if qty > 0:
                            res = place_order(pair, "SELL", qty)
                            scan_log.append(f"{ist_now()} | {pair} | {'EXIT LONG & ' if had_pos else ''}SELL qty={qty} @ {entry} | {res}")
                            trade_log.append({
                                "time": ist_now(), "pair": pair, "side": "SELL",
                                "entry": entry, "msg": signal["msg"], "qty": qty, "order_result": res
                            })
                            if "error" in res:
                                error_message = res["error"]
                            positions[pair] = None
                            # force balance refresh after trading
                            get_wallet_balances(force=True)
                        else:
                            scan_log.append(f"{ist_now()} | {pair} | SELL signal, no {coin} balance (or below min qty)")
                    else:
                        # Avoid pyramiding: if already long, skip
                        pos = positions.get(pair)
                        if pos and pos["side"] == "BUY" and pos["qty"] > 0:
                            scan_log.append(f"{ist_now()} | {pair} | BUY signal but already long; skip")
                        else:
                            usdt = balances.get("USDT", 0.0)
                            raw_qty = (CONFIG["BUY_USDT_ALLOC"] * usdt) / entry if entry else 0.0
                            qty = clamp_buy_qty(pair, raw_qty)
                            if qty > 0:
                                res = place_order(pair, "BUY", qty)
                                scan_log.append(f"{ist_now()} | {pair} | BUY qty={qty} @ {entry} | {res}")
                                trade_log.append({
                                    "time": ist_now(), "pair": pair, "side": "BUY",
                                    "entry": entry, "msg": signal["msg"], "qty": qty, "order_result": res
                                })
                                if "error" in res:
                                    error_message = res["error"]
                                else:
                                    positions[pair] = {"side": "BUY", "qty": qty, "entry": entry}
                                get_wallet_balances(force=True)
                            else:
                                scan_log.append(f"{ist_now()} | {pair} | BUY signal, qty=0 after clamp; skip")
                else:
                    scan_log.append(f"{ist_now()} | {pair} | No Signal")

        # trim logs for memory stability
        if len(scan_log) > 500: scan_log[:] = scan_log[-500:]
        if len(trade_log) > 250: trade_log[:] = trade_log[-250:]

        status["msg"], status["last"] = "Running", ist_now()
        time.sleep(CONFIG["POLL_SEC"])

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
