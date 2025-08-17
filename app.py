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

# ----------------------------
# Flask App
# ----------------------------
app = Flask(__name__)

# ----------------------------
# CoinDCX Config
# ----------------------------
API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET").encode() if os.environ.get("API_SECRET") else b""
BASE_URL = "https://api.coindcx.com"

# Trade universe
PAIRS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "SHIBUSDT", "SOLUSDT",
    "DOGEUSDT", "ADAUSDT", "AEROUSDT", "BNBUSDT", "LTCUSDT"
]

# Per-pair rules (use API to load for production; kept here for stability)
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

# ----------------------------
# Time helpers (IST)
# ----------------------------
IST = timezone('Asia/Kolkata')
def ist_now(): return datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')
def ist_date(): return datetime.now(IST).strftime('%Y-%m-%d')
def ist_yesterday(): return (datetime.now(IST) - timedelta(days=1)).strftime('%Y-%m-%d')

# ----------------------------
# State
# ----------------------------
tick_logs = {p: [] for p in PAIRS}     # [(ts, price)]
candle_logs = {p: [] for p in PAIRS}   # list of OHLCV dicts
scan_log = []                          # text logs for UI
trade_log = []                         # recent trades
exit_orders = []                       # open exits: dicts with pair/side/qty/tp/sl/entry
daily_profit = {}                      # by date string
running = False
status = {"msg": "Idle", "last": ""}
error_message = ""

# ----------------------------
# Utils
# ----------------------------
def hmac_signature(payload: str) -> str:
    return hmac.new(API_SECRET, payload.encode(), hashlib.sha256).hexdigest()

def get_wallet_balances():
    payload = json.dumps({"timestamp": int(time.time() * 1000)})
    sig = hmac_signature(payload)
    headers = {"X-AUTH-APIKEY": API_KEY, "X-AUTH-SIGNATURE": sig, "Content-Type": "application/json"}
    balances = {}
    try:
        r = requests.post(f"{BASE_URL}/exchange/v1/users/balances", headers=headers, data=payload, timeout=10)
        if r.ok:
            for b in r.json():
                balances[b['currency']] = float(b['balance'])
    except Exception as e:
        scan_log.append(f"{ist_now()} | BAL_ERR: {e}")
    return balances

def fetch_all_prices():
    try:
        r = requests.get(f"{BASE_URL}/exchange/ticker", timeout=10)
        if r.ok:
            now = int(time.time())
            return {item["market"]: {"price": float(item["last_price"]), "ts": now}
                    for item in r.json() if item["market"] in PAIRS}
    except Exception as e:
        scan_log.append(f"{ist_now()} | PRICE_ERR: {e}")
    return {}

def aggregate_candles(pair, interval=60):
    """Build rolling OHLCV candles from ticks (per 'interval' seconds)."""
    ticks = tick_logs[pair]
    if not ticks:
        return
    candles = []
    ticks_sorted = sorted(ticks, key=lambda x: x[0])
    candle = None
    last_window = None
    for ts, price in ticks_sorted:
        wstart = ts - (ts % interval)
        if last_window != wstart:
            if candle:
                candles.append(candle)
            candle = {
                "open": price, "high": price, "low": price, "close": price,
                "volume": 1, "start": wstart
            }
            last_window = wstart
        else:
            candle["high"] = max(candle["high"], price)
            candle["low"] = min(candle["low"], price)
            candle["close"] = price
            candle["volume"] += 1
    if candle:
        candles.append(candle)
    candle_logs[pair] = candles[-500:]  # keep more for SMC calc

# ----------------------------
# PA (Relaxed) Signal Logic
# ----------------------------
def pa_buy_sell_signal(pair):
    candles = candle_logs[pair]
    if len(candles) < 3:
        return None
    prev1, prev2, curr = candles[-3], candles[-2], candles[-1]
    # Relaxed BUY: current close > prev close AND breaks last-two highs
    if curr["close"] > prev2["close"] and curr["high"] > max(prev1["high"], prev2["high"]):
        return {"side": "BUY", "entry": curr["close"], "msg": "PA BUY: close>prev & high>last2"}
    # Relaxed SELL: current close < prev close AND breaks last-two lows
    if curr["close"] < prev2["close"] and curr["low"] < min(prev1["low"], prev2["low"]):
        return {"side": "SELL", "entry": curr["close"], "msg": "PA SELL: close<prev & low<last2"}
    return None

# ----------------------------
# Orders
# ----------------------------
def clamp_qty(pair, qty):
    rule = PAIR_RULES.get(pair, {"precision": 6, "min_qty": 0.0})
    precision = int(rule["precision"])
    min_qty = float(rule["min_qty"])
    # enforce min and precision
    qty = max(qty, min_qty)
    step = 10 ** (-precision)
    qty = (int(qty / step)) * step
    return round(qty, precision)

def place_order(pair, side, qty):
    payload = {
        "market": pair,
        "side": side.lower(),
        "order_type": "market_order",
        "total_quantity": str(qty),
        "timestamp": int(time.time() * 1000)
    }
    body = json.dumps(payload)
    sig = hmac_signature(body)
    headers = {"X-AUTH-APIKEY": API_KEY, "X-AUTH-SIGNATURE": sig, "Content-Type": "application/json"}
    try:
        r = requests.post(f"{BASE_URL}/exchange/v1/orders/create", headers=headers, data=body, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def monitor_exits(prices):
    """Exit TP/SL for open positions."""
    global error_message
    to_remove = []
    for ex in exit_orders:
        pair, side, qty, tp, sl, entry = ex["pair"], ex["side"], ex["qty"], ex["tp"], ex["sl"], ex["entry"]
        price = prices.get(pair, {}).get("price")
        if not price:
            continue
        if side == "BUY" and (price >= tp or price <= sl):
            res = place_order(pair, "SELL", qty)
            scan_log.append(f"{ist_now()} | {pair} | EXIT SELL {qty} @ {price} | {res}")
            pl = (price - entry) * qty
            daily_profit[ist_date()] = daily_profit.get(ist_date(), 0) + pl
            if "error" in res:
                error_message = res["error"]
            to_remove.append(ex)
        elif side == "SELL" and (price <= tp or price >= sl):
            res = place_order(pair, "BUY", qty)
            scan_log.append(f"{ist_now()} | {pair} | EXIT BUY {qty} @ {price} | {res}")
            pl = (entry - price) * qty
            daily_profit[ist_date()] = daily_profit.get(ist_date(), 0) + pl
            if "error" in res:
                error_message = res["error"]
            to_remove.append(ex)
    for ex in to_remove:
        exit_orders.remove(ex)

# ----------------------------
# SMC Logic (adapted to CoinDCX candles)
# ----------------------------
import statistics

def detect_order_blocks(data):
    """
    Expects list of candles: {'date','open','high','low','close','volume'} (date optional)
    Returns (bullish_ob, bearish_ob), each list of dicts with zone_low/zone_high.
    """
    bullish_ob = []
    bearish_ob = []
    n = len(data)
    for i in range(2, n - 2):
        curr = data[i]
        n1 = data[i + 1]
        n2 = data[i + 2]

        # Bearish OB: strong up candle then two closes below its low
        if curr['close'] > curr['open'] and n1['close'] < curr['low'] and n2['close'] < curr['low']:
            zone_low = curr['open']
            zone_high = curr['high']
            mitigated = False
            for j in range(i + 3, n):
                if (data[j]['high'] >= zone_low) and (data[j]['low'] <= zone_high):
                    mitigated = True
                    break
            if not mitigated:
                bearish_ob.append({
                    "index": i, "timestamp": curr.get('date'),
                    "type": "bearish", "zone_low": float(zone_low), "zone_high": float(zone_high),
                    "candle": curr
                })

        # Bullish OB: strong down candle then two closes above its high
        if curr['open'] > curr['close'] and n1['close'] > curr['high'] and n2['close'] > curr['high']:
            zone_low = curr['low']
            zone_high = curr['open']
            mitigated = False
            for j in range(i + 3, n):
                if (data[j]['high'] >= zone_low) and (data[j]['low'] <= zone_high):
                    mitigated = True
                    break
            if not mitigated:
                bullish_ob.append({
                    "index": i, "timestamp": curr.get('date'),
                    "type": "bullish", "zone_low": float(zone_low), "zone_high": float(zone_high),
                    "candle": curr
                })
    return bullish_ob, bearish_ob

def calculate_ema_from_closes(closes, period):
    if not closes:
        return []
    k = 2 / (period + 1)
    ema_values = []
    ema = float(closes[0])
    for c in closes:
        ema = (float(c) * k) + (ema * (1 - k))
        ema_values.append(ema)
    return [round(v, 6) for v in ema_values]

def rsi_wilder_from_closes(closes, period=14):
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, period + 1):
        change = closes[i] - closes[i - 1]
        gains.append(max(0.0, change))
        losses.append(max(0.0, -change))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    for i in range(period + 1, len(closes)):
        change = closes[i] - closes[i - 1]
        gain = max(0.0, change)
        loss = max(0.0, -change)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 2)

def run_smc_scan_coindcx():
    """
    Build SMC view from available aggregated candles (recent intraday).
    """
    results = {}
    # Use last ~300 candles (depending on aggregation window)
    for pair in PAIRS:
        candles = candle_logs.get(pair, [])
        if len(candles) < 60:
            continue
        # Map to SMC-friendly structure
        data = []
        for c in candles:
            data.append({
                "date": datetime.fromtimestamp(c["start"], tz=IST).isoformat(),
                "open": c["open"],
                "high": c["high"],
                "low": c["low"],
                "close": c["close"],
                "volume": c["volume"],
            })

        bullish_obs, bearish_obs = detect_order_blocks(data)
        closes = [d["close"] for d in data]
        last_close = closes[-1]
        ema20 = calculate_ema_from_closes(closes, 20)[-1]
        ema50 = calculate_ema_from_closes(closes, 50)[-1]
        rsi = rsi_wilder_from_closes(closes[-(50 + 1):], 14)

        # Volume spike vs last 10 completed bars (exclude current)
        if len(data) >= 12:
            avg_vol = statistics.mean([d['volume'] for d in data[-11:-1]])
        else:
            avg_vol = statistics.mean([d['volume'] for d in data[:-1]]) if len(data) > 1 else 0
        volume_spike = data[-1]['volume'] > 1.5 * avg_vol if avg_vol else False
        trend_tag = "Bullish" if ema20 > ema50 else ("Bearish" if ema20 < ema50 else "Neutral")

        # Check latest membership in any unmitigated block
        flagged = False
        for ob in reversed(bullish_obs):
            if ob['zone_low'] <= last_close <= ob['zone_high']:
                results[pair] = {
                    "status": "In Buy Block",
                    "zone": [round(ob['zone_low'], 6), round(ob['zone_high'], 6)],
                    "price": last_close,
                    "ema20": ema20, "ema50": ema50, "rsi14": rsi,
                    "volume_spike": volume_spike, "trend": trend_tag,
                    "ob_time": ob["timestamp"]
                }
                flagged = True
                break
        if flagged:
            continue

        for ob in reversed(bearish_obs):
            if ob['zone_low'] <= last_close <= ob['zone_high']:
                results[pair] = {
                    "status": "In Sell Block",
                    "zone": [round(ob['zone_low'], 6), round(ob['zone_high'], 6)],
                    "price": last_close,
                    "ema20": ema20, "ema50": ema50, "rsi14": rsi,
                    "volume_spike": volume_spike, "trend": trend_tag,
                    "ob_time": ob["timestamp"]
                }
                break
    return results

# ----------------------------
# Trading Loop
# ----------------------------
def scan_loop():
    global running, error_message
    scan_log.clear()
    last_candle_ts = {p: 0 for p in PAIRS}
    interval = 60  # 1-minute aggregation

    while running:
        prices = fetch_all_prices()
        now = int(time.time())
        monitor_exits(prices)
        balances = get_wallet_balances()

        for pair in PAIRS:
            info = prices.get(pair)
            if not info:
                continue
            price = info["price"]
            tick_logs[pair].append((now, price))
            if len(tick_logs[pair]) > 5000:
                tick_logs[pair] = tick_logs[pair][-5000:]

            aggregate_candles(pair, interval)
            last_candle = candle_logs[pair][-1] if candle_logs[pair] else None

            if last_candle and last_candle["start"] != last_candle_ts[pair]:
                last_candle_ts[pair] = last_candle["start"]
                signal = pa_buy_sell_signal(pair)

                if signal:
                    error_message = ""
                    coin = pair[:-4]
                    # BUY = risk % of USDT; SELL = use coin balance (spot)
                    if signal["side"] == "BUY":
                        usdt = balances.get("USDT", 0.0)
                        # position: 30% of USDT
                        raw_qty = (0.3 * usdt) / signal["entry"] if signal["entry"] else 0.0
                    else:
                        raw_qty = balances.get(coin, 0.0)

                    qty = clamp_qty(pair, raw_qty)

                    # Risk/Reward (adjust as needed)
                    tp = round(signal['entry'] * 1.003, 6)   # +0.3%
                    sl = round(signal['entry'] * 0.997, 6)   # -0.3%

                    res = place_order(pair, signal["side"], qty)

                    scan_log.append(f"{ist_now()} | {pair} | {signal['side']} qty={qty} @ {signal['entry']} | {res}")
                    trade_log.append({
                        "time": ist_now(), "pair": pair, "side": signal["side"],
                        "entry": signal["entry"], "msg": signal["msg"],
                        "tp": tp, "sl": sl, "qty": qty, "order_result": res
                    })

                    exit_orders.append({
                        "pair": pair, "side": signal["side"], "qty": qty,
                        "tp": tp, "sl": sl, "entry": signal["entry"]
                    })

                    if "error" in res:
                        error_message = res["error"]
                else:
                    # Always log scans for visibility
                    scan_log.append(f"{ist_now()} | {pair} | No Signal")

        status["msg"], status["last"] = "Running", ist_now()
        time.sleep(5)

    status["msg"] = "Idle"

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start", methods=["POST"])
def start():
    global running
    if not running:
        running = True
        t = threading.Thread(target=scan_loop)
        t.daemon = True
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
    return jsonify({
        "status": status["msg"], "last": status["last"],
        "usdt": balances.get("USDT", 0.0),
        "profit_today": round(daily_profit.get(ist_date(), 0), 4),
        "profit_yesterday": round(daily_profit.get(ist_yesterday(), 0), 4),
        "coins": coins,
        "trades": trade_log[-10:][::-1],
        "scans": scan_log[-30:][::-1],
        "error": error_message
    })

@app.route("/api/smc-status")
def api_smc_status():
    # Build SMC signals from aggregated candles
    results = run_smc_scan_coindcx()
    return jsonify(results)

@app.route("/ping")
def ping():
    return "pong"

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # (Optional) You could load live rules via API here to replace PAIR_RULES
    # PAIR_RULES.update(load_pair_rules_from_api())
    app.run(host="0.0.0.0", port=10000)
