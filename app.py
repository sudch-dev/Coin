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

PAIRS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "SHIBUSDT", "SOLUSDT",
    "DOGEUSDT", "ADAUSDT", "AEROUSDT", "BNBUSDT", "LTCUSDT"
]

# Use base-currency (quantity) precision + min qty. (You can swap to dynamic rules if you prefer.)
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

# Risk controls
RISK_PCT = 0.005       # risk 0.5% of USDT per trade
ATR_PERIOD = 14
ATR_K = 0.8            # SL distance = ATR_K * ATR
RR = 1.5               # TP distance = RR * SL distance
MAX_DD_DAY_PCT = 0.02  # daily stop at -2% of day-open USDT
COOLDOWN_BARS = 2      # bars to skip after a losing exit

# =========================
# Time helpers (IST)
# =========================
IST = timezone("Asia/Kolkata")
def ist_now(): return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
def ist_date(): return datetime.now(IST).strftime("%Y-%m-%d")
def ist_yesterday(): return (datetime.now(IST) - timedelta(days=1)).strftime("%Y-%m-%d")

# =========================
# State
# =========================
tick_logs = {p: [] for p in PAIRS}     # [(ts, price)]
candle_logs = {p: [] for p in PAIRS}   # list of dicts: open/high/low/close/volume/start
scan_log = []                          # text logs for UI
trade_log = []                         # recent trades
exit_orders = []                       # open exits to monitor
daily_profit = {}                      # realized P/L by date
running = False
status = {"msg": "Idle", "last": ""}
error_message = ""

pair_cooldown = {p: 0 for p in PAIRS}
usdt_day_open = {"date": None, "balance": None}

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
        r = requests.post(f"{BASE_URL}/exchange/v1/users/balances", headers=headers, data=payload, timeout=10)
        if r.ok:
            for b in r.json():
                balances[b["currency"]] = float(b["balance"])
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
            candle["low"] = min(candle["low"], price)
            candle["close"] = price
            candle["volume"] += 1
    if candle: candles.append(candle)
    candle_logs[pair] = candles[-500:]  # keep enough for ATR/SMC

def clamp_qty(pair, qty):
    rule = PAIR_RULES.get(pair, {"precision": 6, "min_qty": 0.0})
    precision = int(rule["precision"])
    min_qty = float(rule["min_qty"])
    qty = max(qty, min_qty)
    step = 10 ** (-precision)
    qty = (int(qty / step)) * step
    return round(qty, precision)

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
        r = requests.post(f"{BASE_URL}/exchange/v1/orders/create", headers=headers, data=body, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# =========================
# Indicators / Logic
# =========================
def compute_atr(candles, period=14):
    """ Wilder ATR on OHLC candles (oldest->newest). """
    if len(candles) < period + 1: return None
    trs = []
    for i in range(1, len(candles)):
        h = candles[i]["high"]; l = candles[i]["low"]; pc = candles[i-1]["close"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    atr = sum(trs[:period]) / period
    for x in trs[period:]:
        atr = (atr * (period - 1) + x) / period
    return atr

def pa_buy_sell_signal(pair):
    candles = candle_logs[pair]
    if len(candles) < 3: return None
    prev1, prev2, curr = candles[-3], candles[-2], candles[-1]
    if curr["close"] > prev2["close"] and curr["high"] > max(prev1["high"], prev2["high"]):
        return {"side": "BUY", "entry": curr["close"], "msg": "PA BUY: close>prev & high>last2"}
    if curr["close"] < prev2["close"] and curr["low"] < min(prev1["low"], prev2["low"]):
        return {"side": "SELL", "entry": curr["close"], "msg": "PA SELL: close<prev & low<last2"}
    return None

# ===== SMC (Order Blocks) =====
import statistics

def detect_order_blocks(data):
    """ data: list of dicts: {date, open, high, low, close, volume} """
    bullish_ob, bearish_ob = [], []
    n = len(data)
    for i in range(2, n - 2):
        curr = data[i]; n1 = data[i+1]; n2 = data[i+2]
        # Bearish OB: up candle then two closes below its low
        if curr["close"] > curr["open"] and n1["close"] < curr["low"] and n2["close"] < curr["low"]:
            zone_low, zone_high = curr["open"], curr["high"]
            mitigated = any((data[j]["high"] >= zone_low) and (data[j]["low"] <= zone_high) for j in range(i+3, n))
            if not mitigated:
                bearish_ob.append({"index": i, "timestamp": curr.get("date"),
                                   "type": "bearish", "zone_low": float(zone_low), "zone_high": float(zone_high),
                                   "candle": curr})
        # Bullish OB: down candle then two closes above its high
        if curr["open"] > curr["close"] and n1["close"] > curr["high"] and n2["close"] > curr["high"]:
            zone_low, zone_high = curr["low"], curr["open"]
            mitigated = any((data[j]["high"] >= zone_low) and (data[j]["low"] <= zone_high) for j in range(i+3, n))
            if not mitigated:
                bullish_ob.append({"index": i, "timestamp": curr.get("date"),
                                   "type": "bullish", "zone_low": float(zone_low), "zone_high": float(zone_high),
                                   "candle": curr})
    return bullish_ob, bearish_ob

def calculate_ema_from_closes(closes, period):
    if not closes: return []
    k = 2 / (period + 1)
    ema_values = []
    ema = float(closes[0])
    for c in closes:
        ema = (float(c) * k) + (ema * (1 - k))
        ema_values.append(ema)
    return [round(v, 6) for v in ema_values]

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
    return round(100 - (100 / (1 + rs)), 2)

def run_smc_scan_coindcx():
    results = {}
    for pair in PAIRS:
        candles = candle_logs.get(pair, [])
        if len(candles) < 60:
            continue
        data = [{
            "date": datetime.fromtimestamp(c["start"], tz=IST).isoformat(),
            "open": c["open"], "high": c["high"], "low": c["low"], "close": c["close"], "volume": c["volume"]
        } for c in candles]

        bullish_obs, bearish_obs = detect_order_blocks(data)
        closes = [d["close"] for d in data]
        last_close = closes[-1]
        ema20 = calculate_ema_from_closes(closes, 20)[-1]
        ema50 = calculate_ema_from_closes(closes, 50)[-1]
        rsi = rsi_wilder_from_closes(closes[-(50+1):], 14)

        if len(data) >= 12:
            avg_vol = statistics.mean([d["volume"] for d in data[-11:-1]])
        else:
            avg_vol = statistics.mean([d["volume"] for d in data[:-1]]) if len(data) > 1 else 0
        volume_spike = data[-1]["volume"] > 1.5 * avg_vol if avg_vol else False
        trend_tag = "Bullish" if ema20 > ema50 else ("Bearish" if ema20 < ema50 else "Neutral")

        flagged = False
        for ob in reversed(bullish_obs):
            if ob["zone_low"] <= last_close <= ob["zone_high"]:
                results[pair] = {
                    "status": "In Buy Block",
                    "zone": [round(ob["zone_low"], 6), round(ob["zone_high"], 6)],
                    "price": last_close, "ema20": ema20, "ema50": ema50, "rsi14": rsi,
                    "volume_spike": volume_spike, "trend": trend_tag, "ob_time": ob["timestamp"]
                }
                flagged = True
                break
        if flagged: continue
        for ob in reversed(bearish_obs):
            if ob["zone_low"] <= last_close <= ob["zone_high"]:
                results[pair] = {
                    "status": "In Sell Block",
                    "zone": [round(ob["zone_low"], 6), round(ob["zone_high"], 6)],
                    "price": last_close, "ema20": ema20, "ema50": ema50, "rsi14": rsi,
                    "volume_spike": volume_spike, "trend": trend_tag, "ob_time": ob["timestamp"]
                }
                break
    return results

# =========================
# Risk Helpers
# =========================
def daily_stop_hit(current_usdt):
    today = ist_date()
    if usdt_day_open["date"] != today or usdt_day_open["balance"] is None:
        usdt_day_open["date"] = today
        usdt_day_open["balance"] = float(current_usdt)
        return False
    if usdt_day_open["balance"] is None:
        return False
    max_loss = usdt_day_open["balance"] * MAX_DD_DAY_PCT
    pl_today = daily_profit.get(today, 0.0)
    return (-pl_today) >= max_loss

# =========================
# Exits
# =========================
def monitor_exits(prices):
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
            if pl < 0:
                pair_cooldown[pair] = COOLDOWN_BARS
            if "error" in res:
                error_message = res["error"]
            to_remove.append(ex)

        elif side == "SELL" and (price <= tp or price >= sl):
            res = place_order(pair, "BUY", qty)
            scan_log.append(f"{ist_now()} | {pair} | EXIT BUY {qty} @ {price} | {res}")
            pl = (entry - price) * qty
            daily_profit[ist_date()] = daily_profit.get(ist_date(), 0) + pl
            if pl < 0:
                pair_cooldown[pair] = COOLDOWN_BARS
            if "error" in res:
                error_message = res["error"]
            to_remove.append(ex)

    for ex in to_remove:
        exit_orders.remove(ex)

# =========================
# Trading Loop
# =========================
def scan_loop():
    global running, error_message
    scan_log.clear()
    last_candle_ts = {p: 0 for p in PAIRS}
    interval = 60  # 1-minute candles

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
                    # Daily stop
                    if daily_stop_hit(balances.get("USDT", 0.0)):
                        scan_log.append(f"{ist_now()} | DAILY STOP HIT — skipping trades")
                        continue

                    # Cooldown after losing exit
                    if pair_cooldown.get(pair, 0) > 0:
                        pair_cooldown[pair] -= 1
                        scan_log.append(f"{ist_now()} | {pair} | Cooldown active ({pair_cooldown[pair]} left)")
                        continue

                    error_message = ""

                    # ATR-based SL/TP
                    recent = candle_logs[pair][-(ATR_PERIOD + 20):]
                    atr = compute_atr(recent, ATR_PERIOD)
                    if not atr:
                        scan_log.append(f"{ist_now()} | {pair} | ATR not ready")
                        continue

                    entry = signal["entry"]
                    sl_dist = ATR_K * atr
                    tp_dist = RR * sl_dist

                    if signal["side"] == "BUY":
                        sl = round(entry - sl_dist, 6)
                        tp = round(entry + tp_dist, 6)
                    else:
                        sl = round(entry + sl_dist, 6)
                        tp = round(entry - tp_dist, 6)

                    # Sizing
                    usdt = balances.get("USDT", 0.0)
                    coin = pair[:-4]
                    if signal["side"] == "BUY":
                        usdt_risk = usdt * RISK_PCT
                        qty_risk = (usdt_risk / sl_dist) if sl_dist > 0 else 0.0
                        qty_budget = (0.3 * usdt) / entry if entry else 0.0
                        raw_qty = max(0.0, min(qty_risk, qty_budget))
                    else:
                        # SELL: exit spot holding fully
                        raw_qty = balances.get(coin, 0.0)

                    qty = clamp_qty(pair, raw_qty)
                    if qty <= 0:
                        scan_log.append(f"{ist_now()} | {pair} | Qty=0 after clamp — skip")
                        continue

                    res = place_order(pair, signal["side"], qty)

                    scan_log.append(f"{ist_now()} | {pair} | {signal['side']} qty={qty} @ {entry} | TP={tp} SL={sl} | {res}")
                    trade_log.append({
                        "time": ist_now(), "pair": pair, "side": signal["side"], "entry": entry,
                        "msg": signal["msg"], "tp": tp, "sl": sl, "qty": qty, "order_result": res
                    })
                    exit_orders.append({"pair": pair, "side": signal["side"], "qty": qty,
                                        "tp": tp, "sl": sl, "entry": entry})
                    if "error" in res:
                        error_message = res["error"]
                else:
                    scan_log.append(f"{ist_now()} | {pair} | No Signal")

        status["msg"], status["last"] = "Running", ist_now()
        time.sleep(5)

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
    return jsonify({
        "status": status["msg"], "last": status["last"],
        "usdt": balances.get("USDT", 0.0),
        "profit_today": round(daily_profit.get(ist_date(), 0), 4),
        "profit_yesterday": round(daily_profit.get(ist_yesterday(), 0), 4),
        "coins": coins, "trades": trade_log[-10:][::-1], "scans": scan_log[-30:][::-1],
        "error": error_message
    })

@app.route("/api/smc-status")
def api_smc_status():
    return jsonify(run_smc_scan_coindcx())

@app.route("/ping")
def ping():
    return "pong"

# =========================
# Main
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
