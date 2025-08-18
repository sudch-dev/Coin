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

# Pairs to scan (spot)
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

# Candle intervals we track simultaneously (seconds)
INTERVALS = [15, 30, 60]

# History / warm start
WARMUP_BARS = 60          # indicators ready right away (we preload)
PRELOAD_LIMIT = 200       # bars to fetch/seed at startup

# PSAR parameters
PSAR_STEP = 0.02
PSAR_MAX  = 0.20

# Fees and edge gate (~1% round trip)
FEE_ROUND_TRIP = 0.010
EDGE_BUFFER    = 0.002   # extra headroom

# Buying & polling
BUY_USDT_ALLOC = 0.35    # use 35% of USDT per buy
POLL_SEC       = 0.6     # faster loop

# Exits: none (entry/exit = triggers); we ride until opposite signal

# =========================
# Time (IST)
# =========================
IST = timezone("Asia/Kolkata")
def ist_now(): return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

# =========================
# State
# =========================
session = requests.Session()
tick_logs            = {p: [] for p in PAIRS}               # [(ts, price)]
candle_logs_map      = {p: {iv: [] for iv in INTERVALS} for p in PAIRS}  # per interval
last_candle_ts_map   = {p: {iv: 0 for iv in INTERVALS} for p in PAIRS}
scan_log             = []
trade_log            = []
positions            = {p: None for p in PAIRS}  # {"side":"BUY","qty":float,"entry":float}
running              = False
status               = {"msg": "Idle", "last": ""}
error_message        = ""

# =========================
# API utils (order protocol unchanged)
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
            scan_log.append(f"{ist_now()} | BAL_ERR: {r.status_code} {r.text[:160]}")
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
            scan_log.append(f"{ist_now()} | PRICE_ERR: {r.status_code} {r.text[:160]}")
    except Exception as e:
        scan_log.append(f"{ist_now()} | PRICE_ERR: {e}")
    return {}

def clamp_buy_qty(pair, qty):
    rule = PAIR_RULES.get(pair, {"precision": 6, "min_qty": 0.0})
    p = int(rule["precision"]); minq = float(rule["min_qty"])
    step = (10 ** (-p)) if p > 0 else 1.0
    qty = (int(max(qty, 0.0) / step)) * step
    if qty < minq: return 0.0
    return round(qty, p)

def precise_sell_qty(pair, balance_qty):
    rule = PAIR_RULES.get(pair, {"precision": 6, "min_qty": 0.0})
    p = int(rule["precision"]); minq = float(rule["min_qty"])
    step = (10 ** (-p)) if p > 0 else 1.0
    qty = (int(max(balance_qty, 0.0) / step)) * step
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
def sma(values, period):
    if len(values) < period: return None
    return sum(values[-period:]) / period

def ema(values, period):
    if len(values) < period: return None
    k = 2 / (period + 1)
    e = float(values[0])
    for v in values[1:]:
        e = float(v) * k + e * (1 - k)
    return e

def compute_atr(c, period=14):
    if len(c) < period + 1: return None
    trs = []
    for i in range(1, len(c)):
        h,l,pc = c[i]["high"], c[i]["low"], c[i-1]["close"]
        trs.append(max(h-l, abs(h-pc), abs(l-pc)))
    atr = sum(trs[:period]) / period
    for x in trs[period:]:
        atr = (atr*(period-1)+x)/period
    return atr

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

def stoch_kd(c, period=14, smooth_k=3, smooth_d=3):
    if len(c) < period + smooth_k + smooth_d: return None, None
    ks = []
    for i in range(period-1, len(c)):
        win = c[i-period+1:i+1]
        hi = max(x["high"] for x in win); lo = min(x["low"] for x in win)
        cl = c[i]["close"]
        k_raw = 0.0 if hi == lo else (cl - lo) / (hi - lo) * 100.0
        ks.append(k_raw)
    def smooth(arr, times):
        out = list(arr)
        for _ in range(times-1):
            tmp = [out[0]] + [(out[j-1]+out[j])/2.0 for j in range(1, len(out))]
            out = tmp
        return out
    k_s = smooth(ks, smooth_k)
    d_s = smooth(k_s, smooth_d)
    return k_s[-1], d_s[-1]

def psar_series(highs, lows, step=0.02, max_step=0.2):
    n = len(highs)
    if n < 2: return [], []
    sar = [0.0] * n; bull = [False] * n
    uptrend = highs[1] >= highs[0]
    bull[1] = uptrend
    ep = highs[1] if uptrend else lows[1]
    sar[1] = lows[0] if uptrend else highs[0]
    af = step
    for i in range(2, n):
        prev_sar = sar[i-1]; prev_up = bull[i-1]
        sar_i = prev_sar + af * (ep - prev_sar)
        if prev_up:
            sar_i = min(sar_i, lows[i-1], lows[i-2] if i-2 >= 0 else lows[i-1])
        else:
            sar_i = max(sar_i, highs[i-1], highs[i-2] if i-2 >= 0 else highs[i-1])
        if prev_up:
            if lows[i] < sar_i:
                bull[i] = False; sar[i] = ep; af = step; ep = lows[i]
            else:
                bull[i] = True; sar[i] = sar_i
                if highs[i] > ep:
                    ep = highs[i]; af = min(max_step, af + step)
        else:
            if highs[i] > sar_i:
                bull[i] = True; sar[i] = ep; af = step; ep = highs[i]
            else:
                bull[i] = False; sar[i] = sar_i
                if lows[i] < ep:
                    ep = lows[i]; af = min(max_step, af + step)
    sar[0] = sar[1]; bull[0] = bull[1]
    return sar, bull

# =========================
# Candle aggregation
# =========================
def aggregate_for_intervals(pair):
    # Build candles for all INTERVALS from tick log
    ticks = tick_logs[pair]
    if not ticks: return
    ticks_sorted = sorted(ticks, key=lambda x: x[0])
    for interval in INTERVALS:
        candles = []
        candle = None; last_window = None
        for ts, price in ticks_sorted:
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
        candle_logs_map[pair][interval] = candles[-600:]

# =========================
# Preload history (try DCX; else synthetic seed)
# =========================
def _try_fetch_dcx_klines(pair, interval_sec, limit=200):
    # Known DCX public endpoint pattern (may vary). We protect with try/except and fall back.
    try:
        now = int(time.time())
        start = now - limit * interval_sec * 2
        # interval label guess: "1m","30s","15s"
        if interval_sec == 60: ilabel = "1m"
        elif interval_sec == 30: ilabel = "30s"
        else: ilabel = "15s"
        url = f"https://public.coindcx.com/market_data/candles?pair={pair}&interval={ilabel}&from={start}&to={now}"
        r = session.get(url, timeout=6)
        if not r.ok: return None
        data = r.json()
        out = []
        for k in data[-limit:]:
            # Expected (ts, open, high, low, close, volume) — we adapt if keys differ
            ts = int(k[0])
            o,h,l,c,v = float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]) if len(k) > 5 else 1.0
            out.append({"open":o,"high":h,"low":l,"close":c,"volume":v,"start":ts - (ts % interval_sec)})
        return out[-limit:]
    except Exception:
        return None

def _synthetic_seed_from_price(price, now, interval_sec, limit=200):
    # Build mild-noise bars around price so indicators initialize instantly.
    out=[]; ts = now - limit*interval_sec
    base = price
    for i in range(limit):
        drift = (i % 7 - 3) * (base * 0.00015)  # ±0.045% wiggle
        o = base + drift*0.2
        c = base + drift
        h = max(o, c) + abs(base)*0.0002
        l = min(o, c) - abs(base)*0.0002
        out.append({"open":o,"high":h,"low":l,"close":c,"volume":1,"start":ts})
        ts += interval_sec
    return out

def preload_pair(pair, last_price):
    now = int(time.time())
    for iv in INTERVALS:
        bars = _try_fetch_dcx_klines(pair, iv, PRELOAD_LIMIT)
        if bars is None or len(bars) < WARMUP_BARS:
            bars = _synthetic_seed_from_price(last_price, now, iv, PRELOAD_LIMIT)
        candle_logs_map[pair][iv] = bars[-600:]
        last_candle_ts_map[pair][iv] = bars[-1]["start"] if bars else 0
    scan_log.append(f"{ist_now()} | {pair} | Preloaded bars for {INTERVALS} secs")

# =========================
# Regime & edge scoring
# =========================
def atr_pct(c, period=14):
    atr = compute_atr(c, period)
    if not atr: return None
    px = c[-1]["close"] or 1.0
    return atr / px

def market_regime(c):
    if len(c) < WARMUP_BARS: return "neutral"
    closes = [x["close"] for x in c]
    ema20 = ema(closes, 20); ema50 = ema(closes, 50)
    if ema20 is None or ema50 is None: return "neutral"
    ap = atr_pct(c, 14)
    if ap is None: return "neutral"
    if ap < 0.002: return "silent"  # <0.2% per bar
    slope = abs((ema20 - ema50) / (closes[-1] or 1.0))
    if slope > 1.3 * ap: return "trend"
    return "range"

def fee_gate(c):
    ap = atr_pct(c, 14)
    if ap is None: return False
    needed = FEE_ROUND_TRIP + EDGE_BUFFER
    proj = ap * 2.0  # heuristic: can we get ~2*ATR on direction flip/swing
    return proj >= needed

def pick_best_interval(pair):
    best = None; best_score = -1e9; best_regime = "neutral"
    for iv in INTERVALS:
        C = candle_logs_map[pair][iv]
        if len(C) < WARMUP_BARS: continue
        ap = atr_pct(C, 14)
        if ap is None: continue
        rg = market_regime(C)
        if not fee_gate(C): continue
        # favor higher ATR% and trending
        trend_bonus = 1.15 if rg == "trend" else (1.0 if rg == "range" else 0.8)
        score = (ap - (FEE_ROUND_TRIP+EDGE_BUFFER)) * trend_bonus
        if score > best_score:
            best_score = score; best = iv; best_regime = rg
    # fallback to 60s if nothing passed
    return (best if best else 60), (best_regime if best else "neutral")

# =========================
# Signal generation on chosen interval
# =========================
def gen_signal_on(candles, regime):
    if len(candles) < WARMUP_BARS: return None
    closes = [c["close"] for c in candles]
    highs  = [c["high"]  for c in candles]
    lows   = [c["low"]   for c in candles]
    curr   = candles[-1]

    sma5  = sma(closes, 5)
    sma10 = sma(closes, 10)
    rsi14 = rsi_wilder_from_closes(closes[-15:], 14)
    k, d  = stoch_kd(candles, 14, 3, 3)
    sar, bull = psar_series(highs, lows, step=PSAR_STEP, max_step=PSAR_MAX)

    if any(x is None for x in [sma5, sma10, rsi14, k, d]) or len(sar) < 2:
        return None

    crossed_up   = (sma5 > sma10) or (closes[-1] > sma10)
    crossed_down = (sma5 < sma10) or (closes[-1] < sma10)
    sar_bull     = bool(bull[-1])
    sar_bear     = not sar_bull
    gap = abs((k or 0) - (d or 0))

    # TREND: continuation
    if regime == "trend":
        if crossed_up and sar_bull and rsi14 > 50 and k > d and gap >= 1.0:
            return {"side":"BUY","entry":curr["close"],"msg":"TREND BUY"}
        if crossed_down and sar_bear and rsi14 < 50 and k < d and gap >= 1.0:
            return {"side":"SELL","entry":curr["close"],"msg":"TREND SELL"}
        return None

    # RANGE: mean-reversion
    if regime == "range":
        if (rsi14 < 40 and k > d) or (crossed_up and sar_bull and rsi14 < 48):
            return {"side":"BUY","entry":curr["close"],"msg":"RANGE BUY"}
        if (rsi14 > 60 and k < d) or (crossed_down and sar_bear and rsi14 > 52):
            return {"side":"SELL","entry":curr["close"],"msg":"RANGE SELL"}
        return None

    # NEUTRAL: light filters
    if regime == "neutral":
        if crossed_up and sar_bull and rsi14 >= 51:
            return {"side":"BUY","entry":curr["close"],"msg":"NEUTRAL BUY"}
        if crossed_down and sar_bear and rsi14 <= 49:
            return {"side":"SELL","entry":curr["close"],"msg":"NEUTRAL SELL"}
        return None

    # SILENT: skip
    return None

# =========================
# Trading loop
# =========================
def scan_loop():
    global running, error_message
    scan_log.clear()

    # ---- Preload history from last price
    prices0 = fetch_all_prices()
    for p in PAIRS:
        px = prices0.get(p, {}).get("price")
        if px:
            preload_pair(p, px)

    while running:
        prices = fetch_all_prices()
        now = int(time.time())
        balances = get_wallet_balances()

        for pair in PAIRS:
            info = prices.get(pair)
            if not info: 
                continue

            # tick ingest
            price = info["price"]
            tick_logs[pair].append((now, price))
            if len(tick_logs[pair]) > 10000:
                tick_logs[pair] = tick_logs[pair][-10000:]

            # aggregate all intervals
            aggregate_for_intervals(pair)

            # pick best interval for this bar
            best_iv, regime = pick_best_interval(pair)
            C = candle_logs_map[pair][best_iv]
            if not C: 
                continue

            # only act when the chosen interval has a new closed bar
            if C[-1]["start"] == last_candle_ts_map[pair][best_iv]:
                continue
            last_candle_ts_map[pair][best_iv] = C[-1]["start"]

            sig = gen_signal_on(C, regime)
            if not sig:
                scan_log.append(f"{ist_now()} | {pair} | NoSig on {best_iv}s ({regime})")
                continue

            side  = sig["side"]; entry = sig["entry"]
            coin  = pair[:-4]

            if side == "SELL":
                # sell 100% wallet balance of this coin
                bal_qty = balances.get(coin, 0.0)
                qty = precise_sell_qty(pair, bal_qty)
                if qty > 0:
                    res = place_order(pair, "SELL", qty)
                    scan_log.append(f"{ist_now()} | {pair} | SELL {qty} @ {entry} [{best_iv}s/{regime}] | {sig['msg']}")
                    trade_log.append({"time": ist_now(),"pair": pair,"side":"SELL","entry":entry,"qty":qty,"msg":sig["msg"],"order_result":res})
                    if "error" in res: error_message = res["error"]
                    positions[pair] = None
                    balances = get_wallet_balances()
                else:
                    scan_log.append(f"{ist_now()} | {pair} | SELL sig, {coin} balance < min qty")

            else:  # BUY
                usdt = balances.get("USDT", 0.0)
                raw_qty = (BUY_USDT_ALLOC * usdt) / entry if entry > 0 else 0.0
                qty = clamp_buy_qty(pair, raw_qty)
                if qty > 0:
                    res = place_order(pair, "BUY", qty)
                    scan_log.append(f"{ist_now()} | {pair} | BUY {qty} @ {entry} [{best_iv}s/{regime}] | {sig['msg']}")
                    trade_log.append({"time": ist_now(),"pair": pair,"side":"BUY","entry":entry,"qty":qty,"msg":sig["msg"],"order_result":res})
                    if "error" in res: error_message = res["error"]
                    else:
                        positions[pair] = {"side":"BUY","qty":qty,"entry":entry}
                    balances = get_wallet_balances()
                else:
                    scan_log.append(f"{ist_now()} | {pair} | BUY sig, qty=0 after clamp")

        # trim logs
        if len(scan_log) > 800: scan_log[:] = scan_log[-800:]
        if len(trade_log) > 250: trade_log[:] = trade_log[-250:]
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
