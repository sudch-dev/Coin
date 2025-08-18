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

# ============================================================
# Flask
# ============================================================
app = Flask(__name__)

# ============================================================
# Config / Constants
# ============================================================
API_KEY = os.environ.get("API_KEY")
API_SECRET = (os.environ.get("API_SECRET") or "").encode()
BASE_URL = "https://api.coindcx.com"

# Trade universe (spot)
PAIRS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "SHIBUSDT", "SOLUSDT",
    "DOGEUSDT", "ADAUSDT", "AEROUSDT", "BNBUSDT", "LTCUSDT"
]

# Pair precision & min qty (adjust if your DCX market specs differ)
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

# Engine tuning (more trades, fee-aware)
CONFIG = {
    "POLL_SEC": 1.0,            # fast polling
    "BUY_USDT_ALLOC": 0.35,     # 35% of USDT per BUY
    "FEE_RT": 0.01,             # 1% round-trip cost
    "EDGE_BUFFER": 0.005,       # +0.5% buffer => require ~1.5% move
    # regime thresholds (based on 1m ATR%)
    "ATR_HYPER": 0.0060,        # >=0.60% => hyper, use 1m
    "ATR_TREND": 0.0030,        # 0.30%..0.60% => trend, use 5m
    # confirmation gates (looser => more trades)
    "RSI_LONG_MIN": 50.0,       # RSI >= 50 for longs
    "RSI_SHORT_MAX": 50.0,      # RSI <= 50 for sells/exit
    "STOCH_KD_MIN": 20.0,       # require K,D > 20 on longs (leaving oversold)
    "STOCH_KD_MAX": 80.0,       # require K,D < 80 on sells (leaving overbought)
    "USE_TREND_FILTER": True,   # close vs EMA50 alignment
}

# Parabolic SAR (final entry/exit switch)
PSAR_STEP = 0.02
PSAR_MAX  = 0.20

# Intervals we maintain (seconds)
INTERVALS = [60, 300, 900]  # 1m, 5m, 15m

# ============================================================
# Time helpers (IST)
# ============================================================
IST = timezone("Asia/Kolkata")
def ist_now(): return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
def ist_date(): return datetime.now(IST).strftime("%Y-%m-%d")

# ============================================================
# State
# ============================================================
# Multi-interval candles
candle_logs_map = {p: {i: [] for i in INTERVALS} for p in PAIRS}   # closed candles
candle_live_map = {p: {i: None for i in INTERVALS} for p in PAIRS} # building candle

# Small tick buffer (optional, for visibility)
tick_logs = {p: [] for p in PAIRS}

# UI logs
scan_log  = []
trade_log = []

running     = False
status      = {"msg": "Idle", "last": ""}
error_message = ""

# Spot positions (long-only)
# positions[pair] = {"side": "BUY", "qty": float, "entry": float}
positions = {p: None for p in PAIRS}

# HTTP session + balance cache
session = requests.Session()
balances_cache = {"ts": 0, "data": {}}
BAL_TTL_SEC = 10

# ============================================================
# Utils / CoinDCX API
# ============================================================
def hmac_signature(payload: str) -> str:
    return hmac.new(API_SECRET, payload.encode(), hashlib.sha256).hexdigest()

def get_wallet_balances(force: bool = False):
    now = time.time()
    if not force and (now - balances_cache["ts"] < BAL_TTL_SEC):
        return dict(balances_cache["data"])
    payload = json.dumps({"timestamp": int(now * 1000)})
    sig = hmac_signature(payload)
    headers = {"X-AUTH-APIKEY": API_KEY, "X-AUTH-SIGNATURE": sig, "Content-Type": "application/json"}
    out = {}
    try:
        r = session.post(f"{BASE_URL}/exchange/v1/users/balances", headers=headers, data=payload, timeout=8)
        if r.ok:
            for b in r.json():
                out[b["currency"]] = float(b["balance"])
            balances_cache["data"] = out
            balances_cache["ts"] = now
        else:
            scan_log.append(f"{ist_now()} | BAL_ERR: {r.status_code} {r.text[:120]}")
    except Exception as e:
        scan_log.append(f"{ist_now()} | BAL_ERR: {e}")
    return out

def fetch_all_prices():
    try:
        r = session.get(f"{BASE_URL}/exchange/ticker", timeout=8)
        if r.ok:
            now = int(time.time())
            m = {}
            for item in r.json():
                mk = item.get("market")
                if mk in PAIRS:
                    m[mk] = {"price": float(item["last_price"]), "ts": now}
            return m
        else:
            scan_log.append(f"{ist_now()} | PRICE_ERR: {r.status_code} {r.text[:120]}")
    except Exception as e:
        scan_log.append(f"{ist_now()} | PRICE_ERR: {e}")
    return {}

def clamp_buy_qty(pair, qty):
    rule = PAIR_RULES.get(pair, {"precision": 6, "min_qty": 0.0})
    precision = int(rule["precision"]); min_qty = float(rule["min_qty"])
    step = (10 ** (-precision)) if precision > 0 else 1.0
    qty = (int(max(qty, 0.0) / step)) * step
    if qty < min_qty: return 0.0
    return round(qty, precision)

def precise_sell_qty(pair, balance_qty):
    rule = PAIR_RULES.get(pair, {"precision": 6, "min_qty": 0.0})
    precision = int(rule["precision"]); min_qty = float(rule["min_qty"])
    step = (10 ** (-precision)) if precision > 0 else 1.0
    qty = (int(max(balance_qty, 0.0) / step)) * step
    if qty < min_qty: return 0.0
    return round(qty, precision)

def place_order(pair, side, qty):
    """Protocol unchanged: market order with total_quantity."""
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

# ============================================================
# Indicators
# ============================================================
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

def stoch_kd(candles, period=14, smooth_k=3, smooth_d=3):
    if len(candles) < period + smooth_k + smooth_d: return None, None
    ks = []
    for i in range(period-1, len(candles)):
        win = candles[i-period+1:i+1]
        high = max(c["high"] for c in win)
        low  = min(c["low"]  for c in win)
        close = candles[i]["close"]
        k_raw = 0 if high == low else (close - low) / (high - low) * 100.0
        ks.append(k_raw)
    # smooth K
    k_s = ks
    for _ in range(smooth_k-1):
        tmp = []
        for j in range(1, len(k_s)):
            tmp.append((k_s[j-1] + k_s[j]) / 2.0)
        k_s = [k_s[0]] + tmp
    # smooth D from K
    d_s = k_s
    for _ in range(smooth_d-1):
        tmp = []
        for j in range(1, len(d_s)):
            tmp.append((d_s[j-1] + d_s[j]) / 2.0)
        d_s = [d_s[0]] + tmp
    return k_s[-1], d_s[-1]

def psar_series(highs, lows, step=0.02, max_step=0.2):
    n = len(highs)
    if n < 2: return [], []
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

# ============================================================
# Mood / Regime & Interval selection
# ============================================================
def atr_percent(candles, period=14):
    if len(candles) < period + 1: return None
    trs = []
    for i in range(1, len(candles)):
        h,l,pc = candles[i]["high"], candles[i]["low"], candles[i-1]["close"]
        trs.append(max(h-l, abs(h-pc), abs(l-pc)))
    atr = sum(trs[:period]) / period
    for x in trs[period:]:
        atr = (atr*(period-1)+x)/period
    px = candles[-1]["close"] or 1.0
    return atr / px

def classify_regime_1m(candles_1m):
    atrp = atr_percent(candles_1m, 14) or 0.0
    if atrp >= CONFIG["ATR_HYPER"]:
        return "hyper", atrp
    if atrp >= CONFIG["ATR_TREND"]:
        return "trend", atrp
    return "chop", atrp

def select_interval_from_regime(regime):
    if regime == "hyper": return 60   # 1m
    if regime == "trend": return 300  # 5m
    return 900                        # 15m

def projected_move_ok(regime, atrp_interval):
    if not atrp_interval: return False
    # Aggressive multipliers (still fee-aware)
    if regime == "hyper":  proj = 3.0 * atrp_interval
    elif regime == "trend": proj = 2.0 * atrp_interval
    else: return False
    return proj >= (CONFIG["FEE_RT"] + CONFIG["EDGE_BUFFER"])  # ~1.5%

# ============================================================
# Signal: SMA5/10 + RSI14 + Stoch(14,3,3) + SAR as final gate
# ============================================================
def mood_signal(pair, logs_by_interval):
    """
    Returns dict {side, entry, interval, msg} or None.
    - Determine regime on 1m.
    - Choose interval (1m/5m/15m).
    - Confirm with SMA(5/10), RSI(14), Stoch(14,3,3).
    - SAR must align (final switch).
    - Fee-aware: projected edge must exceed ~1.5%.
    """
    C1 = logs_by_interval[60]
    if len(C1) < 25: return None

    regime, atrp_1m = classify_regime_1m(C1)
    interval = select_interval_from_regime(regime)
    C = logs_by_interval[interval]
    if len(C) < 25: return None

    closes = [c["close"] for c in C]
    highs  = [c["high"]  for c in C]
    lows   = [c["low"]   for c in C]
    curr   = C[-1]

    sma5  = sma(closes, 5);  sma10 = sma(closes, 10)
    ema50 = ema(closes, 50) if CONFIG["USE_TREND_FILTER"] else None
    rsi14 = rsi_wilder_from_closes(closes[-15:], 14)
    k, d  = stoch_kd(C, 14, 3, 3)
    sar, bull = psar_series(highs, lows, step=PSAR_STEP, max_step=PSAR_MAX)
    atrp_int = atr_percent(C, 14)

    if any(x is None for x in [sma5, sma10, rsi14, k, d]) or sar is None:
        return None

    # Fee-aware edge check
    if not projected_move_ok(regime, atrp_int):
        return None

    # Cross helpers (recent cross in last bar or currently above/below)
    crossed_up   = (closes[-2] <= sma10) and (closes[-1] > sma10) or (sma5 > sma10 and sma5 - sma10 > 0)
    crossed_down = (closes[-2] >= sma10) and (closes[-1] < sma10) or (sma5 < sma10 and sma10 - sma5 > 0)

    # Long conditions
    long_trend_ok = True if ema50 is None else (closes[-1] > ema50)
    long_mom_ok   = (rsi14 >= CONFIG["RSI_LONG_MIN"]) and (k is not None and d is not None and k > d and k >= CONFIG["STOCH_KD_MIN"] and d >= CONFIG["STOCH_KD_MIN"])
    sar_bull      = bool(bull[-1])  # SAR below price

    if crossed_up and long_trend_ok and long_mom_ok and sar_bull:
        return {
            "side": "BUY",
            "entry": curr["close"],
            "interval": interval,
            "msg": f"BUY[{interval//60}m] {regime}: SMA5>10, RSI={round(rsi14,1)}, Stoch K>D, SAR bullish"
        }

    # Sell/Exit conditions (spot: exit long; may also sell residual balances)
    short_trend_ok = True if ema50 is None else (closes[-1] < ema50)
    short_mom_ok   = (rsi14 <= CONFIG["RSI_SHORT_MAX"]) and (k is not None and d is not None and k < d and k <= CONFIG["STOCH_KD_MAX"] and d <= CONFIG["STOCH_KD_MAX"])
    sar_bear       = not bool(bull[-1])  # SAR above price

    if crossed_down and short_trend_ok and short_mom_ok and sar_bear:
        return {
            "side": "SELL",
            "entry": curr["close"],
            "interval": interval,
            "msg": f"SELL[{interval//60}m] {regime}: SMA5<10, RSI={round(rsi14,1)}, Stoch K<D, SAR bearish"
        }

    return None

# ============================================================
# Incremental candle builder (no sorting; fast)
# ============================================================
def on_price(pair, ts, price):
    """
    Feed one tick -> update all live candles.
    Returns True if any candle closed (so we evaluate once per batch).
    """
    any_closed = False
    for interval in INTERVALS:
        wstart = ts - (ts % interval)
        c = candle_live_map[pair][interval]
        if (c is None) or (c["start"] != wstart):
            # finalize previous
            if c is not None:
                candle_logs_map[pair][interval].append(c)
                if len(candle_logs_map[pair][interval]) > 900:
                    candle_logs_map[pair][interval] = candle_logs_map[pair][interval][-900:]
                any_closed = True
            # start new
            candle_live_map[pair][interval] = {
                "open": price, "high": price, "low": price, "close": price,
                "volume": 1, "start": wstart
            }
        else:
            c["high"] = max(c["high"], price)
            c["low"]  = min(c["low"], price)
            c["close"] = price
            c["volume"] += 1
    return any_closed

# ============================================================
# Trading Loop (entry + reverse-exit on opposite signal)
# ============================================================
def scan_loop():
    global running, error_message
    scan_log.clear()

    while running:
        prices = fetch_all_prices()
        now = int(time.time())
        balances = get_wallet_balances(force=False)

        for pair in PAIRS:
            info = prices.get(pair)
            if not info: continue

            price = info["price"]
            # keep tiny tick buffer for visibility
            tl = tick_logs[pair]; tl.append((now, price))
            if len(tl) > 60: tick_logs[pair] = tl[-60:]

            # update candles for all intervals
            closed = on_price(pair, now, price)
            if not closed:
                continue  # wait for a close to evaluate

            # evaluate mood signal
            signal = mood_signal(pair, candle_logs_map[pair])

            if signal:
                side  = signal["side"]
                entry = signal["entry"]
                coin  = pair[:-4]

                if side == "SELL":
                    had_pos = positions.get(pair) is not None
                    raw_qty = balances.get(coin, 0.0)
                    qty = precise_sell_qty(pair, raw_qty)
                    if qty > 0:
                        res = place_order(pair, "SELL", qty)
                        tag = "EXIT LONG & SELL" if had_pos else "SELL"
                        scan_log.append(f"{ist_now()} | {pair} | {tag} qty={qty} @ {entry} | {res} | {signal['msg']}")
                        trade_log.append({
                            "time": ist_now(), "pair": pair, "side": "SELL",
                            "entry": entry, "msg": signal["msg"], "qty": qty, "order_result": res
                        })
                        if "error" in res: error_message = res["error"]
                        positions[pair] = None
                        get_wallet_balances(force=True)
                    else:
                        scan_log.append(f"{ist_now()} | {pair} | SELL signal, no {coin} balance (or < min qty)")

                else:  # BUY
                    pos = positions.get(pair)
                    if pos and pos["side"] == "BUY" and pos["qty"] > 0:
                        scan_log.append(f"{ist_now()} | {pair} | BUY signal but already long; skip")
                    else:
                        usdt = balances.get("USDT", 0.0)
                        raw_qty = (CONFIG["BUY_USDT_ALLOC"] * usdt) / entry if entry else 0.0
                        qty = clamp_buy_qty(pair, raw_qty)
                        if qty > 0:
                            res = place_order(pair, "BUY", qty)
                            scan_log.append(f"{ist_now()} | {pair} | BUY qty={qty} @ {entry} | {res} | {signal['msg']}")
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

        # Trim logs
        if len(scan_log) > 600: scan_log[:] = scan_log[-600:]
        if len(trade_log) > 300: trade_log[:] = trade_log[-300:]

        status["msg"], status["last"] = "Running", ist_now()
        time.sleep(CONFIG["POLL_SEC"])

    status["msg"] = "Idle"

# ============================================================
# Routes (structure intact)
# ============================================================
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

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # Program ID: ABCD
    app.run(host="0.0.0.0", port=10000)
