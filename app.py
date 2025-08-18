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
import math
import random

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

# Intervals (seconds)
INTERVALS = [60, 300, 900]  # 1m, 5m, 15m

# Fees & sizing
FEE_ROUND_TRIP = 0.01       # ~1% round trip
EDGE_BUFFER    = 0.003      # +0.3% headroom
BUY_USDT_ALLOC = 0.35       # 35% of USDT per buy

# Parabolic SAR params
PSAR_STEP = 0.02
PSAR_MAX  = 0.20

# Poll speed
POLL_SEC = 1.0

# =========================
# Time (IST)
# =========================
IST = timezone("Asia/Kolkata")
def ist_now(): return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
def ist_date(): return datetime.now(IST).strftime("%Y-%m-%d")

# =========================
# State
# =========================
# live + closed candles per interval
candle_logs_map = {p: {i: [] for i in INTERVALS} for p in PAIRS}
candle_live_map = {p: {i: None for i in INTERVALS} for p in PAIRS}
tick_logs = {p: [] for p in PAIRS}

scan_log, trade_log = [], []
running = False
status = {"msg": "Idle", "last": ""}
error_message = ""

# Positions: track last BUY to compute realized PnL at SELL
# positions[pair] = {"qty": float, "entry": float, "arm": (interval, aggression)}
positions = {p: None for p in PAIRS}

# Session + cached balances
session = requests.Session()
balances_cache = {"ts": 0, "data": {}}
BAL_TTL_SEC = 8

# =========================
# Adaptive controller (Multi-armed bandit per pair)
# Arms = all combinations of (interval ∈ {1m,5m,15m}) × (aggression ∈ {loose, base, strict})
# UCB1 implementation (fast, simple)
# =========================
AGG_LEVELS = ["loose", "base", "strict"]
ARMS = [(i, a) for i in INTERVALS for a in AGG_LEVELS]

class UCB1:
    def __init__(self, arms):
        self.arms = list(arms)
        self.n = {arm: 0 for arm in self.arms}       # pulls
        self.q = {arm: 0.0 for arm in self.arms}     # mean reward
        self.t = 0                                   # total pulls

    def select(self):
        self.t += 1
        # try all arms at least once
        for arm in self.arms:
            if self.n[arm] == 0:
                return arm
        # UCB score
        scores = []
        for arm in self.arms:
            bonus = math.sqrt(2.0 * math.log(self.t) / self.n[arm])
            scores.append((self.q[arm] + bonus, arm))
        scores.sort(reverse=True)
        return scores[0][1]

    def update(self, arm, reward):
        # reward is realized PnL% net of fees (e.g., +0.6% => 0.006)
        self.n[arm] += 1
        lr = 1.0 / self.n[arm]
        self.q[arm] = self.q[arm] + lr * (reward - self.q[arm])

bandits = {p: UCB1(ARMS) for p in PAIRS}

# =========================
# API utils
# =========================
def hmac_signature(payload: str) -> str:
    return hmac.new(API_SECRET, payload.encode(), hashlib.sha256).hexdigest()

def get_wallet_balances(force=False):
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
            scan_log.append(f"{ist_now()} | BAL_ERR: {r.status_code} {r.text[:100]}")
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
            scan_log.append(f"{ist_now()} | PRICE_ERR: {r.status_code} {r.text[:100]}")
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
    payload = {
        "market": pair,
        "side": side.lower(),
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
        hi = max(c["high"] for c in win); lo = min(c["low"] for c in win)
        cl = candles[i]["close"]
        k_raw = 0.0 if hi == lo else (cl - lo) / (hi - lo) * 100.0
        ks.append(k_raw)
    # simple smoothing
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

# aggression thresholds
AGG_THRESH = {
    "loose":  {"rsi_buy": 48.0, "rsi_sell": 52.0, "stoch_gap": 0.0,  "edge_mult": 1.2},
    "base":   {"rsi_buy": 50.0, "rsi_sell": 50.0, "stoch_gap": 2.0,  "edge_mult": 1.4},
    "strict": {"rsi_buy": 52.0, "rsi_sell": 48.0, "stoch_gap": 5.0,  "edge_mult": 1.6},
}

# =========================
# Candle building (all intervals)
# =========================
def on_price(pair, ts, price):
    closed_any = False
    for interval in INTERVALS:
        wstart = ts - (ts % interval)
        c = candle_live_map[pair][interval]
        if (c is None) or (c["start"] != wstart):
            if c is not None:
                candle_logs_map[pair][interval].append(c)
                if len(candle_logs_map[pair][interval]) > 1000:
                    candle_logs_map[pair][interval] = candle_logs_map[pair][interval][-1000:]
                closed_any = True
            candle_live_map[pair][interval] = {
                "open": price, "high": price, "low": price, "close": price,
                "volume": 1, "start": wstart
            }
        else:
            c["high"] = max(c["high"], price)
            c["low"]  = min(c["low"], price)
            c["close"] = price
            c["volume"] += 1
    return closed_any

# =========================
# Signal using chosen arm (interval, aggression) by bandit
# =========================
def signal_for_arm(pair, interval, aggression):
    C = candle_logs_map[pair][interval]
    if len(C) < 25: 
        return None

    closes = [c["close"] for c in C]
    highs  = [c["high"]  for c in C]
    lows   = [c["low"]   for c in C]
    curr   = C[-1]

    # Indicators
    sma5  = sma(closes, 5); sma10 = sma(closes, 10)
    rsi14 = rsi_wilder_from_closes(closes[-15:], 14)
    k, d  = stoch_kd(C, 14, 3, 3)
    sar, bull = psar_series(highs, lows, step=PSAR_STEP, max_step=PSAR_MAX)
    atrp = atr_percent(C, 14)

    if any(x is None for x in [sma5, sma10, rsi14, k, d]) or len(sar) < 2:
        return None

    # Edge gate (fee-aware) with aggression
    need = (FEE_ROUND_TRIP + EDGE_BUFFER) * AGG_THRESH[aggression]["edge_mult"] / 1.2
    if not atrp or atrp * 2.0 < need:  # 2.0 is a heuristic multiplier
        return None

    crossed_up   = (sma5 > sma10) or (closes[-1] > sma10)
    crossed_down = (sma5 < sma10) or (closes[-1] < sma10)
    sar_bull     = bool((psar_series(highs, lows, PSAR_STEP, PSAR_MAX)[1])[-1])  # recompute small but ok
    sar_bear     = not sar_bull
    gap = abs((k or 0) - (d or 0))

    rsi_buy  = AGG_THRESH[aggression]["rsi_buy"]
    rsi_sell = AGG_THRESH[aggression]["rsi_sell"]
    st_gap   = AGG_THRESH[aggression]["stoch_gap"]

    # BUY
    if crossed_up and sar_bull and (rsi14 >= rsi_buy) and (k is not None and d is not None and (k > d) and (gap >= st_gap)):
        return {"side": "BUY", "entry": curr["close"], "msg": f"BUY[{interval//60}m-{aggression}] SMA/RSI/Stoch/SAR OK"}

    # SELL
    if crossed_down and sar_bear and (rsi14 <= rsi_sell) and (k is not None and d is not None and (k < d) and (gap >= st_gap)):
        return {"side": "SELL", "entry": curr["close"], "msg": f"SELL[{interval//60}m-{aggression}] SMA/RSI/Stoch/SAR OK"}

    return None

# =========================
# Trading loop
# =========================
def scan_loop():
    global running, error_message
    scan_log.clear()
    last_price_seen = {}

    while running:
        prices = fetch_all_prices()
        now = int(time.time())
        balances = get_wallet_balances(force=False)

        for pair in PAIRS:
            info = prices.get(pair)
            if not info: 
                continue

            px = info["price"]
            last_price_seen[pair] = px

            tl = tick_logs[pair]; tl.append((now, px))
            if len(tl) > 90: tick_logs[pair] = tl[-90:]

            closed = on_price(pair, now, px)
            if not closed:
                continue

            # 1) pick arm with bandit
            arm = bandits[pair].select()
            interval, aggression = arm

            # 2) get signal for selected arm
            sig = signal_for_arm(pair, interval, aggression)

            if not sig:
                scan_log.append(f"{ist_now()} | {pair} | NoSig on {interval//60}m-{aggression}")
                continue

            side = sig["side"]; entry = sig["entry"]
            coin = pair[:-4]

            if side == "SELL":
                # Realize PnL if we had a BUY
                had_pos = positions.get(pair)
                raw_qty = balances.get(coin, 0.0)
                qty = precise_sell_qty(pair, raw_qty)
                if qty > 0:
                    res = place_order(pair, "SELL", qty)
                    scan_log.append(f"{ist_now()} | {pair} | SELL {qty} @ {entry} | {res} | {sig['msg']}")
                    trade_log.append({
                        "time": ist_now(), "pair": pair, "side": "SELL",
                        "entry": entry, "msg": sig["msg"], "qty": qty, "order_result": res
                    })
                    if "error" in res:
                        error_message = res["error"]
                    # reward update if we had a tracked BUY arm
                    if had_pos:
                        # approximate PnL% net of fee (double-count fee once for round-trip)
                        buy_entry = had_pos["entry"]; buy_qty = had_pos["qty"]
                        gross = (entry - buy_entry) / buy_entry if buy_entry else 0.0
                        reward = gross - FEE_ROUND_TRIP
                        bandits[pair].update(had_pos["arm"], reward)
                        scan_log.append(f"{ist_now()} | {pair} | UCB UPDATE arm={had_pos['arm']} reward={round(reward*100,2)}%")
                    positions[pair] = None
                    get_wallet_balances(force=True)
                else:
                    scan_log.append(f"{ist_now()} | {pair} | SELL sig, no {coin} balance")

            else:
                # BUY only if not already long
                if positions.get(pair):
                    scan_log.append(f"{ist_now()} | {pair} | BUY sig but already long; skip")
                    continue
                usdt = balances.get("USDT", 0.0)
                raw_qty = (BUY_USDT_ALLOC * usdt) / entry if entry else 0.0
                qty = clamp_buy_qty(pair, raw_qty)
                if qty > 0:
                    res = place_order(pair, "BUY", qty)
                    scan_log.append(f"{ist_now()} | {pair} | BUY {qty} @ {entry} | {res} | {sig['msg']}")
                    trade_log.append({
                        "time": ist_now(), "pair": pair, "side": "BUY",
                        "entry": entry, "msg": sig["msg"], "qty": qty, "order_result": res
                    })
                    if "error" in res:
                        error_message = res["error"]
                    else:
                        positions[pair] = {"qty": qty, "entry": entry, "arm": arm}
                    get_wallet_balances(force=True)
                else:
                    scan_log.append(f"{ist_now()} | {pair} | BUY sig qty=0 after clamp; skip")

        # trim logs
        if len(scan_log) > 700: scan_log[:] = scan_log[-700:]
        if len(trade_log) > 300: trade_log[:] = trade_log[-300:]

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
    # include bandit diagnostics
    diag = {
        p: sorted(
            [((i//60), a, round(bandits[p].q[(i,a)]*100,2), bandits[p].n[(i,a)]) for (i,a) in ARMS],
            key=lambda x: (-x[2], -x[3])
        )[:4]
        for p in PAIRS
    }
    return jsonify({
        "status": status["msg"],
        "last": status["last"],
        "usdt": balances.get("USDT", 0.0),
        "coins": coins,
        "positions": pos_view,
        "trades": trade_log[-12:][::-1],
        "scans": scan_log[-40:][::-1],
        "bandit": diag,
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
