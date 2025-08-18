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

# ===== Trading knobs (tuned for speed but with protection) =====
CANDLE_INTERVAL_SEC = 60      # 1m candles
POLL_SEC = 1.0                # fetch frequency (fast)

# Fee model (~1% round trip on spot)
FEE_ROUND_TRIP = 0.010        # 1.0%
EDGE_BUFFER    = 0.002        # +0.20% headroom

# Sizing & risk controls
BUY_USDT_ALLOC = 0.35         # buy with 35% of USDT
RISK_PCT       = 0.002        # 0.2% of USDT per trade for sizing cap
DAILY_STOP_PCT = 0.02         # stop trading after -2% day

# Exits (fast but safer)
ATR_PERIOD = 14
SL_ATR_K   = 0.7              # SL distance = 0.7 * ATR
TP_RR      = 1.3              # TP distance = 1.3 * SL
BE_AT_R    = 0.7              # Move SL to BE at +0.7R
TRAIL_ATR  = 0.8              # ATR trailing factor after BE

# PSAR (momentum)
PSAR_STEP = 0.02
PSAR_MAX  = 0.20

# Regime thresholds
WARMUP_BARS  = 30             # start after enough bars
TREND_SLOPE_MULT = 1.3        # |EMA20-EMA50|/Price > TREND_SLOPE_MULT*ATR% -> trend
MIN_ATR_PCT  = 0.002          # ignore ultra-low vol (0.2%/bar)

# =========================
# Time (IST)
# =========================
IST = timezone("Asia/Kolkata")
def ist_now(): return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
def ist_date(): return datetime.now(IST).strftime("%Y-%m-%d")

# =========================
# State
# =========================
tick_logs   = {p: [] for p in PAIRS}      # [(ts, price)]
candle_logs = {p: [] for p in PAIRS}      # [{open,high,low,close,volume,start}]
scan_log    = []
trade_log   = []
running     = False
status      = {"msg": "Idle", "last": ""}
error_message = ""
positions   = {p: None for p in PAIRS}    # {"side":"BUY","qty":float,"entry":float}
daily_profit = {}
usdt_day_open = {"date": None, "balance": None}

session = requests.Session()

# =========================
# API utils (unchanged order protocol)
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
            scan_log.append(f"{ist_now()} | BAL_ERR: {r.status_code} {r.text[:120]}")
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
            scan_log.append(f"{ist_now()} | PRICE_ERR: {r.status_code} {r.text[:120]}")
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

def compute_atr(candles, period=14):
    if len(candles) < period + 1: return None
    trs = []
    for i in range(1, len(candles)):
        h,l,pc = candles[i]["high"], candles[i]["low"], candles[i-1]["close"]
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

def stoch_kd(candles, period=14, smooth_k=3, smooth_d=3):
    if len(candles) < period + smooth_k + smooth_d: return None, None
    ks = []
    for i in range(period-1, len(candles)):
        win = candles[i-period+1:i+1]
        hi = max(c["high"] for c in win); lo = min(c["low"] for c in win)
        cl = candles[i]["close"]
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
# Candle aggregation (1m)
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
            candle["low"] = min(candle["low"], price)
            candle["close"] = price
            candle["volume"] += 1
    if candle: candles.append(candle)
    candle_logs[pair] = candles[-600:]  # ~10 hours of 1m bars

# =========================
# Regime detection (trend vs range) + fee-aware gate
# =========================
def market_regime(candles):
    if len(candles) < WARMUP_BARS: return "neutral"
    closes = [c["close"] for c in candles]
    ema20 = ema(closes, 20); ema50 = ema(closes, 50)
    if ema20 is None or ema50 is None: return "neutral"
    atr = compute_atr(candles, ATR_PERIOD)
    if not atr: return "neutral"
    px = closes[-1] or 1.0
    atr_pct = atr / px
    if atr_pct < MIN_ATR_PCT:  # too quiet, skip most trades
        return "silent"
    slope = abs(ema20 - ema50) / px
    # If EMA separation dominates typical bar size -> trending
    if slope > TREND_SLOPE_MULT * atr_pct:
        return "trend"
    # Else mean-reversion regime
    return "range"

def fee_gate(candles):
    atr = compute_atr(candles, ATR_PERIOD)
    if not atr: return False
    px = candles[-1]["close"] or 1.0
    edge_needed = FEE_ROUND_TRIP + EDGE_BUFFER
    proj = (atr / px) * 2.0  # heuristic: can we capture ~2*ATR in the move/flip
    return proj >= edge_needed

# =========================
# Signals (fast & regime-adaptive)
# =========================
def gen_signal(pair):
    C = candle_logs[pair]
    if len(C) < WARMUP_BARS: 
        return None

    if not fee_gate(C):
        return None

    closes = [c["close"] for c in C]
    highs  = [c["high"]  for c in C]
    lows   = [c["low"]   for c in C]
    curr   = C[-1]

    sma5  = sma(closes, 5)
    sma10 = sma(closes, 10)
    rsi14 = rsi_wilder_from_closes(closes[-15:], 14)
    k, d  = stoch_kd(C, 14, 3, 3)
    sar, bull = psar_series(highs, lows, step=PSAR_STEP, max_step=PSAR_MAX)

    if any(x is None for x in [sma5, sma10, rsi14, k, d]) or len(sar) < 2:
        return None

    crossed_up   = (sma5 > sma10) or (closes[-1] > sma10)
    crossed_down = (sma5 < sma10) or (closes[-1] < sma10)
    sar_bull     = bool(bull[-1])
    sar_bear     = not sar_bull

    regime = market_regime(C)

    # ---- Trend mode: continuation (more trades, better expectancy in trends)
    if regime == "trend":
        if crossed_up and sar_bull and rsi14 > 50 and k > d:
            return {"side": "BUY", "entry": curr["close"], "msg": "TREND BUY"}
        if crossed_down and sar_bear and rsi14 < 50 and k < d:
            return {"side": "SELL", "entry": curr["close"], "msg": "TREND SELL"}
        return None

    # ---- Range mode: mean-reversion (buy dips, sell rips)
    if regime == "range":
        # BUY near oversold with turn
        if (rsi14 < 38 and k > d) or (crossed_up and rsi14 < 45 and sar_bull):
            return {"side": "BUY", "entry": curr["close"], "msg": "RANGE BUY"}
        # SELL if we have coin and market is topping
        if (rsi14 > 62 and k < d) or (crossed_down and rsi14 > 55 and sar_bear):
            return {"side": "SELL", "entry": curr["close"], "msg": "RANGE SELL"}
        return None

    # ---- Neutral/silent: be picky (still allow some)
    if regime == "neutral":
        if crossed_up and sar_bull and rsi14 >= 52:
            return {"side": "BUY", "entry": curr["close"], "msg": "NEUTRAL BUY"}
        if crossed_down and sar_bear and rsi14 <= 48:
            return {"side": "SELL", "entry": curr["close"], "msg": "NEUTRAL SELL"}
        return None

    if regime == "silent":
        return None

# =========================
# Risk helpers & exits (fast but protected)
# =========================
exit_orders = []  # {"pair","side","qty","tp","sl","entry","half_taken","be_moved","trail_active"}

def daily_stop_hit(current_usdt):
    today = ist_date()
    if usdt_day_open["date"] != today:
        usdt_day_open["date"] = today
        usdt_day_open["balance"] = float(current_usdt)
        daily_profit[today] = 0.0
        return False
    start = usdt_day_open["balance"] or 0.0
    pl_today = daily_profit.get(today, 0.0)
    return (start > 0) and ((-pl_today) >= start * DAILY_STOP_PCT)

def monitor_exits(prices):
    global error_message
    done = []
    for ex in exit_orders:
        pair = ex["pair"]; side = ex["side"]; qty = ex["qty"]
        tp = ex["tp"]; sl = ex["sl"]; entry = ex["entry"]
        be_moved = ex.get("be_moved", False)
        half_taken = ex.get("half_taken", False)
        trail_active = ex.get("trail_active", True)

        price = prices.get(pair, {}).get("price")
        if not price: 
            continue

        risk = abs(entry - sl)

        # partial at +1R
        if not half_taken and risk > 0:
            hit1R = (side == "BUY" and price >= entry + risk) or (side == "SELL" and price <= entry - risk)
            if hit1R:
                half = precise_sell_qty(pair, qty*0.5) if side=="BUY" else precise_sell_qty(pair, qty*0.5)
                if half > 0:
                    res = place_order(pair, "SELL" if side=="BUY" else "BUY", half)
                    scan_log.append(f"{ist_now()} | {pair} | PART {('SELL' if side=='BUY' else 'BUY')} {half} @ {price}")
                    pl = (price - entry) * half if side=="BUY" else (entry - price)*half
                    daily_profit[ist_date()] = daily_profit.get(ist_date(),0)+pl
                    ex["qty"] = precise_sell_qty(pair, qty - half)
                    ex["half_taken"] = True
                    qty = ex["qty"]
                    if qty <= 0: 
                        done.append(ex); 
                        continue

        # move to BE at +0.7R
        if not be_moved and risk > 0:
            hitBE = (side == "BUY" and price >= entry + BE_AT_R*risk) or (side=="SELL" and price <= entry - BE_AT_R*risk)
            if hitBE:
                ex["sl"] = entry
                ex["be_moved"] = True
                scan_log.append(f"{ist_now()} | {pair} | SL->BE")

        # ATR trail after BE
        if trail_active:
            recent = candle_logs[pair][-(ATR_PERIOD+5):]
            atr_now = compute_atr(recent, ATR_PERIOD) if len(recent) >= ATR_PERIOD+1 else None
            if atr_now:
                if side=="BUY":
                    ex["sl"] = max(ex["sl"], price - TRAIL_ATR*atr_now)
                else:
                    ex["sl"] = min(ex["sl"], price + TRAIL_ATR*atr_now)

        # hard exits
        if qty <= 0:
            done.append(ex); 
            continue

        if side=="BUY" and (price >= tp or price <= ex["sl"]):
            res = place_order(pair, "SELL", qty)
            pl = (price - entry) * qty
            daily_profit[ist_date()] = daily_profit.get(ist_date(),0)+pl
            scan_log.append(f"{ist_now()} | {pair} | EXIT SELL {qty} @ {price} | PnL {round(pl,6)}")
            if "error" in res: error_message = res["error"]
            done.append(ex)

        if side=="SELL" and (price <= tp or price >= ex["sl"]):
            res = place_order(pair, "BUY", qty)
            pl = (entry - price) * qty
            daily_profit[ist_date()] = daily_profit.get(ist_date(),0)+pl
            scan_log.append(f"{ist_now()} | {pair} | EXIT BUY {qty} @ {price} | PnL {round(pl,6)}")
            if "error" in res: error_message = res["error"]
            done.append(ex)

    for ex in done:
        if ex in exit_orders:
            exit_orders.remove(ex)

# =========================
# Trading Loop
# =========================
def scan_loop():
    global running, error_message
    scan_log.clear()
    last_candle_ts = {p: 0 for p in PAIRS}

    while running:
        prices = fetch_all_prices()
        now = int(time.time())
        balances = get_wallet_balances()

        # Daily stop
        if daily_stop_hit(balances.get("USDT", 0.0)):
            scan_log.append(f"{ist_now()} | DAILY STOP HIT — pausing entries")
        else:
            monitor_exits(prices)

        for pair in PAIRS:
            info = prices.get(pair)
            if not info: 
                continue

            # ingest ticks
            price = info["price"]
            tick_logs[pair].append((now, price))
            if len(tick_logs[pair]) > 5000:
                tick_logs[pair] = tick_logs[pair][-5000:]

            # candle aggregation
            aggregate_candles(pair, CANDLE_INTERVAL_SEC)
            last_candle = candle_logs[pair][-1] if candle_logs[pair] else None

            # act on new bar close
            if last_candle and last_candle["start"] != last_candle_ts[pair]:
                last_candle_ts[pair] = last_candle["start"]
                sig = gen_signal(pair)

                # no entry/exit if daily stop hit
                if daily_stop_hit(balances.get("USDT", 0.0)):
                    scan_log.append(f"{ist_now()} | {pair} | Entry blocked by daily stop")
                    continue

                if sig:
                    side  = sig["side"]
                    entry = sig["entry"]
                    coin  = pair[:-4]

                    # ATR-based exits
                    recent = candle_logs[pair][-(ATR_PERIOD+20):]
                    atr = compute_atr(recent, ATR_PERIOD)
                    if not atr:
                        scan_log.append(f"{ist_now()} | {pair} | ATR not ready, skip signal")
                        continue

                    sl_dist = SL_ATR_K * atr
                    tp_dist = TP_RR * sl_dist

                    if side == "SELL":
                        # SELL whatever coin you hold
                        raw_qty = balances.get(coin, 0.0)
                        qty = precise_sell_qty(pair, raw_qty)
                        if qty > 0:
                            res = place_order(pair, "SELL", qty)
                            scan_log.append(f"{ist_now()} | {pair} | SELL {qty} @ {entry} | {sig['msg']}")
                            trade_log.append({
                                "time": ist_now(), "pair": pair, "side": "SELL",
                                "entry": entry, "msg": sig["msg"], "qty": qty, "order_result": res
                            })
                            if "error" in res:
                                error_message = res["error"]
                            else:
                                # track reverse-exit (for symmetry; rare on spot)
                                exit_orders.append({
                                    "pair": pair, "side": "SELL", "qty": qty,
                                    "tp": round(entry - tp_dist, 6), "sl": round(entry + sl_dist, 6),
                                    "entry": entry, "half_taken": False, "be_moved": False, "trail_active": True
                                })
                            # realized PnL will be captured on exit_orders; for direct SELL (no short), we skip
                            balances = get_wallet_balances()
                        else:
                            scan_log.append(f"{ist_now()} | {pair} | SELL sig, {coin} balance < min qty")

                    else:  # BUY
                        usdt = balances.get("USDT", 0.0)
                        # risk-based sizing + budget cap
                        risk_usdt = usdt * RISK_PCT
                        qty_risk  = (risk_usdt / sl_dist) if sl_dist > 0 else 0.0
                        qty_budget= (BUY_USDT_ALLOC * usdt) / entry if entry > 0 else 0.0
                        raw_qty   = max(0.0, min(qty_risk, qty_budget))
                        qty = clamp_buy_qty(pair, raw_qty)

                        if qty > 0:
                            res = place_order(pair, "BUY", qty)
                            scan_log.append(f"{ist_now()} | {pair} | BUY {qty} @ {entry} | {sig['msg']}")
                            trade_log.append({
                                "time": ist_now(), "pair": pair, "side": "BUY",
                                "entry": entry, "msg": sig["msg"], "qty": qty, "order_result": res
                            })
                            if "error" in res:
                                error_message = res["error"]
                            else:
                                positions[pair] = {"side":"BUY","qty":qty,"entry":entry}
                                exit_orders.append({
                                    "pair": pair, "side": "BUY", "qty": qty,
                                    "tp": round(entry + tp_dist, 6), "sl": round(entry - sl_dist, 6),
                                    "entry": entry, "half_taken": False, "be_moved": False, "trail_active": True
                                })
                            balances = get_wallet_balances()
                        else:
                            scan_log.append(f"{ist_now()} | {pair} | BUY sig, qty=0 after clamp")
                else:
                    scan_log.append(f"{ist_now()} | {pair} | No Signal")

        # trim logs
        if len(scan_log) > 600: scan_log[:] = scan_log[-600:]
        if len(trade_log) > 200: trade_log[:] = trade_log[-200:]

        status["msg"], status["last"] = "Running", ist_now()
        time.sleep(POLL_SEC)

    status["msg"] = "Idle"

# =========================
# Routes (unchanged)
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
    today = ist_date()
    return jsonify({
        "status": status["msg"],
        "last": status["last"],
        "usdt": balances.get("USDT", 0.0),
        "profit_today": round(daily_profit.get(today, 0.0), 4),
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
