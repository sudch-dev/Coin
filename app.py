import os
import time
import threading
import hmac
import hashlib
import requests
import json
import traceback
from flask import Flask, render_template, jsonify
from datetime import datetime, timedelta
from pytz import timezone
from collections import deque

# -------------------- Flask --------------------
app = Flask(__name__)

APP_BASE_URL = os.environ.get("APP_BASE_URL", "https://coin-4k37.onrender.com")

# -------------------- API / Markets --------------------
API_KEY = os.environ.get("API_KEY")
API_SECRET_RAW = os.environ.get("API_SECRET", "")
API_SECRET = API_SECRET_RAW.encode() if isinstance(API_SECRET_RAW, str) else API_SECRET_RAW
BASE_URL = "https://api.coindcx.com"

PAIRS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "SHIBUSDT", "SOLUSDT",
    "DOGEUSDT", "ADAUSDT", "AEROUSDT", "BNBUSDT", "LTCUSDT"
]

# local fallbacks; we also fetch precision at boot
PAIR_RULES = {
    "BTCUSDT": {"precision": 6, "min_qty": 0.0005},
    "ETHUSDT": {"precision": 6, "min_qty": 0.0001},
    "XRPUSDT": {"precision": 1, "min_qty": 1},
    "SHIBUSDT": {"precision": 0, "min_qty": 10000},
    "DOGEUSDT": {"precision": 1, "min_qty": 5},
    "SOLUSDT": {"precision": 4, "min_qty": 0.01},
    "AEROUSDT": {"precision": 2, "min_qty": 1},
    "ADAUSDT": {"precision": 1, "min_qty": 2},
    "LTCUSDT": {"precision": 4, "min_qty": 0.01},
    "BNBUSDT": {"precision": 4, "min_qty": 0.01}
}

# -------------------- Maker / HFT settings --------------------
MODE = "maker"                 # quoting engine (spread capture, inventory control)

# fast, tight management to avoid stale quotes
CANDLE_INTERVAL = 10           # 10s bars for trend/vol/SMC gates
POLL_SEC = 1.0                 # quote loop cadence
QUOTE_TTL_SEC = 4              # replace quotes quickly
DRIFT_REQUOTE_PCT = 0.0003     # 0.03% drift => reprice

# fees + required edge
FEE_PCT_PER_SIDE = 0.0010      # set your maker fee tier (0.10%)
TP_BUFFER_PCT = 0.0006         # extra buffer > fees (0.06%)
SPREAD_OFFSET_PCT = None       # if None, auto from fees; else fixed

# ATR / slope gating to avoid getting picked off in fast moves
ATR_WINDOW = 18                # ~3 minutes of 10s bars
ATR_SPREAD_MULT = 1.5          # half-spread bumps with ATR
MAX_ATR_PCT = 0.0020           # if ATR% > 0.20%, pause new bids (danger zone)
MAX_SLOPE_PCT = 0.0020         # if |EMA5-EMA20|/last > 0.20%, only quote away from trend

EMA_FAST, EMA_SLOW = 5, 20

# sizing & inventory control
MAX_PER_PAIR_USDT = 40.0       # per-pair cap
QUOTE_USDT = 12.0              # per-quote notional
INVENTORY_USDT_CAP = 120.0     # total gross inventory cap
INVENTORY_REDUCE_BIAS = 2.5    # stronger bias to unwind inventory

# anti-sudden-drop protections
FAST_MOVE_PCT = 0.0015         # 0.15% move within window -> cancel & freeze
FAST_MOVE_WINDOW_SEC = 5
FREEZE_SEC_AFTER_FAST_MOVE = 6

KILL_SWITCH_PCT = 0.0030       # 0.3% below latest BUY fill entry -> market-reduce
KILL_SWITCH_SELL_FRAC = 0.6    # sell 60% of recent-filled size
KILL_SWITCH_COOLDOWN_SEC = 20  # cooldown to avoid repeated triggers

# last-resort inventory reducer (market)
ORPHAN_MAX_SEC = 900

# keep Render awake
KEEPALIVE_SEC = 240

# -------------------- SMC settings --------------------
SMC_LOOKBACK = 60              # number of completed candles to scan swings/BOS
SMC_LEFT = 2                   # swing pivot strength (left/right)
SMC_RIGHT = 2
SMC_DISPLACEMENT_MULT = 1.2    # BOS displacement must exceed (mult * ATR)
FVG_LOOKBACK = 30              # scan last N completed candles for FVG
FVG_MIN_FILL = 0.5             # require ≥50% fill before quoting into an FVG
SMC_LOG = True                 # add SMC decisions into scan_log

# -------------------- Time helpers --------------------
IST = timezone('Asia/Kolkata')
def ist_now(): return datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')
def ist_date(): return datetime.now(IST).strftime('%Y-%m-%d')
def ist_yesterday(): return (datetime.now(IST) - timedelta(days=1)).strftime('%Y-%m-%d')

# -------------------- State --------------------
tick_logs, candle_logs = {p: [] for p in PAIRS}, {p: [] for p in PAIRS}
scan_log, trade_log = [], []
daily_profit, pair_precision = {}, {}
running = False
status = {"msg": "Idle", "last": ""}
status_epoch = 0
error_message = ""

# active quotes {pair: {"bid": {...}, "ask": {...}}}
active_quotes = {p: {"bid": None, "ask": None} for p in PAIRS}
inventory_timer = {}           # pair -> ts when long inventory began
last_price_state = {p: {"last": None, "ts": 0} for p in PAIRS}
pair_freeze_until = {p: 0 for p in PAIRS}
kill_cooldown_until = {p: 0 for p in PAIRS}
last_buy_fill = {p: {"entry": None, "qty": 0.0, "ts": 0} for p in PAIRS}

# -------------------- Persistent P&L (FIFO) --------------------
PROFIT_STATE_FILE = "profit_state.json"
profit_state = {
    "cumulative_pnl": 0.0,  # USDT
    "daily": {},
    "inventory": {},        # market -> [[qty, avg_price], ...]
    "processed_orders": []
}

def load_profit_state():
    global profit_state
    try:
        with open(PROFIT_STATE_FILE, "r") as f:
            data = json.load(f)
        profit_state["cumulative_pnl"] = float(data.get("cumulative_pnl", 0.0))
        profit_state["daily"] = dict(data.get("daily", {}))
        profit_state["inventory"] = data.get("inventory", {})
        profit_state["processed_orders"] = list(data.get("processed_orders", []))
    except:
        pass

def save_profit_state():
    tmp = {
        "cumulative_pnl": round(profit_state.get("cumulative_pnl", 0.0), 6),
        "daily": {k: round(v, 6) for k, v in profit_state.get("daily", {}).items()},
        "inventory": profit_state.get("inventory", {}),
        "processed_orders": profit_state.get("processed_orders", [])
    }
    try:
        with open(PROFIT_STATE_FILE, "w") as f:
            json.dump(tmp, f)
    except:
        pass

def _get_inventory_deque(market):
    inv = profit_state["inventory"].get(market, [])
    dq = deque()
    for lot in inv:
        try:
            q, c = float(lot[0]), float(lot[1])
            if q > 0 and c > 0:
                dq.append([q, c])
        except:
            continue
    return dq

def _set_inventory_from_deque(market, dq):
    profit_state["inventory"][market] = [[float(q), float(c)] for (q, c) in dq]

def apply_fill_update(market, side, price, qty, ts_ms, order_id):
    if not order_id or order_id in profit_state["processed_orders"]:
        return
    try:
        price = float(price); qty = float(qty)
    except:
        return
    if price <= 0 or qty <= 0:
        return

    inv = _get_inventory_deque(market)
    realized = 0.0

    if side.lower() == "buy":
        inv.append([qty, price])
    else:
        sell_q = qty
        while sell_q > 1e-18 and inv:
            lot_q, lot_px = inv[0]
            used = min(sell_q, lot_q)
            realized += (price - lot_px) * used
            lot_q -= used
            sell_q -= used
            if lot_q <= 1e-18:
                inv.popleft()
            else:
                inv[0][0] = lot_q

    _set_inventory_from_deque(market, inv)
    profit_state["processed_orders"].append(order_id)
    profit_state["cumulative_pnl"] = float(profit_state.get("cumulative_pnl", 0.0) + realized)
    dkey = ist_date()
    profit_state["daily"][dkey] = float(profit_state["daily"].get(dkey, 0.0) + realized)
    save_profit_state()

# -------------------- HTTP helpers --------------------
def hmac_signature(payload):
    return hmac.new(API_SECRET, payload.encode(), hashlib.sha256).hexdigest()

def _log_http_issue(prefix, r):
    try:
        body = r.text[:240] if hasattr(r, "text") else ""
        scan_log.append(f"{ist_now()} | {prefix} HTTP {r.status_code} | {body}")
    except Exception as e:
        scan_log.append(f"{ist_now()} | {prefix} log-fail: {e}")

def _signed_post(url, body):
    payload = json.dumps(body, separators=(',', ':'))
    sig = hmac_signature(payload)
    headers = {"X-AUTH-APIKEY": API_KEY or "", "X-AUTH-SIGNATURE": sig, "Content-Type": "application/json"}
    try:
        r = requests.post(url, headers=headers, data=payload, timeout=12)
        if not r.ok:
            _log_http_issue(f"POST {url}", r)
        if r.headers.get("content-type", "").startswith("application/json"):
            return r.json()
        return {}
    except Exception as e:
        scan_log.append(f"{ist_now()} | POST fail {url} | {e.__class__.__name__}: {e}")
        scan_log.append(traceback.format_exc().splitlines()[-1])
        return {}

def _keepalive_ping():
    try:
        requests.get(f"{APP_BASE_URL}/ping", timeout=5)
    except Exception:
        pass

def fetch_pair_precisions():
    try:
        r = requests.get(f"{BASE_URL}/exchange/v1/markets_details", timeout=10)
        if r.ok:
            for item in r.json():
                if item.get("pair") in PAIRS:
                    pair_precision[item["pair"]] = int(item.get("target_currency_precision", 6))
    except Exception as e:
        scan_log.append(f"{ist_now()} | markets_details fail: {e}")

def get_wallet_balances():
    payload = json.dumps({"timestamp": int(time.time() * 1000)})
    sig = hmac_signature(payload)
    headers = {"X-AUTH-APIKEY": API_KEY or "", "X-AUTH-SIGNATURE": sig, "Content-Type": "application/json"}
    balances = {}
    try:
        r = requests.post(f"{BASE_URL}/exchange/v1/users/balances", headers=headers, data=payload, timeout=10)
        if r.ok:
            for b in r.json():
                balances[b['currency']] = float(b['balance'])
        else:
            _log_http_issue("balances", r)
    except Exception as e:
        scan_log.append(f"{ist_now()} | balances fail: {e}")
    return balances

def fetch_all_prices():
    try:
        r = requests.get(f"{BASE_URL}/exchange/ticker", timeout=10)
        if r.ok:
            now = int(time.time())
            return {item["market"]: {"price": float(item["last_price"]), "ts": now}
                    for item in r.json() if item.get("market") in PAIRS}
        else:
            _log_http_issue("ticker", r)
    except Exception as e:
        scan_log.append(f"{ist_now()} | ticker fail: {e}")
    return {}

# -------------------- Candles / Indicators --------------------
def aggregate_candles(pair, interval=CANDLE_INTERVAL):
    ticks = tick_logs[pair]
    if not ticks:
        return
    candles, candle, last_window = [], None, None
    for ts, price in sorted(ticks, key=lambda x: x[0]):
        wstart = ts - (ts % interval)
        if last_window != wstart:
            if candle:
                candles.append(candle)
            candle = {"open": price, "high": price, "low": price, "close": price, "volume": 1, "start": wstart}
            last_window = wstart
        else:
            candle["high"] = max(candle["high"], price)
            candle["low"] = min(candle["low"], price)
            candle["close"] = price
            candle["volume"] += 1
    if candle:
        candles.append(candle)
    candle_logs[pair] = candles[-240:]  # ~40 m of 10s bars

def _ema(vals, n):
    if len(vals) < n:
        return None
    k = 2 / (n + 1)
    ema = sum(vals[:n]) / n
    for v in vals[n:]:
        ema = v * k + ema * (1 - k)
    return ema

def _atr_and_pct(pair, last, window=ATR_WINDOW):
    cs = candle_logs.get(pair) or []
    if len(cs) <= window:
        return None, None
    trs = []
    prev_close = cs[-window-1]["close"]
    for c in cs[-window:]:
        tr = max(c["high"] - c["low"], abs(c["high"] - prev_close), abs(c["low"] - prev_close))
        trs.append(tr)
        prev_close = c["close"]
    atr = sum(trs)/len(trs) if trs else None
    atr_pct = (atr/last) if (atr and last > 0) else None
    return atr, atr_pct

# ----------- SMC helpers (swings, BOS, sweeps, FVG) -----------
def _find_swings(cs, left=2, right=2):
    """Return indices for swing highs and lows in completed candles."""
    highs, lows = [], []
    n = len(cs)
    for i in range(left, n-right):
        h = cs[i]["high"]
        l = cs[i]["low"]
        if all(h >= cs[i-k]["high"] for k in range(1, left+1)) and all(h >= cs[i+k]["high"] for k in range(1, right+1)):
            highs.append(i)
        if all(l <= cs[i-k]["low"] for k in range(1, left+1)) and all(l <= cs[i+k]["low"] for k in range(1, right+1)):
            lows.append(i)
    return highs, lows

def _recent_bos(cs, highs, lows, atr, side="up", lookback=60, displacement_mult=1.2):
    """
    Detect recent Break of Structure:
    - side="up": close > last swing high + displacement
    - side="down": close < last swing low  - displacement
    Returns (True/False, level_index, level_price)
    """
    if not cs or atr is None:
        return False, None, None
    start = max(0, len(cs)-lookback)
    segment = cs[start:-1] if len(cs) > 1 else cs
    if side == "up" and highs:
        hidx = [i for i in highs if i >= start]
        if not hidx: return False, None, None
        last_high_i = hidx[-1]
        key_level = cs[last_high_i]["high"]
        for i in range(last_high_i+1, len(cs)-1):
            if cs[i]["close"] > key_level + displacement_mult*atr:
                return True, last_high_i, key_level
    if side == "down" and lows:
        lidx = [i for i in lows if i >= start]
        if not lidx: return False, None, None
        last_low_i = lidx[-1]
        key_level = cs[last_low_i]["low"]
        for i in range(last_low_i+1, len(cs)-1):
            if cs[i]["close"] < key_level - displacement_mult*atr:
                return True, last_low_i, key_level
    return False, None, None

def _recent_sweep(cs, highs, lows, lookback=60):
    """
    Liquidity sweep:
    - Sell-side sweep (SSL): price takes out recent low (wick closes back within prior range)
    - Buy-side sweep (BSL): price takes out recent high (wick closes back)
    Returns one of: "SSL", "BSL", None
    """
    if len(cs) < 5:
        return None
    start = max(0, len(cs)-lookback-1)
    prev = cs[start: -1]  # completed candles excluding last closed
    last_closed = cs[-2]  # the most recent completed candle
    if not prev:
        return None
    recent_low = min(c["low"] for c in prev[-10:])
    recent_high = max(c["high"] for c in prev[-10:])

    # SSL: low pierces below recent_low but closes back above it
    if last_closed["low"] < recent_low and last_closed["close"] > recent_low:
        return "SSL"
    # BSL: high pierces above recent_high but closes back below it
    if last_closed["high"] > recent_high and last_closed["close"] < recent_high:
        return "BSL"
    return None

def _recent_fvg(cs, lookback=30):
    """
    Fair Value Gap (3-candle): bullish if L1 > H-1; bearish if H1 < L-1
    Return dict {"type": "bull"/"bear", "gap": (low, high), "filled_pct": 0..1}
    using last discovered FVG within lookback, else None.
    """
    n = len(cs)
    start = max(0, n - lookback)
    fvg = None
    for i in range(start+2, n):  # need i-2, i-1, i
        c0, c1, c2 = cs[i-2], cs[i-1], cs[i]
        # bullish FVG: low(c2) > high(c0)
        if c2["low"] > c0["high"]:
            gap_low, gap_high = c0["high"], c2["low"]
            # fill measure on subsequent candles (including c2 close onward)
            post = cs[i:]
            min_after = min([x["low"] for x in post]) if post else c2["low"]
            filled = max(0.0, min(1.0, (gap_high - max(gap_low, min_after)) / max(1e-12, (gap_high-gap_low))))
            fvg = {"type": "bull", "gap": (gap_low, gap_high), "filled_pct": filled}
        # bearish FVG: high(c2) < low(c0)
        if c2["high"] < c0["low"]:
            gap_low, gap_high = c2["high"], c0["low"]
            post = cs[i:]
            max_after = max([x["high"] for x in post]) if post else c2["high"]
            filled = max(0.0, min(1.0, (min(gap_high, max_after) - gap_low) / max(1e-12, (gap_high-gap_low))))
            fvg = {"type": "bear", "gap": (gap_low, gap_high), "filled_pct": filled}
    return fvg

def _smc_bias(pair, last, atr):
    """
    Compute SMC context using completed candles:
    - BOS up/down using swings + displacement
    - recent liquidity sweep (SSL/BSL)
    - last FVG and its filled percentage
    Returns dict with 'bias' ('bull'/'bear'/None), 'sweep', 'fvg'
    """
    cs = candle_logs.get(pair) or []
    if len(cs) < max(SMC_LOOKBACK, ATR_WINDOW) + 3:
        return {"bias": None, "sweep": None, "fvg": None}

    completed = cs[:-1] if len(cs) >= 2 else cs
    highs, lows = _find_swings(completed[-SMC_LOOKBACK:], SMC_LEFT, SMC_RIGHT)
    # adjust indices relative to completed window
    offset = len(completed[-SMC_LOOKBACK:])
    highs = [i + len(completed) - offset for i in highs]
    lows  = [i + len(completed) - offset for i in lows]

    bos_up, _, key_high = _recent_bos(completed, highs, lows, atr, side="up",
                                      lookback=SMC_LOOKBACK, displacement_mult=SMC_DISPLACEMENT_MULT)
    bos_dn, _, key_low  = _recent_bos(completed, highs, lows, atr, side="down",
                                      lookback=SMC_LOOKBACK, displacement_mult=SMC_DISPLACEMENT_MULT)
    bias = "bull" if (bos_up and not bos_dn) else "bear" if (bos_dn and not bos_up) else None

    sweep = _recent_sweep(completed, highs, lows, lookback=SMC_LOOKBACK)
    fvg   = _recent_fvg(completed, lookback=FVG_LOOKBACK)

    return {"bias": bias, "sweep": sweep, "fvg": fvg}

# -------------------- Exchange calls --------------------
def place_order(pair, side, qty):
    payload = {
        "market": pair, "side": side.lower(), "order_type": "market_order",
        "total_quantity": str(qty), "timestamp": int(time.time() * 1000)
    }
    return _signed_post(f"{BASE_URL}/exchange/v1/orders/create", payload) or {}

def place_limit_order(pair, side, qty, price):
    payload = {
        "market": pair, "side": side.lower(), "order_type": "limit_order",
        "price_per_unit": str(price), "total_quantity": str(qty),
        "timestamp": int(time.time() * 1000)
        # If supported by CoinDCX, add post-only flag here to avoid taker fills.
    }
    return _signed_post(f"{BASE_URL}/exchange/v1/orders/create", payload) or {}

def cancel_order(order_id=None, client_order_id=None):
    body = {"timestamp": int(time.time() * 1000)}
    if order_id: body["id"] = order_id
    if client_order_id: body["client_order_id"] = client_order_id
    return _signed_post(f"{BASE_URL}/exchange/v1/orders/cancel", body) or {}

def get_order_status(order_id=None, client_order_id=None):
    body = {"timestamp": int(time.time() * 1000)}
    if order_id: body["id"] = order_id
    if client_order_id: body["client_order_id"] = client_order_id
    res = _signed_post(f"{BASE_URL}/exchange/v1/orders/status", body)
    return res if isinstance(res, dict) else {}

def _extract_order_id(res: dict):
    if not isinstance(res, dict):
        return None
    try:
        if isinstance(res.get("orders"), list) and res["orders"]:
            o = res["orders"][0]
            return str(o.get("id") or o.get("order_id") or o.get("client_order_id") or "")
    except:
        pass
    return str(res.get("id") or res.get("order_id") or res.get("client_order_id") or res.get("orderId") or "") or None

def _fnum(x, d=0.0):
    try:
        return float(x)
    except:
        return d

def _record_fill_from_status(market, side, st, order_id):
    if not isinstance(st, dict): return 0.0, 0.0
    total_q  = _fnum(st.get("total_quantity", st.get("quantity", st.get("orig_qty", 0))))
    remain_q = _fnum(st.get("remaining_quantity", st.get("remaining_qty", st.get("leaves_qty", 0))))
    exec_q   = _fnum(st.get("executed_quantity", st.get("filled_qty", st.get("executedQty", 0))))
    filled   = exec_q if exec_q > 0 else max(0.0, total_q - remain_q)
    avg_px   = _fnum(st.get("avg_price", st.get("average_price", st.get("avg_execution_price", st.get("price", 0)))))
    if filled > 0 and avg_px > 0:
        ts_field = st.get("updated_at") or st.get("created_at") or st.get("timestamp") or int(time.time() * 1000)
        try:
            ts_ms = int(ts_field)
            if ts_ms < 10 ** 12: ts_ms *= 1000
        except:
            ts_ms = int(time.time() * 1000)
        apply_fill_update(market, side, avg_px, filled, ts_ms, order_id)
    return filled, avg_px

# -------------------- Maker quoting helpers --------------------
def _effective_half_spread_pct(adapt_with_atr_pct=None):
    base = (2 * FEE_PCT_PER_SIDE + TP_BUFFER_PCT) / 2.0
    if adapt_with_atr_pct:
        base = max(base, adapt_with_atr_pct * ATR_SPREAD_MULT)
    return max(base, (SPREAD_OFFSET_PCT or 0.0))

def _quote_prices(last, atr_pct=None):
    off = _effective_half_spread_pct(atr_pct)
    bid = round(last * (1.0 - off), 6)
    ask = round(last * (1.0 + off), 6)
    return bid, ask

def _qty_for_pair(pair, price, usdt_avail, coin_avail, side, trend_skew):
    q = max(1e-12, QUOTE_USDT) / max(price, 1e-9)
    q *= trend_skew
    rule = PAIR_RULES.get(pair, {"precision": 6, "min_qty": 0.0001})
    q = min(q, MAX_PER_PAIR_USDT / max(price, 1e-9))
    if side == "BUY":
        q = min(q, usdt_avail / max(price, 1e-9))
    else:
        q = min(q, coin_avail)
    q = max(q, rule["min_qty"])
    q = round(q, rule["precision"])
    return q

def _net_inventory_units(pair):
    dq = _get_inventory_deque(pair)
    return sum(q for q, _ in dq)

def _gross_inventory_usdt(prices):
    total = 0.0
    for p in PAIRS:
        q = abs(_net_inventory_units(p))
        px = prices.get(p, {}).get("price", 0.0)
        total += q * px
    return total

def _cancel_quote(pair, side_key):
    q = active_quotes.get(pair, {}).get(side_key)
    if not q: return
    oid = q.get("id")
    if oid:
        cancel_order(order_id=oid)
        scan_log.append(f"{ist_now()} | {pair} | cancel {side_key} quote {oid}")
    active_quotes[pair][side_key] = None

def cancel_all_quotes(pair):
    _cancel_quote(pair, "bid")
    _cancel_quote(pair, "ask")

def _place_quote(pair, side_word, price, qty):
    # side_word: "BUY" or "SELL"
    res = place_limit_order(pair, side_word, qty, price)
    oid = _extract_order_id(res)
    side_key = "bid" if side_word == "BUY" else "ask"
    active_quotes[pair][side_key] = {"id": oid, "px": price, "qty": qty, "ts": int(time.time())}
    scan_log.append(f"{ist_now()} | {pair} | quote {side_word} {qty} @ {price} | id={oid} | res={res}")
    return oid

def _check_quote_fill(pair, side_key, last_price):
    """
    side_key: 'bid' or 'ask'
    Translates to order side 'BUY' (bid) or 'SELL' (ask) for P&L accounting.
    Also updates last_buy_fill for kill-switch after BUY fills.
    """
    q = active_quotes.get(pair, {}).get(side_key)
    if not q or not q.get("id"):
        return False

    st = get_order_status(order_id=q["id"])

    rem = _fnum(st.get("remaining_quantity", st.get("remaining_qty", st.get("leaves_qty", 0))))
    status_txt = (st.get("status") or "").lower()
    filled = (rem == 0) or ("filled" in status_txt and "part" not in status_txt)

    if filled:
        order_side = "BUY" if side_key == "bid" else "SELL"
        filled_qty, avg_px = _record_fill_from_status(pair, order_side, st, q["id"])
        active_quotes[pair][side_key] = None
        scan_log.append(f"{ist_now()} | {pair} | FILL {order_side} {filled_qty} @ {avg_px} | st={st}")

        # Track the latest BUY fill for kill-switch
        if order_side == "BUY" and filled_qty > 0 and avg_px > 0:
            last_buy_fill[pair] = {"entry": float(avg_px), "qty": float(filled_qty), "ts": int(time.time())}
        return True
    return False

def _manage_orphan_inventory(pair, now_ts, prices, balances):
    q_units = _net_inventory_units(pair)
    if q_units <= 0:
        inventory_timer.pop(pair, None)
        return
    if pair not in inventory_timer:
        inventory_timer[pair] = now_ts
        return
    if now_ts - inventory_timer[pair] < ORPHAN_MAX_SEC:
        return
    # last resort: market-out a chunk
    coin = pair[:-4]
    coin_bal = balances.get(coin, 0.0)
    qty = min(0.3 * q_units, coin_bal)
    rule = PAIR_RULES.get(pair, {"precision": 6, "min_qty": 0.0001})
    qty = max(qty, rule["min_qty"])
    qty = round(qty, rule["precision"])
    if qty <= 0: return
    res = place_order(pair, "SELL", qty)
    oid = _extract_order_id(res)
    if oid:
        st = get_order_status(order_id=oid)
        _record_fill_from_status(pair, "SELL", st, oid)
    scan_log.append(f"{ist_now()} | {pair} | ORPHAN TIMEOUT market SELL {qty} | res={res}")
    inventory_timer[pair] = now_ts

# -------------------- Main loop (autostart-safe) --------------------
last_keepalive = 0
_autostart_lock = threading.Lock()

def scan_loop():
    global running, error_message, status_epoch, last_keepalive
    scan_log.clear()
    running = True

    while running:
        # keep Render awake
        now_real = time.time()
        if now_real - last_keepalive >= KEEPALIVE_SEC:
            _keepalive_ping()
            last_keepalive = now_real

        prices = fetch_all_prices()
        now_ts = int(time.time())
        balances = get_wallet_balances()

        # ticks & candles
        for pair in PAIRS:
            if pair in prices:
                px = prices[pair]["price"]
                tick_logs[pair].append((now_ts, px))
                if len(tick_logs[pair]) > 4000:
                    tick_logs[pair] = tick_logs[pair][-4000:]
                aggregate_candles(pair, CANDLE_INTERVAL)

        # risk guard
        gross_usdt = _gross_inventory_usdt(prices)
        if gross_usdt > INVENTORY_USDT_CAP:
            scan_log.append(f"{ist_now()} | RISK | Gross inventory {round(gross_usdt,2)} > cap {INVENTORY_USDT_CAP} — pause new bids")

        # per-pair logic
        for pair in PAIRS:
            if pair not in prices:
                continue

            last = prices[pair]["price"]

            # -------- Fast-move detection & freeze --------
            prev = last_price_state[pair]["last"]
            pts  = last_price_state[pair]["ts"]
            last_price_state[pair] = {"last": last, "ts": now_ts}

            if prev and (now_ts - pts) <= FAST_MOVE_WINDOW_SEC:
                move_pct = abs(last - prev) / max(prev, 1e-9)
                if move_pct >= FAST_MOVE_PCT:
                    cancel_all_quotes(pair)
                    pair_freeze_until[pair] = now_ts + FREEZE_SEC_AFTER_FAST_MOVE
                    scan_log.append(f"{ist_now()} | {pair} | FAST MOVE {round(move_pct*100,3)}% — freeze until {pair_freeze_until[pair]}")
                    # skip quoting this tick
                    continue

            if now_ts < pair_freeze_until[pair]:
                # while frozen, still check fills (in case partials were resting)
                for side_key in ("bid", "ask"):
                    _check_quote_fill(pair, side_key, last)
                continue

            # -------- Indicators / gates --------
            closes = [c["close"] for c in (candle_logs.get(pair) or [])[:-1]]
            ema_fast = _ema(closes[-(EMA_SLOW+EMA_FAST):] + [last], EMA_FAST) if len(closes) >= EMA_SLOW else None
            ema_slow = _ema(closes[-(EMA_SLOW+EMA_FAST):] + [last], EMA_SLOW) if len(closes) >= EMA_SLOW else None
            bullish = (ema_fast is not None and ema_slow is not None and ema_fast >= ema_slow)
            slope_pct = None
            if ema_fast is not None and ema_slow is not None and last > 0:
                slope_pct = abs(ema_fast - ema_slow) / last

            atr_abs, atr_pct = _atr_and_pct(pair, last, ATR_WINDOW)

            # -------- SMC context --------
            smc = _smc_bias(pair, last, atr_abs)
            smc_bias = smc.get("bias")
            smc_sweep = smc.get("sweep")
            smc_fvg = smc.get("fvg")

            # base quotes with adaptive spread
            bid_px, ask_px = _quote_prices(last, atr_pct)

            usdt = balances.get("USDT", 0.0)
            coin = pair[:-4]
            coin_bal = balances.get(coin, 0.0)
            net_units = _net_inventory_units(pair)

            # inventory & trend bias
            reduce_bias_bid = INVENTORY_REDUCE_BIAS if net_units < 0 else 1.0
            reduce_bias_ask = INVENTORY_REDUCE_BIAS if net_units > 0 else 1.0
            trend_bias_bid = 1.2 if bullish else 0.9
            trend_bias_ask = 1.2 if not bullish else 0.9

            # -------- Allowed sides (ATR/slope + SMC) --------
            allow_bid, allow_ask = True, True
            if gross_usdt > INVENTORY_USDT_CAP:
                allow_bid = False
            if atr_pct is not None and atr_pct > MAX_ATR_PCT:
                allow_bid = False  # turbulence: avoid catching knives

            if slope_pct is not None and slope_pct > MAX_SLOPE_PCT:
                # steep trend: quote only away from direction
                if bullish:
                    allow_ask = False
                    allow_bid = True
                else:
                    allow_bid = False
                    allow_ask = True

            # SMC bias: prefer side with structure (BOS) and react to sweeps
            if smc_bias == "bull":
                allow_ask = min(allow_ask, True)
                allow_bid = True
            elif smc_bias == "bear":
                allow_bid = min(allow_bid, True)
                allow_ask = True

            # Liquidity sweeps: SSL -> favor BUY (mean reversion); BSL -> favor SELL
            if smc_sweep == "SSL":
                allow_ask = False  # avoid offering into recovery; let it revert then sell later
                allow_bid = True
            elif smc_sweep == "BSL":
                allow_bid = False
                allow_ask = True

            # FVG filter: avoid quoting INTO an unfilled adverse FVG
            # - If bullish FVG and filled < 50%, prefer buys, avoid sells into the gap
            # - If bearish FVG and filled < 50%, prefer sells, avoid buys into the gap
            if smc_fvg and smc_fvg.get("filled_pct", 1.0) < FVG_MIN_FILL:
                if smc_fvg.get("type") == "bull":
                    allow_ask = False
                    allow_bid = allow_bid and True
                elif smc_fvg.get("type") == "bear":
                    allow_bid = False
                    allow_ask = allow_ask and True

            if SMC_LOG:
                scan_log.append(f"{ist_now()} | {pair} | SMC bias={smc_bias} sweep={smc_sweep} fvg={smc_fvg}")

            # cancel stale/moved quotes
            for side_key, q in list(active_quotes[pair].items()):
                if not q:
                    continue
                age = now_ts - int(q.get("ts", now_ts))
                px = q.get("px", last)
                drift = abs(last - px) / max(px, 1e-9)
                if age >= QUOTE_TTL_SEC or drift >= DRIFT_REQUOTE_PCT:
                    _cancel_quote(pair, side_key)

            # check fills (also updates last_buy_fill on BUY)
            for side_key in ("bid", "ask"):
                _check_quote_fill(pair, side_key, last)

            # -------- Inventory kill-switch after BUY fill --------
            lb = last_buy_fill.get(pair, {})
            if lb and lb.get("entry") and now_ts >= kill_cooldown_until[pair]:
                entry = float(lb["entry"]); filled_qty = float(lb["qty"])
                if last <= entry * (1.0 - KILL_SWITCH_PCT):
                    qty_to_sell = filled_qty * KILL_SWITCH_SELL_FRAC
                    qty_to_sell = min(qty_to_sell, coin_bal)
                    rule = PAIR_RULES.get(pair, {"precision": 6, "min_qty": 0.0001})
                    qty_to_sell = max(qty_to_sell, rule["min_qty"])
                    qty_to_sell = round(qty_to_sell, rule["precision"])
                    if qty_to_sell > 0:
                        _cancel_quote(pair, "bid")
                        res = place_order(pair, "SELL", qty_to_sell)
                        oid = _extract_order_id(res)
                        if oid:
                            st = get_order_status(order_id=oid)
                            _record_fill_from_status(pair, "SELL", st, oid)
                        scan_log.append(f"{ist_now()} | {pair} | KILL-SWITCH: SOLD {qty_to_sell} due to {round(100*(entry-last)/entry,3)}% drop | res={res}")
                        kill_cooldown_until[pair] = now_ts + KILL_SWITCH_COOLDOWN_SEC
                        last_buy_fill[pair]["qty"] = max(0.0, filled_qty - qty_to_sell)

            # sizing for quotes
            qty_bid = _qty_for_pair(pair, bid_px, usdt, coin_bal, "BUY", reduce_bias_bid * trend_bias_bid) if allow_bid else 0.0
            qty_ask = _qty_for_pair(pair, ask_px, usdt, coin_bal, "SELL", reduce_bias_ask * trend_bias_ask) if allow_ask else 0.0

            # place quotes if empty
            if qty_bid > 0 and not active_quotes[pair]["bid"]:
                bpx = min(bid_px, round(last * (1 - 1e-6), 6))  # do not cross
                _place_quote(pair, "BUY", bpx, qty_bid)

            if qty_ask > 0 and coin_bal >= PAIR_RULES.get(pair, {"min_qty": 0.0})["min_qty"]:
                if not active_quotes[pair]["ask"]:
                    apx = max(ask_px, round(last * (1 + 1e-6), 6))
                    _place_quote(pair, "SELL", apx, qty_ask)

            _manage_orphan_inventory(pair, now_ts, prices, balances)

        status["msg"], status["last"] = "Running", ist_now()
        status_epoch = int(time.time())
        print(f"[{ist_now()}] Loop active — quoting…")
        time.sleep(POLL_SEC)

    status["msg"] = "Idle"

def compute_realized_pnl_today():
    return round(profit_state["daily"].get(ist_date(), 0.0), 6)

# -------------------- Routes --------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start", methods=["POST"])
def start():
    _start_loop_once()
    return jsonify({"status": "started"})

@app.route("/stop", methods=["POST"])
def stop():
    global running
    running = False
    try:
        for p in PAIRS:
            for s in ("bid", "ask"):
                if active_quotes[p][s] and active_quotes[p][s].get("id"):
                    cancel_order(order_id=active_quotes[p][s]["id"])
                    active_quotes[p][s] = None
    except:
        pass
    return jsonify({"status": "stopped"})

@app.route("/status")
def get_status():
    balances = get_wallet_balances()
    coins = {pair[:-4]: balances.get(pair[:-4], 0.0) for pair in PAIRS}
    profit_today = compute_realized_pnl_today()
    profit_yesterday = round(profit_state["daily"].get(ist_yesterday(), 0.0), 6)
    cumulative_pnl = round(profit_state.get("cumulative_pnl", 0.0), 6)

    visible_quotes = {}
    for p in PAIRS:
        v = {}
        for side in ("bid", "ask"):
            q = active_quotes[p][side]
            if q:
                v[side] = {"id": q.get("id"), "px": q.get("px"), "qty": q.get("qty"), "ts": q.get("ts")}
        if v:
            visible_quotes[p] = v

    return jsonify({
        "status": status["msg"],
        "last": status["last"],
        "status_epoch": status_epoch,
        "usdt": balances.get("USDT", 0.0),
        "profit_today": profit_today,
        "profit_yesterday": profit_yesterday,
        "pnl_cumulative": cumulative_pnl,
        "processed_orders": len(profit_state.get("processed_orders", [])),
        "inventory_markets": list(profit_state.get("inventory", {}).keys()),
        "quotes": visible_quotes,
        "coins": coins,
        "trades": trade_log[-10:][::-1],
        "scans": scan_log[-60:][::-1],
        "error": error_message
    })

@app.route("/ping")
def ping():
    return "pong"

# -------------------- Safe autostart --------------------
_autostart_lock = threading.Lock()
def _start_loop_once():
    global running
    with _autostart_lock:
        if not running:
            running = True
            t = threading.Thread(target=scan_loop, daemon=True)
            t.start()

# Kick the loop shortly after module import
def _boot_kick():
    try:
        time.sleep(1.0)
        _start_loop_once()
    except Exception as e:
        print("boot kick failed:", e)

if os.environ.get("AUTOSTART", "1") == "1":
    threading.Thread(target=_boot_kick, daemon=True).start()

# Also autostart when running directly
if __name__ == "__main__":
    load_profit_state()
    fetch_pair_precisions()
    _start_loop_once()
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
