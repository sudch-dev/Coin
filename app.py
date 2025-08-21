import os
import time
import threading
import hmac
import hashlib
import requests
import json
import traceback
from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta
from pytz import timezone
from collections import deque

# -------------------- Flask --------------------
app = Flask(__name__)

APP_BASE_URL = os.environ.get("APP_BASE_URL", "https://coin-4k37.onrender.com")
KEEPALIVE_TOKEN = os.environ.get("KEEPALIVE_TOKEN", "")

# -------------------- API / Markets --------------------
API_KEY = os.environ.get("API_KEY")
API_SECRET_RAW = os.environ.get("API_SECRET", "")
API_SECRET = API_SECRET_RAW.encode() if isinstance(API_SECRET_RAW, str) else API_SECRET_RAW
BASE_URL = "https://api.coindcx.com"

PAIRS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "SHIBUSDT", "SOLUSDT",
    "DOGEUSDT", "ADAUSDT", "AEROUSDT", "BNBUSDT"
]

# Local fallbacks; overwritten by live fetch at boot and periodically thereafter
PAIR_RULES = {
    "BTCUSDT":  {"price_precision": 1, "qty_precision": 4, "min_qty": 0.001,  "min_notional": 0.0},
    "ETHUSDT":  {"price_precision": 2, "qty_precision": 6, "min_qty": 0.0001, "min_notional": 0.0},
    "XRPUSDT":  {"price_precision": 4, "qty_precision": 4, "min_qty": 0.1,    "min_notional": 0.0},
    "SHIBUSDT": {"price_precision": 8, "qty_precision": 4, "min_qty": 10000,  "min_notional": 0.0},
    "DOGEUSDT": {"price_precision": 5, "qty_precision": 4, "min_qty": 0.001,  "min_notional": 0.0},
    "SOLUSDT":  {"price_precision": 2, "qty_precision": 4, "min_qty": 0.01,   "min_notional": 0.0},
    "AEROUSDT": {"price_precision": 3, "qty_precision": 2, "min_qty": 0.01,   "min_notional": 0.0},
    "ADAUSDT":  {"price_precision": 4, "qty_precision": 2, "min_qty": 0.1,    "min_notional": 0.0},
    "BNBUSDT":  {"price_precision": 3, "qty_precision": 4, "min_qty": 0.001,  "min_notional": 0.0},
}

# -------------------- Maker / HFT settings --------------------
MODE = "maker"

CANDLE_INTERVAL = 10
POLL_SEC = 1.0
# Safer defaults (reduce cancel spam)
QUOTE_TTL_SEC = 10          # was 4
DRIFT_REQUOTE_PCT = 0.0006  # was 0.0003

FEE_PCT_PER_SIDE = 0.0010
TP_BUFFER_PCT = 0.0006
SPREAD_OFFSET_PCT = None

ATR_WINDOW = 18
ATR_SPREAD_MULT = 1.5
MAX_ATR_PCT = 0.0020
MAX_SLOPE_PCT = 0.0020

EMA_FAST, EMA_SLOW = 5, 20

MAX_PER_PAIR_USDT = 40.0
QUOTE_USDT = 12.0
INVENTORY_USDT_CAP = 120.0
INVENTORY_REDUCE_BIAS = 2.5

FAST_MOVE_PCT = 0.0015
FAST_MOVE_WINDOW_SEC = 5
FREEZE_SEC_AFTER_FAST_MOVE = 6

KILL_SWITCH_PCT = 0.0030
KILL_SWITCH_SELL_FRAC = 0.4  # was 0.6
KILL_SWITCH_COOLDOWN_SEC = 20

ORPHAN_MAX_SEC = 900
KEEPALIVE_SEC = 240

# --- New knobs for stickier quotes & rate-limited requotes
MIN_QUOTE_LIFETIME_SEC = 8
DRIFT_REQUOTE_ATR_MULT = 0.6
MAX_REQUOTE_PER_MIN = 10

# --- Buy headroom so rounded qty always fits fees/balance
BUY_HEADROOM = 1.0005  # 5 bps safety

# -------------------- SMC settings --------------------
SMC_LOOKBACK = 60
SMC_LEFT = 2
SMC_RIGHT = 2
SMC_DISPLACEMENT_MULT = 1.2
FVG_LOOKBACK = 30
FVG_MIN_FILL = 0.5
SMC_LOG = True

# SMC confirmation-only mode: require an explicit structure signal to quote
SMC_CONFIRM_ONLY = True

# --- Kill-switch refinements
KILL_SWITCH_CONFIRM_SEC = 3            # sustained breach
KILL_SWITCH_REQUIRE_SMC = True         # require bearish structure to dump
KILL_SWITCH_LIMIT_OFFSET = 0.0005      # sell with a small-through limit, not market

# -------------------- Time helpers --------------------
IST = timezone('Asia/Kolkata')
def ist_now(): return datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')
def ist_date(): return datetime.now(IST).strftime('%Y-%m-%d')
def ist_yesterday(): return (datetime.now(IST) - timedelta(days=1)).strftime('%Y-%m-%d')

# -------------------- State --------------------
tick_logs, candle_logs = {p: [] for p in PAIRS}, {p: [] for p in PAIRS}
scan_log, trade_log = [], []
daily_profit = {}
running = False
status = {"msg": "Idle", "last": ""}
status_epoch = 0
error_message = ""

active_quotes = {p: {"bid": None, "ask": None} for p in PAIRS}
inventory_timer = {}
last_price_state = {p: {"last": None, "ts": 0} for p in PAIRS}
pair_freeze_until = {p: 0 for p in PAIRS}
kill_cooldown_until = {p: 0 for p in PAIRS}
last_buy_fill = {p: {"entry": None, "qty": 0.0, "ts": 0} for p in PAIRS}

# new: per-pair requote counters & kill-breach timers
requote_counter = {p: {"t": 0, "n": 0} for p in PAIRS}
kill_breach_since = {p: 0 for p in PAIRS}

# rules refresh cadence
_last_rules_refresh = 0
RULES_REFRESH_SEC = 1800  # 30 mins

# -------------------- Persistent P&L (FIFO) --------------------
PROFIT_STATE_FILE = "profit_state.json"
profit_state = {
    "cumulative_pnl": 0.0,
    "daily": {},
    "inventory": {},
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
        if not APP_BASE_URL:
            return
        url = f"{APP_BASE_URL}/ping"
        if KEEPALIVE_TOKEN:
            url = f"{url}?t={KEEPALIVE_TOKEN}"
            headers = {"X-Keepalive-Token": KEEPALIVE_TOKEN}
        else:
            headers = {}
        requests.get(url, headers=headers, timeout=5)
    except Exception:
        pass

# -------------------- Market rules (precision / min qty / notional) --------------------
def fetch_pair_precisions():
    """
    Pull live market metadata and cache per-pair rules:
    - price_precision  := target_currency_precision
    - qty_precision    := base_currency_precision
    - min_qty          := min_quantity
    - min_notional     := min_notional (if provided)
    """
    try:
        r = requests.get(f"{BASE_URL}/exchange/v1/markets_details", timeout=12)
        if not r.ok:
            _log_http_issue("markets_details", r)
            return
        data = r.json()
        for item in data:
            p = item.get("pair") or item.get("market") or item.get("coindcx_name")
            if p in PAIRS:
                PAIR_RULES[p] = {
                    "price_precision": int(item.get("target_currency_precision", 6)),
                    "qty_precision":   int(item.get("base_currency_precision", 6)),
                    "min_qty":         float(item.get("min_quantity", 0.0) or 0.0),
                    "min_notional":    float(item.get("min_notional", 0.0) or 0.0)
                }
        scan_log.append(f"{ist_now()} | market rules refreshed")
    except Exception as e:
        scan_log.append(f"{ist_now()} | markets_details fail: {e}")

# -------------------- Balances / Prices --------------------
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

# -------------------- Precision, fees & min-qty helpers --------------------
def _rules(pair): return PAIR_RULES.get(pair, {})
def _min_qty(pair): return float(_rules(pair).get("min_qty", 0.0) or 0.0)
def _qty_prec(pair): return int(_rules(pair).get("qty_precision", 6))
def _min_notional(pair): return float(_rules(pair).get("min_notional", 0.0) or 0.0)

def fmt_price(pair, price):
    pp = int(_rules(pair).get("price_precision", 6))
    return float(f"{float(price):.{pp}f}")

def fmt_qty(pair, qty):
    qp = _qty_prec(pair)
    mq = _min_qty(pair)
    q = max(float(qty), mq)
    q = float(f"{q:.{qp}f}")
    if q < mq:
        q = float(f"{mq:.{qp}f}")
    return q

def _qty_step(pair):
    return 10 ** (-_qty_prec(pair))

def meets_min_notional(pair, price, qty):
    mn = _min_notional(pair)
    return (price * qty) >= mn

# --- Fee-aware affordability gate
def _fee_multiplier(side):
    return 1.0 + FEE_PCT_PER_SIDE if side.upper() == "BUY" else 1.0 - FEE_PCT_PER_SIDE

def _affordable_buy_qty(pair, price, usdt_avail):
    if price <= 0:
        return 0.0
    eff = usdt_avail / (price * _fee_multiplier("BUY"))
    return max(0.0, eff)

# --- Parse precision from exchange error and update rules live
def _learn_precision_from_error(pair, msg_lower):
    if not isinstance(msg_lower, str):
        return False
    import re
    m = re.search(r'([A-Z]{3,5})\s+precision\s+should\s+be\s+(\d+)', msg_lower.upper())
    if not m:
        return False
    cur = m.group(1)
    pval = int(m.group(2))
    base = pair[:-4]
    quote = pair[-4:]
    if cur == base:
        PAIR_RULES.setdefault(pair, {})["qty_precision"] = pval
        return True
    if cur == quote:
        PAIR_RULES.setdefault(pair, {})["price_precision"] = pval
        return True
    return False

# --- Locked vs free balances (avoid oversubscription) ---
def _locked_from_active_quotes():
    locked_usdt = 0.0
    locked_coin = {}
    for pair, sides in active_quotes.items():
        bid = sides.get("bid")
        ask = sides.get("ask")
        if bid and bid.get("qty") and bid.get("px"):
            px = float(bid["px"]); q = float(bid["qty"])
            locked_usdt += px * q * (1.0 + FEE_PCT_PER_SIDE + 1e-4)  # tiny cushion
        if ask and ask.get("qty"):
            coin = pair[:-4]
            locked_coin[coin] = locked_coin.get(coin, 0.0) + float(ask["qty"])
    return locked_usdt, locked_coin

def _compute_free_balances(balances):
    free = dict(balances or {})
    locked_usdt, locked_coin = _locked_from_active_quotes()
    free["USDT"] = max(0.0, float(free.get("USDT", 0.0)) - locked_usdt)
    for coin, amt in locked_coin.items():
        free[coin] = max(0.0, float(free.get(coin, 0.0)) - amt)
    return free

def normalize_qty_for_side(pair, side, price, qty, usdt_avail, coin_avail):
    """
    Enforce:
      1) precision (qty), 2) min_qty, 3) wallet caps (fee-aware for BUY with headroom), 4) min_notional.
    Returns precision-safe qty or 0.0 to skip.
    """
    q = fmt_qty(pair, qty)
    mq = _min_qty(pair)

    if q < mq:
        q = fmt_qty(pair, mq)

    if side.upper() == "BUY":
        # fee + headroom aware
        denom = max(price * _fee_multiplier("BUY") * BUY_HEADROOM, 1e-12)
        max_buyable = usdt_avail / denom
        if q > max_buyable:
            q = fmt_qty(pair, max_buyable)
        # After rounding, ensure it *still* fits. Step down if needed.
        step = _qty_step(pair)
        while q >= mq and (price * q * _fee_multiplier("BUY") * BUY_HEADROOM) > usdt_avail + 1e-12:
            q = fmt_qty(pair, max(0.0, q - step))
    else:
        if q > coin_avail:
            q = fmt_qty(pair, coin_avail)

    if q < mq or q <= 0:
        return 0.0

    mn = _min_notional(pair)
    if mn > 0 and (price * q) < mn:
        needed_q = mn / max(price, 1e-9)
        q = fmt_qty(pair, max(q, needed_q))
        if side.upper() == "BUY":
            denom = max(price * _fee_multiplier("BUY") * BUY_HEADROOM, 1e-12)
            max_buyable = usdt_avail / denom
            if q > max_buyable:
                q = fmt_qty(pair, max_buyable)
            step = _qty_step(pair)
            while q >= mq and (price * q * _fee_multiplier("BUY") * BUY_HEADROOM) > usdt_avail + 1e-12:
                q = fmt_qty(pair, max(0.0, q - step))
        else:
            if q > coin_avail:
                q = fmt_qty(pair, coin_avail)
        if q < mq or (mn > 0 and (price * q) < mn):
            return 0.0

    return q

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

# ----------- SMC helpers -----------
def _find_swings(cs, left=2, right=2):
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
    if not cs or atr is None:
        return False, None, None
    start = max(0, len(cs)-lookback)
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
    if len(cs) < 5:
        return None
    start = max(0, len(cs)-lookback-1)
    prev = cs[start: -1]
    last_closed = cs[-2]
    if not prev:
        return None
    recent_low = min(c["low"] for c in prev[-10:])
    recent_high = max(c["high"] for c in prev[-10:])
    if last_closed["low"] < recent_low and last_closed["close"] > recent_low:
        return "SSL"
    if last_closed["high"] > recent_high and last_closed["close"] < recent_high:
        return "BSL"
    return None

def _recent_fvg(cs, lookback=30):
    n = len(cs)
    start = max(0, n - lookback)
    fvg = None
    for i in range(start+2, n):
        c0, c1, c2 = cs[i-2], cs[i-1], cs[i]
        if c2["low"] > c0["high"]:
            gap_low, gap_high = c0["high"], c2["low"]
            post = cs[i:]
            min_after = min([x["low"] for x in post]) if post else c2["low"]
            filled = max(0.0, min(1.0, (gap_high - max(gap_low, min_after)) / max(1e-12, (gap_high-gap_low))))
            fvg = {"type": "bull", "gap": (gap_low, gap_high), "filled_pct": filled}
        if c2["high"] < c0["low"]:
            gap_low, gap_high = c2["high"], c0["low"]
            post = cs[i:]
            max_after = max([x["high"] for x in post]) if post else c2["high"]
            filled = max(0.0, min(1.0, (min(gap_high, max_after) - gap_low) / max(1e-12, (gap_high-gap_low))))
            fvg = {"type": "bear", "gap": (gap_low, gap_high), "filled_pct": filled}
    return fvg

def _smc_bias(pair, last, atr):
    cs = candle_logs.get(pair) or []
    if len(cs) < max(SMC_LOOKBACK, ATR_WINDOW) + 3:
        return {"bias": None, "sweep": None, "fvg": None}
    completed = cs[:-1] if len(cs) >= 2 else cs
    highs, lows = _find_swings(completed[-SMC_LOOKBACK:], SMC_LEFT, SMC_RIGHT)
    offset = len(completed[-SMC_LOOKBACK:])
    highs = [i + len(completed) - offset for i in highs]
    lows  = [i + len(completed) - offset for i in lows]
    bos_up, _, _ = _recent_bos(completed, highs, lows, atr, side="up",
                               lookback=SMC_LOOKBACK, displacement_mult=SMC_DISPLACEMENT_MULT)
    bos_dn, _, _ = _recent_bos(completed, highs, lows, atr, side="down",
                               lookback=SMC_LOOKBACK, displacement_mult=SMC_DISPLACEMENT_MULT)
    bias = "bull" if (bos_up and not bos_dn) else "bear" if (bos_dn and not bos_up) else None
    sweep = _recent_sweep(completed, highs, lows, lookback=SMC_LOOKBACK)
    fvg   = _recent_fvg(completed, lookback=FVG_LOOKBACK)
    return {"bias": bias, "sweep": sweep, "fvg": fvg}

# -------------- SMC confirmation-only decision --------------
def _smc_confirms(side, smc):
    """
    Returns True ONLY if there's explicit structure confirmation
    for the given side.
    BUY: bias=='bull' or sweep=='SSL' or bull FVG (>= fill threshold)
    SELL: bias=='bear' or sweep=='BSL' or bear FVG (>= fill threshold)
    """
    s = side.upper()
    bias  = smc.get("bias")
    sweep = smc.get("sweep")
    fvg   = smc.get("fvg") or {}
    filled_ok = float(fvg.get("filled_pct", 0.0)) >= FVG_MIN_FILL

    if s == "BUY":
        return (bias == "bull") or (sweep == "SSL") or (fvg.get("type") == "bull" and filled_ok)
    else:  # SELL
        return (bias == "bear") or (sweep == "BSL") or (fvg.get("type") == "bear" and filled_ok)

# -------------------- Exchange calls (with precision/min qty & adaptive retry) --------------------
def place_order(pair, side, qty, price_hint=None, balances=None):
    usdt_avail = (balances or {}).get("USDT", 0.0)
    coin_avail = (balances or {}).get(pair[:-4], 0.0)
    price = float(price_hint or 0.0)

    q_pre = fmt_qty(pair, qty)
    q = normalize_qty_for_side(pair, side, price if price > 0 else 1.0, q_pre, usdt_avail, coin_avail)
    if q <= 0:
        scan_log.append(f"{ist_now()} | {pair} | SKIP market {side} — min/fees/wallet/notional fail")
        return {}
    payload = {
        "market": pair,
        "side": side.lower(),
        "order_type": "market_order",
        "total_quantity": f"{q}",
        "timestamp": int(time.time() * 1000)
    }
    scan_log.append(f"{ist_now()} | {pair} | PRE-ORDER {side} qty={payload['total_quantity']} @ MKT "
                    f"(min_qty={_min_qty(pair)}, min_notional={_min_notional(pair)}, qp={_qty_prec(pair)})")
    res = _signed_post(f"{BASE_URL}/exchange/v1/orders/create", payload) or {}
    msg = (res.get("message") or "").lower() if isinstance(res, dict) else ""
    if res and ("precision" in msg or "min" in msg):
        learned = _learn_precision_from_error(pair, msg)
        q_retry = normalize_qty_for_side(pair, side, price if price > 0 else 1.0, q, usdt_avail, coin_avail)
        if q_retry > 0 and (learned or q_retry != q):
            payload["total_quantity"] = f"{fmt_qty(pair, q_retry)}"
            res = _signed_post(f"{BASE_URL}/exchange/v1/orders/create", payload) or {}
            scan_log.append(f"{ist_now()} | {pair} | RETRY market {side} qty={payload['total_quantity']} (learned_precision={learned})")
    return res

def place_limit_order(pair, side, qty, price, balances=None):
    """
    Returns (res, q_used, p_used).
    """
    p = fmt_price(pair, price)
    usdt_avail = (balances or {}).get("USDT", 0.0)
    coin_avail = (balances or {}).get(pair[:-4], 0.0)

    q = normalize_qty_for_side(pair, side, p, fmt_qty(pair, qty), usdt_avail, coin_avail)
    if q <= 0:
        scan_log.append(f"{ist_now()} | {pair} | SKIP limit {side} {p} — min/fees/wallet/notional fail")
        return {}, 0.0, p

    payload = {
        "market": pair,
        "side": side.lower(),
        "order_type": "limit_order",
        "price_per_unit": f"{p}",
        "total_quantity": f"{q}",
        "timestamp": int(time.time() * 1000)
    }
    scan_log.append(f"{ist_now()} | {pair} | PRE-ORDER {side} qty={payload['total_quantity']} @ {payload['price_per_unit']} "
                    f"(min_qty={_min_qty(pair)}, min_notional={_min_notional(pair)}, qp={_qty_prec(pair)})")

    res = _signed_post(f"{BASE_URL}/exchange/v1/orders/create", payload) or {}
    msg = (res.get("message") or "").lower() if isinstance(res, dict) else ""
    if res and ("precision" in msg or "min" in msg):
        learned = _learn_precision_from_error(pair, msg)
        p2 = fmt_price(pair, price)
        q2 = normalize_qty_for_side(pair, side, p2, q, usdt_avail, coin_avail)
        if q2 > 0 and (learned or p2 != p or q2 != q):
            payload["price_per_unit"] = f"{p2}"
            payload["total_quantity"] = f"{q2}"
            res = _signed_post(f"{BASE_URL}/exchange/v1/orders/create", payload) or {}
            scan_log.append(f"{ist_now()} | {pair} | RETRY limit {side} qty={payload['total_quantity']} @ {payload['price_per_unit']} (learned_precision={learned})")
            p, q = p2, q2
    return res, q, p

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

def _quote_prices(pair, last, atr_pct=None):
    off = _effective_half_spread_pct(atr_pct)
    bid_raw = last * (1.0 - off)
    ask_raw = last * (1.0 + off)
    bid = fmt_price(pair, bid_raw)
    ask = fmt_price(pair, ask_raw)
    if bid >= ask:
        step = 10 ** (-int(_rules(pair).get("price_precision", 6)))
        bid = fmt_price(pair, bid - step)
        ask = fmt_price(pair, ask + step)
    return bid, ask

def _qty_for_pair(pair, price, usdt_avail, coin_avail, side, trend_skew):
    notional_target = max(1e-12, QUOTE_USDT) * trend_skew
    q = notional_target / max(price, 1e-9)
    q = min(q, MAX_PER_PAIR_USDT / max(price, 1e-9))
    if side == "BUY":
        q = min(q, usdt_avail / max(price, 1e-9))
    else:
        q = min(q, coin_avail)
    q = fmt_qty(pair, q)
    if q < _min_qty(pair):
        return 0.0
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

def _place_quote(pair, side_word, price, qty, balances):
    # Place, then only record if order actually opened
    res, q_used, p_used = place_limit_order(pair, side_word, qty, price, balances=balances)
    oid = _extract_order_id(res)
    side_key = "bid" if side_word == "BUY" else "ask"
    if oid:
        active_quotes[pair][side_key] = {"id": oid, "px": p_used, "qty": q_used, "ts": int(time.time())}
    scan_log.append(f"{ist_now()} | {pair} | quote {side_word} {q_used} @ {p_used} | id={oid} | res={res}")
    return oid

def _check_quote_fill(pair, side_key, last_price):
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

    # Require bearish SMC confirmation before forced liquidation
    last_px = prices.get(pair, {}).get("price", 0.0)
    atr_abs, _ = _atr_and_pct(pair, last_px, ATR_WINDOW)
    smc_here = _smc_bias(pair, last_px, atr_abs)
    if not _smc_confirms("SELL", smc_here):
        scan_log.append(f"{ist_now()} | {pair} | ORPHAN hold — no bearish structure")
        inventory_timer[pair] = now_ts
        return

    coin = pair[:-4]
    coin_bal = balances.get(coin, 0.0)
    qty = min(0.3 * q_units, coin_bal)
    qty = fmt_qty(pair, qty)
    if qty <= 0 or qty < _min_qty(pair):
        scan_log.append(f"{ist_now()} | {pair} | ORPHAN skip — qty<{_min_qty(pair)}")
        inventory_timer[pair] = now_ts
        return

    limit_px = fmt_price(pair, last_px * 0.999)  # patient limit slightly through
    res, q_used, p_used = place_limit_order(pair, "SELL", qty, limit_px, balances=balances)
    oid = _extract_order_id(res)
    if oid:
        st = get_order_status(order_id=oid)
        _record_fill_from_status(pair, "SELL", st, oid)
    scan_log.append(f"{ist_now()} | {pair} | ORPHAN LIMIT SELL {q_used} @ {p_used} | res={res}")
    inventory_timer[pair] = now_ts

# -------------------- Main loop (autostart-safe) --------------------
last_keepalive = 0
_autostart_lock = threading.Lock()

def scan_loop():
    global running, error_message, status_epoch, last_keepalive, _last_rules_refresh
    scan_log.clear()
    running = True

    while running:
        now_real = time.time()
        if now_real - last_keepalive >= KEEPALIVE_SEC:
            _keepalive_ping()
            last_keepalive = now_real

        prices = fetch_all_prices()

        if (time.time() - _last_rules_refresh) >= RULES_REFRESH_SEC:
            fetch_pair_precisions()
            _last_rules_refresh = time.time()

        now_ts = int(time.time())
        balances = get_wallet_balances()
        free_balances = _compute_free_balances(balances)

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
                    continue

            if now_ts < pair_freeze_until[pair]:
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
            bid_px, ask_px = _quote_prices(pair, last, atr_pct)

            usdt = free_balances.get("USDT", 0.0)
            coin = pair[:-4]
            coin_bal = free_balances.get(coin, 0.0)
            net_units = _net_inventory_units(pair)

            # inventory & EMA trend biases (used only for sizing; SMC decides whether to shoot)
            reduce_bias_bid = INVENTORY_REDUCE_BIAS if net_units < 0 else 1.0
            reduce_bias_ask = INVENTORY_REDUCE_BIAS if net_units > 0 else 1.0
            trend_bias_bid = 1.2 if bullish else 0.9
            trend_bias_ask = 1.2 if not bullish else 0.9

            # -------- Risk/vol filters --------
            allow_bid, allow_ask = True, True
            if gross_usdt > INVENTORY_USDT_CAP:
                allow_bid = False
            if atr_pct is not None and atr_pct > MAX_ATR_PCT:
                allow_bid = False
            if slope_pct is not None and slope_pct > MAX_SLOPE_PCT:
                if bullish:
                    allow_ask = False; allow_bid = True
                else:
                    allow_bid = False; allow_ask = True

            # -------- SMC confirmation-only gating --------
            if SMC_CONFIRM_ONLY:
                allow_bid = allow_bid and _smc_confirms("BUY", smc)
                allow_ask = allow_ask and _smc_confirms("SELL", smc)

            if SMC_LOG:
                scan_log.append(
                    f"{ist_now()} | {pair} | SMC bias={smc_bias} sweep={smc_sweep} fvg={smc_fvg} "
                    f"| confirmBUY={_smc_confirms('BUY', smc)} confirmSELL={_smc_confirms('SELL', smc)} "
                    f"| allow_bid={allow_bid} allow_ask={allow_ask}"
                )

            # ------- Stickier cancel/re-quote logic (ATR-aware + rate-limited) -------
            for side_key, q in list(active_quotes[pair].items()):
                if not q:
                    continue
                age = now_ts - int(q.get("ts", now_ts))
                px = q.get("px", last)
                drift = abs(last - px) / max(px, 1e-9)

                # reset per-minute counter
                if now_ts - requote_counter[pair]["t"] >= 60:
                    requote_counter[pair] = {"t": now_ts, "n": 0}

                # ATR-aware dynamic drift threshold
                dynamic_drift = DRIFT_REQUOTE_PCT
                if atr_pct is not None:
                    dynamic_drift = max(DRIFT_REQUOTE_PCT, atr_pct * DRIFT_REQUOTE_ATR_MULT)

                # protect young quotes unless drift is clearly large
                if age < MIN_QUOTE_LIFETIME_SEC and drift < 2 * dynamic_drift:
                    continue

                # decide cancel: (age big OR drift big) AND under rate limit
                if (age >= max(MIN_QUOTE_LIFETIME_SEC, QUOTE_TTL_SEC) or drift >= dynamic_drift) \
                   and requote_counter[pair]["n"] < MAX_REQUOTE_PER_MIN:
                    _cancel_quote(pair, side_key)
                    requote_counter[pair]["n"] += 1

            # check fills
            for side_key in ("bid", "ask"):
                _check_quote_fill(pair, side_key, last)

            # -------- Inventory kill-switch after BUY fill (safer: sustained+SMC+limit) --------
            lb = last_buy_fill.get(pair, {})
            if lb and lb.get("entry") and now_ts >= kill_cooldown_until[pair]:
                entry = float(lb["entry"]); filled_qty = float(lb["qty"])
                breach = (last <= entry * (1.0 - KILL_SWITCH_PCT))

                if breach:
                    if kill_breach_since.get(pair, 0) == 0:
                        kill_breach_since[pair] = now_ts
                    sustained = (now_ts - kill_breach_since[pair]) >= KILL_SWITCH_CONFIRM_SEC
                    smc_ok = True
                    if KILL_SWITCH_REQUIRE_SMC:
                        smc_ok = (_smc_confirms("SELL", smc) is True)

                    if sustained and smc_ok:
                        qty_to_sell = min(filled_qty * KILL_SWITCH_SELL_FRAC, coin_bal)
                        qty_to_sell = fmt_qty(pair, qty_to_sell)
                        if qty_to_sell >= _min_qty(pair):
                            _cancel_quote(pair, "bid")
                            limit_px = fmt_price(pair, last * (1.0 - KILL_SWITCH_LIMIT_OFFSET))
                            res, q_used, p_used = place_limit_order(pair, "SELL", qty_to_sell, limit_px, balances=free_balances)
                            oid = _extract_order_id(res)
                            if oid:
                                st = get_order_status(order_id=oid)
                                _record_fill_from_status(pair, "SELL", st, oid)
                            scan_log.append(f"{ist_now()} | {pair} | KILL-SWITCH LIMIT: {q_used} @ {p_used} | res={res}")
                            kill_cooldown_until[pair] = now_ts + KILL_SWITCH_COOLDOWN_SEC
                            last_buy_fill[pair]["qty"] = max(0.0, filled_qty - q_used)
                else:
                    kill_breach_since[pair] = 0

            # sizing for quotes (only if allowed post-SMC confirmation)
            qty_bid = _qty_for_pair(pair, bid_px, usdt, coin_bal, "BUY", reduce_bias_bid * trend_bias_bid) if allow_bid else 0.0
            qty_ask = _qty_for_pair(pair, ask_px, usdt, coin_bal, "SELL", reduce_bias_ask * trend_bias_ask) if allow_ask else 0.0

            # place quotes if empty (keep prices precision-safe and non-crossing)
            if qty_bid > 0 and not active_quotes[pair]["bid"]:
                bpx = fmt_price(pair, min(bid_px, last * (1 - 1e-6)))
                _place_quote(pair, "BUY", bpx, qty_bid, free_balances)

            if qty_ask > 0 and coin_bal >= _min_qty(pair):
                if not active_quotes[pair]["ask"]:
                    apx = fmt_price(pair, max(ask_px, last * (1 + 1e-6)))
                    _place_quote(pair, "SELL", apx, qty_ask, free_balances)

            _manage_orphan_inventory(pair, now_ts, prices, free_balances)

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
    usdt_total = balances.get("USDT", 0.0)
    locked_usdt, locked_coin = _locked_from_active_quotes()
    free_balances = _compute_free_balances(balances)
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

    # ---- keepalive status for UI ----
    now_real = time.time()
    last_age = now_real - (last_keepalive or 0)
    keepalive_info = {
        "enabled": bool(KEEPALIVE_TOKEN),               # true if token set on server
        "interval_sec": KEEPALIVE_SEC,
        "last_ping_epoch": int(last_keepalive or 0),
        "last_ping_age_sec": (max(0, int(last_age)) if last_keepalive else None),
        "next_due_sec": (max(0, int(KEEPALIVE_SEC - last_age)) if last_keepalive else None),
        "app_base_url": APP_BASE_URL,
    }

    return jsonify({
        "status": status["msg"],
        "last": status["last"],
        "status_epoch": status_epoch,
        "usdt": usdt_total,
        "usdt_free": free_balances.get("USDT", 0.0),
        "usdt_locked": locked_usdt,
        "locked_coin": locked_coin,
        "profit_today": profit_today,
        "profit_yesterday": profit_yesterday,
        "pnl_cumulative": cumulative_pnl,
        "processed_orders": len(profit_state.get("processed_orders", [])),
        "inventory_markets": list(profit_state.get("inventory", {}).keys()),
        "quotes": visible_quotes,
        "coins": coins,
        "trades": trade_log[-10:][::-1],
        "scans": scan_log[-60:][::-1],
        "error": error_message,
        "keepalive": keepalive_info,   # <<< new
    })

@app.route("/ping", methods=["GET", "HEAD"])
def ping():
    token = os.environ.get("KEEPALIVE_TOKEN", "")
    provided = (request.args.get("t") or
                request.headers.get("X-Keepalive-Token") or "")
    if token and provided != token:
        print(f"[{ist_now()}] /ping forbidden (bad token) method={request.method}")
        return "forbidden", 403

    # Log, but avoid printing body for HEAD
    print(f"[{ist_now()}] /ping ok method={request.method}")
    if request.method == "HEAD":
        # Return 200 with no body for uptime monitors
        return ("", 200)
    return ("pong", 200)

# -------------------- Safe autostart --------------------
_autostart_lock = threading.Lock()
def _start_loop_once():
    global running
    with _autostart_lock:
        if not running:
            running = True
            t = threading.Thread(target=scan_loop, daemon=True)
            t.start()

def _boot_kick():
    try:
        load_profit_state()
        fetch_pair_precisions()
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
