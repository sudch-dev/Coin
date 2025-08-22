# app.py

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

APP_BASE_URL = os.environ.get("APP_BASE_URL", "")
KEEPALIVE_TOKEN = os.environ.get("KEEPALIVE_TOKEN", "")
TRADE_ENABLED = os.environ.get("TRADE_ENABLED", "1")  # "1" to enable trading, else signals only

# -------------------- API / Markets --------------------
API_KEY = os.environ.get("API_KEY", "").strip()
API_SECRET_RAW = os.environ.get("API_SECRET", "")
API_SECRET = API_SECRET_RAW.encode() if isinstance(API_SECRET_RAW, str) else API_SECRET_RAW
BASE_URL = "https://api.coindcx.com"

# Pairs (LTC removed per request)
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

# -------------------- Strategy / runtime settings --------------------
CANDLE_INTERVAL = 5        # seconds (your request)
POLL_SEC = 1.0             # tick poll
KEEPALIVE_SEC = 240

# fees are used for min-notional checks; we aren’t adding/taking here
FEE_PCT_PER_SIDE = 0.0010

# Position sizing
BUY_FRACTION_USDT = 0.30   # 30% of free USDT per entry
BUY_HEADROOM = 1.0005      # 5 bps cushion for fees/rounding

# Take-profit (no SL)
TP_PCT = 0.01              # 1% target

# -------------------- Time helpers --------------------
IST = timezone('Asia/Kolkata')
def ist_now(): return datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')
def ist_date(): return datetime.now(IST).strftime('%Y-%m-%d')
def ist_yesterday(): return (datetime.now(IST) - timedelta(days=1)).strftime('%Y-%m-%d')

# -------------------- State --------------------
tick_logs = {p: [] for p in PAIRS}     # (ts, price) raw ticks
candle_logs = {p: [] for p in PAIRS}   # 5s candles
scan_log, trade_log = [], []
running = False
status = {"msg": "Idle", "last": ""}
status_epoch = 0
error_message = ""

# inventory lots (FIFO P&L)
PROFIT_STATE_FILE = "profit_state.json"
profit_state = {
    "cumulative_pnl": 0.0,
    "daily": {},
    "inventory": {},         # per market: [[qty, cost], ...]
    "processed_orders": []
}

# track last entry per pair for TP management
last_entry = {p: {"entry": None, "qty": 0.0, "tp_order_id": None} for p in PAIRS}

# keepalive
_last_keepalive = 0

# rules refresh cadence
_last_rules_refresh = 0
RULES_REFRESH_SEC = 1800  # 30 mins

# -------------------- Persistence helpers --------------------
from collections import deque

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
        # track last entry for TP
        last_entry[market]["entry"] = price
        last_entry[market]["qty"] = last_entry[market].get("qty", 0.0) + qty
    else:
        # FIFO realization
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
        # reduce last_entry qty
        last_entry[market]["qty"] = max(0.0, last_entry[market].get("qty", 0.0) - qty)
        if last_entry[market]["qty"] <= 1e-18:
            last_entry[market] = {"entry": None, "qty": 0.0, "tp_order_id": None}

    _set_inventory_from_deque(market, inv)
    profit_state["processed_orders"].append(order_id)
    profit_state["cumulative_pnl"] = float(profit_state.get("cumulative_pnl", 0.0) + realized)
    dkey = ist_date()
    profit_state["daily"][dkey] = float(profit_state["daily"].get(dkey, 0.0) + realized)
    save_profit_state()

def compute_realized_pnl_today():
    return round(profit_state["daily"].get(ist_date(), 0.0), 6)

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
        headers = {}
        if KEEPALIVE_TOKEN:
            url = f"{url}?t={KEEPALIVE_TOKEN}"
            headers = {"X-Keepalive-Token": KEEPALIVE_TOKEN}
        requests.request("HEAD", url, headers=headers, timeout=5)
    except Exception:
        pass

# -------------------- Market rules (precision / min qty / notional) --------------------
def fetch_pair_precisions():
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

# -------------------- Precision helpers --------------------
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

# -------------------- Orders --------------------
def _why_trade_blocked():
    reasons = []
    if TRADE_ENABLED != "1": reasons.append("TRADE_ENABLED!=1")
    if not API_KEY: reasons.append("API_KEY missing")
    if not API_SECRET_RAW: reasons.append("API_SECRET missing")
    return "; ".join(reasons) or None

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

def get_order_status(order_id=None, client_order_id=None):
    body = {"timestamp": int(time.time() * 1000)}
    if order_id: body["id"] = order_id
    if client_order_id: body["client_order_id"] = client_order_id
    res = _signed_post(f"{BASE_URL}/exchange/v1/orders/status", body)
    return res if isinstance(res, dict) else {}

def place_market(pair, side, qty):
    payload = {
        "market": pair,
        "side": side.lower(),
        "order_type": "market_order",
        "total_quantity": f"{qty}",
        "timestamp": int(time.time() * 1000)
    }
    scan_log.append(f"{ist_now()} | {pair} | PRE-ORDER {side} MKT qty={payload['total_quantity']}"
                    f" (min_qty={_min_qty(pair)}, min_notional={_min_notional(pair)}, qp={_qty_prec(pair)})")
    res = _signed_post(f"{BASE_URL}/exchange/v1/orders/create", payload) or {}
    msg = (res.get("message") or "").lower() if isinstance(res, dict) else ""
    if res and ("precision" in msg or "min" in msg):
        learned = _learn_precision_from_error(pair, msg)
        if learned:
            # retry with rounded qty again
            payload["total_quantity"] = f"{fmt_qty(pair, float(qty))}"
            res = _signed_post(f"{BASE_URL}/exchange/v1/orders/create", payload) or {}
            scan_log.append(f"{ist_now()} | {pair} | RETRY market {side} qty={payload['total_quantity']} (learned_precision={learned})")
    return res

def place_limit(pair, side, qty, price):
    p = fmt_price(pair, price)
    q = fmt_qty(pair, qty)
    payload = {
        "market": pair,
        "side": side.lower(),
        "order_type": "limit_order",
        "price_per_unit": f"{p}",
        "total_quantity": f"{q}",
        "timestamp": int(time.time() * 1000)
    }
    scan_log.append(f"{ist_now()} | {pair} | PRE-ORDER {side} LIMIT qty={payload['total_quantity']} @ {payload['price_per_unit']}"
                    f" (min_qty={_min_qty(pair)}, min_notional={_min_notional(pair)}, qp={_qty_prec(pair)})")
    res = _signed_post(f"{BASE_URL}/exchange/v1/orders/create", payload) or {}
    msg = (res.get("message") or "").lower() if isinstance(res, dict) else ""
    if res and ("precision" in msg or "min" in msg):
        learned = _learn_precision_from_error(pair, msg)
        if learned:
            payload["price_per_unit"] = f"{fmt_price(pair, price)}"
            payload["total_quantity"] = f"{fmt_qty(pair, q)}"
            res = _signed_post(f"{BASE_URL}/exchange/v1/orders/create", payload) or {}
            scan_log.append(f"{ist_now()} | {pair} | RETRY limit {side} qty={payload['total_quantity']} @ {payload['price_per_unit']} (learned_precision={learned})")
    return res

def _record_fill_from_status(market, side, st, order_id):
    def _fnum(x, d=0.0):
        try:
            return float(x)
        except:
            return d
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

# -------------------- Indicators --------------------
def _ema(vals, n):
    if len(vals) < n:
        return None
    k = 2 / (n + 1)
    ema = sum(vals[:n]) / n
    for v in vals[n:]:
        ema = v * k + ema * (1 - k)
    return ema

def _macd(vals, fast=12, slow=26, signal=9):
    if len(vals) < slow + signal:
        return None, None, None
    # EMA series
    def ema_series(data, n):
        k = 2 / (n + 1)
        out = []
        ema = sum(data[:n]) / n
        out.extend([None]*(n-1))
        out.append(ema)
        for v in data[n:]:
            ema = v * k + ema * (1 - k)
            out.append(ema)
        return out
    ema_fast_s = ema_series(vals, fast)
    ema_slow_s = ema_series(vals, slow)
    macd_line_s = []
    for a, b in zip(ema_fast_s, ema_slow_s):
        macd_line_s.append(None if (a is None or b is None) else (a - b))
    # signal line
    macd_clean = [m for m in macd_line_s if m is not None]
    if len(macd_clean) < signal:
        return None, None, None
    sig_s = ema_series(macd_clean, signal)
    # align lengths
    pad = len(macd_line_s) - len(sig_s)
    signal_line_s = [None]*pad + sig_s
    hist_s = []
    for m, s in zip(macd_line_s, signal_line_s):
        hist_s.append(None if (m is None or s is None) else (m - s))
    return macd_line_s, signal_line_s, hist_s

def _divergence_note(prices, macd_line):
    """
    Lightweight divergence note:
    - If price makes higher high but MACD doesn't -> bearish divergence
    - If price makes lower low but MACD doesn't -> bullish divergence
    Only checks last ~10 closes. Log only (does not gate).
    """
    if len(prices) < 15 or macd_line is None:
        return None
    closes = prices[-15:]
    macds  = [m for m in macd_line[-15:] if m is not None]
    if len(macds) < 5:
        return None
    ph = max(closes[-10:-1]); ch = closes[-1]
    mh = max(macds[-10:-1]);   mhc = macds[-1]
    pl = min(closes[-10:-1]); cl = closes[-1]
    ml = min(macds[-10:-1]);   mlc = macds[-1]

    if ch > ph and mhc <= mh:
        return "bearish_divergence"
    if cl < pl and mlc >= ml:
        return "bullish_divergence"
    return None

# -------------------- Strategy logic --------------------
def _compute_signals(pair):
    cs = candle_logs.get(pair) or []
    if len(cs) < 35:  # enough history for MACD 26+9 and EMAs
        return None

    closes = [c["close"] for c in cs]
    ema5  = _ema(closes, 5)
    ema20 = _ema(closes, 20)

    # emulate last two points for cross
    ema5_prev  = _ema(closes[:-1], 5)
    ema20_prev = _ema(closes[:-1], 20)

    macd_line, signal_line, hist = _macd(closes)
    if macd_line is None:  # not enough history
        return None

    m_curr = macd_line[-1]; s_curr = signal_line[-1]
    m_prev = macd_line[-2] if len(macd_line) >= 2 else None
    s_prev = signal_line[-2] if len(signal_line) >= 2 else None

    bull_cross_ema = (ema5_prev is not None and ema20_prev is not None and ema5_prev <= ema20_prev and ema5 >= ema20)
    bear_cross_ema = (ema5_prev is not None and ema20_prev is not None and ema5_prev >= ema20_prev and ema5 <= ema20)

    bull_cross_macd = (m_prev is not None and s_prev is not None and m_prev <= s_prev and m_curr > s_curr)
    bear_cross_macd = (m_prev is not None and s_prev is not None and m_prev >= s_prev and m_curr < s_curr)

    div_note = _divergence_note(closes, macd_line)

    return {
        "ema5": ema5, "ema20": ema20,
        "macd": m_curr, "signal": s_curr, "hist": (m_curr - s_curr) if (m_curr is not None and s_curr is not None) else None,
        "bull_cross_ema": bool(bull_cross_ema),
        "bear_cross_ema": bool(bear_cross_ema),
        "bull_cross_macd": bool(bull_cross_macd),
        "bear_cross_macd": bool(bear_cross_macd),
        "divergence": div_note
    }

def _can_buy(signals):
    return signals["bull_cross_ema"] and signals["bull_cross_macd"]

def _can_sell(signals):
    return signals["bear_cross_ema"] and signals["bear_cross_macd"]

# -------------------- Aggregation --------------------
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
    candle_logs[pair] = candles[-600:]  # ~50 min of 5s bars

# -------------------- Main loop --------------------
_autostart_lock = threading.Lock()

def scan_loop():
    global running, status_epoch, _last_keepalive, _last_rules_refresh
    scan_log.clear()
    running = True
    scan_log.append(f"{ist_now()} | LOOP started (signals + trading)")

    while running:
        now_real = time.time()
        if now_real - _last_keepalive >= KEEPALIVE_SEC:
            _keepalive_ping()
            _last_keepalive = now_real

        prices = fetch_all_prices()

        if (time.time() - _last_rules_refresh) >= RULES_REFRESH_SEC:
            fetch_pair_precisions()
            _last_rules_refresh = time.time()

        now_ts = int(time.time())
        balances = get_wallet_balances()

        # ticks & candles
        for pair in PAIRS:
            if pair in prices:
                px = prices[pair]["price"]
                tick_logs[pair].append((now_ts, px))
                if len(tick_logs[pair]) > 5000:
                    tick_logs[pair] = tick_logs[pair][-5000:]
                aggregate_candles(pair, CANDLE_INTERVAL)

        # per-pair strategy
        blocked_reason = _why_trade_blocked()
        if blocked_reason:
            scan_log.append(f"{ist_now()} | TRADE BLOCKED | {blocked_reason} — signals only (no orders)")

        usdt = balances.get("USDT", 0.0)

        for pair in PAIRS:
            if pair not in prices:
                continue
            last_px = prices[pair]["price"]
            signals = _compute_signals(pair)
            if not signals:
                continue

            # log signals
            scan_log.append(
                f"{ist_now()} | {pair} | "
                f"EMA5={round(signals['ema5'],6) if signals['ema5'] else None} "
                f"EMA20={round(signals['ema20'],6) if signals['ema20'] else None} "
                f"MACD={round(signals['macd'],6) if signals['macd'] is not None else None} "
                f"SIG={round(signals['signal'],6) if signals['signal'] is not None else None} "
                f"HIST={round(signals['hist'],6) if signals['hist'] is not None else None} "
                f"DIV={signals['divergence']} | "
                f"Gates: BUY={_can_buy(signals)} SELL={_can_sell(signals)}"
            )

            # compute free coin balance for possible sells
            coin = pair[:-4]
            coin_bal = float(balances.get(coin, 0.0))

            # ---- SELL path (whichever comes first: 1% TP or Sell signal) ----
            entry = last_entry[pair]["entry"]
            qty_held = last_entry[pair]["qty"]
            if entry and qty_held > 0:
                up_pct = (last_px/entry - 1.0) if entry > 0 else 0.0
                tp_hit = (up_pct >= TP_PCT)
                sell_signal = _can_sell(signals)

                if tp_hit or sell_signal:
                    if blocked_reason:
                        scan_log.append(f"{ist_now()} | {pair} | SELL would place (tp_hit={tp_hit} sell_signal={sell_signal}) but blocked: {blocked_reason}")
                    else:
                        sell_qty = min(qty_held, coin_bal)
                        sell_qty = fmt_qty(pair, sell_qty)
                        if sell_qty >= _min_qty(pair):
                            res = place_market(pair, "SELL", sell_qty)
                            oid = _extract_order_id(res)
                            if oid:
                                st = get_order_status(order_id=oid)
                                filled, avg_px = _record_fill_from_status(pair, "SELL", st, oid)
                                trade_log.append({"time": ist_now(), "pair": pair, "side": "SELL", "qty": filled, "px": avg_px, "reason": "TP" if tp_hit else "SellSignal"})
                                scan_log.append(f"{ist_now()} | {pair} | SOLD {filled} @ {avg_px} (reason: {'TP' if tp_hit else 'SellSignal'})")
                            else:
                                scan_log.append(f"{ist_now()} | {pair} | SELL create failed | res={res}")
                        else:
                            scan_log.append(f"{ist_now()} | {pair} | SELL skip — qty<{_min_qty(pair)}")

                    # proceed to next pair after attempting sell
                    continue

            # ---- BUY path (only if flat on that coin) ----
            if not entry or qty_held <= 1e-12:
                if _can_buy(signals):
                    if blocked_reason:
                        scan_log.append(f"{ist_now()} | {pair} | BUY would place but blocked: {blocked_reason}")
                    else:
                        # 30% of free USDT -> qty (fee/headroom aware rounding)
                        spend_usdt = float(usdt) * BUY_FRACTION_USDT
                        if spend_usdt <= 0:
                            scan_log.append(f"{ist_now()} | {pair} | BUY skip — USDT=0")
                        else:
                            # max affordable qty with fees cushion
                            denom = max(last_px * (1.0 + FEE_PCT_PER_SIDE) * BUY_HEADROOM, 1e-12)
                            raw_qty = spend_usdt / denom
                            qty = fmt_qty(pair, raw_qty)

                            # ensure min_notional
                            mn = _min_notional(pair)
                            if mn > 0 and (qty * last_px) < mn:
                                qty = fmt_qty(pair, mn / last_px)

                            if qty >= _min_qty(pair) and (qty * last_px) > 0:
                                res = place_market(pair, "BUY", qty)
                                oid = _extract_order_id(res)
                                if oid:
                                    st = get_order_status(order_id=oid)
                                    filled, avg_px = _record_fill_from_status(pair, "BUY", st, oid)
                                    trade_log.append({"time": ist_now(), "pair": pair, "side": "BUY", "qty": filled, "px": avg_px, "reason": "BuySignal"})
                                    scan_log.append(f"{ist_now()} | {pair} | BOUGHT {filled} @ {avg_px}")
                                    # place TP limit (optional). We do NOT need a resting TP since we exit on either TP or sell signal.
                                    # Keeping simple: we monitor and sell when tp_hit or sell_signal; no resting order needed.
                                else:
                                    scan_log.append(f"{ist_now()} | {pair} | BUY create failed | res={res}")
                            else:
                                scan_log.append(f"{ist_now()} | {pair} | BUY skip — qty<{_min_qty(pair)} or notional too small")

        status["msg"], status["last"] = "Running", ist_now()
        status_epoch = int(time.time())
        time.sleep(POLL_SEC)

    status["msg"] = "Idle"
    scan_log.append(f"{ist_now()} | LOOP stopped")

# -------------------- Routes & UI --------------------
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
    return jsonify({"status": "stopped"})

@app.route("/status")
def get_status():
    balances = get_wallet_balances()
    coins = {pair[:-4]: balances.get(pair[:-4], 0.0) for pair in PAIRS}
    profit_today = compute_realized_pnl_today()
    profit_yesterday = round(profit_state["daily"].get(ist_yesterday(), 0.0), 6)
    cumulative_pnl = round(profit_state.get("cumulative_pnl", 0.0), 6)

    # visible positions
    positions = {p: {"entry": last_entry[p]["entry"], "qty": last_entry[p]["qty"]} for p in PAIRS}

    # keepalive ui info
    now_real = time.time()
    ka_last_age = now_real - (_last_keepalive or 0)
    keepalive_info = {
        "enabled": bool(KEEPALIVE_TOKEN),
        "interval_sec": KEEPALIVE_SEC,
        "last_ping_epoch": int(_last_keepalive or 0),
        "last_ping_age_sec": (max(0, int(ka_last_age)) if _last_keepalive else None),
        "next_due_sec": (max(0, int(KEEPALIVE_SEC - ka_last_age)) if _last_keepalive else None),
        "app_base_url": APP_BASE_URL,
    }

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
        "positions": positions,
        "coins": coins,
        "trades": trade_log[-12:][::-1],
        "scans": scan_log[-120:][::-1],
        "error": error_message,
        "keepalive": keepalive_info
    })

@app.route("/ping", methods=["GET", "HEAD"])
def ping():
    token = os.environ.get("KEEPALIVE_TOKEN", "")
    provided = (request.args.get("t") or request.headers.get("X-Keepalive-Token") or "")
    if token and provided != token:
        print(f"[{ist_now()}] /ping forbidden (bad token) method={request.method}")
        return "forbidden", 403
    print(f"{ist_now()} | /ping ok method={request.method}")
    if request.method == "HEAD":
        return ("", 200)
    return ("pong", 200)

# -------------------- Boot helpers --------------------
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

if __name__ == "__main__":
    load_profit_state()
    fetch_pair_precisions()
    _start_loop_once()
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
