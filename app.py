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

APP_BASE_URL    = os.environ.get("APP_BASE_URL", "https://coin-4k37.onrender.com")
KEEPALIVE_TOKEN = os.environ.get("KEEPALIVE_TOKEN", "")

# -------------------- Exchange creds --------------------
API_KEY = os.environ.get("API_KEY")
API_SECRET_RAW = os.environ.get("API_SECRET", "")
API_SECRET = API_SECRET_RAW.encode() if isinstance(API_SECRET_RAW, str) else API_SECRET_RAW
BASE_URL = "https://api.coindcx.com"

# -------------------- Markets --------------------
# (Removed LTC as requested)
PAIRS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "SHIBUSDT", "SOLUSDT",
    "DOGEUSDT", "ADAUSDT", "AEROUSDT", "BNBUSDT"
]

# Local fallbacks; refreshed from live rules
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

# -------------------- Strategy knobs --------------------
CANDLE_INTERVAL = 5     # seconds (you asked for 5s)
POLL_SEC        = 1.0
TP_TARGET_PCT   = 0.010  # 1% target (revised)
BUY_FRACTION_USDT = 0.30 # use 30% of free USDT on buys
SELL_ALL_COIN     = True # exit uses full coin wallet

# EMA & MACD (entry/exit gates)
EMA_FAST = 5
EMA_SLOW = 10
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Fees and safety headroom
FEE_PCT_PER_SIDE = 0.0010
BUY_HEADROOM     = 1.0005  # 5bp cushion so rounding always fits

# Keepalive
KEEPALIVE_SEC = 240

# Rules refresh cadence
RULES_REFRESH_SEC = 1800  # 30 min
_last_rules_refresh = 0

# -------------------- Time helpers --------------------
IST = timezone('Asia/Kolkata')
def ist_now(): return datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')
def ist_date(): return datetime.now(IST).strftime('%Y-%m-%d')
def ist_yesterday(): return (datetime.now(IST) - timedelta(days=1)).strftime('%Y-%m-%d')

# -------------------- State --------------------
tick_logs     = {p: [] for p in PAIRS}   # (ts, price) ticks for making 5s candles
candle_logs   = {p: [] for p in PAIRS}   # list of dicts {open,high,low,close,start}
scan_log      = []                       # textual decisions
trade_log     = []                       # successful trades
running       = False
status        = {"msg": "Idle", "last": ""}
status_epoch  = 0
error_message = ""
last_keepalive = 0

# Position tracking per pair (simple swing system)
positions = {
    # pair: {"qty": float, "entry": float, "ts": int}
}

# P&L persistence
PROFIT_STATE_FILE = "profit_state.json"
profit_state = {
    "cumulative_pnl": 0.0,
    "daily": {},
    "processed_orders": []
}

# -------------------- P&L helpers --------------------
def load_profit_state():
    try:
        with open(PROFIT_STATE_FILE, "r") as f:
            data = json.load(f)
        profit_state["cumulative_pnl"] = float(data.get("cumulative_pnl", 0.0))
        profit_state["daily"] = dict(data.get("daily", {}))
        profit_state["processed_orders"] = list(data.get("processed_orders", []))
    except:
        pass

def save_profit_state():
    tmp = {
        "cumulative_pnl": round(profit_state.get("cumulative_pnl", 0.0), 6),
        "daily": {k: round(v, 6) for k, v in profit_state.get("daily", {}).items()},
        "processed_orders": profit_state.get("processed_orders", [])
    }
    try:
        with open(PROFIT_STATE_FILE, "w") as f:
            json.dump(tmp, f)
    except:
        pass

def record_realized_pnl(pnl):
    pnl = float(pnl or 0.0)
    profit_state["cumulative_pnl"] = float(profit_state.get("cumulative_pnl", 0.0) + pnl)
    dkey = ist_date()
    profit_state["daily"][dkey] = float(profit_state["daily"].get(dkey, 0.0) + pnl)
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
            headers["X-Keepalive-Token"] = KEEPALIVE_TOKEN
        requests.get(url, headers=headers, timeout=5)
    except Exception:
        pass

# -------------------- Exchange helpers --------------------
def fetch_pair_precisions():
    try:
        r = requests.get(f"{BASE_URL}/exchange/v1/markets_details", timeout=12)
        if not r.ok:
            _log_http_issue("markets_details", r)
            return
        data = r.json()
        by_pair = {}
        for item in data:
            p = item.get("pair") or item.get("market") or item.get("coindcx_name")
            if not p:
                continue
            by_pair[p] = {
                "price_precision": int(item.get("target_currency_precision", 6)),
                "qty_precision":   int(item.get("base_currency_precision", 6)),
                "min_qty":         float(item.get("min_quantity", 0.0) or 0.0),
                "min_notional":    float(item.get("min_notional", 0.0) or 0.0)
            }
        for p in PAIRS:
            if p in by_pair:
                PAIR_RULES[p] = by_pair[p]
        scan_log.append(f"{ist_now()} | market rules refreshed")
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

# -------------------- Precision helpers --------------------
def _rules(pair): return PAIR_RULES.get(pair, {})
def _min_qty(pair): return float(_rules(pair).get("min_qty", 0.0) or 0.0)
def _qty_prec(pair): return int(_rules(pair).get("qty_precision", 6))

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

def _fee_multiplier(side):
    return 1.0 + FEE_PCT_PER_SIDE if side.upper() == "BUY" else 1.0 - FEE_PCT_PER_SIDE

def normalize_qty_for_buy(pair, price, usdt_avail):
    """
    Turn a USDT notional into a precision/min-qty safe quantity that fits fees.
    """
    if price <= 0:
        return 0.0
    # fee + headroom aware
    denom = max(price * _fee_multiplier("BUY") * BUY_HEADROOM, 1e-12)
    q = usdt_avail / denom
    q = fmt_qty(pair, q)
    # after rounding make sure still fits
    step = _qty_step(pair)
    while q >= _min_qty(pair) and (price * q * _fee_multiplier("BUY") * BUY_HEADROOM) > usdt_avail + 1e-12:
        q = fmt_qty(pair, max(0.0, q - step))
    if q < _min_qty(pair): return 0.0
    return q

def place_market(pair, side, qty):
    payload = {
        "market": pair,
        "side": side.lower(),
        "order_type": "market_order",
        "total_quantity": f"{fmt_qty(pair, qty)}",
        "timestamp": int(time.time() * 1000)
    }
    scan_log.append(f"{ist_now()} | {pair} | PRE-ORDER {side} qty={payload['total_quantity']} @ MKT")
    res = _signed_post(f"{BASE_URL}/exchange/v1/orders/create", payload) or {}
    return res

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

def _filled_avg_from_status(st):
    try:
        total_q  = float(st.get("total_quantity", st.get("quantity", 0)))
        remain_q = float(st.get("remaining_quantity", st.get("remaining_qty", 0)))
        exec_q   = float(st.get("executed_quantity", st.get("filled_qty", 0)))
        filled   = exec_q if exec_q > 0 else max(0.0, total_q - remain_q)
        avg_px   = float(st.get("avg_price", st.get("average_price", st.get("avg_execution_price", st.get("price", 0)))))
        return filled, avg_px
    except:
        return 0.0, 0.0

# -------------------- Candle & indicators --------------------
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
    candle_logs[pair] = candles[-300:]  # keep ~25 min of 5s bars

def _ema_series(vals, n):
    if len(vals) == 0:
        return []
    k = 2 / (n + 1)
    ema_vals = []
    ema = vals[0]
    for v in vals:
        ema = v * k + ema * (1 - k)
        ema_vals.append(ema)
    return ema_vals

def _macd_series(closes):
    if len(closes) < MACD_SLOW + MACD_SIGNAL:
        return [], [], []
    ema_fast = _ema_series(closes, MACD_FAST)
    ema_slow = _ema_series(closes, MACD_SLOW)
    macd = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal = _ema_series(macd, MACD_SIGNAL)
    hist = [m - s for m, s in zip(macd[-len(signal):], signal)]
    # align lengths: take last len(signal)
    macd_al = macd[-len(signal):]
    return macd_al, signal, hist

def _cross_up(a_prev, a_now, b_prev, b_now):
    return a_prev is not None and b_prev is not None and a_prev <= b_prev and a_now > b_now

def _cross_dn(a_prev, a_now, b_prev, b_now):
    return a_prev is not None and b_prev is not None and a_prev >= b_prev and a_now < b_now

def _swing_highs(vals, look=5):
    idx = []
    for i in range(look, len(vals)-look):
        if all(vals[i] >= vals[i-k] for k in range(1, look+1)) and all(vals[i] >= vals[i+k] for k in range(1, look+1)):
            idx.append(i)
    return idx

def _swing_lows(vals, look=5):
    idx = []
    for i in range(look, len(vals)-look):
        if all(vals[i] <= vals[i-k] for k in range(1, look+1)) and all(vals[i] <= vals[i+k] for k in range(1, look+1)):
            idx.append(i)
    return idx

def _macd_convergence_divergence(closes, macd_line):
    """
    Heuristic:
    - look at last 2 swing highs and last 2 swing lows on price and macd_line
    - if price HH but MACD LH -> bearish divergence
    - if price LL but MACD HL -> bullish divergence
    Else report convergence (or not enough data).
    Also include simple slope agreement over last N.
    """
    note = "n/a"
    if len(closes) < 30 or len(macd_line) < 30:
        return "insufficient"

    N = 20
    p_seg = closes[-N:]
    m_seg = macd_line[-N:]

    # slope sign
    p_slope = p_seg[-1] - p_seg[0]
    m_slope = m_seg[-1] - m_seg[0]
    slope_note = f"slope price={'↑' if p_slope>0 else '↓' if p_slope<0 else '→'}, macd={'↑' if m_slope>0 else '↓' if m_slope<0 else '→'}"

    # swing points based on short lookback
    look = 3
    ph = _swing_highs(closes, look)
    mh = _swing_highs(macd_line, look)
    pl = _swing_lows(closes, look)
    ml = _swing_lows(macd_line, look)

    bear_div = bull_div = False
    if len(ph) >= 2 and len(mh) >= 2:
        # last two highs
        p1, p2 = closes[ph[-2]], closes[ph[-1]]
        m1, m2 = macd_line[mh[-2]], macd_line[mh[-1]]
        if p2 > p1 and m2 <= m1:
            bear_div = True
    if len(pl) >= 2 and len(ml) >= 2:
        # last two lows
        p1, p2 = closes[pl[-2]], closes[pl[-1]]
        m1, m2 = macd_line[ml[-2]], macd_line[ml[-1]]
        if p2 < p1 and m2 >= m1:
            bull_div = True

    if bear_div:
        note = f"Bearish divergence ({slope_note})"
    elif bull_div:
        note = f"Bullish divergence ({slope_note})"
    else:
        # slopes agree?
        agree = (p_slope >= 0 and m_slope >= 0) or (p_slope <= 0 and m_slope <= 0)
        note = f"Convergence ({'agree' if agree else 'mixed'}; {slope_note})"
    return note

# -------------------- Strategy core --------------------
def strategy_scan():
    prices = fetch_all_prices()
    now_ts = int(time.time())

    # build ticks and candles
    for pair in PAIRS:
        if pair in prices:
            px = prices[pair]["price"]
            tick_logs[pair].append((now_ts, px))
            if len(tick_logs[pair]) > 6000:
                tick_logs[pair] = tick_logs[pair][-6000:]
            aggregate_candles(pair, CANDLE_INTERVAL)

    balances = get_wallet_balances()
    usdt_free = balances.get("USDT", 0.0)

    # per pair decision
    for pair in PAIRS:
        cs = candle_logs.get(pair) or []
        if len(cs) < max(EMA_SLOW, MACD_SLOW + MACD_SIGNAL) + 3:
            continue

        closes = [c["close"] for c in cs]
        last = closes[-1]

        # EMAs
        ema5_series = _ema_series(closes, EMA_FAST)
        ema10_series = _ema_series(closes, EMA_SLOW)
        e5_prev, e5_now = (ema5_series[-2], ema5_series[-1]) if len(ema5_series) >= 2 else (None, None)
        e10_prev, e10_now = (ema10_series[-2], ema10_series[-1]) if len(ema10_series) >= 2 else (None, None)

        # MACD
        macd_line, macd_signal, macd_hist = _macd_series(closes)
        if not macd_line or not macd_signal:
            continue
        m_prev, m_now = (macd_line[-2], macd_line[-1])
        s_prev, s_now = (macd_signal[-2], macd_signal[-1])

        macd_state = "bullish" if m_now > s_now else "bearish" if m_now < s_now else "flat"
        macd_xup   = _cross_up(m_prev, m_now, s_prev, s_now)
        macd_xdn   = _cross_dn(m_prev, m_now, s_prev, s_now)

        # MACD convergence/divergence with price
        # (align macd_line length with closes tail for check)
        m_for_cd = [None]*(len(closes)-len(macd_line)) + macd_line
        m_for_cd = [x for x in m_for_cd if x is not None]
        cd_note  = _macd_convergence_divergence(closes, m_for_cd)

        # gates
        buy_gate  = (e5_now is not None and e10_now is not None and e5_now > e10_now) and (m_now > s_now)
        sell_gate = (e5_now is not None and e10_now is not None and e5_now < e10_now) and (m_now < s_now)

        # log the full signal status
        scan_log.append(
            f"{ist_now()} | {pair} | "
            f"EMA5={round(e5_now,6)} EMA10={round(e10_now,6)} | "
            f"MACD={round(m_now,6)} SIG={round(s_now,6)} "
            f"(state={macd_state}, xup={macd_xup}, xdn={macd_xdn}) | "
            f"CD={cd_note} | Gates: Buy={buy_gate} Sell={sell_gate} | Px={last}"
        )

        # ---- Entry (BUY) ----
        have_pos = pair in positions and positions[pair].get("qty", 0.0) > 0.0
        if (not have_pos) and buy_gate:
            notional = BUY_FRACTION_USDT * usdt_free
            if notional >= 1.0:  # simple tiny floor
                qty = normalize_qty_for_buy(pair, last, notional)
                if qty > 0:
                    res = place_market(pair, "BUY", qty)
                    oid = _extract_order_id(res)
                    if oid:
                        st = get_order_status(order_id=oid)
                        filled, avg_px = _filled_avg_from_status(st)
                        if filled > 0 and avg_px > 0:
                            positions[pair] = {"qty": filled, "entry": avg_px, "ts": now_ts}
                            trade_log.append(f"{ist_now()} | {pair} | BUY {filled} @ {avg_px} | oid={oid}")
                            scan_log.append(f"{ist_now()} | {pair} | BUY filled qty={filled} @ {avg_px}")
                        else:
                            scan_log.append(f"{ist_now()} | {pair} | BUY no fill | res={res}")
                    else:
                        scan_log.append(f"{ist_now()} | {pair} | BUY failed | res={res}")
                else:
                    scan_log.append(f"{ist_now()} | {pair} | BUY qty=0 (precision/min-qty/fee)")
            continue  # after a buy attempt, skip sell checks this tick

        # ---- Exit (SELL) ----
        if have_pos:
            entry = positions[pair]["entry"]
            qty   = positions[pair]["qty"]
            take_profit = (last >= entry * (1.0 + TP_TARGET_PCT))
            sell_signal = sell_gate

            # log the exit conditions check
            scan_log.append(
                f"{ist_now()} | {pair} | EXIT-CHECK entry={entry} last={last} "
                f"TP@{round(100*TP_TARGET_PCT,2)}% hit={take_profit} | SellSignal={sell_signal}"
            )

            if take_profit or sell_signal:
                sell_qty = qty  # full position (and we assume wallet has it)
                res = place_market(pair, "SELL", sell_qty)
                oid = _extract_order_id(res)
                if oid:
                    st = get_order_status(order_id=oid)
                    filled, avg_px = _filled_avg_from_status(st)
                    if filled > 0 and avg_px > 0:
                        pnl = (avg_px - entry) * min(filled, qty)
                        record_realized_pnl(pnl)
                        trade_log.append(f"{ist_now()} | {pair} | SELL {filled} @ {avg_px} | PNL={round(pnl,6)} | oid={oid}")
                        scan_log.append(f"{ist_now()} | {pair} | SELL filled qty={filled} @ {avg_px} | PNL={round(pnl,6)}")
                        positions.pop(pair, None)
                    else:
                        scan_log.append(f"{ist_now()} | {pair} | SELL no fill | res={res}")
                else:
                    scan_log.append(f"{ist_now()} | {pair} | SELL failed | res={res}")

# -------------------- Main loop --------------------
_autostart_lock = threading.Lock()

def scan_loop():
    global running, status_epoch, last_keepalive, _last_rules_refresh
    scan_log.clear()
    running = True

    while running:
        now_real = time.time()

        # keepalive
        if now_real - last_keepalive >= KEEPALIVE_SEC:
            _keepalive_ping()
            last_keepalive = now_real

        # refresh pair rules periodically
        if (time.time() - _last_rules_refresh) >= RULES_REFRESH_SEC:
            fetch_pair_precisions()
            _last_rules_refresh = time.time()

        try:
            strategy_scan()
        except Exception as e:
            msg = f"{ist_now()} | scan_loop error: {e}"
            scan_log.append(msg)
            scan_log.append(traceback.format_exc().splitlines()[-1])

        status["msg"], status["last"] = "Running", ist_now()
        status_epoch = int(time.time())
        time.sleep(POLL_SEC)

    status["msg"] = "Idle"

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
    return jsonify({"status": "stopped"})

@app.route("/status")
def get_status():
    balances = get_wallet_balances()
    usdt_total = balances.get("USDT", 0.0)

    # current positions snapshot
    pos = {}
    for p, v in positions.items():
        pos[p] = {"qty": v["qty"], "entry": v["entry"], "ts": v["ts"]}

    # keepalive status
    now_real = time.time()
    last_age = now_real - (last_keepalive or 0)
    keepalive_info = {
        "enabled": bool(KEEPALIVE_TOKEN),
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
        "positions": pos,
        "profit_today": compute_realized_pnl_today(),
        "pnl_cumulative": round(profit_state.get("cumulative_pnl", 0.0), 6),
        "trades": trade_log[-12:][::-1],
        "scans": scan_log[-120:][::-1],
        "keepalive": keepalive_info
    })

@app.route("/ping", methods=["GET", "HEAD"])
def ping():
    token = os.environ.get("KEEPALIVE_TOKEN", "")
    provided = (request.args.get("t") or
                request.headers.get("X-Keepalive-Token") or "")
    if token and provided != token:
        print(f"[{ist_now()}] /ping forbidden (bad token) method={request.method}")
        return "forbidden", 403

    print(f"[{ist_now()}] /ping ok method={request.method}")
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
        time.sleep(0.5)
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
