# ===================== TRADER ADD-ON (price log → signal → orders) =====================
import threading
from collections import deque

# ---- Tunables ----
TRADE_INTERVAL_SEC = 15 * 60      # candle interval built from tick log (15m)
POLL_EVERY_SEC      = 5           # how often to poll ticker
COOLDOWN_SEC        = 60          # cooldown after an exit per pair
USDT_PER_TRADE      = 30.0        # position size per trade (buy side)
TP_MULT             = 1.2         # TP = entry + MULT * risk_unit
SL_MULT             = 1.0         # SL = entry - MULT * risk_unit
MIN_TICK_RISK_FRAC  = 0.0015      # min risk per unit = entry * this

# Basic min-qty/precision guards (edit to your needs)
PAIR_RULES = {
    "BTCUSDT": {"precision": 6, "min_qty": 0.0001},
    "ETHUSDT": {"precision": 6, "min_qty": 0.001},
    "BNBUSDT": {"precision": 3, "min_qty": 0.01},
    "SOLUSDT": {"precision": 3, "min_qty": 0.01},
    "DOGEUSDT":{"precision": 0, "min_qty": 10},
}

# ---- State ----
_ticks         = {p: deque(maxlen=5000) for p in DEFAULT_PAIRS}  # (ts, price)
_candles       = {p: [] for p in DEFAULT_PAIRS}                  # rolling candles
_exit_orders   = []  # [{pair, side, qty, tp, sl, entry}]
_pair_cooldown = {p: 0 for p in DEFAULT_PAIRS}
_trade_log     = deque(maxlen=100)
_scan_log      = deque(maxlen=400)
_running       = False
_worker        = None

def _now(): return int(time.time())

def _psar_last(candles, step=0.02, mx=0.2):
    """Return last psar dict using existing engine."""
    ps = psar_series(candles, step=step, max_step=mx)
    return ps[-2:] if ps else []

def _aggregate_candles(pair: str, interval_sec: int):
    """Aggregate ticks → OHLCV candles in-place (keeps last ~200)."""
    ticks = list(_ticks[pair])
    if not ticks: return
    candles = []
    last_bucket = None
    cur = None
    for ts, px in ticks:
        bucket = ts - (ts % interval_sec)
        if bucket != last_bucket:
            if cur: candles.append(cur)
            cur = {"start": bucket, "open": px, "high": px, "low": px, "close": px, "volume": 1.0}
            last_bucket = bucket
        else:
            cur["high"] = max(cur["high"], px)
            cur["low"]  = min(cur["low"], px)
            cur["close"] = px
            cur["volume"] += 1.0
    if cur: candles.append(cur)
    _candles[pair] = candles[-200:]

def _round_qty(pair, qty):
    rule = PAIR_RULES.get(pair, {"precision": 6, "min_qty": 0.0001})
    q = max(qty, rule["min_qty"])
    prec = rule["precision"]
    return float(f"{q:.{prec}f}")

def _balances_map():
    """Return {'USDT': x, 'BTC': y, ...}"""
    try:
        bals = client.balances()
        out = {}
        for b in bals:
            try:
                out[b["currency"]] = float(b["balance"])
            except:
                pass
        return out
    except Exception:
        return {}

def _place_market(pair, side, qty):
    """Place market order; returns json."""
    payload = {
        "market": pair,
        "side": side.lower(),                 # "buy" or "sell"
        "order_type": "market_order",
        "total_quantity": str(qty),
        "timestamp": int(time.time() * 1000)
    }
    headers, body = client._sign(payload)
    try:
        r = requests.post(f"{BASE_URL}/exchange/v1/orders/create", headers=headers, data=body, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def _get_order_status(order_id=None, client_order_id=None):
    body = {"timestamp": int(time.time() * 1000)}
    if order_id: body["id"] = order_id
    if client_order_id: body["client_order_id"] = client_order_id
    headers, signed = client._sign(body)
    try:
        r = requests.post(f"{BASE_URL}/exchange/v1/orders/status", headers=headers, data=signed, timeout=10)
        return r.json()
    except Exception:
        return {}

def _append_scan(msg): _scan_log.append(f"[{_now()}] {msg}")
def _append_trade(msg): _trade_log.append(f"[{_now()}] {msg}")

def _monitor_exits(prices_by_pair):
    to_remove = []
    now = _now()
    for ex in list(_exit_orders):
        pair, side, qty, tp, sl = ex["pair"], ex["side"], ex["qty"], ex["tp"], ex["sl"]
        px = prices_by_pair.get(pair)
        if not px: continue
        if side == "BUY" and (px >= tp or px <= sl):
            res = _place_market(pair, "SELL", qty)
            _append_trade(f"EXIT SELL {pair} qty={qty} px={px} -> {res}")
            to_remove.append(ex)
            _pair_cooldown[pair] = now + COOLDOWN_SEC
        elif side == "SELL" and (px <= tp or px >= sl):
            res = _place_market(pair, "BUY", qty)
            _append_trade(f"EXIT BUY {pair} qty={qty} px={px} -> {res}")
            to_remove.append(ex)
            _pair_cooldown[pair] = now + COOLDOWN_SEC
    for ex in to_remove:
        try: _exit_orders.remove(ex)
        except ValueError: pass

def _signal_for_pair(pair, live_price):
    """PSAR flip signal on completed candles + live price risk sizing."""
    c = _candles[pair]
    if len(c) < 20: 
        return None
    # Use completed candles only for PSAR flip check
    completed = c[:-1] if len(c) >= 2 else c
    ps_last = _psar_last(completed)
    if len(ps_last) < 2:
        return None
    prev_bull = ps_last[-2]["bull"]
    last_bull = ps_last[-1]["bull"]
    if last_bull != prev_bull:
        # flip happened on last completed candle → act using live price
        side = "BUY" if last_bull else "SELL"
        entry = float(live_price)
        # risk unit: min between half of last candle range and min tick risk
        rng = completed[-1]["high"] - completed[-1]["low"]
        min_risk = entry * MIN_TICK_RISK_FRAC
        risk_unit = max(rng * 0.5, min_risk)
        if side == "BUY":
            sl = round(entry - SL_MULT * risk_unit, 6)
            tp = round(entry + TP_MULT * risk_unit, 6)
        else:
            sl = round(entry + SL_MULT * risk_unit, 6)
            tp = round(entry - TP_MULT * risk_unit, 6)
        return {"side": side, "entry": entry, "sl": sl, "tp": tp, "note": "PSAR flip"}
    return None

def _trade_loop(pairs, interval_sec=TRADE_INTERVAL_SEC):
    global _running
    last_poll = 0
    while _running:
        now = _now()
        # throttle polling
        if now - last_poll >= POLL_EVERY_SEC:
            last_poll = now
            # 1) fetch prices once
            prices = {}
            try:
                t = client.ticker()
                for row in t:
                    m = row.get("market")
                    if m in pairs:
                        try:
                            prices[m] = float(row["last_price"])
                            _ticks[m].append((now, prices[m]))
                        except: pass
            except Exception as e:
                _append_scan(f"ticker error: {e}")

            # 2) maintain candles
            for p in pairs:
                _aggregate_candles(p, interval_sec)

            # 3) monitor exits
            _monitor_exits(prices)

            # 4) check entries
            bals = _balances_map()
            usdt = float(bals.get("USDT", 0.0))
            for p in pairs:
                # cooldown or active exit?
                if now < _pair_cooldown.get(p, 0): 
                    _append_scan(f"{p} cooldown; skip")
                    continue
                if any(ex for ex in _exit_orders if ex["pair"] == p):
                    _append_scan(f"{p} has exit pending; skip")
                    continue

                live = prices.get(p)
                if not live: 
                    continue

                sig = _signal_for_pair(p, live)
                if not sig:
                    _append_scan(f"{p} no signal")
                    continue

                side, entry, sl, tp = sig["side"], sig["entry"], sig["sl"], sig["tp"]
                # qty sizing
                if side == "BUY":
                    if usdt < 5: 
                        _append_scan(f"{p} BUY signal but low USDT {usdt}")
                        continue
                    qty = USDT_PER_TRADE / entry
                else:
                    coin = p[:-4]
                    qty = float(bals.get(coin, 0.0))

                qty = _round_qty(p, qty)
                if qty <= 0: 
                    _append_scan(f"{p} {side} qty too small")
                    continue

                res = _place_market(p, side, qty)
                _append_trade(f"ENTRY {side} {p} qty={qty} @~{entry} -> {res}")

                # track exit order
                _exit_orders.append({"pair": p, "side": side, "qty": qty, "tp": tp, "sl": sl, "entry": entry})

        time.sleep(1)

# --------------------------- Endpoints ---------------------------
@app.route("/trade/start", methods=["POST"])
def trade_start():
    """Start the background trading loop.
       Optional json: {"pairs":"BTCUSDT,ETHUSDT","interval_sec":900}
    """
    global _running, _worker
    if _running:
        return jsonify({"ok": True, "msg": "already running"})
    data = request.get_json(silent=True) or {}
    pairs_csv = (data.get("pairs") or "")
    pairs = [p.strip().upper() for p in pairs_csv.split(",") if p.strip()] or DEFAULT_PAIRS
    interval_sec = int(data.get("interval_sec") or TRADE_INTERVAL_SEC)
    _running = True
    _worker = threading.Thread(target=_trade_loop, args=(pairs, interval_sec), daemon=True)
    _worker.start()
    return jsonify({"ok": True, "msg": "started", "pairs": pairs, "interval_sec": interval_sec})

@app.route("/trade/stop", methods=["POST"])
def trade_stop():
    global _running
    _running = False
    return jsonify({"ok": True, "msg": "stopping"})

@app.route("/trade/status")
def trade_status():
    def _last(lst, n): 
        return list(lst)[-n:] if isinstance(lst, deque) else lst[-n:]
    return jsonify({
        "running": _running,
        "pairs": list(_ticks.keys()),
        "open_exits": _exit_orders,
        "cooldowns": _pair_cooldown,
        "last_scans": _last(_scan_log, 40),
        "last_trades": _last(_trade_log, 20),
        "now": _now(),
        "interval_sec": TRADE_INTERVAL_SEC,
        "poll_sec": POLL_EVERY_SEC
    })
# =================== END TRADER ADD-ON ===================
