import os
import time
import threading
import requests
from collections import deque
from flask import Flask, render_template_string, jsonify

PAIRS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "SHIBUSDT", "SOLUSDT",
    "DOGEUSDT", "ADAUSDT", "MATICUSDT", "BNBUSDT", "LTCUSDT"
]

# Buffer size for rolling price log (number of ticks to keep)
BUFFER_SIZE = 100

app = Flask(__name__)
price_buffers = {pair: deque(maxlen=BUFFER_SIZE) for pair in PAIRS}
signal_logs = {pair: deque(maxlen=20) for pair in PAIRS}
trade_log = []
status = {"running": False, "msg": "Idle"}
scan_interval = 5  # seconds

def fetch_price(pair):
    url = f"https://public.coindcx.com/market_data/current_price?pair={pair}"
    try:
        r = requests.get(url, timeout=5)
        if r.ok:
            return float(r.json()["last_price"])
    except Exception:
        return None

def calc_ema(buffer, n):
    if len(buffer) < n:
        return None
    prices = list(buffer)[-n:]
    ema = sum(prices) / n
    alpha = 2 / (n + 1)
    for price in prices[-n+1:]:
        ema = alpha * price + (1 - alpha) * ema
    return ema

def log_signal(pair, msg):
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    signal_logs[pair].appendleft(f"{now} | {msg}")

def scan_loop():
    last_cross = {pair: None for pair in PAIRS}
    status["msg"] = "Running"
    while status["running"]:
        for pair in PAIRS:
            price = fetch_price(pair)
            if price:
                price_buffers[pair].append(price)
                ema5 = calc_ema(price_buffers[pair], 5)
                ema10 = calc_ema(price_buffers[pair], 10)
                if ema5 and ema10:
                    prev_ema5 = calc_ema(list(price_buffers[pair])[:-1], 5)
                    prev_ema10 = calc_ema(list(price_buffers[pair])[:-1], 10)
                    # Simple crossover signal
                    if prev_ema5 and prev_ema10:
                        if prev_ema5 < prev_ema10 and ema5 > ema10:
                            log_signal(pair, f"BUY signal: EMA5 crossed above EMA10 | Price: {price}")
                            trade_log.append({"pair": pair, "side": "BUY", "price": price, "time": time.strftime('%Y-%m-%d %H:%M:%S')})
                        elif prev_ema5 > prev_ema10 and ema5 < ema10:
                            log_signal(pair, f"SELL signal: EMA5 crossed below EMA10 | Price: {price}")
                            trade_log.append({"pair": pair, "side": "SELL", "price": price, "time": time.strftime('%Y-%m-%d %H:%M:%S')})
                        else:
                            log_signal(pair, f"Scanned: Price {price} | EMA5 {ema5:.4f} | EMA10 {ema10:.4f}")
                    else:
                        log_signal(pair, f"Scanned: Price {price} | EMA5 {ema5:.4f} | EMA10 {ema10:.4f}")
                else:
                    log_signal(pair, f"Scanned: Price {price} | Not enough data for EMA")
            else:
                log_signal(pair, f"Failed to fetch price.")
        time.sleep(scan_interval)
    status["msg"] = "Stopped"

@app.route("/")
def index():
    return render_template_string("""
    <html>
    <head>
        <title>CoinDCX Live EMA(5,10) Scanner</title>
        <style>
            body { font-family: Arial; background: #f6fbfc; }
            .container { max-width: 900px; margin: 24px auto; padding: 24px; background: #fff; border-radius: 8px; }
            .btn { padding: 7px 20px; margin-right: 8px; border: none; border-radius: 4px; background: #008cff; color: #fff; font-weight: bold;}
            .logbox { background: #eef; font-family: monospace; font-size: 13px; margin-bottom: 20px; max-height: 150px; overflow: auto; border-radius: 4px; padding: 6px 8px;}
        </style>
    </head>
    <body>
    <div class="container">
        <h2>CoinDCX Live EMA(5,10) Spot Scanner</h2>
        <button class="btn" onclick="startScan()">START</button>
        <button class="btn" onclick="stopScan()">STOP</button>
        <span id="status" style="font-weight: bold; margin-left:24px"></span>
        <hr/>
        <h3>Live Price Logs and Signals</h3>
        <div id="logcontainer">
        </div>
        <h4>Trade Log</h4>
        <div class="logbox" id="tradebox"></div>
    </div>
    <script>
        function fetchLogs() {
            fetch('/logs').then(res => res.json()).then(data => {
                document.getElementById("status").innerText = "Status: " + data.status;
                let html = '';
                for (const pair of data.logs_order) {
                    html += `<b>${pair}</b><div class='logbox'>` + data.logs[pair].join('<br/>') + "</div>";
                }
                document.getElementById("logcontainer").innerHTML = html;
                let tlog = data.trade_log.map(x => `[${x.time}] ${x.pair} ${x.side} @ ${x.price}`).join('<br/>');
                document.getElementById("tradebox").innerHTML = tlog;
            });
        }
        function startScan() {
            fetch('/start', {method: "POST"}).then(_=>fetchLogs());
        }
        function stopScan() {
            fetch('/stop', {method: "POST"}).then(_=>fetchLogs());
        }
        setInterval(fetchLogs, 3000);
        fetchLogs();
    </script>
    </body>
    </html>
    """)

@app.route("/start", methods=["POST"])
def start():
    if not status["running"]:
        status["running"] = True
        thread = threading.Thread(target=scan_loop)
        thread.daemon = True
        thread.start()
    return jsonify({"ok": True})

@app.route("/stop", methods=["POST"])
def stop():
    status["running"] = False
    return jsonify({"ok": True})

@app.route("/logs")
def logs():
    logs = {pair: list(signal_logs[pair]) for pair in PAIRS}
    return jsonify({
        "status": status["msg"],
        "logs": logs,
        "logs_order": PAIRS,
        "trade_log": trade_log[-20:]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
