"""
EMA Crossover + EMA200 Trend Gate — Multi-Pair Live Trading Bot
===============================================================
Trades up to MAX_OPEN_POSITIONS pairs simultaneously.
Each pair runs its own independent signal, position, and risk tracking.
Pairs are taken from the WATCHLIST — set these to your screener top picks.

Strategy: matches backtest_ema_rsi.py exactly (1h candles, long only)
  Entry : EMA12 > EMA26 crossover AND close > EMA200 AND ADX>=20
          AND volume >= 1.2x avg AND RSI < 60
  Exit  : EMA crossover down OR trailing stop OR take-profit OR RSI > 70

Requirements: pip install python-binance pandas ta-lib requests
"""

import sys
import time
import threading
import logging
import requests
import pandas as pd
import talib
from binance.client import Client
from binance.exceptions import BinanceAPIException
from requests.exceptions import ReadTimeout, ConnectionError as ReqConnectionError
from datetime import datetime
import configparser

# ── CREDENTIALS ───────────────────────────────────────────────────────────────
# Initialize config
config = configparser.ConfigParser()
config.read('config.ini')
    
API_KEY = config['mainnet_spot'].get("API_KEY")
API_SECRET = config['mainnet_spot'].get("API_SECRET")
TESTNET    = True

# ── WATCHLIST — set from screener top picks ───────────────────────────────────
WATCHLIST = [
    "BTCUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "INJUSDT",
    "ATOMUSDT"
]
MAX_OPEN_POSITIONS = 3      # max simultaneous open trades across all pairs
                            # keeps total exposure manageable

# ── SIZING ────────────────────────────────────────────────────────────────────
TRADE_SIZE_PCT  = 0.10      # 10% of available USDT balance per trade
MIN_TRADE_USDT  = 10.0
MAX_TRADE_USDT  = 200.0

# ── STRATEGY PARAMS — must match backtest_ema_rsi.py ─────────────────────────
EMA_FAST          = 12
EMA_SLOW          = 26
EMA_TREND         = 90
RSI_PERIOD        = 14
RSI_BUY_MAX       = 60
RSI_SELL_MIN      = 70
ADX_MIN_TREND     = 20
STOP_LOSS_PCT     = 0.020
TAKE_PROFIT_PCT   = 0.045
TRAILING_STOP_PCT = 0.020
MIN_HOLD_CANDLES  = 6
COOLDOWN_CANDLES  = 4
VOLUME_FILTER     = True
VOLUME_MA_PERIOD  = 20
VOLUME_MIN_MULT   = 1.2
CANDLE_INTERVAL   = Client.KLINE_INTERVAL_1HOUR
LOOP_SLEEP_SEC    = 60

# ── COMMISSION ────────────────────────────────────────────────────────────────
PAY_FEES_IN_BNB = False
COMMISSION_PCT  = 0.00075 if PAY_FEES_IN_BNB else 0.001

# ── RISK ──────────────────────────────────────────────────────────────────────
MAX_DAILY_LOSS_USDT = 30.0  # halt all pairs if total daily loss exceeds this

# ── RETRY ─────────────────────────────────────────────────────────────────────
REQUEST_TIMEOUT  = 15
MAX_RETRIES      = 4
RETRY_BASE_SLEEP = 5

# ── TELEGRAM ──────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = "8691926463:AAFqTIz2XCgFC4dQDgwWwV3VXbt_wMqb6TU"        # e.g. "7123456789:AAFxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
TELEGRAM_CHAT_ID = "1026673157" 
TELEGRAM_ENABLED = bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)

# ── LOGGING — UTF-8 forced to avoid Windows cp1252 errors ────────────────────
_stream_handler = logging.StreamHandler(stream=sys.stdout)
_stream_handler.setFormatter(logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
))
try:
    _stream_handler.stream.reconfigure(encoding="utf-8")
except AttributeError:
    pass

_file_handler = logging.FileHandler("bot.log", encoding="utf-8")
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
))
logging.basicConfig(level=logging.INFO, handlers=[_stream_handler, _file_handler])
log = logging.getLogger("bot")

# ── PER-PAIR STATE ────────────────────────────────────────────────────────────
class PairState:
    """Tracks position and metrics for one trading pair independently."""
    def __init__(self, symbol: str):
        self.symbol          = symbol
        self.in_position     = False
        self.entry_price     = 0.0
        self.qty             = 0.0
        self.peak_price      = 0.0
        self.candles_held    = 0
        self.candles_cooldown= COOLDOWN_CANDLES  # start ready
        self.total_pnl       = 0.0
        self.total_trades    = 0
        self.winning_trades  = 0
        self.total_commission= 0.0

    def summary_line(self) -> str:
        wr = (self.winning_trades / self.total_trades * 100) if self.total_trades else 0
        return (f"{self.symbol:<12} trades={self.total_trades}  "
                f"wr={wr:.0f}%  net={self.total_pnl:+.4f} USDT  "
                f"comm={self.total_commission:.4f} USDT  "
                f"{'[IN POSITION]' if self.in_position else '[FLAT]'}")

# ── GLOBAL SESSION STATE ──────────────────────────────────────────────────────
class SessionState:
    def __init__(self):
        self.pairs          = {s: PairState(s) for s in WATCHLIST}
        self.daily_pnl      = 0.0
        self.session_start  = datetime.utcnow().date()
        self.bot_start_time = datetime.utcnow()

    def reset_daily_if_needed(self):
        today = datetime.utcnow().date()
        if today != self.session_start:
            log.info(f"New day — resetting daily P&L (was {self.daily_pnl:.2f} USDT)")
            self.daily_pnl     = 0.0
            self.session_start = today

    def open_positions(self) -> list:
        return [p for p in self.pairs.values() if p.in_position]

    def total_pnl(self) -> float:
        return sum(p.total_pnl for p in self.pairs.values())

    def total_trades(self) -> int:
        return sum(p.total_trades for p in self.pairs.values())

    def total_commission(self) -> float:
        return sum(p.total_commission for p in self.pairs.values())

    def summary(self) -> str:
        runtime  = datetime.utcnow() - self.bot_start_time
        h, rem   = divmod(int(runtime.total_seconds()), 3600)
        m        = rem // 60
        total_t  = self.total_trades()
        total_w  = sum(p.winning_trades for p in self.pairs.values())
        wr       = (total_w / total_t * 100) if total_t else 0
        lines    = [
            "", "=" * 60, "  SESSION SUMMARY", "=" * 60,
            f"  Runtime         : {h}h {m}m",
            f"  Total trades    : {total_t}  (win rate: {wr:.1f}%)",
            f"  Net P&L         : {self.total_pnl():+.4f} USDT",
            f"  Commission paid : {self.total_commission():.4f} USDT",
            f"  Daily P&L today : {self.daily_pnl:+.4f} USDT",
            "", "  Per-pair breakdown:", "  " + "-"*56,
        ]
        for p in self.pairs.values():
            lines.append(f"    {p.summary_line()}")
        lines.append("=" * 60)
        return "\n".join(lines)

session = SessionState()

# ── TELEGRAM ──────────────────────────────────────────────────────────────────
def notify(message: str, silent: bool = False):
    if not TELEGRAM_ENABLED:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": message,
                  "parse_mode": "Markdown", "disable_notification": silent},
            timeout=5,
        )
    except Exception as e:
        log.warning(f"Telegram failed: {e}")

# ── CLIENT & RETRY ────────────────────────────────────────────────────────────
def make_client() -> Client:
    return Client(API_KEY, API_SECRET, testnet=TESTNET,
                  requests_params={"timeout": REQUEST_TIMEOUT})

def api_call_with_retry(fn, *args, label="call", **kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except (ReadTimeout, ReqConnectionError) as e:
            wait = RETRY_BASE_SLEEP * (2 ** (attempt - 1))
            if attempt < MAX_RETRIES:
                log.warning(f"{label} timeout (attempt {attempt}/{MAX_RETRIES}) — retry in {wait}s")
                time.sleep(wait)
            else:
                log.error(f"{label} failed after {MAX_RETRIES} attempts")
                raise

# ── DATA ──────────────────────────────────────────────────────────────────────
def fetch_candles(client: Client, symbol: str, n: int = 250) -> pd.DataFrame:
    raw = api_call_with_retry(
        client.get_klines, label=f"fetch_{symbol}",
        symbol=symbol, interval=CANDLE_INTERVAL, limit=n + 1,
    )
    df = pd.DataFrame(raw, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","trades",
        "taker_buy_base","taker_buy_quote","ignore",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    return df.iloc[:-1]   # drop the still-open candle

def compute_signal(df: pd.DataFrame) -> tuple[int, bool]:
    """Returns (signal, in_uptrend). signal: 1=buy, -1=sell, 0=hold."""
    df  = df.copy()
    c, h, l = df["close"].values, df["high"].values, df["low"].values
    df[f"ema_{EMA_FAST}"]  = talib.EMA(c, timeperiod=EMA_FAST)
    df[f"ema_{EMA_SLOW}"]  = talib.EMA(c, timeperiod=EMA_SLOW)
    df[f"ema_{EMA_TREND}"] = talib.EMA(c, timeperiod=EMA_TREND)
    df["rsi"]              = talib.RSI(c, timeperiod=RSI_PERIOD)
    df["adx"]              = talib.ADX(h, l, c, timeperiod=14)
    df["vol_ma"]           = talib.SMA(df["volume"].values, timeperiod=VOLUME_MA_PERIOD)
    df.dropna(inplace=True)
    if len(df) < 2:
        return 0, False
    last, prev = df.iloc[-1], df.iloc[-2]
    fast, slow = f"ema_{EMA_FAST}", f"ema_{EMA_SLOW}"
    cross_up   = last[fast] > last[slow] and prev[fast] <= prev[slow]
    cross_down = last[fast] < last[slow] and prev[fast] >= prev[slow]
    uptrend    = last["close"] > last[f"ema_{EMA_TREND}"]
    trending   = last["adx"] >= ADX_MIN_TREND
    high_vol   = last["volume"] >= last["vol_ma"] * VOLUME_MIN_MULT if VOLUME_FILTER else True
    if cross_up and uptrend and trending and high_vol and last["rsi"] < RSI_BUY_MAX:
        return 1, uptrend
    if cross_down or last["rsi"] > RSI_SELL_MIN:
        return -1, uptrend
    return 0, uptrend

# ── ORDER HELPERS ─────────────────────────────────────────────────────────────
def get_price(client: Client, symbol: str) -> float:
    result = api_call_with_retry(
        client.get_symbol_ticker, label=f"price_{symbol}", symbol=symbol
    )
    return float(result["price"])

def get_usdt_balance(client: Client) -> float:
    account = api_call_with_retry(client.get_account, label="balance")
    for asset in account["balances"]:
        if asset["asset"] == "USDT":
            return float(asset["free"])
    return 0.0

def get_qty_precision(client: Client, symbol: str) -> int:
    info = client.get_symbol_info(symbol)
    for f in info["filters"]:
        if f["filterType"] == "LOT_SIZE":
            step = f["stepSize"]
            return len(step.rstrip("0").split(".")[-1]) if "." in step else 0
    return 5

def calc_trade_size(balance: float) -> float:
    return max(MIN_TRADE_USDT, min(MAX_TRADE_USDT, balance * TRADE_SIZE_PCT))

def place_buy(client: Client, pair: PairState) -> bool:
    balance    = get_usdt_balance(client)
    trade_usdt = calc_trade_size(balance)
    if balance < MIN_TRADE_USDT:
        log.warning(f"{pair.symbol} balance too low ({balance:.2f} USDT) — skip buy")
        return False
    price = get_price(client, pair.symbol)
    prec  = get_qty_precision(client, pair.symbol)
    qty   = round(trade_usdt / price, prec)
    try:
        order          = client.order_market_buy(symbol=pair.symbol, quantity=qty)
        commission     = trade_usdt * COMMISSION_PCT
        pair.total_commission += commission
        pair.in_position   = True
        pair.entry_price   = price
        pair.qty           = qty
        pair.peak_price    = price
        pair.candles_held  = 0
        pair.candles_cooldown = 0
        log.info(f"[{pair.symbol}] BUY  qty={qty}  price={price:,.2f}  "
                 f"size={trade_usdt:.2f} USDT  comm={commission:.4f}  "
                 f"order={order['orderId']}")
        notify(
            f"*BUY* {pair.symbol}\n"
            f"Price: `${price:,.2f}`  Qty: `{qty}`\n"
            f"TP: `${price*(1+TAKE_PROFIT_PCT):,.2f}`  "
            f"SL: `${price*(1-STOP_LOSS_PCT):,.2f}`"
        )
        return True
    except BinanceAPIException as e:
        log.error(f"[{pair.symbol}] Buy failed: {e}")
        return False

def place_sell(client: Client, pair: PairState, reason: str) -> bool:
    try:
        order       = client.order_market_sell(symbol=pair.symbol, quantity=pair.qty)
        exit_price  = get_price(client, pair.symbol)
        gross_pnl   = (exit_price - pair.entry_price) * pair.qty
        commission  = exit_price * pair.qty * COMMISSION_PCT
        net_pnl     = gross_pnl - commission
        pair.total_commission  += commission
        pair.total_pnl         += net_pnl
        pair.total_trades      += 1
        if net_pnl > 0:
            pair.winning_trades += 1
        session.daily_pnl      += net_pnl
        pair.in_position        = False
        pair.candles_cooldown   = 0
        log.info(f"[{pair.symbol}] SELL qty={pair.qty}  price={exit_price:,.2f}  "
                 f"gross={gross_pnl:+.4f}  comm={commission:.4f}  "
                 f"net={net_pnl:+.4f}  reason={reason}  order={order['orderId']}")
        notify(
            f"*SELL* {pair.symbol}  _{reason}_\n"
            f"Price: `${exit_price:,.2f}`\n"
            f"*Net P&L: `${net_pnl:+.4f} USDT`*\n"
            f"Session P&L [{pair.symbol}]: `${pair.total_pnl:+.4f}`"
        )
        return True
    except BinanceAPIException as e:
        log.error(f"[{pair.symbol}] Sell failed: {e}")
        return False

# ── RISK GUARD (per pair) ─────────────────────────────────────────────────────
def check_stop_take(client: Client, pair: PairState) -> bool:
    if not pair.in_position:
        return False
    price          = get_price(client, pair.symbol)
    pair.peak_price= max(pair.peak_price, price)
    change         = (price - pair.entry_price) / pair.entry_price
    trail_drop     = (pair.peak_price - price) / pair.peak_price if pair.peak_price > 0 else 0
    if change <= -STOP_LOSS_PCT:
        log.warning(f"[{pair.symbol}] Stop-loss hit @ {price:,.2f} ({change*100:+.2f}%)")
        notify(f"*STOP-LOSS* {pair.symbol} @ `${price:,.2f}` ({change*100:+.2f}%)")
        return place_sell(client, pair, "STOP-LOSS")
    if trail_drop >= TRAILING_STOP_PCT and price > pair.entry_price:
        log.info(f"[{pair.symbol}] Trailing stop @ {price:,.2f} "
                 f"({trail_drop*100:.2f}% below peak)")
        return place_sell(client, pair, "TRAILING-STOP")
    if change >= TAKE_PROFIT_PCT:
        log.info(f"[{pair.symbol}] Take-profit @ {price:,.2f} ({change*100:+.2f}%)")
        return place_sell(client, pair, "TAKE-PROFIT")
    return False

# ── GRACEFUL SHUTDOWN ─────────────────────────────────────────────────────────
def graceful_shutdown(client: Client):
    print("\n")
    log.info("Shutdown requested (Ctrl+C)")
    open_pairs = session.open_positions()

    if open_pairs:
        print(f"\n  {len(open_pairs)} OPEN POSITION(S) DETECTED:\n")
        for pair in open_pairs:
            try:
                price      = get_price(client, pair.symbol)
                unrealised = (price - pair.entry_price) * pair.qty
                pct        = (price - pair.entry_price) / pair.entry_price * 100
                print(f"  {pair.symbol:<12} entry={pair.entry_price:,.2f}  "
                      f"now={price:,.2f} ({pct:+.2f}%)  "
                      f"unrealised={unrealised:+.4f} USDT")
            except Exception:
                print(f"  {pair.symbol:<12} (could not fetch price)")

        print("\n  Close ALL open positions before exiting? [Y/n] (auto-closes in 10s): ",
              end="", flush=True)

        answer_holder = [None]
        def read_input():
            try:
                answer_holder[0] = sys.stdin.readline().strip().lower()
            except Exception:
                answer_holder[0] = ""
        t = threading.Thread(target=read_input, daemon=True)
        t.start()
        t.join(timeout=10)
        answer = answer_holder[0] if answer_holder[0] is not None else ""
        if answer_holder[0] is None:
            print("(timed out — closing all positions)")

        if answer in ("", "y", "yes"):
            for pair in open_pairs:
                log.info(f"[{pair.symbol}] Closing position before shutdown...")
                try:
                    place_sell(client, pair, "SHUTDOWN")
                except Exception as e:
                    log.error(f"[{pair.symbol}] Shutdown sell failed: {e}")
                    notify(f"*WARNING* Could not close {pair.symbol} — close manually!")
        else:
            for pair in open_pairs:
                log.warning(f"[{pair.symbol}] Exiting WITH open position — close manually!")
                notify(f"*WARNING* {pair.symbol} position still open — close manually!")
    else:
        log.info("No open positions — clean shutdown.")

    summary = session.summary()
    print(summary)
    log.info("Bot shut down.")
    notify(
        f"*Bot stopped*\n"
        f"Total trades: `{session.total_trades()}`\n"
        f"Net P&L: `${session.total_pnl():+.4f} USDT`\n"
        f"Commission: `${session.total_commission():.4f} USDT`"
    )

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
def run(client: Client):
    log.info(f"Bot started — watching {len(WATCHLIST)} pairs: {WATCHLIST}")
    log.info(f"Max simultaneous positions: {MAX_OPEN_POSITIONS}")
    log.info(f"Strategy: EMA{EMA_FAST}/{EMA_SLOW}/{EMA_TREND} | "
             f"SL {STOP_LOSS_PCT*100:.1f}% | TP {TAKE_PROFIT_PCT*100:.1f}% | "
             f"TSL {TRAILING_STOP_PCT*100:.1f}%")
    notify(
        f"*Bot started* — {len(WATCHLIST)} pairs\n"
        f"Pairs: {', '.join(WATCHLIST)}\n"
        f"Max positions: {MAX_OPEN_POSITIONS} | "
        f"SL {STOP_LOSS_PCT*100:.1f}% | TP {TAKE_PROFIT_PCT*100:.1f}%"
    )

    while True:
        try:
            session.reset_daily_if_needed()

            # Daily loss circuit breaker — halt all trading
            if session.daily_pnl <= -MAX_DAILY_LOSS_USDT:
                log.warning(f"Daily loss limit hit ({session.daily_pnl:.2f} USDT) — "
                            "pausing all pairs until tomorrow")
                notify(f"*Daily loss limit hit* `{session.daily_pnl:.2f} USDT` — paused")
                time.sleep(3600)
                continue

            open_count = len(session.open_positions())

            for symbol, pair in session.pairs.items():
                try:
                    # Always check SL/TSL/TP for open positions
                    if check_stop_take(client, pair):
                        continue

                    # Fetch candles and compute signal
                    df             = fetch_candles(client, symbol)
                    signal, uptrend= compute_signal(df)
                    price          = float(df["close"].iloc[-1])

                    # Increment candles_held and cooldown
                    if pair.in_position:
                        pair.candles_held += 1
                    else:
                        if pair.candles_cooldown < COOLDOWN_CANDLES:
                            pair.candles_cooldown += 1

                    trend_str = "UP" if uptrend else "DOWN"
                    log.info(f"[{symbol}] price={price:,.2f}  signal={signal:+d}  "
                             f"trend={trend_str}  "
                             f"{'pos=YES held=' + str(pair.candles_held) if pair.in_position else 'pos=NO cool=' + str(pair.candles_cooldown)}")

                    # Entry — only if under position limit and cooldown elapsed
                    can_enter = (not pair.in_position
                                 and open_count < MAX_OPEN_POSITIONS
                                 and pair.candles_cooldown >= COOLDOWN_CANDLES
                                 and capital_ok(client))

                    if signal == 1 and can_enter:
                        if place_buy(client, pair):
                            open_count += 1

                    # Exit — only after minimum hold
                    elif (signal == -1
                          and pair.in_position
                          and pair.candles_held >= MIN_HOLD_CANDLES):
                        if place_sell(client, pair, "SIGNAL"):
                            open_count -= 1

                    time.sleep(1)   # small gap between pairs to avoid rate limits

                except (ReadTimeout, ReqConnectionError) as e:
                    log.error(f"[{symbol}] Network error — skipping this loop: {e}")
                except BinanceAPIException as e:
                    log.error(f"[{symbol}] API error: {e}")
                except Exception as e:
                    log.exception(f"[{symbol}] Unexpected error: {e}")

        except Exception as e:
            log.exception(f"Outer loop error: {e}")

        time.sleep(LOOP_SLEEP_SEC)

def capital_ok(client: Client) -> bool:
    """Check we have at least MIN_TRADE_USDT available."""
    try:
        return get_usdt_balance(client) >= MIN_TRADE_USDT
    except Exception:
        return False

# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _client = None
    try:
        _client = make_client()
        info    = _client.get_system_status()
        log.info(f"Connected to Binance {'TESTNET' if TESTNET else 'LIVE'} — {info}")
        run(_client)
    except KeyboardInterrupt:
        graceful_shutdown(_client if _client else make_client())