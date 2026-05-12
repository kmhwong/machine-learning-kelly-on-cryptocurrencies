"""
kelly_live.py  —  Kelly Bot Live / Paper Trading
=================================================
Loads a trained model saved by kelly_bot_ml.py and executes real or paper
trades on Binance at every bar close.

Architecture
------------
  kelly_bot_ml.py  ->  trains, validates, saves model_BTCUSDT_1h.pkl
  kelly_live.py    ->  loads model, polls bar closes, executes orders

Modes
-----
  --mode paper   : full simulation with real market data, NO real orders
                   Tracks a virtual wallet, logs what it WOULD have done.
                   Run for 2-4 weeks before switching to live.

  --mode live    : real orders on Binance (requires API key with trade perms)
                   Start with small capital. Never risk more than you can lose.

Safety features
---------------
  - Drawdown circuit breaker (default 10%): halts trading if portfolio
    drops more than X% from its peak. Requires manual restart to resume.
  - Max order cap: single order never exceeds max_order_usdt (default $500).
  - Duplicate guard: skips the bar if the last trade was in the same bar.
  - Dry-run check: prints the order it would place before actually placing it.
  - State persistence: saves wallet state to disk every bar so restarts
    don't lose position tracking.
  - All activity logged to kelly_live.log with timestamps.

Usage
-----
  # Paper trade (safe, no real orders)
  python kelly_live.py --model model_BTCUSDT_1h.pkl --mode paper

  # Live trade (real money)
  python kelly_live.py --model model_BTCUSDT_1h.pkl --mode live \\
      --api-key YOUR_KEY --api-secret YOUR_SECRET

  # Testnet (Binance sandbox, fake money but real API calls)
  python kelly_live.py --model model_BTCUSDT_1h.pkl --mode live \\
      --api-key TESTNET_KEY --api-secret TESTNET_SECRET --testnet

  # Override max drawdown circuit breaker
  python kelly_live.py --model model_BTCUSDT_1h.pkl --mode paper \\
      --max-drawdown 0.15

Requirements
------------
  pip install requests joblib numpy pandas scipy scikit-learn xgboost schedule
"""

import argparse
import hashlib
import hmac
import json
import logging
import os
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import schedule
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

# ---------------------------------------------------------------------------
# Logging — both console and file
# ---------------------------------------------------------------------------
# Force UTF-8 on all handlers so Unicode chars (-> ✓ ⚠) work on Windows
import io
_file_handler   = logging.FileHandler('kelly_live.log', encoding='utf-8')
_stream_handler = logging.StreamHandler(
    io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stdout, 'buffer') else sys.stdout
)
_fmt = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
_file_handler.setFormatter(_fmt)
_stream_handler.setFormatter(_fmt)
logging.basicConfig(level=logging.INFO, handlers=[_file_handler, _stream_handler])
log = logging.getLogger(__name__)


# ===========================================================================
# TELEGRAM NOTIFIER
# ===========================================================================

class TelegramNotifier:
    """
    Sends messages to a Telegram chat via Bot API.

    Setup (one-time, takes 2 minutes):
      1. Open Telegram, search for @BotFather
      2. Send /newbot, follow prompts -> you get a BOT_TOKEN
      3. Start a chat with your new bot (send it any message)
      4. Visit https://api.telegram.org/bot<BOT_TOKEN>/getUpdates
         Find "chat":{"id": XXXXXXX} — that is your CHAT_ID
      5. Pass --tg-token and --tg-chat-id when running kelly_live.py

    Message levels:
      notify()  — every bar summary (trade or no trade)
      alert()   — important events: trades, circuit breaker, errors
                  Sends regardless of --tg-quiet setting
    """

    BASE_URL = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self, token: str, chat_id: str, quiet: bool = False):
        self.token   = token
        self.chat_id = chat_id
        self.quiet   = quiet          # if True, only send alerts not bar summaries
        self.enabled = bool(token and chat_id)

        if self.enabled:
            log.info(f"Telegram notifications enabled (chat_id={chat_id}  quiet={quiet})")
        else:
            log.info("Telegram notifications disabled (no --tg-token / --tg-chat-id)")

    def _send(self, text: str) -> bool:
        """Send a message. Returns True on success, False on failure."""
        if not self.enabled:
            return False
        try:
            r = requests.post(
                self.BASE_URL.format(token=self.token),
                data={
                    'chat_id':    self.chat_id,
                    'text':       text,
                    'parse_mode': 'HTML',
                },
                timeout=10,
            )
            if not r.ok:
                log.warning(f"Telegram send failed: {r.status_code} {r.text[:100]}")
                return False
            return True
        except Exception as e:
            log.warning(f"Telegram error: {e}")
            return False

    def alert(self, text: str) -> bool:
        """Always send — used for trades, errors, circuit breaker."""
        return self._send(text)

    def notify(self, text: str) -> bool:
        """Send only if not in quiet mode — used for routine bar summaries."""
        if self.quiet:
            return False
        return self._send(text)

    def send_startup(self, symbol: str, interval: str, mode: str,
                     starting_capital: float) -> None:
        self.alert(
            f"<b>Kelly Bot Started</b>\n"
            f"Symbol   : {symbol} {interval}\n"
            f"Mode     : {mode.upper()}\n"
            f"Capital  : ${starting_capital:,.2f}\n"
            f"Time     : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )

    def send_bar_summary(self, symbol: str, price: float, p_shrunk: float,
                         target_frac: float, total_val: float,
                         roi: float, drawdown: float, traded: bool) -> None:
        trade_marker = "TRADE" if traded else "—"
        self.notify(
            f"<b>[{trade_marker}] {symbol} Bar Close</b>\n"
            f"Price    : ${price:,.2f}\n"
            f"p        : {p_shrunk:.4f}\n"
            f"Target   : {target_frac:.0%}\n"
            f"Portfolio: ${total_val:,.2f}  (ROI {roi:+.2f}%)\n"
            f"Drawdown : {drawdown:.2f}%"
        )

    def send_trade(self, side: str, symbol: str, value_usdt: float,
                   price: float, new_frac: float, fee: float) -> None:
        emoji = "BUY" if side == "BUY" else "SELL"
        self.alert(
            f"<b>{emoji} {symbol}</b>\n"
            f"Side     : {side}\n"
            f"Amount   : ${value_usdt:,.2f} USDT\n"
            f"Price    : ${price:,.2f}\n"
            f"New frac : {new_frac:.0%}\n"
            f"Fee      : ${fee:.4f}"
        )

    def send_circuit_breaker(self, symbol: str, drawdown: float,
                              limit: float, portfolio: float) -> None:
        self.alert(
            f"<b>CIRCUIT BREAKER - {symbol}</b>\n"
            f"Drawdown : {drawdown:.2f}% (limit {limit:.0%})\n"
            f"Portfolio: ${portfolio:,.2f}\n"
            f"Status   : TRADING HALTED\n"
            f"Action   : Restart bot manually to resume"
        )

    def send_error(self, symbol: str, context: str, error: str) -> None:
        self.alert(
            f"<b>ERROR - {symbol}</b>\n"
            f"Context  : {context}\n"
            f"Error    : {error[:200]}"
        )


# ===========================================================================
# CONSTANTS
# ===========================================================================

BINANCE_BASE     = 'https://api.binance.com'
BINANCE_TESTNET  = 'https://testnet.binance.vision'

# Minimum order sizes enforced by Binance (USDT notional)
MIN_ORDER_USDT = 10.0

# How many bars of history to fetch for feature computation
# Must be >= the longest rolling window in build_features (200 bars for SMA)
FEATURE_LOOKBACK = 300


# ===========================================================================
# BINANCE REST CLIENT
# ===========================================================================

class BinanceClient:
    """
    Thin wrapper around Binance REST API.
    Handles HMAC signing, error parsing, and endpoint fallback.
    Paper mode skips all order endpoints.
    """

    def __init__(self, api_key: str = '', api_secret: str = '',
                 testnet: bool = False, paper: bool = False):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.paper      = paper
        self.base       = BINANCE_TESTNET if testnet else BINANCE_BASE
        self.session    = requests.Session()
        self.session.headers.update({'X-MBX-APIKEY': api_key})

    def _sign(self, params: dict) -> dict:
        params['timestamp'] = int(time.time() * 1000)
        query    = urllib.parse.urlencode(params)
        sig      = hmac.new(
            self.api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        params['signature'] = sig
        return params

    def _get(self, path: str, params: dict = None, signed: bool = False) -> dict:
        if signed:
            params = self._sign(params or {})
        r = self.session.get(self.base + path, params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, params: dict) -> dict:
        params = self._sign(params)
        r = self.session.post(self.base + path, params=params, timeout=10)
        if not r.ok:
            err = r.json() if r.headers.get('content-type','').startswith('application/json') else {'msg': r.text}
            code = err.get('code', r.status_code)
            msg  = err.get('msg', r.text)
            log.error(f"Order failed [{code}]: {msg}")
            if r.status_code in (401, 403):
                log.error(
                    "API key issue detected. Check:\n"
                    "  1. Key has 'Spot & Margin Trading' permission enabled\n"
                    "  2. IP whitelist includes your current IP (or disable IP restriction)\n"
                    "  3. Key is for the correct environment (live vs testnet)\n"
                    "  Returning FAILED status — bot will continue to next bar."
                )
                return {'status': 'FAILED', 'code': code, 'msg': msg}
            r.raise_for_status()
        return r.json()

    # ---- Public endpoints ----

    def get_klines(self, symbol: str, interval: str, limit: int = 300) -> list:
        """Fetch OHLCV bars. Returns raw list from Binance."""
        return self._get('/api/v3/klines', {
            'symbol': symbol, 'interval': interval, 'limit': limit
        })

    def get_symbol_info(self, symbol: str) -> dict:
        """Fetch exchange filters for a symbol (min qty, step size, etc.)."""
        info = self._get('/api/v3/exchangeInfo', {'symbol': symbol})
        for s in info.get('symbols', []):
            if s['symbol'] == symbol:
                return s
        raise ValueError(f"Symbol {symbol} not found on exchange")

    def get_server_time(self) -> int:
        return self._get('/api/v3/time')['serverTime']

    # ---- Private endpoints (require API key) ----

    def get_account(self) -> dict:
        return self._get('/api/v3/account', {}, signed=True)

    def get_balances(self, assets: list[str]) -> dict[str, float]:
        """Returns {asset: free_balance} for each requested asset."""
        account = self.get_account()
        result  = {}
        for bal in account['balances']:
            if bal['asset'] in assets:
                result[bal['asset']] = float(bal['free'])
        return result

    def place_market_order(self, symbol: str, side: str,
                           quote_qty: float = None,
                           base_qty: float = None) -> dict:
        """
        Place a market order.
          side       : 'BUY' or 'SELL'
          quote_qty  : spend this many USDT (for BUY)
          base_qty   : sell this many base asset (for SELL)

        In paper mode, logs the order but does NOT call the API.
        """
        if self.paper:
            log.info(
                f"[PAPER] {side} {symbol}  "
                f"quote_qty={quote_qty:.2f}" if quote_qty else
                f"[PAPER] {side} {symbol}  base_qty={base_qty:.6f}"
            )
            return {'status': 'PAPER', 'side': side,
                    'quoteOrderQty': quote_qty, 'origQty': base_qty}

        params = {'symbol': symbol, 'side': side, 'type': 'MARKET'}
        if side == 'BUY' and quote_qty is not None:
            params['quoteOrderQty'] = round(quote_qty, 2)
        elif side == 'SELL' and base_qty is not None:
            params['quantity'] = self._round_qty(symbol, base_qty)
        else:
            raise ValueError("BUY needs quote_qty, SELL needs base_qty")

        return self._post('/api/v3/order', params)

    def _round_qty(self, symbol: str, qty: float) -> str:
        """Round quantity to exchange step size."""
        try:
            info = self.get_symbol_info(symbol)
            for f in info['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    step = float(f['stepSize'])
                    precision = len(str(step).rstrip('0').split('.')[-1])
                    return f"{qty:.{precision}f}"
        except Exception:
            pass
        return f"{qty:.6f}"


# ===========================================================================
# STATE PERSISTENCE
# ===========================================================================

class WalletState:
    """
    Persists portfolio state to disk so restarts don't lose position.

    In paper mode: tracks virtual usdt and asset balances.
    In live mode:  reads real balances from exchange at startup,
                   then tracks locally (reconciles at each bar).

    State file format (JSON):
    {
        "usdt":          9800.00,
        "asset":         0.025,
        "peak_value":    10200.00,
        "last_bar_time": "2025-01-15T10:00:00+00:00",
        "total_trades":  12,
        "total_commission": 15.23
    }
    """

    def __init__(self, filepath: str, starting_capital: float = 10_000.0):
        self.filepath        = Path(filepath)
        self.starting_capital = starting_capital
        self._load()

    def _load(self):
        if self.filepath.exists():
            with open(self.filepath) as f:
                d = json.load(f)
            self.usdt             = d['usdt']
            self.asset            = d['asset']
            self.peak_value       = d['peak_value']
            self.last_bar_time    = d.get('last_bar_time', '')
            self.total_trades     = d.get('total_trades', 0)
            self.total_commission = d.get('total_commission', 0.0)
            log.info(
                f"State loaded: USDT={self.usdt:.2f}  "
                f"Asset={self.asset:.6f}  "
                f"Peak={self.peak_value:.2f}  "
                f"Trades={self.total_trades}"
            )
            # Warn if loaded balance differs significantly from expected starting capital
            if abs(self.usdt - self.starting_capital) > self.starting_capital * 0.5:
                log.warning(
                    f"Loaded USDT ({self.usdt:.2f}) differs significantly from "
                    f"--starting-capital ({self.starting_capital:.2f}). "
                    f"If this is a new session, use --reset-state to start fresh, "
                    f"or check that --starting-capital matches your actual wallet balance."
                )
        else:
            log.info(f"No state file found — starting fresh with ${self.starting_capital:.2f}")
            self.usdt             = self.starting_capital
            self.asset            = 0.0
            self.peak_value       = self.starting_capital
            self.last_bar_time    = ''
            self.total_trades     = 0
            self.total_commission = 0.0

    def save(self):
        with open(self.filepath, 'w') as f:
            json.dump({
                'usdt':             self.usdt,
                'asset':            self.asset,
                'peak_value':       self.peak_value,
                'last_bar_time':    self.last_bar_time,
                'total_trades':     self.total_trades,
                'total_commission': self.total_commission,
            }, f, indent=2)

    def total_value(self, price: float) -> float:
        return self.usdt + self.asset * price

    def current_frac(self, price: float) -> float:
        tv = self.total_value(price)
        return (self.asset * price) / tv if tv > 0 else 0.0

    def sync_from_exchange(self, client: BinanceClient, base_asset: str):
        """
        In live mode: read actual balances from exchange and overwrite local
        state. Protects against drift from manual trades or partial fills.
        """
        try:
            balances = client.get_balances(['USDT', base_asset])
            self.usdt  = balances.get('USDT', self.usdt)
            self.asset = balances.get(base_asset, self.asset)
            log.info(f"Synced from exchange: USDT={self.usdt:.2f}  {base_asset}={self.asset:.6f}")
        except Exception as e:
            log.warning(f"Balance sync failed: {e} — using cached state")


# ===========================================================================
# FEATURE ENGINEERING  (mirrors kelly_bot_ml.py exactly)
# ===========================================================================

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl  = df['high'] - df['low']
    hpc = (df['high'] - df['close'].shift(1)).abs()
    lpc = (df['low']  - df['close'].shift(1)).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identical to kelly_bot_ml.py build_features.
    Must stay in sync if the backtest feature set changes.
    """
    df    = df.copy()
    c     = df['close']
    v     = df['volume']
    ret_1 = np.log(c / c.shift(1))

    for n in [3, 6, 12, 24, 48]:
        df[f'ret_{n}'] = np.log(c / c.shift(n))

    df['rsi_14'] = compute_rsi(c, 14)
    df['roc_12'] = (c - c.shift(12)) / c.shift(12)

    ema12             = c.ewm(span=12, adjust=False).mean()
    ema26             = c.ewm(span=26, adjust=False).mean()
    macd_line         = ema12 - ema26
    df['macd_signal'] = macd_line - macd_line.ewm(span=9, adjust=False).mean()

    sma_200           = c.rolling(200, min_periods=50).mean()
    df['trend_200']   = (c - sma_200) / sma_200
    sma_50            = c.rolling(50, min_periods=20).mean()
    df['trend_slope'] = sma_50.diff(5) / c
    df['up_vol_ratio'] = ret_1.rolling(24).apply(
        lambda x: (x > 0).mean(), raw=True
    )
    df['vol_x_trend'] = (
        df['trend_200'].rolling(200, min_periods=50).rank(pct=True) *
        np.sign(df['trend_200'])
    )

    for n in [6, 24, 48]:
        df[f'vol_std_{n}'] = ret_1.rolling(n).std()

    df['atr_14']  = compute_atr(df, 14)
    df['atr_pct'] = df['atr_14'] / c
    df['vol_ratio'] = df['vol_std_6'] / df['vol_std_48'].replace(0, np.nan)

    df['vol_pct_rank_200'] = (
        df['vol_std_24'].rolling(200, min_periods=50).rank(pct=True)
    )
    atr_med          = df['atr_pct'].rolling(100, min_periods=30).median().replace(0, np.nan)
    df['vol_regime'] = (df['atr_pct'] / atr_med) * df['vol_pct_rank_200']

    df['dist_to_high_24'] = (c.rolling(24).max() - c) / c
    df['dist_to_low_24']  = (c - c.rolling(24).min()) / c

    vol_mean          = v.rolling(24).mean()
    vol_std_v         = v.rolling(24).std().replace(0, np.nan)
    df['vol_zscore']  = (v - vol_mean) / vol_std_v

    obv               = (np.sign(ret_1) * v).cumsum()
    df['obv_slope']   = (
        obv.rolling(12).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
        ) / (v.rolling(12).mean() + 1e-9)
    )
    df['buy_imbalance'] = df['taker_base'] / v.replace(0, np.nan) - 0.5

    return df


# ===========================================================================
# CALIBRATED PREDICTION  (mirrors kelly_bot_ml.py exactly)
# ===========================================================================

def _platt_calibrate(raw_tr, y_tr, raw_val):
    lr = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000)
    lr.fit(raw_tr.reshape(-1, 1), y_tr)
    return lr.predict_proba(raw_val.reshape(-1, 1))[:, 1]


def _isotonic_calibrate(raw_tr, y_tr, raw_val):
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(raw_tr, y_tr)
    return np.clip(iso.predict(raw_val), 0.0, 1.0)


def _beta_calibrate(raw_tr, y_tr, raw_val):
    eps    = 1e-7
    p      = np.clip(raw_tr, eps, 1 - eps)
    pt     = np.clip(raw_val, eps, 1 - eps)
    X_fit  = np.column_stack([np.log(p),  np.log(1 - p)])
    X_pred = np.column_stack([np.log(pt), np.log(1 - pt)])
    lr     = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000)
    lr.fit(X_fit, y_tr)
    return lr.predict_proba(X_pred)[:, 1]


def predict_single_bar(model_dict: dict, x: np.ndarray) -> float:
    """
    Predict calibrated p for a single bar feature vector x (shape: [n_features]).
    Returns a scalar float.
    """
    X      = x.reshape(1, -1)
    base   = model_dict['base']
    method = model_dict['best_method']
    raw    = base.predict_proba(X)[:, 1]
    raw_tr = model_dict['best_raw_tr']
    y_tr   = model_dict['best_y_tr']

    if method == 'platt':
        p = _platt_calibrate(raw_tr, y_tr, raw)[0]
    elif method == 'isotonic':
        p = _isotonic_calibrate(raw_tr, y_tr, raw)[0]
    else:
        p = _beta_calibrate(raw_tr, y_tr, raw)[0]

    return float(p)


def shrink_probability(p: float, n_train: int, alpha: float = 3.0) -> float:
    if np.isnan(p):
        return np.nan
    wins = p * n_train
    return float(np.clip((wins + alpha) / (n_train + 2 * alpha), 0.0, 1.0))


def kelly_fraction(p: float, gain: float, loss: float, mult: float) -> float:
    if np.isnan(p):
        return 0.0
    f = (p / loss - (1 - p) / gain) * mult
    return float(np.clip(f, 0.0, 1.0))


# ===========================================================================
# TREND FLOOR  (single-bar version, mirrors kelly_bot_ml.py logic)
# ===========================================================================

class TrendFloorTracker:
    """
    Stateful tracker for the 12-bar trend floor hold-off.
    The backtest computed this over an array; live trading needs it bar-by-bar.
    """

    def __init__(self, floor_frac: float, trend_threshold: float,
                 hold_bars: int = 12):
        self.floor_frac       = floor_frac
        self.trend_threshold  = trend_threshold
        self.hold_bars        = hold_bars
        self._hold_count      = 0

    def update(self, trend_200: float, rsi_14: float) -> float:
        """
        Call once per bar with the latest trend_200 and rsi_14 values.
        Returns the floor fraction for this bar (0.0 or floor_frac).
        """
        if self.floor_frac <= 0:
            return 0.0

        bull_signal = (trend_200 > self.trend_threshold) and (rsi_14 < 70)

        if bull_signal:
            self._hold_count = self.hold_bars
            return self.floor_frac
        elif self._hold_count > 0:
            self._hold_count -= 1
            return self.floor_frac
        else:
            return 0.0


# ===========================================================================
# BAR FETCH AND PARSE
# ===========================================================================

def fetch_recent_bars(client: BinanceClient, symbol: str,
                      interval: str, limit: int = FEATURE_LOOKBACK) -> pd.DataFrame:
    """
    Fetches the most recent `limit` closed bars.
    Excludes the currently-open bar (last item from Binance).
    """
    raw = client.get_klines(symbol, interval, limit=limit + 1)

    df = pd.DataFrame(raw, columns=[
        'open_dt', 'open', 'high', 'low', 'close', 'volume',
        'close_dt', 'quote_vol', 'trades', 'taker_base', 'taker_quote', 'ignore'
    ])
    df['open_dt']  = pd.to_datetime(df['open_dt'],  unit='ms', utc=True)
    df['close_dt'] = pd.to_datetime(df['close_dt'], unit='ms', utc=True)
    for col in ['open', 'high', 'low', 'close', 'volume',
                'quote_vol', 'taker_base', 'taker_quote']:
        df[col] = df[col].astype(float)

    # Drop the currently open bar (its OHLCV is incomplete)
    df = df.iloc[:-1].reset_index(drop=True)
    return df


def get_latest_closed_bar_time(client: BinanceClient,
                                symbol: str, interval: str) -> str:
    """Returns the close_dt of the most recently closed bar as ISO string."""
    raw = client.get_klines(symbol, interval, limit=2)
    # raw[-1] is the open bar, raw[-2] is the last closed bar
    close_ms = int(raw[-2][6])
    return pd.Timestamp(close_ms, unit='ms', tz='UTC').isoformat()


# ===========================================================================
# CIRCUIT BREAKER
# ===========================================================================

def check_circuit_breaker(state: WalletState, price: float,
                           max_drawdown: float,
                           tg: 'TelegramNotifier' = None,
                           symbol: str = '') -> bool:
    """
    Returns True if trading should be halted.
    Updates peak_value if portfolio has grown.
    """
    current_value = state.total_value(price)

    # Update peak
    if current_value > state.peak_value:
        state.peak_value = current_value
        state.save()

    drawdown = (current_value - state.peak_value) / state.peak_value
    if drawdown <= -abs(max_drawdown):
        log.error(
            f"CIRCUIT BREAKER TRIGGERED: drawdown={drawdown:.2%}  "
            f"(limit={max_drawdown:.2%})  "
            f"Portfolio={current_value:.2f}  Peak={state.peak_value:.2f}  "
            f"Trading HALTED. Restart manually to resume."
        )
        if tg:
            tg.send_circuit_breaker(symbol, drawdown * 100, max_drawdown,
                                    current_value)
        return True
    return False


# ===========================================================================
# ORDER EXECUTION
# ===========================================================================

def execute_rebalance(client: BinanceClient, state: WalletState,
                      symbol: str, base_asset: str,
                      target_frac: float, current_price: float,
                      deadband: float, max_order_usdt: float,
                      fee_rate: float = 0.001,
                      tg: 'TelegramNotifier' = None) -> bool:
    """
    Computes required trade and executes if it exceeds the deadband.

    Returns True if a trade was executed, False otherwise.
    """
    total_value  = state.total_value(current_price)
    current_frac = state.current_frac(current_price)
    diff         = target_frac - current_frac

    if abs(diff) < deadband:
        log.info(
            f"  No trade: |Δfrac|={abs(diff):.4f} < deadband={deadband:.4f}"
        )
        return False

    trade_value = diff * total_value  # positive = BUY, negative = SELL

    # Cap single order size
    if abs(trade_value) > max_order_usdt:
        log.warning(
            f"  Order capped: {abs(trade_value):.2f} -> {max_order_usdt:.2f} USDT"
        )
        trade_value = max_order_usdt * np.sign(trade_value)

    # Minimum order check
    if abs(trade_value) < MIN_ORDER_USDT:
        log.info(f"  Order too small ({abs(trade_value):.2f} USDT < {MIN_ORDER_USDT}), skipping")
        return False

    if trade_value > 0:  # BUY
        fee      = trade_value * fee_rate
        if state.usdt < trade_value + fee:
            trade_value = (state.usdt / (1 + fee_rate)) * 0.99  # use 99% of available
            log.warning(f"  Insufficient USDT — reduced buy to {trade_value:.2f}")

        log.info(
            f"  BUY  {trade_value:.2f} USDT of {symbol}  "
            f"@ ~{current_price:.2f}  frac: {current_frac:.3f} -> {target_frac:.3f}"
        )
        result = client.place_market_order(symbol, 'BUY', quote_qty=trade_value)

        if result.get('status') == 'FAILED':
            log.warning("  BUY not executed (order failed) — wallet state unchanged")
            return False
        if result.get('status') in ('FILLED', 'PAPER'):
            qty = trade_value / current_price
            fee_paid = trade_value * fee_rate
            state.usdt  -= (trade_value + fee_paid)
            state.asset += qty
            state.total_trades     += 1
            state.total_commission += fee_paid
            log.info(f"  BUY filled: +{qty:.6f} {base_asset}  fee={fee_paid:.4f} USDT")
            if tg:
                tg.send_trade('BUY', symbol, trade_value, current_price,
                              state.current_frac(current_price), fee_paid)

    else:  # SELL
        sell_value = abs(trade_value)
        qty        = sell_value / current_price
        fee        = sell_value * fee_rate

        if state.asset < qty:
            qty        = state.asset * 0.99  # sell 99% of holdings
            sell_value = qty * current_price
            log.warning(f"  Insufficient {base_asset} — reduced sell to {qty:.6f}")

        log.info(
            f"  SELL {sell_value:.2f} USDT of {symbol}  "
            f"@ ~{current_price:.2f}  frac: {current_frac:.3f} -> {target_frac:.3f}"
        )
        result = client.place_market_order(symbol, 'SELL', base_qty=qty)

        if result.get('status') == 'FAILED':
            log.warning("  SELL not executed (order failed) — wallet state unchanged")
            return False
        if result.get('status') in ('FILLED', 'PAPER'):
            fee_paid = sell_value * fee_rate
            state.usdt  += (sell_value - fee_paid)
            state.asset -= qty
            state.total_trades     += 1
            state.total_commission += fee_paid
            log.info(f"  SELL filled: -{qty:.6f} {base_asset}  fee={fee_paid:.4f} USDT")
            if tg:
                tg.send_trade('SELL', symbol, sell_value, current_price,
                              state.current_frac(current_price), fee_paid)

    state.save()
    return True


# ===========================================================================
# MAIN BAR-CLOSE HANDLER
# ===========================================================================

def on_bar_close(client: BinanceClient, state: WalletState,
                 payload: dict, trend_tracker: TrendFloorTracker,
                 args, tg: TelegramNotifier = None) -> None:
    """
    Called once per bar close. Full pipeline:
      1. Fetch recent closed bars
      2. Check for duplicate bar (already processed this close time)
      3. Build features on latest bar
      4. Predict p, apply shrinkage
      5. Compute Kelly fraction + trend floor
      6. Check circuit breaker
      7. Execute rebalance if needed
      8. Log portfolio state
    """
    symbol       = payload['symbol']
    interval     = payload['interval']
    model_dict   = payload['model_dict']
    feature_cols = payload['feature_cols']
    gain         = payload['gain']
    loss         = payload['loss']
    kelly_mult   = payload['kelly_mult']
    min_edge     = payload['min_edge']
    n_train      = payload['n_train']
    base_asset   = symbol.replace('USDT', '')

    log.info(f"{'='*55}")
    log.info(f"Bar close check: {symbol} {interval}")

    # ---- 1. Fetch bars ----
    try:
        df = fetch_recent_bars(client, symbol, interval, limit=FEATURE_LOOKBACK)
    except Exception as e:
        log.error(f"Failed to fetch bars: {e}")
        if tg:
            tg.send_error(symbol, "fetch bars", str(e))
        return

    if len(df) < 200:
        log.warning(f"Only {len(df)} bars available — need 200+ for features. Skipping.")
        return

    latest_bar_time = df['close_dt'].iloc[-1].isoformat()

    # ---- 2. Duplicate check ----
    if latest_bar_time == state.last_bar_time:
        log.info(f"Already processed bar {latest_bar_time} — skipping")
        return

    current_price = float(df['close'].iloc[-1])
    log.info(f"Processing bar: {latest_bar_time}  price={current_price:.2f}")

    # ---- 3. Sync balance from exchange (live mode only) ----
    if not client.paper:
        state.sync_from_exchange(client, base_asset)

    # ---- 4. Circuit breaker ----
    if check_circuit_breaker(state, current_price, args.max_drawdown, tg=tg, symbol=symbol):
        state.last_bar_time = latest_bar_time
        state.save()
        return

    # ---- 5. Build features ----
    try:
        feat_df = build_features(df)
    except Exception as e:
        log.error(f"Feature build failed: {e}")
        if tg:
            tg.send_error(symbol, "feature build", str(e))
        return

    latest_features = feat_df[feature_cols].iloc[-1].values
    if np.any(np.isnan(latest_features)):
        nan_cols = [feature_cols[i] for i, v in enumerate(latest_features) if np.isnan(v)]
        log.warning(f"NaN features on latest bar: {nan_cols} — skipping")
        return

    # ---- 6. Predict ----
    try:
        p_raw    = predict_single_bar(model_dict, latest_features)
        p_shrunk = shrink_probability(p_raw, n_train)
    except Exception as e:
        log.error(f"Prediction failed: {e}")
        if tg:
            tg.send_error(symbol, "prediction", str(e))
        return

    breakeven_p = loss / (gain + loss)
    min_edge_p  = breakeven_p + min_edge

    log.info(
        f"  p_raw={p_raw:.4f}  p_shrunk={p_shrunk:.4f}  "
        f"breakeven={breakeven_p:.4f}  min_edge={min_edge_p:.4f}"
    )

    # ---- 7. Kelly fraction ----
    k_frac = kelly_fraction(p_shrunk, gain, loss, kelly_mult) \
             if p_shrunk > min_edge_p else 0.0

    # ---- 8. Trend floor ----
    trend_200 = float(feat_df['trend_200'].iloc[-1])
    rsi_14    = float(feat_df['rsi_14'].iloc[-1])
    floor     = trend_tracker.update(trend_200, rsi_14)

    target_frac = min(max(k_frac, floor), 1.0)

    log.info(
        f"  trend_200={trend_200:.4f}  rsi_14={rsi_14:.2f}  "
        f"kelly_frac={k_frac:.4f}  floor={floor:.4f}  "
        f"target_frac={target_frac:.4f}"
    )

    # ---- 9. Execute ----
    traded = execute_rebalance(
        client, state, symbol, base_asset,
        target_frac, current_price,
        deadband, args.max_order_usdt,
        tg=tg,
    )

    # ---- 10. Portfolio summary ----
    total_val    = state.total_value(current_price)
    current_frac = state.current_frac(current_price)
    roi          = (total_val - args.starting_capital) / args.starting_capital * 100
    drawdown     = (total_val - state.peak_value) / state.peak_value * 100

    log.info(
        f"  Portfolio: {total_val:.2f} USDT  "
        f"ROI={roi:+.2f}%  DD={drawdown:.2f}%  "
        f"frac={current_frac:.3f}  trades={state.total_trades}"
    )

    if tg:
        tg.send_bar_summary(
            symbol, current_price, p_shrunk, target_frac,
            total_val, roi, drawdown,
            traded=locals().get('traded', False)
        )

    state.last_bar_time = latest_bar_time
    state.save()


# ===========================================================================
# GRACEFUL SHUTDOWN — close all positions and show P&L
# ===========================================================================

def shutdown_and_close_positions(
    client: BinanceClient,
    state: WalletState,
    symbol: str,
    base_asset: str,
    starting_capital: float,
    tg: TelegramNotifier = None,
) -> None:
    """
    Called on KeyboardInterrupt (Ctrl+C).

    Steps:
      1. Fetch current market price
      2. If holding any asset, place a SELL to return to 100% USDT (flat)
      3. Print and log the full P&L report
      4. Send P&L summary to Telegram
      5. Delete the state file (clean slate for next run)
    """
    log.info("=" * 55)
    log.info("SHUTDOWN INITIATED — closing all positions")

    # ---- 1. Get current price ----
    try:
        raw     = client.get_klines(symbol, '1m', limit=2)
        price   = float(raw[-2][4])   # last closed 1m bar close
        log.info(f"Current {symbol} price: ${price:,.2f}")
    except Exception as e:
        log.error(f"Could not fetch price for close-out: {e}")
        log.warning("Using last known price from state for P&L calculation only")
        price = None

    # ---- 2. Close position if holding asset ----
    closed_value = 0.0
    if state.asset > 0 and price is not None:
        asset_value = state.asset * price
        log.info(f"Closing position: {state.asset:.6f} {base_asset} = ${asset_value:.2f}")

        if client.paper:
            # Paper: simulate the sell
            fee              = asset_value * 0.001
            state.usdt      += asset_value - fee
            closed_value     = asset_value - fee
            state.total_commission += fee
            state.total_trades += 1
            log.info(f"[PAPER] SELL {state.asset:.6f} {base_asset} @ ${price:.2f} -> +${closed_value:.2f} USDT")
            state.asset = 0.0
        else:
            # Live/testnet: real market sell
            try:
                result = client.place_market_order(symbol, 'SELL', base_qty=state.asset)
                if result.get('status') not in ('FAILED',):
                    fee             = asset_value * 0.001
                    state.usdt     += asset_value - fee
                    closed_value    = asset_value - fee
                    state.total_commission += fee
                    state.total_trades += 1
                    log.info(f"SELL executed: {state.asset:.6f} {base_asset} -> +${closed_value:.2f} USDT")
                    state.asset = 0.0
                else:
                    log.error("Close-out SELL failed — check position manually on Binance")
            except Exception as e:
                log.error(f"Close-out SELL error: {e}")
    elif state.asset == 0:
        log.info("No open position — already flat")
    elif price is None:
        log.warning(f"Cannot close {state.asset:.6f} {base_asset} — no price available")

    # ---- 3. P&L Report ----
    final_usdt    = state.usdt + (state.asset * price if price else 0)
    pnl_usdt      = final_usdt - starting_capital
    pnl_pct       = pnl_usdt / starting_capital * 100
    commission    = state.total_commission
    net_pnl_usdt  = pnl_usdt - commission   # already deducted in simulation but shown separately
    peak_dd       = (final_usdt - state.peak_value) / state.peak_value * 100

    separator = "=" * 55
    report_lines = [
        separator,
        "FINAL P&L REPORT",
        separator,
        f"  Symbol         : {symbol}",
        f"  Starting capital: ${starting_capital:>10,.2f}",
        f"  Final value     : ${final_usdt:>10,.2f}",
        f"  P&L (USDT)      : ${pnl_usdt:>+10,.2f}",
        f"  P&L (%)         : {pnl_pct:>+10.2f}%",
        f"  Total commission: ${commission:>10.4f}",
        f"  Total trades    : {state.total_trades}",
        f"  Peak drawdown   : {peak_dd:>10.2f}%",
        separator,
    ]

    for line in report_lines:
        log.info(line)

    # ---- 4. Telegram P&L summary ----
    if tg:
        msg = (
            "<b>Kelly Bot Stopped - P&L Report</b>\n"
            f"Symbol    : {symbol}\n"
            f"Start     : ${starting_capital:,.2f}\n"
            f"Final     : ${final_usdt:,.2f}\n"
            f"P&L       : ${pnl_usdt:+,.2f}  ({pnl_pct:+.2f}%)\n"
            f"Trades    : {state.total_trades}\n"
            f"Commission: ${commission:.4f}\n"
            f"Peak DD   : {peak_dd:.2f}%"
        )
        tg.alert(msg)

    # ---- 5. Save final state then clean up ----
    state.usdt = final_usdt if price else state.usdt
    state.save()
    log.info("State saved. Bot stopped cleanly.")


# ===========================================================================
# SCHEDULER
# ===========================================================================

INTERVAL_TO_SECONDS = {
    '1m':  60,    '3m':  180,   '5m':  300,   '15m': 900,
    '30m': 1800,  '1h':  3600,  '2h':  7200,  '4h':  14400,
    '6h':  21600, '8h':  28800, '12h': 43200, '1d':  86400,
}


def run_scheduler(client: BinanceClient, state: WalletState,
                  payload: dict, trend_tracker: TrendFloorTracker,
                  args, tg: TelegramNotifier = None) -> None:
    """
    Runs a blocking loop that fires on_bar_close at every bar boundary.

    Strategy: sleep until 5 seconds after the expected bar close time,
    then fire. The 5-second buffer ensures Binance has the closed bar ready.
    """
    interval      = payload['interval']
    bar_seconds   = INTERVAL_TO_SECONDS.get(interval, 3600)
    buffer_secs   = 5

    log.info(
        f"Scheduler started: {payload['symbol']} {interval}  "
        f"bar_seconds={bar_seconds}  "
        f"mode={'PAPER' if client.paper else 'LIVE'}"
    )

    # Fire once immediately at startup (catch up if a bar just closed)
    on_bar_close(client, state, payload, trend_tracker, args, tg=tg)

    while True:
        now           = time.time()
        # Next bar close = ceiling of now to bar_seconds boundary
        next_close    = (int(now // bar_seconds) + 1) * bar_seconds
        wait_until    = next_close + buffer_secs
        sleep_seconds = wait_until - now

        next_dt = datetime.fromtimestamp(wait_until, tz=timezone.utc)
        log.info(f"Next bar check at {next_dt.strftime('%Y-%m-%d %H:%M:%S UTC')} "
                 f"(sleeping {sleep_seconds:.0f}s)")

        try:
            time.sleep(max(sleep_seconds, 1))
        except KeyboardInterrupt:
            raise   # propagate up to main block handler
        on_bar_close(client, state, payload, trend_tracker, args, tg=tg)


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Kelly Bot Live/Paper Trading'
    )

    # Model
    parser.add_argument('--model',   type=str, required=True,
                        help='Path to .pkl file saved by kelly_bot_ml.py')

    # Mode
    parser.add_argument('--mode',    type=str, default='paper',
                        choices=['paper', 'live'],
                        help='paper = simulate only | live = real orders')
    parser.add_argument('--testnet', action='store_true',
                        help='Use Binance testnet (https://testnet.binance.vision)')

    # API credentials (required for live, optional for paper)
    parser.add_argument('--api-key',    type=str, default='',
                        help='Binance API key (required for live)')
    parser.add_argument('--api-secret', type=str, default='',
                        help='Binance API secret (required for live)')

    # Risk controls
    parser.add_argument('--max-drawdown',   type=float, default=0.10,
                        help='Circuit breaker: halt if drawdown exceeds this (default 10%%)')
    parser.add_argument('--max-order-usdt', type=float, default=500.0,
                        help='Maximum single order size in USDT (default $500)')
    parser.add_argument('--starting-capital', type=float, default=10_000.0,
                        help='Starting capital in USDT for paper mode (default $10,000)')
    parser.add_argument('--deadband',       type=float, default=None,
                        help='Override deadband from model file')

    # State
    parser.add_argument('--state-file', type=str, default=None,
                        help='Path to state JSON file (default: state_SYMBOL_INTERVAL.json)')
    parser.add_argument('--reset-state', action='store_true',
                        help='Delete existing state file and start fresh '
                             '(use when switching between testnet/live or restarting cleanly)')

    # Telegram notifications
    parser.add_argument('--tg-token',   type=str, default='',
                        help='Telegram Bot token (from @BotFather)')
    parser.add_argument('--tg-chat-id', type=str, default='',
                        help='Telegram chat ID to send messages to')
    parser.add_argument('--tg-quiet',   action='store_true',
                        help='Only send trade/error alerts, skip routine bar summaries')

    args = parser.parse_args()

    # ---- Validate ----
    if args.mode == 'live' and (not args.api_key or not args.api_secret):
        parser.error("--mode live requires --api-key and --api-secret")

    # ---- Load model ----
    log.info(f"Loading model: {args.model}")
    try:
        saved = joblib.load(args.model)
    except FileNotFoundError:
        log.error(f"Model file not found: {args.model}  "
                  f"Run kelly_bot_ml.py first to generate it.")
        sys.exit(1)

    model_dict   = saved['model_dict']
    feature_cols = saved['feature_cols']
    symbol       = saved['symbol']
    interval     = saved['interval']
    gain         = saved['gain']
    loss         = saved['loss']
    kelly_mult   = saved['kelly_mult']
    min_edge     = saved['min_edge']
    trend_floor  = saved['trend_floor']
    trend_thresh = saved['trend_threshold']
    # Resolve deadband: CLI arg > model file > hardcoded default (0.08)
    # Handles older model files that may not have 'deadband' in the payload
    deadband = (
        args.deadband
        if args.deadband is not None
        else saved.get('deadband', 0.08)
    )
    if deadband is None:
        deadband = 0.08
        log.warning("deadband not found in model file — using default 0.08")

    # n_train: number of samples the model was trained on (for shrinkage)
    n_train = len(model_dict['best_y_tr'])

    log.info(
        f"Model loaded: {symbol} {interval}  "
        f"gain={gain}  loss={loss}  kelly_mult={kelly_mult}  "
        f"trend_floor={trend_floor}  deadband={deadband}"
    )

    # ---- Build payload ----
    payload = {
        'symbol':       symbol,
        'interval':     interval,
        'model_dict':   model_dict,
        'feature_cols': feature_cols,
        'gain':         gain,
        'loss':         loss,
        'kelly_mult':   kelly_mult,
        'min_edge':     min_edge,
        'n_train':      n_train,
    }

    # ---- Client ----
    client = BinanceClient(
        api_key    = args.api_key,
        api_secret = args.api_secret,
        testnet    = args.testnet,
        paper      = (args.mode == 'paper'),
    )

    # Verify connectivity
    try:
        server_time = client.get_server_time()
        log.info(f"Binance connected. Server time: {server_time}")
    except Exception as e:
        log.error(f"Cannot reach Binance: {e}")
        sys.exit(1)

    # Verify API key permissions (live mode only)
    if args.mode == 'live':
        try:
            acct = client.get_account()
            can_trade = acct.get('canTrade', False)
            if not can_trade:
                log.error(
                    "API key does NOT have trading permission.\n"
                    "  Go to Binance -> API Management -> Edit -> enable 'Spot Trading'.\n"
                    "  Then restart kelly_live.py."
                )
                sys.exit(1)
            log.info("API key verified: trading permission confirmed")
        except requests.exceptions.HTTPError as e:
            if '401' in str(e) or '403' in str(e):
                log.error(
                    f"API key rejected ({e}).\n"
                    "  Possible causes:\n"
                    "  1. Wrong key/secret for this environment (live vs testnet)\n"
                    "  2. IP not whitelisted — go to Binance API settings and\n"
                    "     either add your IP or disable IP restriction\n"
                    "  3. Key has expired or been deleted"
                )
            else:
                log.error(f"Account check failed: {e}")
            sys.exit(1)

    # ---- State ----
    state_file = args.state_file or f"state_{symbol}_{interval}.json"

    # Reset state if requested or if switching environments
    if args.reset_state and Path(state_file).exists():
        Path(state_file).unlink()
        log.info(f"State file reset: {state_file} deleted — starting fresh")

    state = WalletState(state_file, starting_capital=args.starting_capital)

    # In live mode, sync balances from exchange on startup
    if args.mode == 'live':
        base_asset   = symbol.replace('USDT', '')
        state.sync_from_exchange(client, base_asset)
        last_price   = float(client.get_klines(symbol, interval, limit=2)[-2][4])
        actual_value = state.total_value(last_price)

        # FIX: if this is a fresh session (no prior trades recorded),
        # update starting_capital to match the actual wallet value.
        # This prevents ROI being calculated against a wrong baseline
        # (e.g. testnet pre-funded with $100k when user set --starting-capital 10000).
        if state.total_trades == 0:
            args.starting_capital = actual_value
            log.info(
                f"Starting capital set to actual wallet value: ${actual_value:,.2f}  "
                f"(overrides --starting-capital {args.starting_capital:,.2f})"
            )
        elif 'session_start_value' not in vars(state):
            # Resuming existing session — keep original starting capital from args
            log.info(
                f"Resuming session. Using --starting-capital ${args.starting_capital:,.2f} "
                f"for ROI baseline. Current wallet: ${actual_value:,.2f}"
            )

        state.peak_value = max(state.peak_value, actual_value)
        state.save()

    # ---- Trend tracker ----
    trend_tracker = TrendFloorTracker(
        floor_frac      = trend_floor,
        trend_threshold = trend_thresh,
        hold_bars       = 12,
    )

    # ---- Start ----
    # ---- Telegram ----
    tg = TelegramNotifier(
        token   = args.tg_token,
        chat_id = args.tg_chat_id,
        quiet   = args.tg_quiet,
    )

    log.info("=" * 55)
    log.info(f"Kelly Live Bot starting")
    log.info(f"  Mode        : {'PAPER (no real orders)' if args.mode == 'paper' else 'LIVE'}")
    log.info(f"  Symbol      : {symbol}  {interval}")
    log.info(f"  Testnet     : {args.testnet}")
    log.info(f"  Max DD      : {args.max_drawdown:.0%}")
    log.info(f"  Max order   : ${args.max_order_usdt:.0f}")
    log.info(f"  Deadband    : {deadband:.0%}")
    log.info(f"  Trend floor : {trend_floor:.0%} (threshold {trend_thresh:.0%})")
    log.info(f"  State file  : {state_file}")
    log.info("=" * 55)

    tg.send_startup(symbol, interval, args.mode, args.starting_capital)
    try:
        run_scheduler(client, state, payload, trend_tracker, args, tg=tg)
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt received — initiating graceful shutdown")
        shutdown_and_close_positions(
            client          = client,
            state           = state,
            symbol          = symbol,
            base_asset      = symbol.replace('USDT', ''),
            starting_capital= args.starting_capital,
            tg              = tg,
        )