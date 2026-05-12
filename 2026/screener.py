"""
Multi-Pair Screener — EMA/RSI + EMA200 Trend Gate (Long Only)
=============================================================
Ranks pairs by composite score:
  40% win rate | 30% ATR fit | 20% ADX trend | 10% Fear & Greed sentiment

Requirements:
  pip install python-binance pandas ta-lib matplotlib requests
"""

import os
import time
import requests
import numpy as np
import pandas as pd
import talib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from binance.client import Client
from datetime import datetime, timedelta

# ── CONFIG — keep in sync with backtest_ema_rsi.py ───────────────────────────
API_KEY    = os.environ.get("BINANCE_API_KEY", "")
API_SECRET = os.environ.get("BINANCE_API_SECRET", "")

WATCHLIST = [
    "BTCUSDT",  "ETHUSDT",  "BNBUSDT",  "SOLUSDT",  "XRPUSDT",
    "ADAUSDT",  "AVAXUSDT", "DOTUSDT",  "LINKUSDT",  "DOGEUSDT",
    "MATICUSDT","UNIUSDT",  "ATOMUSDT", "LTCUSDT",  "NEARUSDT",
    "APTUSDT",  "ARBUSDT",  "OPUSDT",   "INJUSDT",  "SUIUSDT", "TRONUSDT",
]

INTERVAL         = Client.KLINE_INTERVAL_1HOUR
LOOKBACK_DAYS    = 180

EMA_FAST         = 12
EMA_SLOW         = 26
EMA_TREND        = 90
RSI_PERIOD       = 14
RSI_BUY_MAX      = 60
RSI_SELL_MIN     = 70
ADX_MIN_TREND    = 20

STOP_LOSS_PCT    = 0.020
TAKE_PROFIT_PCT  = 0.045
TRAILING_STOP_PCT= 0.020
MIN_HOLD_CANDLES = 6
COOLDOWN_CANDLES = 4
INITIAL_CAPITAL  = 1000.0

VOLUME_FILTER    = True
VOLUME_MA_PERIOD = 20
VOLUME_MIN_MULT  = 1.2

PAY_FEES_IN_BNB  = False
COMMISSION_PCT   = 0.00075 if PAY_FEES_IN_BNB else 0.001
MIN_TRADES       = 2            # skip pairs with fewer trades than this

# ATR sweet spot — calibrated to real 4h crypto ATR (0.5-2.0% typical)
ATR_MIN_PCT = 0.3   # floor — very calm pairs
ATR_MAX_PCT = 3.0   # ceiling — too wild for clean exits
ATR_IDEAL   = 1.0   # peak score — enough to reach TP without excessive noise

# Scoring weights
W_WINRATE   = 0.40
W_ATR       = 0.30
W_ADX       = 0.20
W_SENTIMENT = 0.10

ADX_WEAK    = 20
ADX_STRONG  = 40

DELAY_BETWEEN_PAIRS = 0.5       # seconds between API calls

# ── CLIENT ────────────────────────────────────────────────────────────────────
client = Client(API_KEY, API_SECRET)

# ── FETCH ─────────────────────────────────────────────────────────────────────
def fetch_ohlcv(symbol: str) -> pd.DataFrame | None:
    try:
        start = str(int((datetime.utcnow() - timedelta(days=LOOKBACK_DAYS)).timestamp() * 1000))
        raw   = client.get_historical_klines(symbol, INTERVAL, start)
        if len(raw) < 250:      # need 200+ for EMA warmup + trading candles
            return None
        df = pd.DataFrame(raw, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_volume","trades",
            "taker_buy_base","taker_buy_quote","ignore"
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("open_time", inplace=True)
        for col in ["open","high","low","close","volume"]:
            df[col] = df[col].astype(float)
        return df[["open","high","low","close","volume"]]
    except Exception as e:
        print(f"  fetch failed: {e}")
        return None

# ── INDICATORS ────────────────────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l = df["close"].values, df["high"].values, df["low"].values
    df[f"ema_{EMA_FAST}"]  = talib.EMA(c, timeperiod=EMA_FAST)
    df[f"ema_{EMA_SLOW}"]  = talib.EMA(c, timeperiod=EMA_SLOW)
    df[f"ema_{EMA_TREND}"] = talib.EMA(c, timeperiod=EMA_TREND)
    df["rsi"]              = talib.RSI(c, timeperiod=RSI_PERIOD)
    df["adx"]              = talib.ADX(h, l, c, timeperiod=14)
    df["atr"]              = talib.ATR(h, l, c, timeperiod=14)
    df["vol_ma"]           = talib.SMA(df["volume"].values, timeperiod=VOLUME_MA_PERIOD)
    return df.dropna()

# ── SIGNALS ───────────────────────────────────────────────────────────────────
def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df    = df.copy()
    fast  = f"ema_{EMA_FAST}"
    slow  = f"ema_{EMA_SLOW}"
    trend = f"ema_{EMA_TREND}"

    cross_up   = (df[fast] > df[slow]) & (df[fast].shift(1) <= df[slow].shift(1))
    cross_down = (df[fast] < df[slow]) & (df[fast].shift(1) >= df[slow].shift(1))
    uptrend    = df["close"] > df[trend]
    trending   = df["adx"] >= ADX_MIN_TREND
    high_vol   = (df["volume"] >= df["vol_ma"] * VOLUME_MIN_MULT) if VOLUME_FILTER \
                 else pd.Series(True, index=df.index)

    df["signal"]     = 0
    df["in_uptrend"] = uptrend.astype(int)
    df.loc[cross_up & uptrend & trending & high_vol & (df["rsi"] < RSI_BUY_MAX), "signal"] = 1
    df.loc[cross_down,                "signal"] = -1
    df.loc[df["rsi"] > RSI_SELL_MIN, "signal"] = -1
    return df

# ── MINI BACKTEST ─────────────────────────────────────────────────────────────
def mini_backtest(df: pd.DataFrame) -> dict:
    capital          = INITIAL_CAPITAL
    position         = 0.0
    entry_px         = 0.0
    peak_price       = 0.0
    candles_held     = 0
    candles_cooldown = COOLDOWN_CANDLES
    trades           = []

    def close_trade(exit_price, exit_type):
        nonlocal capital, position, entry_px
        nonlocal peak_price, candles_held, candles_cooldown
        entry_value      = position * entry_px
        exit_value       = position * exit_price
        commission       = (entry_value + exit_value) * COMMISSION_PCT
        gross_pnl        = exit_value - entry_value
        net_pnl          = gross_pnl - commission
        pct              = net_pnl / entry_value if entry_value > 0 else 0
        capital         += exit_value - (exit_value * COMMISSION_PCT)
        position         = 0.0
        peak_price       = 0.0
        candles_held     = 0
        candles_cooldown = 0
        trades.append({"pnl": net_pnl, "pnl_pct": pct * 100, "commission": commission})

    for ts, row in df.iterrows():
        price = row["close"]

        if position > 0:
            candles_held += 1
            peak_price    = max(peak_price, price)
            pct_change    = (price - entry_px) / entry_px
            trail_drop    = (peak_price - price) / peak_price if peak_price > 0 else 0

            if candles_held >= MIN_HOLD_CANDLES:
                if pct_change <= -STOP_LOSS_PCT:
                    close_trade(price, "SL")
                elif trail_drop >= TRAILING_STOP_PCT and price > entry_px:
                    close_trade(price, "TSL")
                elif pct_change >= TAKE_PROFIT_PCT:
                    close_trade(price, "TP")
                elif row["signal"] == -1:
                    close_trade(price, "Signal")
        else:
            if candles_cooldown < COOLDOWN_CANDLES:
                candles_cooldown += 1

        if row["signal"] == 1 and position == 0 and capital > 0 \
                and candles_cooldown >= COOLDOWN_CANDLES:
            comm             = capital * COMMISSION_PCT
            capital         -= comm
            position         = capital / price
            entry_px         = price
            peak_price       = price
            capital          = 0.0
            candles_held     = 0
            candles_cooldown = 0

    if position > 0:
        close_trade(df["close"].iloc[-1], "End")

    n          = len(trades)
    wins       = [t for t in trades if t["pnl"] > 0]
    total_ret  = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    total_comm = sum(t["commission"] for t in trades)
    return {
        "n_trades":         n,
        "win_rate":         len(wins) / n if n else 0,
        "total_return":     total_ret,
        "total_commission": total_comm,
        "pct_uptrend":      df["in_uptrend"].mean() * 100,
    }

# ── SENTIMENT ─────────────────────────────────────────────────────────────────
def fetch_fear_greed() -> tuple[int, str]:
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
        d = r.json()["data"][0]
        return int(d["value"]), d["value_classification"]
    except Exception:
        return 50, "Neutral"

def sentiment_score(fg: int) -> float:
    if fg <= 25:  return 1.0
    if fg <= 50:  return 0.5 + (25 - (fg - 25)) / 50
    if fg <= 75:  return 0.5 - (fg - 50) / 50
    return 0.1

# ── SCORE HELPERS ─────────────────────────────────────────────────────────────
def score_atr(atr_pct: float) -> float:
    if atr_pct < ATR_MIN_PCT or atr_pct > ATR_MAX_PCT: return 0.0
    if atr_pct <= ATR_IDEAL:
        return (atr_pct - ATR_MIN_PCT) / (ATR_IDEAL - ATR_MIN_PCT)
    return (ATR_MAX_PCT - atr_pct) / (ATR_MAX_PCT - ATR_IDEAL)

def score_adx(adx: float) -> float:
    return min(max((adx - ADX_WEAK) / (ADX_STRONG - ADX_WEAK), 0.0), 1.0)

# ── SCREENER ──────────────────────────────────────────────────────────────────
def screen_all() -> pd.DataFrame:
    fg_value, fg_label = fetch_fear_greed()
    sent               = sentiment_score(fg_value)
    print(f"\n  Fear & Greed: {fg_value} — {fg_label}  (sentiment score: {sent:.2f})")
    print(f"  Screening {len(WATCHLIST)} pairs | {INTERVAL} | {LOOKBACK_DAYS}d\n")

    rows = []
    for symbol in WATCHLIST:
        print(f"  [{symbol}]", end=" ", flush=True)
        df = fetch_ohlcv(symbol)
        if df is None:
            print("skipped (insufficient data)")
            continue

        df  = add_indicators(df)
        df  = generate_signals(df)
        bt  = mini_backtest(df)

        if bt["n_trades"] < MIN_TRADES:
            print(f"only {bt['n_trades']} trades — skipped (need ≥{MIN_TRADES})")
            continue

        atr_pct = (df["atr"].iloc[-1] / df["close"].iloc[-1]) * 100
        adx_avg = df["adx"].tail(20).mean()
        comm_pct= (bt["total_commission"] / INITIAL_CAPITAL * 100)

        s_wr   = bt["win_rate"]
        s_atr  = score_atr(atr_pct)
        s_adx  = score_adx(adx_avg)
        score  = W_WINRATE*s_wr + W_ATR*s_atr + W_ADX*s_adx + W_SENTIMENT*sent

        rows.append({
            "symbol":       symbol,
            "composite":    round(score, 4),
            "win_rate":     round(bt["win_rate"] * 100, 1),
            "total_return": round(bt["total_return"], 2),
            "n_trades":     bt["n_trades"],
            "comm_pct":     round(comm_pct, 1),
            "atr_pct":      round(atr_pct, 2),
            "adx_avg":      round(adx_avg, 1),
            "pct_uptrend":  round(bt["pct_uptrend"], 1),
            "fg_value":     fg_value,
            "fg_label":     fg_label,
            "s_winrate":    round(s_wr, 3),
            "s_atr":        round(s_atr, 3),
            "s_adx":        round(s_adx, 3),
            "s_sentiment":  round(sent, 3),
        })
        comm_flag = " !" if comm_pct > 10 else ""
        print(f"win={bt['win_rate']*100:.0f}%  up={bt['pct_uptrend']:.0f}%  "
              f"ATR={atr_pct:.1f}%  ADX={adx_avg:.0f}  "
              f"comm={comm_pct:.1f}%{comm_flag}  score={score:.3f}")
        time.sleep(DELAY_BETWEEN_PAIRS)

    df_out = pd.DataFrame(rows).sort_values("composite", ascending=False).reset_index(drop=True)
    df_out.index += 1
    return df_out

# ── PRINT TABLE ───────────────────────────────────────────────────────────────
def print_table(df: pd.DataFrame):
    print("\n" + "="*90)
    print("  SCREENER RESULTS — ranked by composite score")
    print("="*90)
    print(f"  {'#':<3} {'Symbol':<12} {'Score':>6} {'WinRate':>8} {'Return%':>8} "
          f"{'Trades':>7} {'Up%':>5} {'Comm%':>6} {'ATR%':>6} {'ADX':>5}")
    print("  " + "-"*87)
    for rank, row in df.iterrows():
        star = " ★" if rank == 1 else ""
        comm_flag = "!" if row.get("comm_pct", 0) > 10 else " "
        print(f"  {rank:<3} {row['symbol']:<12} {row['composite']:>6.3f} "
              f"{row['win_rate']:>7.1f}% {row['total_return']:>+8.2f}% "
              f"{row['n_trades']:>7} {row.get('pct_uptrend',0):>4.0f}% "
              f"{row.get('comm_pct',0):>5.1f}%{comm_flag} "
              f"{row['atr_pct']:>5.1f}% {row['adx_avg']:>5.0f}{star}")
    print("="*90)
    if len(df):
        best = df.iloc[0]
        print(f"\n  ✦ Top pick: {best['symbol']}  "
              f"win={best['win_rate']}%  return={best['total_return']:+.2f}%  "
              f"uptrend={best.get('pct_uptrend',0):.0f}% of period")
        print(f"    Fear & Greed: {best['fg_value']} — {best['fg_label']}\n")

# ── CHART ─────────────────────────────────────────────────────────────────────
def plot_screener(df: pd.DataFrame):
    if df.empty: return
    BG, SURFACE = "#0d1117", "#161b22"
    GRID, TEXT, MUTED = "#30363d", "#c9d1d9", "#8b949e"
    C = {"s_winrate":"#58a6ff","s_atr":"#3fb950","s_adx":"#f0883e","s_sentiment":"#a371f7"}
    LABELS = {
        "s_winrate": f"Win rate (×{W_WINRATE})",
        "s_atr":     f"ATR fit  (×{W_ATR})",
        "s_adx":     f"ADX trend(×{W_ADX})",
        "s_sentiment":f"Sentiment(×{W_SENTIMENT})",
    }
    metrics  = list(C.keys())
    weights  = [W_WINRATE, W_ATR, W_ADX, W_SENTIMENT]
    symbols  = df["symbol"].tolist()
    n        = len(symbols)
    x        = np.arange(n)

    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.3)
    ax_main = fig.add_subplot(gs[0, :])
    ax_wr   = fig.add_subplot(gs[1, 0])
    ax_atr  = fig.add_subplot(gs[1, 1])

    def style(ax, title):
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_title(title, color=TEXT, fontsize=9, pad=6)
        for spine in ax.spines.values(): spine.set_edgecolor(GRID)
        ax.yaxis.grid(True, color=GRID, lw=0.4, linestyle="--")
        ax.set_axisbelow(True)

    # Stacked score bars
    style(ax_main, "Composite score breakdown")
    bottom = np.zeros(n)
    for m, w in zip(metrics, weights):
        vals = df[m].values * w
        ax_main.bar(x, vals, bottom=bottom, color=C[m], label=LABELS[m], width=0.6, zorder=3)
        bottom += vals
    for i, (score, sym) in enumerate(zip(df["composite"], symbols)):
        ax_main.text(i, score + 0.005, f"{score:.3f}", ha="center", color=TEXT, fontsize=7.5)
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(symbols, rotation=25, ha="right", color=TEXT, fontsize=8)
    ax_main.set_ylim(0, 1.05)
    ax_main.set_ylabel("Score", color=MUTED, fontsize=8)
    ax_main.legend(facecolor=SURFACE, edgecolor=GRID, labelcolor=TEXT, fontsize=7.5, loc="upper right")

    # Win rate + return
    style(ax_wr, "Win rate & total return %")
    ax_wr.bar(x - 0.2, df["win_rate"].values, width=0.35, color="#58a6ff", label="Win rate %", zorder=3)
    ax_wr.axhline(50, color=MUTED, lw=0.6, linestyle="--")
    ax2b = ax_wr.twinx()
    colors_ret = ["#3fb950" if v >= 0 else "#f85149" for v in df["total_return"].values]
    ax2b.bar(x + 0.2, df["total_return"].values, width=0.35, color=colors_ret, alpha=0.8, label="Return %", zorder=3)
    ax2b.set_ylabel("Return %", color=MUTED, fontsize=7)
    ax2b.tick_params(colors=MUTED, labelsize=7)
    ax2b.spines["right"].set_edgecolor(GRID)
    ax_wr.set_xticks(x)
    ax_wr.set_xticklabels(symbols, rotation=30, ha="right", color=TEXT, fontsize=7)
    ax_wr.set_ylabel("Win rate %", color=MUTED, fontsize=7)

    # ATR + ADX
    style(ax_atr, "ATR volatility & ADX trend")
    ax_atr.bar(x - 0.2, df["atr_pct"].values, width=0.35, color="#f0883e", label="ATR %", zorder=3)
    ax_atr.axhline(ATR_MIN_PCT, color="#f85149", lw=0.6, linestyle=":")
    ax_atr.axhline(ATR_MAX_PCT, color="#f85149", lw=0.6, linestyle=":")
    ax_atr.axhline(ATR_IDEAL,   color="#3fb950", lw=0.8, linestyle="--", alpha=0.6)
    ax4b = ax_atr.twinx()
    ax4b.plot(x, df["adx_avg"].values, color="#a371f7", marker="o", ms=4, lw=1.2, label="ADX avg", zorder=4)
    ax4b.axhline(ADX_WEAK,   color=MUTED, lw=0.5, linestyle=":")
    ax4b.axhline(ADX_STRONG, color="#3fb950", lw=0.5, linestyle="--", alpha=0.5)
    ax4b.set_ylabel("ADX", color=MUTED, fontsize=7)
    ax4b.tick_params(colors=MUTED, labelsize=7)
    ax4b.spines["right"].set_edgecolor(GRID)
    ax_atr.set_xticks(x)
    ax_atr.set_xticklabels(symbols, rotation=30, ha="right", color=TEXT, fontsize=7)
    ax_atr.set_ylabel("ATR %", color=MUTED, fontsize=7)

    fg = df.iloc[0]["fg_value"]; fg_lbl = df.iloc[0]["fg_label"]
    fig.suptitle(f"Screener — {INTERVAL} / {LOOKBACK_DAYS}d  |  Fear & Greed: {fg} ({fg_lbl})",
                 color=TEXT, fontsize=11, y=0.98)
    plt.savefig("screener_results.png", dpi=150, bbox_inches="tight", facecolor=BG)
    print("  Chart → screener_results.png")
    plt.show()

# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = screen_all()
    print_table(results)
    plot_screener(results)
    results.to_csv("screener_results.csv", index_label="rank")
    print("  Data  → screener_results.csv\n")
    if len(results):
        winner = results.iloc[0]["symbol"]
        print(f"  → Run: python backtest_ema_rsi.py --symbol {winner}\n")