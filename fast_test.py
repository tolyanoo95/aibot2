#!/usr/bin/env python3
"""Fast backtest from cached data — no API fetch."""
import sys, pickle, functools, logging, warnings
import numpy as np, pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
print = functools.partial(print, flush=True)
console = Console()

from src.config import config
from src.indicators import TechnicalIndicators
from backtest_volbars import resample_to_volume_bars
from backtest_wf import simulate_dca_trades

CACHE_FILE = "data/ohlcv_cache_15m.pkl"


def load_and_prepare():
    with open(CACHE_FILE, "rb") as f:
        raw = pickle.load(f)
    indicators = TechnicalIndicators()
    all_pair_data = {}
    for symbol, df in raw.items():
        tdf = indicators.calculate_all(df.copy())
        vdf = resample_to_volume_bars(df)
        if len(vdf) < 1300:
            continue
        warmup = min(1920, len(df))
        htf_vdf = resample_to_volume_bars(df, initial_threshold=df["volume"].iloc[:warmup].median() * 10)
        if len(htf_vdf) > 20:
            htf_vdf = indicators.calculate_all(htf_vdf)
            for col in ["ema_9", "ema_21", "ema_50"]:
                if col in htf_vdf.columns:
                    vdf[f"htf_{col}"] = htf_vdf[col].reindex(vdf.index, method="ffill")
            import pandas_ta as pta
            st = pta.supertrend(htf_vdf["high"], htf_vdf["low"], htf_vdf["close"], length=9, multiplier=3.0)
            if st is not None:
                for sc in st.columns:
                    if "SUPERTd" in sc:
                        vdf["htf_supertrend"] = st[sc].reindex(vdf.index, method="ffill")
        vdf = indicators.calculate_all(vdf)
        time_atr = tdf["atr"].reindex(vdf.index, method="ffill")
        vdf["atr"] = time_atr.values
        all_pair_data[symbol] = vdf
    console.print(f"Loaded {len(all_pair_data)} pairs from cache")
    return all_pair_data


def run_test(data, sl_mult=2.0, tp_mult=4.0, train_bars=1000, test_bars=300):
    all_trades = []
    fold = 0; start = 0
    global_locked_dir = None
    global_sl_streak = {"LONG": 0, "SHORT": 0}
    min_len = min(len(df) for df in data.values())

    while start + train_bars + test_bars <= min_len:
        fold += 1
        train_end = start + train_bars
        test_end = train_end + test_bars
        fold_trades = []
        for symbol, df_full in data.items():
            if len(df_full) < test_end:
                continue
            df_test = df_full.iloc[train_end:test_end]
            if len(df_test) < 20:
                continue
            close_test = df_test["close"].values
            atr_test = df_test["atr"].values
            atr_ma20 = pd.Series(atr_test).rolling(20, min_periods=1).mean().values
            signals = []
            for j in range(len(df_test)):
                roc = float(df_test["roc_12"].iloc[j]) if "roc_12" in df_test.columns else 0
                if roc > 0.30: direction = "LONG"
                elif roc < -0.10: direction = "SHORT"
                else: direction = "NEUTRAL"
                conf = 0
                if direction != "NEUTRAL":
                    if global_locked_dir == direction: direction = "NEUTRAL"; continue
                    htf_st = float(df_test["htf_supertrend"].iloc[j]) if "htf_supertrend" in df_test.columns else 0
                    if not np.isnan(htf_st) and htf_st != 0:
                        if direction == "LONG" and htf_st < 0: direction = "NEUTRAL"; continue
                        if direction == "SHORT" and htf_st > 0: direction = "NEUTRAL"; continue
                    atr_exp = atr_test[j] / atr_ma20[j] if atr_ma20[j] > 0 else 1.0
                    if atr_exp > 1.5: direction = "NEUTRAL"; continue
                    adx = float(df_test["ADX_14"].iloc[j]) if "ADX_14" in df_test.columns else 25
                    if adx < 25: direction = "NEUTRAL"; continue
                    rsi_s6 = float(df_test["rsi"].diff(6).iloc[j]) if "rsi" in df_test.columns else 0
                    if direction == "LONG" and rsi_s6 < 0: direction = "NEUTRAL"; continue
                    if direction == "SHORT" and rsi_s6 > 0: direction = "NEUTRAL"; continue
                    conf = 0.90
                signals.append({"bar_idx": j, "symbol": symbol, "direction": direction, "confidence": conf})
            trades = simulate_dca_trades(
                df_test, signals, tp_mult=tp_mult, dca_step_mult=1.0,
                max_entries=3, hard_sl_mult=sl_mult, max_hold=24,
                max_open=config.MAX_OPEN_TRADES, cooldown=3, threshold=0.10, full_size_dca=True)
            fold_trades.extend(trades)
        for tr in sorted(fold_trades, key=lambda x: getattr(x, 'exit_bar', getattr(x, 'first_bar', 0))):
            d = tr.direction
            is_dca_sl = tr.exit_reason in ("SL","HARD_SL") and hasattr(tr,'total_size') and tr.total_size > 1
            if is_dca_sl: global_locked_dir = d
            elif tr.exit_reason == "TP":
                global_sl_streak[d] = 0
                if global_locked_dir == d: global_locked_dir = None
            elif tr.exit_reason in ("SL","HARD_SL"):
                global_sl_streak[d] += 1
                if global_sl_streak[d] >= 2: global_locked_dir = d
        all_trades.extend(fold_trades)
        start += test_bars

    if not all_trades:
        return {}
    total_pnl = sum(t.pnl_pct for t in all_trades)
    n = len(all_trades)
    wr = sum(1 for t in all_trades if t.pnl_pct > 0) / n * 100
    pos_size = 1000 * 5 / 11
    net = pos_size * total_pnl / 100 - 2 * pos_size * 0.0002 * n
    tp = sum(1 for t in all_trades if t.exit_reason == "TP")
    sl = sum(1 for t in all_trades if t.exit_reason in ("SL","HARD_SL"))
    return {"trades": n, "wr": wr, "pnl": total_pnl, "net": net, "tp": tp, "sl": sl}


if __name__ == "__main__":
    data = load_and_prepare()
    tests = [
        ("SL=2.0 TP=4.0 (current)", 2.0, 4.0),
        ("SL=1.5 TP=3.0", 1.5, 3.0),
        ("SL=1.5 TP=4.0", 1.5, 4.0),
        ("SL=2.0 TP=3.0", 2.0, 3.0),
        ("SL=1.0 TP=3.0", 1.0, 3.0),
        ("SL=2.5 TP=5.0", 2.5, 5.0),
    ]
    results = []
    for name, sl, tp in tests:
        r = run_test(data, sl_mult=sl, tp_mult=tp)
        r["name"] = name
        results.append(r)
        console.print(f"{name}: Trades={r['trades']} WR={r['wr']:.1f}% NET=${r['net']:+.0f} TP={r['tp']} SL={r['sl']}")

    table = Table(title="SL/TP Comparison (435d, cached)", box=box.ROUNDED)
    table.add_column("Variant", style="bold")
    table.add_column("Trades", justify="right")
    table.add_column("WR", justify="right")
    table.add_column("NET", justify="right")
    table.add_column("vs Current", justify="right")
    table.add_column("TP", justify="right")
    table.add_column("SL", justify="right")
    base = results[0]["net"]
    for r in results:
        d = r["net"] - base
        table.add_row(r["name"], str(r["trades"]), f"{r['wr']:.1f}%", f"${r['net']:+.0f}",
                      f"{d:+.0f}" if r["name"] != results[0]["name"] else "—",
                      str(r["tp"]), str(r["sl"]))
    console.print(table)
