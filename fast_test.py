#!/usr/bin/env python3
"""
Fast backtest from cached data — no API fetch.
Matches live bot: HTF Supertrend, ADX>20, no global lock, 3-step Move SL + trail.

Usage:
    python fast_test.py                    # default SL/TP grid
    python fast_test.py --sl 2.0 --tp 4.0  # single test
"""
import argparse, pickle, functools, logging, warnings
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

ADX_MIN = 20
LONG_MOM = 0.30
SHORT_MOM = 0.10
MOVE_SL_STEPS = [(0.15, 0.20), (0.5, 0.25), (1.0, 0.5), (2.0, 1.0)]
MOVE_SL_TRAIL = 1.0


def load_and_prepare():
    with open(CACHE_FILE, "rb") as f:
        raw = pickle.load(f)
    import pandas_ta as pta
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


def generate_signals(df_test, atr_test, atr_ma20):
    signals = []
    for j in range(len(df_test)):
        roc = float(df_test["roc_12"].iloc[j]) if "roc_12" in df_test.columns else 0
        if roc > LONG_MOM:
            direction = "LONG"
        elif roc < -SHORT_MOM:
            direction = "SHORT"
        else:
            continue

        htf_st = float(df_test["htf_supertrend"].iloc[j]) if "htf_supertrend" in df_test.columns else 0
        if not np.isnan(htf_st) and htf_st != 0:
            if direction == "LONG" and htf_st < 0: continue
            if direction == "SHORT" and htf_st > 0: continue

        atr_exp = atr_test[j] / atr_ma20[j] if atr_ma20[j] > 0 else 1.0
        if atr_exp > 1.5: continue

        adx = float(df_test["ADX_14"].iloc[j]) if "ADX_14" in df_test.columns else 25
        if adx < ADX_MIN: continue

        rsi_s6 = float(df_test["rsi"].diff(6).iloc[j]) if "rsi" in df_test.columns else 0
        if direction == "LONG" and rsi_s6 < 0: continue
        if direction == "SHORT" and rsi_s6 > 0: continue

        signals.append({"bar_idx": j, "symbol": "", "direction": direction, "confidence": 0.90})
    return signals


def run_test(data, sl_mult=2.0, tp_mult=4.0, train_bars=1000, test_bars=300):
    all_trades = []
    start = 0
    min_len = min(len(df) for df in data.values())

    while start + train_bars + test_bars <= min_len:
        train_end = start + train_bars
        test_end = train_end + test_bars
        for symbol, df_full in data.items():
            if len(df_full) < test_end:
                continue
            df_test = df_full.iloc[train_end:test_end]
            if len(df_test) < 20:
                continue
            atr_test = df_test["atr"].values
            atr_ma20 = pd.Series(atr_test).rolling(20, min_periods=1).mean().values
            signals = generate_signals(df_test, atr_test, atr_ma20)
            for s in signals:
                s["symbol"] = symbol

            trades = simulate_dca_trades(
                df_test, signals, tp_mult=tp_mult, dca_step_mult=1.0,
                max_entries=3, hard_sl_mult=sl_mult, max_hold=24,
                max_open=11, cooldown=3, threshold=0.10, full_size_dca=True,
                move_sl_at=MOVE_SL_STEPS[0][0], move_sl_to=MOVE_SL_STEPS[0][1],
                move_sl_steps=MOVE_SL_STEPS, move_sl_trail=MOVE_SL_TRAIL,
            )
            all_trades.extend(trades)
        start += test_bars

    if not all_trades:
        return {}
    n = len(all_trades)
    total_pnl = sum(t.pnl_pct for t in all_trades)
    wr = sum(1 for t in all_trades if t.pnl_pct > 0) / n * 100
    pos_size = 1000 * 5 / 11
    net = pos_size * total_pnl / 100 - 2 * pos_size * 0.0002 * n
    tp = sum(1 for t in all_trades if t.exit_reason == "TP")
    sl = sum(1 for t in all_trades if t.exit_reason in ("SL", "HARD_SL"))
    sl_moved = sum(1 for t in all_trades if t.exit_reason == "SL_MOVED")
    return {"trades": n, "wr": wr, "pnl": total_pnl, "net": net, "tp": tp, "sl": sl, "sl_moved": sl_moved}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sl", type=float, default=0, help="Single SL test")
    parser.add_argument("--tp", type=float, default=0, help="Single TP test")
    args = parser.parse_args()

    data = load_and_prepare()

    if args.sl > 0 and args.tp > 0:
        r = run_test(data, sl_mult=args.sl, tp_mult=args.tp)
        console.print(f"SL={args.sl} TP={args.tp}: Trades={r['trades']} WR={r['wr']:.1f}% "
                      f"NET=${r['net']:+,.0f} TP={r['tp']} SL={r['sl']} SL_MOV={r['sl_moved']}")
    else:
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
            console.print(f"{name}: Trades={r['trades']} WR={r['wr']:.1f}% NET=${r['net']:+,.0f} "
                          f"TP={r['tp']} SL={r['sl']} SL_MOV={r['sl_moved']}")

        table = Table(title="SL/TP Grid (HTF Supertrend + 3-step Move SL + trail)", box=box.ROUNDED)
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
            dp = d / abs(base) * 100 if base else 0
            table.add_row(r["name"], str(r["trades"]), f"{r['wr']:.1f}%", f"${r['net']:+,.0f}",
                          f"${d:+,.0f} ({dp:+.1f}%)" if r["name"] != results[0]["name"] else "—",
                          str(r["tp"]), str(r["sl"]))
        console.print(table)
