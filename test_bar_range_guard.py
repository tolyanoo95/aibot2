#!/usr/bin/env python3
"""Test bar range / ATR guard to avoid entering on oversized bars."""
import pickle, functools, logging, warnings
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
SL_MULT = 2.0
TP_MULT = 4.0
LONG_MOM = 0.30
SHORT_MOM = 0.10
MAX_DCA = 3
MAX_OPEN = 11


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
        vdf = indicators.calculate_all(vdf)
        time_atr = tdf["atr"].reindex(vdf.index, method="ffill")
        vdf["atr"] = time_atr.values
        all_pair_data[symbol] = vdf
    console.print(f"Loaded {len(all_pair_data)} pairs")
    return all_pair_data


def run_test(data, bar_range_max=0, train_bars=1000, test_bars=300):
    """Run backtest with optional bar_range/ATR guard."""
    all_trades = []
    start = 0
    min_len = min(len(df) for df in data.values())
    skipped_by_guard = 0
    total_signals = 0

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
            high_test = df_test["high"].values
            low_test = df_test["low"].values

            signals = []
            for j in range(len(df_test)):
                roc = float(df_test["roc_12"].iloc[j]) if "roc_12" in df_test.columns else 0
                if roc > LONG_MOM:
                    direction = "LONG"
                elif roc < -SHORT_MOM:
                    direction = "SHORT"
                else:
                    continue

                htf_e9 = float(df_test["htf_ema_9"].iloc[j]) if "htf_ema_9" in df_test.columns else 0
                htf_e21 = float(df_test["htf_ema_21"].iloc[j]) if "htf_ema_21" in df_test.columns else 0
                if htf_e9 > 0 and htf_e21 > 0:
                    if direction == "LONG" and htf_e9 < htf_e21: continue
                    if direction == "SHORT" and htf_e9 > htf_e21: continue

                atr_exp = atr_test[j] / atr_ma20[j] if atr_ma20[j] > 0 else 1.0
                if atr_exp > 1.5: continue

                adx = float(df_test["ADX_14"].iloc[j]) if "ADX_14" in df_test.columns else 25
                if adx < ADX_MIN: continue

                rsi_s6 = float(df_test["rsi"].diff(6).iloc[j]) if "rsi" in df_test.columns else 0
                if direction == "LONG" and rsi_s6 < 0: continue
                if direction == "SHORT" and rsi_s6 > 0: continue

                total_signals += 1

                # Bar range guard
                if bar_range_max > 0:
                    bar_range = high_test[j] - low_test[j]
                    bar_atr = atr_test[j] if atr_test[j] > 0 else 1
                    if bar_range / bar_atr > bar_range_max:
                        skipped_by_guard += 1
                        continue

                signals.append({"bar_idx": j, "symbol": symbol, "direction": direction, "confidence": 0.90})

            trades = simulate_dca_trades(
                df_test, signals, tp_mult=TP_MULT, dca_step_mult=1.0,
                max_entries=MAX_DCA, hard_sl_mult=SL_MULT, max_hold=24,
                max_open=MAX_OPEN, cooldown=3, threshold=0.10,
                full_size_dca=True, move_sl_at=2.0, move_sl_to=1.0,
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
    timeout = sum(1 for t in all_trades if t.exit_reason == "TIMEOUT")
    vol_drop = sum(1 for t in all_trades if t.exit_reason == "VOL_DROP")
    sl_moved = sum(1 for t in all_trades if t.exit_reason == "SL_MOVED")

    bars1_sl = sum(1 for t in all_trades
                   if t.exit_reason in ("SL", "HARD_SL")
                   and hasattr(t, 'first_bar') and hasattr(t, 'exit_bar')
                   and t.exit_bar - t.first_bar <= 1)

    return {
        "trades": n, "wr": wr, "net": net, "tp": tp, "sl": sl,
        "timeout": timeout, "vol_drop": vol_drop, "sl_moved": sl_moved,
        "bars1_sl": bars1_sl, "skipped": skipped_by_guard, "total_sigs": total_signals,
    }


if __name__ == "__main__":
    data = load_and_prepare()

    tests = [
        ("Baseline (no guard)", 0),
        ("bar_range < 2.0x ATR", 2.0),
        ("bar_range < 2.5x ATR", 2.5),
        ("bar_range < 3.0x ATR", 3.0),
        ("bar_range < 3.5x ATR", 3.5),
        ("bar_range < 4.0x ATR", 4.0),
        ("bar_range < 5.0x ATR", 5.0),
    ]

    results = []
    for name, threshold in tests:
        console.print(f"Running: {name}...")
        r = run_test(data, bar_range_max=threshold)
        r["name"] = name
        results.append(r)
        console.print(f"  Trades={r['trades']} WR={r['wr']:.1f}% NET=${r['net']:+,.0f} "
                      f"SL={r['sl']} SL_MOVED={r['sl_moved']} Bars1_SL={r['bars1_sl']} "
                      f"Skipped={r['skipped']}/{r['total_sigs']}")

    table = Table(title="Bar Range Guard Test (3yr)", box=box.ROUNDED)
    table.add_column("Guard", style="bold")
    table.add_column("Trades", justify="right")
    table.add_column("WR", justify="right")
    table.add_column("NET", justify="right")
    table.add_column("vs Base", justify="right")
    table.add_column("SL", justify="right")
    table.add_column("SL_MOV", justify="right")
    table.add_column("B1_SL", justify="right")
    table.add_column("Skipped", justify="right")

    base_net = results[0]["net"]
    for r in results:
        delta = r["net"] - base_net
        delta_pct = (delta / abs(base_net) * 100) if base_net != 0 else 0
        vs = f"${delta:+,.0f} ({delta_pct:+.1f}%)" if r["name"] != results[0]["name"] else "—"
        table.add_row(
            r["name"], str(r["trades"]), f"{r['wr']:.1f}%",
            f"${r['net']:+,.0f}", vs,
            str(r["sl"]), str(r.get("sl_moved", 0)),
            str(r.get("bars1_sl", "?")),
            str(r["skipped"]),
        )
    console.print(table)
