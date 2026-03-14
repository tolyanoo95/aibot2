#!/usr/bin/env python3
"""
Volume Bars Backtest — matches live bot (main_volbars.py) exactly.

Rules-based: ROC momentum → HTF Supertrend → ATR expansion → ADX → RSI slope
No ML. DCA + flip + vol_drop + 3-step Move SL + trail.

Usage:
    python backtest_volbars.py                       # full test from cache
    python backtest_volbars.py --days 365            # last 365 days
    python backtest_volbars.py --sl 2.0 --tp 4.0     # custom SL/TP
    python backtest_volbars.py --no-cache             # fetch from API
"""

import argparse
import functools
import logging
import os
import warnings
from typing import List

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

from src.config import config
from src.indicators import TechnicalIndicators
from backtest_wf import simulate_dca_trades, DcaTrade

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
print = functools.partial(print, flush=True)
console = Console()

ADX_MIN = 20
LONG_MOM = 0.30
SHORT_MOM = 0.10
MOVE_SL_STEPS = [(0.5, 0.20), (1.0, 0.5), (2.0, 1.0)]
MOVE_SL_TRAIL = 1.0


def resample_to_volume_bars(df: pd.DataFrame, initial_threshold: float = None,
                            warmup_bars: int = 1920) -> pd.DataFrame:
    """Resample time-based OHLCV into volume-based bars.

    No lookahead: threshold is computed from a rolling window of PAST data only.
    First `warmup_bars` (20 days of 15m) are used to compute initial threshold.
    After that, threshold adapts using rolling median of last 960 bars (10 days).
    """
    if initial_threshold is None:
        if len(df) < warmup_bars:
            initial_threshold = df["volume"].median() * 2
        else:
            initial_threshold = df["volume"].iloc[:warmup_bars].median() * 2

    vol_threshold = initial_threshold
    vol_history = list(df["volume"].values[:warmup_bars])

    bars = []
    cum_vol = 0
    bar_open = bar_high = bar_low = bar_close = None
    bar_start_time = None
    time_bar_idx = 0

    for idx, row in df.iterrows():
        time_bar_idx += 1

        if bar_open is None:
            bar_open = row["open"]
            bar_high = row["high"]
            bar_low = row["low"]
            bar_start_time = idx

        bar_high = max(bar_high, row["high"])
        bar_low = min(bar_low, row["low"])
        bar_close = row["close"]
        cum_vol += row["volume"]

        if cum_vol >= vol_threshold:
            bars.append({
                "timestamp": bar_start_time,
                "open": bar_open,
                "high": bar_high,
                "low": bar_low,
                "close": bar_close,
                "volume": cum_vol,
            })
            cum_vol = 0
            bar_open = None

        if time_bar_idx > warmup_bars:
            vol_history.append(row["volume"])
            if len(vol_history) > 960:
                vol_history = vol_history[-960:]
            if time_bar_idx % 960 == 0:
                vol_threshold = float(np.median(vol_history)) * 2

    if not bars:
        return pd.DataFrame()

    result = pd.DataFrame(bars)
    result.set_index("timestamp", inplace=True)
    return result


def run_backtest(
    total_days: int = 1200,
    train_bars: int = 1000,
    test_bars: int = 300,
    sl_mult: float = 2.0,
    tp_mult: float = 4.0,
    use_cache: bool = True,
):
    console.print(f"\n[bold cyan]Volume Bars Backtest (matches live bot)[/bold cyan]")
    console.print(f"HTF: Supertrend(9, 3.0) | ADX>{ADX_MIN} | No global lock")
    console.print(f"Move SL: {len(MOVE_SL_STEPS)}-step {MOVE_SL_STEPS} + trail {MOVE_SL_TRAIL}x ATR")
    console.print(f"SL={sl_mult}x TP={tp_mult}x | DCA=3 | Days: {total_days}\n")

    indicators = TechnicalIndicators()
    import pandas_ta as pta

    cache_file = "data/ohlcv_cache_15m.pkl"
    raw_data = {}
    BARS_15M = 96
    total_candles = total_days * BARS_15M

    if use_cache and os.path.exists(cache_file):
        import pickle
        with open(cache_file, "rb") as f:
            raw_data = pickle.load(f)
        console.print(f"[cyan]Loaded {len(raw_data)} pairs from cache[/cyan]")
    else:
        from src.data_fetcher import BinanceDataFetcher
        console.print(f"[cyan]Fetching from API ({total_days} days)...[/cyan]")
        fetcher = BinanceDataFetcher(config)
        for symbol in config.TRADING_PAIRS:
            df = fetcher.fetch_ohlcv_extended(symbol, "15m", total_candles=total_candles)
            if not df.empty:
                raw_data[symbol] = df

    all_pair_data = {}
    for symbol in config.TRADING_PAIRS:
        if symbol not in raw_data:
            continue
        df = raw_data[symbol]
        if len(df) > total_candles:
            df = df.iloc[-total_candles:]
        if len(df) < 200:
            continue

        tdf = indicators.calculate_all(df.copy())
        vdf = resample_to_volume_bars(df)
        if len(vdf) < train_bars + test_bars:
            console.print(f"  [yellow]{symbol}: only {len(vdf)} vol bars, skipping[/yellow]")
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
        console.print(f"  {symbol}: {len(vdf)} vol bars")

    if not all_pair_data:
        console.print("[red]No data![/red]")
        return

    # Walk-forward
    all_trades = []
    fold = 0
    start = 0
    min_len = min(len(df) for df in all_pair_data.values())
    fold_results = []

    while start + train_bars + test_bars <= min_len:
        fold += 1
        train_end = start + train_bars
        test_end = train_end + test_bars
        fold_trades = []

        for symbol, df_full in all_pair_data.items():
            if len(df_full) < test_end:
                continue
            df_test = df_full.iloc[train_end:test_end]
            if len(df_test) < 20:
                continue

            atr_test = df_test["atr"].values
            atr_ma20 = pd.Series(atr_test).rolling(20, min_periods=1).mean().values

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

                signals.append({"bar_idx": j, "symbol": symbol, "direction": direction, "confidence": 0.90})

            trades = simulate_dca_trades(
                df_test, signals, tp_mult=tp_mult, dca_step_mult=1.0,
                max_entries=3, hard_sl_mult=sl_mult, max_hold=24,
                max_open=11, cooldown=3, threshold=0.10, full_size_dca=True,
                move_sl_at=MOVE_SL_STEPS[0][0], move_sl_to=MOVE_SL_STEPS[0][1],
                move_sl_steps=MOVE_SL_STEPS, move_sl_trail=MOVE_SL_TRAIL,
            )
            fold_trades.extend(trades)

        if fold_trades:
            pnl = sum(t.pnl_pct for t in fold_trades)
            wins = sum(1 for t in fold_trades if t.pnl_pct > 0)
            wr = wins / len(fold_trades) * 100
            test_start = list(all_pair_data.values())[0].index[train_end]
            test_end_dt = list(all_pair_data.values())[0].index[min(test_end - 1, len(list(all_pair_data.values())[0]) - 1)]
            fold_results.append({"fold": fold, "trades": len(fold_trades), "wr": wr, "pnl": pnl,
                                 "start": str(test_start)[:10], "end": str(test_end_dt)[:10]})

        all_trades.extend(fold_trades)
        start += test_bars

    # === SUMMARY ===
    console.print("\n" + "=" * 70)
    if not all_trades:
        console.print("[red]No trades![/red]")
        return

    n = len(all_trades)
    total_pnl = sum(t.pnl_pct for t in all_trades)
    wr = sum(1 for t in all_trades if t.pnl_pct > 0) / n * 100

    long_trades = [t for t in all_trades if t.direction == "LONG"]
    short_trades = [t for t in all_trades if t.direction == "SHORT"]
    long_pnl = sum(t.pnl_pct for t in long_trades)
    short_pnl = sum(t.pnl_pct for t in short_trades)
    long_wr = sum(1 for t in long_trades if t.pnl_pct > 0) / max(len(long_trades), 1) * 100
    short_wr = sum(1 for t in short_trades if t.pnl_pct > 0) / max(len(short_trades), 1) * 100

    tp_count = sum(1 for t in all_trades if t.exit_reason == "TP")
    sl_count = sum(1 for t in all_trades if t.exit_reason in ("SL", "HARD_SL"))
    sl_moved = sum(1 for t in all_trades if t.exit_reason == "SL_MOVED")
    vol_drop = sum(1 for t in all_trades if t.exit_reason == "VOL_DROP")
    timeout = sum(1 for t in all_trades if t.exit_reason in ("TIMEOUT", "END"))

    pos_size = 1000 * 5 / 11
    fee_per_trade = 2 * pos_size * 0.0002
    total_fees = fee_per_trade * n
    gross = pos_size * total_pnl / 100
    net = gross - total_fees
    days = total_days
    months = days / 30

    table = Table(title="Volume Bars Backtest Results", box=box.ROUNDED)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Total trades", str(n))
    table.add_row("Win rate", f"{wr:.1f}%")
    table.add_row("Folds (positive/total)", f"{sum(1 for f in fold_results if f['pnl']>0)}/{len(fold_results)}")
    table.add_row("", "")
    table.add_row("LONG", f"{len(long_trades)} (WR {long_wr:.0f}%, PnL {long_pnl:+.2f}%)")
    table.add_row("SHORT", f"{len(short_trades)} (WR {short_wr:.0f}%, PnL {short_pnl:+.2f}%)")
    table.add_row("", "")
    table.add_row("TP", str(tp_count))
    table.add_row("HARD_SL", str(sl_count))
    table.add_row("SL_MOVED", str(sl_moved))
    table.add_row("VOL_DROP", str(vol_drop))
    table.add_row("TIMEOUT", str(timeout))
    table.add_row("", "")
    table.add_row("[bold]NET profit ($1K×5x)[/bold]", f"[bold]${net:+,.0f}[/bold]")
    table.add_row("NET/month", f"${net/months:+,.0f}")
    table.add_row("Gross profit", f"${gross:+,.0f}")
    table.add_row("Total fees", f"-${total_fees:,.0f}")
    table.add_row("Fee/trade", f"${fee_per_trade:.2f}")
    table.add_row("", "")
    table.add_row("Period", f"{days} days ({months:.0f} months)")
    table.add_row("Folds", str(fold))
    console.print(table)

    # Per-symbol
    symbols = sorted(set(t.symbol for t in all_trades))
    sym_table = Table(title="Per Symbol", box=box.SIMPLE)
    sym_table.add_column("Symbol", style="bold")
    sym_table.add_column("Trades", justify="right")
    sym_table.add_column("WR", justify="right")
    sym_table.add_column("NET", justify="right")
    sym_table.add_column("TP", justify="right")
    sym_table.add_column("SL", justify="right")
    for s in symbols:
        st = [t for t in all_trades if t.symbol == s]
        s_pnl = sum(t.pnl_pct for t in st)
        s_wr = sum(1 for t in st if t.pnl_pct > 0) / max(len(st), 1) * 100
        s_net = pos_size * s_pnl / 100 - fee_per_trade * len(st)
        s_tp = sum(1 for t in st if t.exit_reason == "TP")
        s_sl = sum(1 for t in st if t.exit_reason in ("SL", "HARD_SL"))
        style = "green" if s_net > 0 else "red"
        sym_table.add_row(s.replace("/USDT", ""), str(len(st)), f"{s_wr:.0f}%",
                          f"${s_net:+,.0f}", str(s_tp), str(s_sl), style=style)
    console.print(sym_table)

    # Per-year
    if fold_results:
        years = {}
        for t in all_trades:
            if hasattr(t, 'first_bar'):
                continue
            y = 0
            for fr in fold_results:
                y = int(fr['start'][:4]) if fr['start'] else 0
            if y not in years:
                years[y] = []
            years[y].append(t)

    return all_trades


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Volume Bars Backtest (matches live bot)")
    parser.add_argument("--days", type=int, default=1200, help="Days of history (default: 1200)")
    parser.add_argument("--train-bars", type=int, default=1000, help="Train window in vol bars")
    parser.add_argument("--test-bars", type=int, default=300, help="Test window in vol bars")
    parser.add_argument("--sl", type=float, default=2.0, help="SL multiplier (default: 2.0)")
    parser.add_argument("--tp", type=float, default=4.0, help="TP multiplier (default: 4.0)")
    parser.add_argument("--no-cache", action="store_true", help="Force API fetch")
    args = parser.parse_args()

    run_backtest(
        total_days=args.days, train_bars=args.train_bars,
        test_bars=args.test_bars, sl_mult=args.sl, tp_mult=args.tp,
        use_cache=not args.no_cache,
    )
