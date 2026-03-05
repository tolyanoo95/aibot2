#!/usr/bin/env python3
"""
Walk-Forward Backtest with realistic SL/TP/Trailing simulation.

Trains on window [0..train_end], tests on [train_end..test_end],
then slides forward. Reports PnL as if the bot actually traded.

Usage:
    python backtest_wf.py
    python backtest_wf.py --days 60 --train-days 20 --test-days 5
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from src.config import config
from src.data_fetcher import BinanceDataFetcher
from src.features import FeatureEngineer
from src.indicators import TechnicalIndicators
from src.ml_model import MLSignalModel
from src.trend_health_model import TrendHealthModel
from src.signal_generator import SignalGenerator

logging.basicConfig(level=logging.WARNING)
console = Console()


@dataclass
class SimTrade:
    symbol: str
    direction: str  # LONG / SHORT
    entry_price: float
    sl: float
    tp: float
    confidence: float
    entry_bar: int
    exit_bar: int = -1
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl_pct: float = 0.0


def simulate_trades(
    df: pd.DataFrame,
    signals: List[dict],
    sl_mult: float,
    tp_mult: float,
    max_hold: int = 18,
    max_open: int = 2,
    cooldown: int = 4,
    adx_col: str = "ADX_14",
) -> List[SimTrade]:
    """Simulate trading with SL/TP checked against high/low each bar."""
    trades: List[SimTrade] = []
    open_trades: List[SimTrade] = []
    cooldowns: dict = {}

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = df["atr"].values
    adx = df[adx_col].values if adx_col in df.columns else np.full(len(df), 25.0)

    for i in range(len(signals)):
        sig = signals[i]
        bar_idx = sig["bar_idx"]

        # Close expired trades
        for t in list(open_trades):
            bars_held = bar_idx - t.entry_bar
            if bars_held <= 0:
                continue

            hit_tp = hit_sl = False
            if t.direction == "LONG":
                hit_tp = high[bar_idx] >= t.tp
                hit_sl = low[bar_idx] <= t.sl
            else:
                hit_tp = low[bar_idx] <= t.tp
                hit_sl = high[bar_idx] >= t.sl

            closed = False
            if hit_sl and hit_tp:
                t.exit_price = t.sl
                t.exit_reason = "SL"
                closed = True
            elif hit_sl:
                t.exit_price = t.sl
                t.exit_reason = "SL"
                closed = True
            elif hit_tp:
                t.exit_price = t.tp
                t.exit_reason = "TP"
                closed = True
            elif bars_held >= max_hold:
                t.exit_price = close[bar_idx]
                t.exit_reason = "TIMEOUT"
                closed = True

            if closed:
                t.exit_bar = bar_idx
                if t.direction == "LONG":
                    t.pnl_pct = (t.exit_price - t.entry_price) / t.entry_price * 100
                else:
                    t.pnl_pct = (t.entry_price - t.exit_price) / t.entry_price * 100
                open_trades.remove(t)
                trades.append(t)
                cooldowns[t.symbol] = bar_idx + cooldown

        # Try to open new trade
        direction = sig.get("direction")
        conf = sig.get("confidence", 0)
        symbol = sig.get("symbol", "")

        if direction not in ("LONG", "SHORT"):
            continue
        if conf < config.PREDICTION_THRESHOLD:
            continue
        if len(open_trades) >= max_open:
            continue
        if cooldowns.get(symbol, 0) > bar_idx:
            continue
        if any(t.symbol == symbol for t in open_trades):
            continue

        a = atr[bar_idx]
        if np.isnan(a) or a <= 0:
            continue

        price = close[bar_idx]
        tp_mult_adj = tp_mult
        if adx[bar_idx] < 20:
            tp_mult_adj = min(tp_mult, 1.5)

        if direction == "LONG":
            sl_price = price - a * sl_mult
            tp_price = price + a * tp_mult_adj
        else:
            sl_price = price + a * sl_mult
            tp_price = price - a * tp_mult_adj

        trade = SimTrade(
            symbol=symbol, direction=direction,
            entry_price=price, sl=sl_price, tp=tp_price,
            confidence=conf, entry_bar=bar_idx,
        )
        open_trades.append(trade)

    # Close remaining open trades at last bar
    for t in open_trades:
        t.exit_bar = len(close) - 1
        t.exit_price = close[-1]
        t.exit_reason = "END"
        if t.direction == "LONG":
            t.pnl_pct = (t.exit_price - t.entry_price) / t.entry_price * 100
        else:
            t.pnl_pct = (t.entry_price - t.exit_price) / t.entry_price * 100
        trades.append(t)

    return trades


def run_walk_forward(
    total_days: int = 45,
    train_days: int = 20,
    test_days: int = 5,
    sl_mult: float = 1.5,
    tp_mult: float = 3.0,
):
    total_candles = total_days * 288
    train_candles = train_days * 288
    test_candles = test_days * 288

    console.print(Panel(
        f"[bold]Walk-Forward Backtest[/bold]\n"
        f"Total: {total_days}d | Train: {train_days}d | Test: {test_days}d\n"
        f"SL={sl_mult}x TP={tp_mult}x | Threshold={config.PREDICTION_THRESHOLD}\n"
        f"Pairs: {len(config.TRADING_PAIRS)}",
        box=box.DOUBLE,
    ))

    fetcher = BinanceDataFetcher(config)
    indicators = TechnicalIndicators()
    fe = FeatureEngineer()
    sig_gen = SignalGenerator(config)

    # Fetch all data once
    console.print("\n[cyan]Fetching data...[/cyan]")
    all_pair_data = {}
    for symbol in config.TRADING_PAIRS:
        df = fetcher.fetch_ohlcv_extended(symbol, "5m", total_candles=total_candles)
        if df.empty or len(df) < train_candles + test_candles:
            console.print(f"  [red]{symbol}: not enough data ({len(df)} bars)[/red]")
            continue
        df = indicators.calculate_all(df)
        all_pair_data[symbol] = df
        console.print(f"  {symbol}: {len(df)} bars")

    if not all_pair_data:
        console.print("[red]No data![/red]")
        return

    # Walk-forward windows
    all_trades: List[SimTrade] = []
    fold = 0
    start = 0

    while start + train_candles + test_candles <= total_candles:
        fold += 1
        train_end = start + train_candles
        test_end = train_end + test_candles

        console.print(f"\n[bold]Fold {fold}[/bold]: train bars {start}-{train_end}, test bars {train_end}-{test_end}")

        # Build training data from ALL pairs
        train_X_all, train_y_all = [], []
        feat_names_ref = None

        for symbol, df_full in all_pair_data.items():
            if len(df_full) < test_end:
                continue
            df_train = df_full.iloc[start:train_end].copy()
            if len(df_train) < 200:
                continue

            feat = fe.create_features(df_train)
            feat = fe.add_rolling_htf_features(feat)
            cols = fe.get_feature_columns(feat, model_type="trend")
            X = feat[cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

            y = fe.create_labels(
                df_train,
                tp_multiplier=config.LABEL_TP_MULTIPLIER,
                sl_multiplier=config.LABEL_SL_MULTIPLIER,
                max_bars=config.LABEL_MAX_BARS,
                ternary=True,
                uncertain_threshold_pct=config.UNCERTAIN_THRESHOLD_PCT,
            )
            common = X.index.intersection(y.index)
            X_c = X.loc[common].iloc[:-config.LABEL_MAX_BARS]
            y_c = y.loc[common].iloc[:-config.LABEL_MAX_BARS]

            train_X_all.append(X_c)
            train_y_all.append(y_c)
            feat_names_ref = cols

        if not train_X_all:
            start += test_candles
            continue

        X_train = pd.concat(train_X_all, ignore_index=True)
        y_train = pd.concat(train_y_all, ignore_index=True)

        # Train fresh model for this fold (temp path, not saved permanently)
        import tempfile, os
        tmp_path = os.path.join(tempfile.gettempdir(), f"wf_model_fold{fold}.json")
        model = MLSignalModel(tmp_path)

        sw = fe.compute_sample_weights(y_train)
        model.train(X_train, y_train, feat_names_ref, sample_weights=sw)

        n_buy = int((y_train == 1).sum())
        n_sell = int((y_train == -1).sum())
        n_hold = int((y_train == 0).sum())
        console.print(f"  Trained on {len(X_train)} samples (B:{n_buy} S:{n_sell} H:{n_hold})")

        # Test on out-of-sample data
        # Compute features on ALL data up to test_end (so rolling windows have full lookback)
        fold_trades = []
        for symbol, df_full in all_pair_data.items():
            if len(df_full) < test_end:
                continue

            df_with_lookback = df_full.iloc[:test_end].copy()
            feat_all = fe.create_features(df_with_lookback)
            feat_all = fe.add_rolling_htf_features(feat_all)
            cols = fe.get_feature_columns(feat_all, model_type="trend")
            X_all_feat = feat_all[cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

            # Slice only test window for signals (but features had full lookback)
            X = X_all_feat.iloc[train_end:test_end]
            df_test = df_with_lookback.iloc[train_end:test_end]
            if len(X) < 50:
                continue

            # Generate signals
            signals = []
            for j in range(len(X)):
                pred = model.predict(X.iloc[[j]])
                sig_val = pred.get("signal", 0)
                conf = pred.get("confidence", 0)

                # Confidence cap
                if conf > 0.90:
                    conf = 0.85

                # Dead hours check using bar time (not datetime.now)
                bar_time = df_test.index[j]
                bar_hour = bar_time.hour if hasattr(bar_time, 'hour') else 0
                dead_hours = config.FILTER_DEAD_HOURS
                if dead_hours and bar_hour in dead_hours:
                    direction = "NEUTRAL"
                    conf *= 0.3
                else:
                    direction = "SHORT" if sig_val == -1 else ("LONG" if sig_val == 1 else "NEUTRAL")

                # Exhaustion guard check
                row = X.iloc[j]
                guard_feats = {k: float(row.get(k, 0)) for k in [
                    "dist_from_high_96", "dist_from_low_96",
                    "dist_from_high_288", "dist_from_low_288",
                    "max_drawdown_48h", "roc_deceleration"]}

                if direction != "NEUTRAL":
                    guard = sig_gen._exhaustion_guard(direction, guard_feats)
                    if guard:
                        direction = "NEUTRAL"
                        conf *= 0.3

                signals.append({
                    "bar_idx": j,
                    "symbol": symbol,
                    "direction": direction,
                    "confidence": conf,
                })

            trades = simulate_trades(
                df_test, signals,
                sl_mult=sl_mult, tp_mult=tp_mult,
                max_hold=config.MAX_HOLD_BARS,
                max_open=config.MAX_OPEN_TRADES,
                cooldown=4,
            )
            fold_trades.extend(trades)

        # Fold summary
        if fold_trades:
            pnl = sum(t.pnl_pct for t in fold_trades)
            wins = sum(1 for t in fold_trades if t.pnl_pct > 0)
            total = len(fold_trades)
            wr = wins / total * 100 if total > 0 else 0
            console.print(f"  [bold]Result: {total} trades, WR {wr:.0f}%, PnL {pnl:+.2f}%[/bold]")
        else:
            console.print(f"  [dim]No trades in test window[/dim]")

        all_trades.extend(fold_trades)
        start += test_candles

    # Final summary
    console.print("\n" + "=" * 60)

    if not all_trades:
        console.print("[red]No trades across all folds![/red]")
        return

    total_pnl = sum(t.pnl_pct for t in all_trades)
    total_wins = sum(1 for t in all_trades if t.pnl_pct > 0)
    total_count = len(all_trades)
    total_wr = total_wins / total_count * 100

    long_trades = [t for t in all_trades if t.direction == "LONG"]
    short_trades = [t for t in all_trades if t.direction == "SHORT"]
    long_pnl = sum(t.pnl_pct for t in long_trades)
    short_pnl = sum(t.pnl_pct for t in short_trades)
    long_wr = sum(1 for t in long_trades if t.pnl_pct > 0) / len(long_trades) * 100 if long_trades else 0
    short_wr = sum(1 for t in short_trades if t.pnl_pct > 0) / len(short_trades) * 100 if short_trades else 0

    tp_count = sum(1 for t in all_trades if t.exit_reason == "TP")
    sl_count = sum(1 for t in all_trades if t.exit_reason == "SL")
    to_count = sum(1 for t in all_trades if t.exit_reason == "TIMEOUT")

    table = Table(title="Walk-Forward Backtest Results", box=box.ROUNDED)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total trades", str(total_count))
    table.add_row("Win rate", f"{total_wr:.1f}%")
    table.add_row("Total PnL", f"{total_pnl:+.2f}%")
    table.add_row("", "")
    table.add_row("LONG trades", f"{len(long_trades)} (WR {long_wr:.0f}%, PnL {long_pnl:+.2f}%)")
    table.add_row("SHORT trades", f"{len(short_trades)} (WR {short_wr:.0f}%, PnL {short_pnl:+.2f}%)")
    table.add_row("", "")
    table.add_row("TP hits", str(tp_count))
    table.add_row("SL hits", str(sl_count))
    table.add_row("Timeouts", str(to_count))
    table.add_row("", "")
    table.add_row("Folds", str(fold))
    table.add_row("Avg PnL/trade", f"{total_pnl / total_count:+.3f}%")

    console.print(table)

    # Per-symbol breakdown
    symbols = sorted(set(t.symbol for t in all_trades))
    sym_table = Table(title="Per-Symbol Breakdown", box=box.SIMPLE)
    sym_table.add_column("Symbol")
    sym_table.add_column("Trades", justify="right")
    sym_table.add_column("WR", justify="right")
    sym_table.add_column("PnL", justify="right")

    for s in symbols:
        s_trades = [t for t in all_trades if t.symbol == s]
        s_pnl = sum(t.pnl_pct for t in s_trades)
        s_wr = sum(1 for t in s_trades if t.pnl_pct > 0) / len(s_trades) * 100
        sym_table.add_row(s, str(len(s_trades)), f"{s_wr:.0f}%", f"{s_pnl:+.2f}%")

    console.print(sym_table)
    return all_trades


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-Forward Backtest")
    parser.add_argument("--days", type=int, default=45, help="Total days of data")
    parser.add_argument("--train-days", type=int, default=20, help="Training window days")
    parser.add_argument("--test-days", type=int, default=5, help="Test window days")
    parser.add_argument("--sl", type=float, default=1.5, help="SL ATR multiplier")
    parser.add_argument("--tp", type=float, default=3.0, help="TP ATR multiplier")
    args = parser.parse_args()

    run_walk_forward(
        total_days=args.days,
        train_days=args.train_days,
        test_days=args.test_days,
        sl_mult=args.sl,
        tp_mult=args.tp,
    )
