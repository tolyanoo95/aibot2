#!/usr/bin/env python3
"""
Volume Bars ML Backtest.
Instead of fixed 15m time bars, resample into fixed-volume bars.
Each bar = same amount of trading volume → more stable patterns for ML.
"""

import argparse
import functools
import logging
import os
import sys
import tempfile
import warnings
from typing import List

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

from src.config import config
from src.data_fetcher import BinanceDataFetcher
from src.features import FeatureEngineer
from src.indicators import TechnicalIndicators
from src.ml_model import MLSignalModel
from src.signal_generator import SignalGenerator
from backtest_wf import simulate_trades, simulate_dca_trades, SimTrade, DcaTrade

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
print = functools.partial(print, flush=True)
console = Console()


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

        # Update rolling threshold every 960 bars (no lookahead)
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


def run_volbars_backtest(
    total_days: int = 187,
    train_bars: int = 2000,
    test_bars: int = 500,
    vol_threshold: float = 2000,
    sl_mult: float = 1.5,
    tp_mult: float = 3.0,
    long_mom: float = 0.30,
    short_mom: float = 0.10,
    conf_threshold: float = 0.55,
    use_dca: bool = False,
):
    """ML backtest on volume bars instead of time bars."""

    console.print(f"\n[bold cyan]Volume Bars ML Backtest[/bold cyan]")
    console.print(f"Days: {total_days} | Vol threshold: {vol_threshold}")
    console.print(f"Train: {train_bars} bars | Test: {test_bars} bars")
    console.print(f"SL={sl_mult}x TP={tp_mult}x | Conf>{conf_threshold}\n")

    fetcher = BinanceDataFetcher(config)
    indicators = TechnicalIndicators()
    fe = FeatureEngineer()
    sig_gen = SignalGenerator(config)

    trading_pairs = list(config.TRADING_PAIRS)
    pair_to_id = {sym: i for i, sym in enumerate(sorted(trading_pairs))}

    BARS_15M = 96
    total_candles = total_days * BARS_15M

    console.print(f"[cyan]Fetching 15m data and converting to volume bars (no lookahead)...[/cyan]")
    all_pair_data = {}
    all_time_data = {}  # time bars for realistic ATR
    for symbol in trading_pairs:
        df = fetcher.fetch_ohlcv_extended(symbol, "15m", total_candles=total_candles)
        if df.empty or len(df) < 200:
            continue

        # Keep time bars for ATR
        tdf = indicators.calculate_all(df.copy())
        all_time_data[symbol] = tdf

        vdf = resample_to_volume_bars(df)
        if len(vdf) < train_bars + test_bars:
            console.print(f"  [yellow]{symbol}: only {len(vdf)} vol bars, skipping[/yellow]")
            continue

        # HTF volume bars (5x bigger threshold) for trend detection
        htf_vdf = resample_to_volume_bars(df, initial_threshold=df["volume"].iloc[:1920].median() * 10 if len(df) > 1920 else df["volume"].median() * 10)
        if len(htf_vdf) > 20:
            htf_vdf = indicators.calculate_all(htf_vdf)
            # Forward-fill HTF EMA to primary volume bar timestamps
            for col in ["ema_9", "ema_21", "ema_50"]:
                if col in htf_vdf.columns:
                    vdf[f"htf_{col}"] = htf_vdf[col].reindex(vdf.index, method="ffill")

        vdf = indicators.calculate_all(vdf)

        # Replace volume bar ATR with time bar ATR (forward-filled to volume bar timestamps)
        time_atr = tdf["atr"].reindex(vdf.index, method="ffill")
        vdf["atr"] = time_atr.values

        all_pair_data[symbol] = vdf
        vol_atr = (vdf["atr"] / vdf["close"] * 100).mean()
        console.print(f"  {symbol}: {len(vdf)} vol bars (ATR from time bars: {vol_atr:.3f}%)")

    if not all_pair_data:
        console.print("[red]No data![/red]")
        return

    # Compute ratio z-scores vs BTC and ETH (pairs trading signal as feature)
    RATIO_LB = 96
    ratio_zscores = {}
    btc_sym, eth_sym = "BTC/USDT", "ETH/USDT"
    if btc_sym in all_pair_data and eth_sym in all_pair_data:
        btc_c = all_pair_data[btc_sym]["close"]
        eth_c = all_pair_data[eth_sym]["close"]
        for symbol, vdf in all_pair_data.items():
            sc = vdf["close"]
            rf = pd.DataFrame(index=vdf.index)
            for ref_sym, ref_c, col in [(btc_sym, btc_c, "ratio_vs_btc_z"), (eth_sym, eth_c, "ratio_vs_eth_z")]:
                if symbol == ref_sym:
                    rf[col] = 0.0
                else:
                    r = sc / ref_c.reindex(vdf.index, method="ffill")
                    rm = r.rolling(RATIO_LB, min_periods=20).mean()
                    rs = r.rolling(RATIO_LB, min_periods=20).std().replace(0, 1)
                    rf[col] = ((r - rm) / rs).replace([np.inf, -np.inf], 0).fillna(0)
            ratio_zscores[symbol] = rf
        console.print(f"  Ratio z-scores computed (pairs trading signal as features)")

    # Walk-forward on volume bars
    all_trades = []
    fold = 0
    start = 0
    # Global direction lock: only ONE direction locked at a time, never both
    global_locked_dir = None  # None, "LONG", or "SHORT"
    global_sl_streak = {"LONG": 0, "SHORT": 0}

    min_len = min(len(df) for df in all_pair_data.values())

    while start + train_bars + test_bars <= min_len:
        fold += 1
        train_end = start + train_bars
        test_end = train_end + test_bars

        # === TRAIN ===
        train_data = {
            "long_wt": {"X": [], "y": []}, "long_at": {"X": [], "y": []},
            "short_wt": {"X": [], "y": []}, "short_at": {"X": [], "y": []},
        }
        feat_names_refs = {}

        for symbol, df_full in all_pair_data.items():
            if len(df_full) < test_end:
                continue
            df_train = df_full.iloc[start:train_end].copy()
            if len(df_train) < 200:
                continue

            feat = fe.create_features(df_train)
            cols = fe.get_feature_columns(feat, model_type="trend")
            X = feat[cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            X["pair_id"] = pair_to_id.get(symbol, 0)
            if symbol in ratio_zscores:
                rz = ratio_zscores[symbol].iloc[start:train_end].reindex(X.index, method="ffill").fillna(0)
                X["ratio_vs_btc_z"] = rz["ratio_vs_btc_z"].values
                X["ratio_vs_eth_z"] = rz["ratio_vs_eth_z"].values
            else:
                X["ratio_vs_btc_z"] = 0.0
                X["ratio_vs_eth_z"] = 0.0
            feat_names_refs["trend"] = list(cols) + ["pair_id", "ratio_vs_btc_z", "ratio_vs_eth_z"]

            close = df_train["close"].values
            high = df_train["high"].values
            low = df_train["low"].values
            atr_vals = df_train["atr"].values
            roc_12 = df_train["roc_12"].values if "roc_12" in df_train.columns else np.zeros(len(df_train))

            labels = pd.Series(-99, index=df_train.index)
            max_bars = 12

            for i in range(len(df_train) - max_bars):
                a = atr_vals[i]
                if np.isnan(a) or a <= 0:
                    continue
                r = roc_12[i]
                if np.isnan(r):
                    continue
                if r > long_mom:
                    direction = "LONG"
                elif r < -short_mom:
                    direction = "SHORT"
                else:
                    continue

                tp_p = close[i] + a * tp_mult if direction == "LONG" else close[i] - a * tp_mult
                sl_p = close[i] - a * sl_mult if direction == "LONG" else close[i] + a * sl_mult
                result = -99
                for j in range(1, max_bars + 1):
                    idx = i + j
                    if idx >= len(df_train):
                        break
                    if direction == "LONG":
                        if high[idx] >= tp_p:
                            result = 1; break
                        if low[idx] <= sl_p:
                            result = 0; break
                    else:
                        if low[idx] <= tp_p:
                            result = 1; break
                        if high[idx] >= sl_p:
                            result = 0; break
                labels.iloc[i] = result

            ema50 = df_train["ema_50"].values if "ema_50" in df_train.columns else close
            is_uptrend = close > ema50

            for i_row in range(len(labels)):
                lbl = labels.iloc[i_row]
                if lbl == -99:
                    continue
                r = roc_12[i_row]
                if np.isnan(r):
                    continue
                up = is_uptrend[i_row]
                y_val = {0: -1, 1: 1}[lbl]
                if r > long_mom and up:
                    key = "long_wt"
                elif r > long_mom and not up:
                    key = "long_at"
                elif r < -short_mom and not up:
                    key = "short_wt"
                elif r < -short_mom and up:
                    key = "short_at"
                else:
                    continue
                train_data[key]["X"].append(X.iloc[[i_row]])
                train_data[key]["y"].append(y_val)

        models = {}
        if conf_threshold > 0:
            for dk in ["long_wt", "long_at", "short_wt", "short_at"]:
                xl = train_data[dk]["X"]
                yl = train_data[dk]["y"]
                if not xl or len(xl) < 50:
                    continue
                X_tr = pd.concat(xl, ignore_index=True)
                y_tr = pd.Series(yl)
                fn = feat_names_refs["trend"]
                n_good = int((y_tr == 1).sum())
                good_pct = n_good * 100 // len(y_tr)
                tmp_path = os.path.join(tempfile.gettempdir(), f"volbar_{dk}_f{fold}.json")
                m = MLSignalModel(tmp_path)
                sw = fe.compute_sample_weights(y_tr)
                m.train(X_tr, y_tr, fn, sample_weights=sw)
                models[dk] = m

        # === TEST ===
        fold_trades = []
        for symbol, df_full in all_pair_data.items():
            if len(df_full) < test_end:
                continue

            df_with_lookback = df_full.iloc[:test_end].copy()
            feat_all = fe.create_features(df_with_lookback)
            cols = fe.get_feature_columns(feat_all, model_type="trend")
            X_all = feat_all[cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            X_all["pair_id"] = pair_to_id.get(symbol, 0)
            if symbol in ratio_zscores:
                rz = ratio_zscores[symbol].reindex(feat_all.index, method="ffill").fillna(0)
                X_all["ratio_vs_btc_z"] = rz["ratio_vs_btc_z"].values
                X_all["ratio_vs_eth_z"] = rz["ratio_vs_eth_z"].values
            else:
                X_all["ratio_vs_btc_z"] = 0.0
                X_all["ratio_vs_eth_z"] = 0.0

            X_test = X_all.iloc[train_end:test_end]
            df_test = df_with_lookback.iloc[train_end:test_end]
            if len(X_test) < 20:
                continue

            ema50_test = df_test["ema_50"].values if "ema_50" in df_test.columns else np.full(len(df_test), 0)
            close_test = df_test["close"].values
            atr_test = df_test["atr"].values if "atr" in df_test.columns else np.ones(len(df_test))
            atr_ma20 = pd.Series(atr_test).rolling(20, min_periods=1).mean().values

            signals = []
            for j in range(len(X_test)):
                roc = float(df_test["roc_12"].iloc[j]) if "roc_12" in df_test.columns else 0
                if roc > long_mom:
                    direction = "LONG"
                elif roc < -short_mom:
                    direction = "SHORT"
                else:
                    direction = "NEUTRAL"

                conf = 0
                if direction != "NEUTRAL":
                    # Global direction lock (only one dir at a time)
                    if global_locked_dir == direction:
                        direction = "NEUTRAL"; continue

                    # HTF volume bars trend filter
                    htf_e9 = float(df_test["htf_ema_9"].iloc[j]) if "htf_ema_9" in df_test.columns else 0
                    htf_e21 = float(df_test["htf_ema_21"].iloc[j]) if "htf_ema_21" in df_test.columns else 0
                    if htf_e9 > 0 and htf_e21 > 0:
                        htf_downtrend = htf_e9 < htf_e21
                        htf_uptrend = htf_e9 > htf_e21
                        if direction == "LONG" and htf_downtrend:
                            direction = "NEUTRAL"; continue
                        if direction == "SHORT" and htf_uptrend:
                            direction = "NEUTRAL"; continue

                    # Guard: volatility filter
                    atr_exp = atr_test[j] / atr_ma20[j] if atr_ma20[j] > 0 else 1.0
                    if atr_exp > 1.5:
                        direction = "NEUTRAL"; continue

                    # Guard: rsi_slope (momentum alive?)
                    rsi_s6 = float(df_test["rsi"].diff(6).iloc[j]) if "rsi" in df_test.columns else 0
                    if direction == "LONG" and rsi_s6 < 0:
                        direction = "NEUTRAL"; continue
                    if direction == "SHORT" and rsi_s6 > 0:
                        direction = "NEUTRAL"; continue

                    e50 = ema50_test[j] if j < len(ema50_test) else 0
                    trend_up = close_test[j] > e50 if e50 > 0 else True
                    with_trend = (direction == "LONG" and trend_up) or (direction == "SHORT" and not trend_up)
                    model_key = ("long_wt" if direction == "LONG" else "short_wt") if with_trend else ("long_at" if direction == "LONG" else "short_at")
                    X_row = X_test.iloc[[j]]

                    if conf_threshold <= 0:
                        conf = 0.90
                    elif model_key in models:
                        pred = models[model_key].predict(X_row)
                        ml_signal = pred.get("signal", 0)
                        ml_conf = pred.get("confidence", 0)
                        if ml_signal == 1 and ml_conf >= conf_threshold:
                            conf = ml_conf
                        else:
                            direction = "NEUTRAL"
                    else:
                        direction = "NEUTRAL"

                signals.append({
                    "bar_idx": j, "symbol": symbol,
                    "direction": direction, "confidence": conf,
                })

            if use_dca:
                trades = simulate_dca_trades(
                    df_test, signals,
                    tp_mult=tp_mult, dca_step_mult=1.0,
                    max_entries=3, hard_sl_mult=sl_mult,
                    max_hold=24, max_open=config.MAX_OPEN_TRADES,
                    cooldown=3, threshold=0.10 if conf_threshold <= 0 else conf_threshold,
                    full_size_dca=True,
                )
            else:
                trades = simulate_trades(
                    df_test, signals,
                    sl_mult=sl_mult, tp_mult=tp_mult,
                    max_hold=24, max_open=config.MAX_OPEN_TRADES,
                    cooldown=3, threshold=0.10 if conf_threshold <= 0 else conf_threshold,
                )
            fold_trades.extend(trades)

        if fold_trades:
            pnl = sum(t.pnl_pct for t in fold_trades)
            wins = sum(1 for t in fold_trades if t.pnl_pct > 0)
            wr = wins / len(fold_trades) * 100
            console.print(f"  Fold {fold}: {len(fold_trades)} trades, WR {wr:.0f}%, PnL {pnl:+.2f}%")

        # Update global direction lock (only one direction locked at a time)
        for tr in sorted(fold_trades, key=lambda x: getattr(x, 'exit_bar', getattr(x, 'first_bar', 0))):
            d = tr.direction
            if tr.exit_reason in ("SL", "HARD_SL"):
                global_sl_streak[d] += 1
                if global_sl_streak[d] >= 2:
                    global_locked_dir = d  # lock THIS direction
            elif tr.exit_reason == "TP":
                global_sl_streak[d] = 0
                if global_locked_dir == d:
                    global_locked_dir = None  # unlock if TP in locked direction

        all_trades.extend(fold_trades)
        start += test_bars

    # Summary
    console.print("\n" + "=" * 60)
    if not all_trades:
        console.print("[red]No trades![/red]")
        return

    total_pnl = sum(t.pnl_pct for t in all_trades)
    total_count = len(all_trades)
    total_wr = sum(1 for t in all_trades if t.pnl_pct > 0) / total_count * 100

    long_trades = [t for t in all_trades if t.direction == "LONG"]
    short_trades = [t for t in all_trades if t.direction == "SHORT"]
    long_pnl = sum(t.pnl_pct for t in long_trades)
    short_pnl = sum(t.pnl_pct for t in short_trades)
    long_wr = sum(1 for t in long_trades if t.pnl_pct > 0) / max(len(long_trades), 1) * 100
    short_wr = sum(1 for t in short_trades if t.pnl_pct > 0) / max(len(short_trades), 1) * 100

    tp_count = sum(1 for t in all_trades if t.exit_reason == "TP")
    sl_count = sum(1 for t in all_trades if t.exit_reason == "SL")
    to_count = sum(1 for t in all_trades if t.exit_reason in ("TIMEOUT", "END"))

    # Fee calculation ($1000 capital, 5x leverage, maker 0.02%)
    position_size = 1000 * 5 / 11
    fee_per_trade = 2 * position_size * 0.0002  # open + close, maker fee
    total_fees = fee_per_trade * total_count
    gross_profit = position_size * total_pnl / 100
    net_profit = gross_profit - total_fees
    avg_gross = position_size * (total_pnl / total_count) / 100
    avg_net = avg_gross - fee_per_trade

    table = Table(title="Volume Bars ML Results", box=box.ROUNDED)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Total trades", str(total_count))
    table.add_row("Win rate", f"{total_wr:.1f}%")
    table.add_row("Raw PnL", f"{total_pnl:+.2f}%")
    table.add_row("PnL/month", f"{total_pnl / total_days * 30:+.2f}%")
    table.add_row("", "")
    table.add_row("LONG", f"{len(long_trades)} (WR {long_wr:.0f}%, PnL {long_pnl:+.2f}%)")
    table.add_row("SHORT", f"{len(short_trades)} (WR {short_wr:.0f}%, PnL {short_pnl:+.2f}%)")
    table.add_row("", "")
    table.add_row("TP / SL / TO", f"{tp_count} / {sl_count} / {to_count}")
    table.add_row("Avg PnL/trade", f"{total_pnl / total_count:+.4f}%")
    table.add_row("", "")
    table.add_row("[bold]FEE ANALYSIS ($1K×5x)[/bold]", "")
    table.add_row("Position/pair", f"${position_size:.0f}")
    table.add_row("Fee/trade", f"${fee_per_trade:.2f}")
    table.add_row("Total fees", f"${total_fees:.0f}")
    table.add_row("Gross profit", f"${gross_profit:+.0f}")
    table.add_row("NET profit", f"${net_profit:+.0f}")
    table.add_row("Avg net/trade", f"${avg_net:+.2f}")
    table.add_row("", "")
    table.add_row("Days", str(total_days))
    table.add_row("Folds", str(fold))
    console.print(table)

    symbols = sorted(set(t.symbol for t in all_trades))
    sym_table = Table(title="Per-Symbol", box=box.SIMPLE)
    sym_table.add_column("Symbol")
    sym_table.add_column("Trades", justify="right")
    sym_table.add_column("WR", justify="right")
    sym_table.add_column("PnL", justify="right")
    for s in symbols:
        s_trades = [t for t in all_trades if t.symbol == s]
        s_pnl = sum(t.pnl_pct for t in s_trades)
        s_wr = sum(1 for t in s_trades if t.pnl_pct > 0) / max(len(s_trades), 1) * 100
        sym_table.add_row(s, str(len(s_trades)), f"{s_wr:.0f}%", f"{s_pnl:+.2f}%")
    console.print(sym_table)
    return all_trades


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=187)
    parser.add_argument("--train-bars", type=int, default=1000)
    parser.add_argument("--test-bars", type=int, default=300)
    parser.add_argument("--conf", type=float, default=0.0, help="0=no ML, >0=ML filter")
    parser.add_argument("--dca", action="store_true", help="Use DCA (1/3 sizing per entry)")
    args = parser.parse_args()
    run_volbars_backtest(
        total_days=args.days, train_bars=args.train_bars,
        test_bars=args.test_bars, conf_threshold=args.conf,
        use_dca=args.dca,
    )
