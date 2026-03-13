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
    threshold: float = None,
    pair_cooldown_sl: int = 2,
    pair_cooldown_bars: int = 8,
) -> List[SimTrade]:
    """Simulate trading with SL/TP checked against high/low each bar."""
    trades: List[SimTrade] = []
    open_trades: List[SimTrade] = []
    cooldowns: dict = {}
    sl_streaks: dict = {}
    pair_dir_cooldowns: dict = {}

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

            # Tighten SL after price confirms direction (+0.5x ATR)
            entry_atr = atr[t.entry_bar] if t.entry_bar < len(atr) and not np.isnan(atr[t.entry_bar]) else 0
            if entry_atr > 0:
                if t.direction == "LONG" and high[bar_idx] >= t.entry_price + 0.5 * entry_atr:
                    new_sl = t.entry_price - 1.0 * entry_atr
                    t.sl = max(t.sl, new_sl)
                elif t.direction == "SHORT" and low[bar_idx] <= t.entry_price - 0.5 * entry_atr:
                    new_sl = t.entry_price + 1.0 * entry_atr
                    t.sl = min(t.sl, new_sl)

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

                key = (t.symbol, t.direction)
                if t.exit_reason == "SL":
                    sl_streaks[key] = sl_streaks.get(key, 0) + 1
                    if sl_streaks[key] >= pair_cooldown_sl:
                        pair_dir_cooldowns[key] = bar_idx + pair_cooldown_bars
                else:
                    sl_streaks[key] = 0

        # Try to open new trade
        direction = sig.get("direction")
        conf = sig.get("confidence", 0)
        symbol = sig.get("symbol", "")

        if direction not in ("LONG", "SHORT"):
            continue
        thresh = threshold if threshold is not None else config.PREDICTION_THRESHOLD
        if conf < thresh:
            continue
        if len(open_trades) >= max_open:
            continue
        if cooldowns.get(symbol, 0) > bar_idx:
            continue
        if pair_dir_cooldowns.get((symbol, direction), 0) > bar_idx:
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


@dataclass
class DcaTrade:
    symbol: str
    direction: str
    entries: list
    avg_price: float
    total_size: int
    max_entries: int
    hard_sl: float
    tp: float
    first_bar: int
    exit_bar: int = -1
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl_pct: float = 0.0


def simulate_dca_trades(
    df: pd.DataFrame,
    signals: List[dict],
    tp_mult: float = 1.5,
    dca_step_mult: float = 1.0,
    max_entries: int = 3,
    hard_sl_mult: float = 3.5,
    max_hold: int = 36,
    max_open: int = 2,
    cooldown: int = 3,
    threshold: float = 0.10,
    pair_cooldown_sl: int = 2,
    pair_cooldown_bars: int = 8,
    full_size_dca: bool = False,
    early_exit: str = "",
    trail_atr: float = 0,
    trail_activate: float = 0,
    move_sl_at: float = 2.0,
    move_sl_to: float = 1.0,
    move_sl_steps: list = None,
    move_sl_trail: float = 0,
) -> List[DcaTrade]:
    """DCA v2: 5 improvements to reduce HARD_SL losses."""
    trades: List[DcaTrade] = []
    open_positions: List[DcaTrade] = []
    cooldowns: dict = {}
    sl_streaks: dict = {}
    pair_dir_cooldowns: dict = {}

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    vol = df["volume"].values
    atr = df["atr"].values
    roc12 = df["roc_12"].values if "roc_12" in df.columns else np.zeros(len(df))
    adx = df["ADX_14"].values if "ADX_14" in df.columns else np.full(len(df), 25.0)
    # Fix 1: ATR expansion for volatility-scaled SL/step
    atr_ma20 = pd.Series(atr).rolling(20, min_periods=1).mean().values
    vol_ma20 = pd.Series(vol).rolling(20, min_periods=1).mean().values

    # OBV (On Balance Volume)
    obv = np.zeros(len(close))
    for k in range(1, len(close)):
        if close[k] > close[k-1]:
            obv[k] = obv[k-1] + vol[k]
        elif close[k] < close[k-1]:
            obv[k] = obv[k-1] - vol[k]
        else:
            obv[k] = obv[k-1]

    ee_vol_drop = True  # always on (like live bot)
    ee_obv_div = "obv_div" in early_exit
    ee_vol_dry = "vol_dry" in early_exit

    for i in range(len(signals)):
        sig = signals[i]
        bar_idx = sig["bar_idx"]
        if bar_idx >= len(close):
            break

        for pos in list(open_positions):
            bars_held = bar_idx - pos.first_bar
            if bars_held <= 0:
                continue

            entry_atr = atr[pos.first_bar]
            if np.isnan(entry_atr) or entry_atr <= 0:
                entry_atr = abs(pos.entries[0][0]) * 0.01

            # Fix 1: volatility-scaled DCA step (wider in high vol)
            cur_atr = atr[bar_idx] if bar_idx < len(atr) and not np.isnan(atr[bar_idx]) else entry_atr
            atr_exp = cur_atr / atr_ma20[bar_idx] if bar_idx < len(atr_ma20) and atr_ma20[bar_idx] > 0 else 1.0
            vol_scale = max(1.0, atr_exp)
            effective_step = dca_step_mult * vol_scale

            # Fix 2: dynamic max_entries by ADX
            cur_adx = adx[bar_idx] if bar_idx < len(adx) and not np.isnan(adx[bar_idx]) else 25.0
            if cur_adx >= 30:
                dyn_max = 3
            elif cur_adx >= 20:
                dyn_max = 2
            else:
                dyn_max = 1  # no DCA in weak trend

            # Fix 3: stricter roc_ok (must be >= 0, not >= -0.2)
            if pos.total_size < min(pos.max_entries, dyn_max):
                roc_ok = True
                roc_val = roc12[bar_idx] if bar_idx < len(roc12) else 0
                if pos.direction == "LONG" and roc_val < 0:
                    roc_ok = False
                elif pos.direction == "SHORT" and roc_val > 0:
                    roc_ok = False

                next_step = pos.total_size
                if pos.direction == "LONG":
                    add_level = pos.entries[0][0] - next_step * effective_step * entry_atr
                    if low[bar_idx] <= add_level and roc_ok:
                        pos.entries.append((add_level, bar_idx))
                        pos.total_size += 1
                        pos.avg_price = sum(e[0] for e in pos.entries) / pos.total_size
                        pos.tp = pos.avg_price + tp_mult * entry_atr
                else:
                    add_level = pos.entries[0][0] + next_step * effective_step * entry_atr
                    if high[bar_idx] >= add_level and roc_ok:
                        pos.entries.append((add_level, bar_idx))
                        pos.total_size += 1
                        pos.avg_price = sum(e[0] for e in pos.entries) / pos.total_size
                        pos.tp = pos.avg_price - tp_mult * entry_atr

            # Early exit checks (before SL/TP)
            early_closed = False
            if bars_held >= 3:
                # Volume Drop: avg vol last 3 bars < 50% of MA20
                if ee_vol_drop and bar_idx >= 2:
                    vol_avg3 = vol[bar_idx-2:bar_idx+1].mean() if bar_idx >= 2 else vol[bar_idx]
                    vol_ma = vol_ma20[bar_idx] if bar_idx < len(vol_ma20) and vol_ma20[bar_idx] > 0 else 1
                    if vol_avg3 < vol_ma * 0.5:
                        pos.exit_price = close[bar_idx]
                        pos.exit_reason = "VOL_DROP"
                        early_closed = True

                # OBV Divergence: price up but OBV down (LONG) or vice versa
                if ee_obv_div and not early_closed and bar_idx >= 3:
                    price_up = close[bar_idx] > close[bar_idx-3]
                    obv_up = obv[bar_idx] > obv[bar_idx-3]
                    if pos.direction == "LONG" and price_up and not obv_up:
                        pos.exit_price = close[bar_idx]
                        pos.exit_reason = "OBV_DIV"
                        early_closed = True
                    elif pos.direction == "SHORT" and not price_up and obv_up:
                        pos.exit_price = close[bar_idx]
                        pos.exit_reason = "OBV_DIV"
                        early_closed = True

                # Volume Dry-Up: current bar volume < 30% of MA20
                if ee_vol_dry and not early_closed:
                    vol_ma = vol_ma20[bar_idx] if bar_idx < len(vol_ma20) and vol_ma20[bar_idx] > 0 else 1
                    if vol[bar_idx] < vol_ma * 0.3:
                        pos.exit_price = close[bar_idx]
                        pos.exit_reason = "VOL_DRY"
                        early_closed = True

            # Fix 1 continued: volatility-scaled hard SL
            effective_hard_sl_dist = hard_sl_mult * vol_scale * entry_atr
            if pos.direction == "LONG":
                dynamic_hard_sl = pos.entries[0][0] - effective_hard_sl_dist
            else:
                dynamic_hard_sl = pos.entries[0][0] + effective_hard_sl_dist

            hit_tp = hit_sl = False
            if pos.direction == "LONG":
                hit_tp = high[bar_idx] >= pos.tp
                hit_sl = low[bar_idx] <= min(pos.hard_sl, dynamic_hard_sl)
            else:
                hit_tp = low[bar_idx] <= pos.tp
                hit_sl = high[bar_idx] >= max(pos.hard_sl, dynamic_hard_sl)

            # Trailing stop (legacy, off by default)
            trail_hit = False
            if trail_atr > 0 and trail_activate > 0:
                if not hasattr(pos, '_best'):
                    pos._best = pos.entries[0][0]
                if pos.direction == "LONG":
                    pos._best = max(pos._best, high[bar_idx])
                    trail_profit = pos._best - pos.avg_price
                    if trail_profit >= trail_activate * entry_atr:
                        trail_sl = pos._best - trail_atr * entry_atr
                        if low[bar_idx] <= trail_sl:
                            trail_hit = True
                            trail_exit_price = trail_sl
                else:
                    pos._best = min(pos._best, low[bar_idx])
                    trail_profit = pos.avg_price - pos._best
                    if trail_profit >= trail_activate * entry_atr:
                        trail_sl = pos._best + trail_atr * entry_atr
                        if high[bar_idx] >= trail_sl:
                            trail_hit = True
                            trail_exit_price = trail_sl

            # Move SL: single or multi-step, then optional trail
            if not hasattr(pos, '_sl_step'):
                pos._sl_step = 0
            if not hasattr(pos, '_best_price'):
                pos._best_price = pos.avg_price
            if move_sl_at > 0:
                steps = move_sl_steps if move_sl_steps else [(move_sl_at, move_sl_to)]
                if pos._sl_step < len(steps):
                    step_at, step_to = steps[pos._sl_step]
                    if pos.direction == "LONG":
                        if high[bar_idx] - pos.avg_price >= step_at * entry_atr:
                            pos.hard_sl = pos.avg_price + step_to * entry_atr
                            pos._sl_step += 1
                    else:
                        if pos.avg_price - low[bar_idx] >= step_at * entry_atr:
                            pos.hard_sl = pos.avg_price - step_to * entry_atr
                            pos._sl_step += 1
                elif move_sl_trail > 0 and pos._sl_step >= len(steps):
                    if pos.direction == "LONG":
                        pos._best_price = max(pos._best_price, high[bar_idx])
                        new_sl = pos._best_price - move_sl_trail * entry_atr
                        if new_sl > pos.hard_sl:
                            pos.hard_sl = new_sl
                    else:
                        pos._best_price = min(pos._best_price, low[bar_idx])
                        new_sl = pos._best_price + move_sl_trail * entry_atr
                        if new_sl < pos.hard_sl:
                            pos.hard_sl = new_sl

            # Fix 4: DCA timeout — close DCA positions after 24 bars
            dca_timeout = pos.total_size > 1 and bars_held >= 24

            closed = False
            if hit_sl and hit_tp:
                pos.exit_price = pos.hard_sl
                pos.exit_reason = "HARD_SL"
                closed = True
            elif hit_sl:
                pos.exit_price = pos.hard_sl
                pos.exit_reason = "HARD_SL"
                closed = True
            elif hit_tp:
                pos.exit_price = pos.tp
                pos.exit_reason = "TP"
                closed = True
            elif trail_hit:
                pos.exit_price = trail_exit_price
                pos.exit_reason = "TRAIL"
                closed = True
            elif early_closed:
                closed = True
            elif dca_timeout:
                pos.exit_price = close[bar_idx]
                pos.exit_reason = "DCA_TIMEOUT"
                closed = True
            elif bars_held >= max_hold:
                pos.exit_price = close[bar_idx]
                pos.exit_reason = "TIMEOUT"
                closed = True

            if closed:
                pos.exit_bar = bar_idx
                size_mult = pos.total_size if full_size_dca else pos.total_size / max_entries
                if pos.direction == "LONG":
                    pos.pnl_pct = (pos.exit_price - pos.avg_price) / pos.avg_price * 100 * size_mult
                else:
                    pos.pnl_pct = (pos.avg_price - pos.exit_price) / pos.avg_price * 100 * size_mult
                open_positions.remove(pos)
                trades.append(pos)
                cooldowns[pos.symbol] = bar_idx + cooldown

                key = (pos.symbol, pos.direction)
                if pos.exit_reason == "HARD_SL":
                    sl_streaks[key] = sl_streaks.get(key, 0) + 1
                    if sl_streaks[key] >= pair_cooldown_sl:
                        pair_dir_cooldowns[key] = bar_idx + pair_cooldown_bars
                else:
                    sl_streaks[key] = 0

        direction = sig.get("direction")
        conf = sig.get("confidence", 0)
        symbol = sig.get("symbol", "")

        if direction not in ("LONG", "SHORT"):
            continue
        if conf < threshold:
            continue
        if len(open_positions) >= max_open:
            continue
        if cooldowns.get(symbol, 0) > bar_idx:
            continue
        if pair_dir_cooldowns.get((symbol, direction), 0) > bar_idx:
            continue
        existing = [p for p in open_positions if p.symbol == symbol]
        if existing:
            ex = existing[0]
            if ex.direction == direction:
                continue
            # Flip: close opposite position, then open new
            ex.exit_bar = bar_idx
            ex.exit_price = close[bar_idx]
            ex.exit_reason = "FLIP"
            size_mult = ex.total_size if full_size_dca else ex.total_size / max_entries
            if ex.direction == "LONG":
                ex.pnl_pct = (ex.exit_price - ex.avg_price) / ex.avg_price * 100 * size_mult
            else:
                ex.pnl_pct = (ex.avg_price - ex.exit_price) / ex.avg_price * 100 * size_mult
            open_positions.remove(ex)
            trades.append(ex)
            cooldowns[ex.symbol] = bar_idx + cooldown
            key = (ex.symbol, ex.direction)
            if ex.pnl_pct < 0:
                sl_streaks[key] = sl_streaks.get(key, 0) + 1
                if sl_streaks[key] >= pair_cooldown_sl:
                    pair_dir_cooldowns[key] = bar_idx + pair_cooldown_bars
            else:
                sl_streaks[key] = 0

        a = atr[bar_idx]
        if np.isnan(a) or a <= 0:
            continue

        price = close[bar_idx]
        if direction == "LONG":
            hard_sl = price - hard_sl_mult * a
            tp_price = price + tp_mult * a
        else:
            hard_sl = price + hard_sl_mult * a
            tp_price = price - tp_mult * a

        pos = DcaTrade(
            symbol=symbol, direction=direction,
            entries=[(price, bar_idx)],
            avg_price=price, total_size=1,
            max_entries=max_entries,
            hard_sl=hard_sl, tp=tp_price,
            first_bar=bar_idx,
        )
        open_positions.append(pos)

    for pos in open_positions:
        pos.exit_bar = len(close) - 1
        pos.exit_price = close[-1]
        pos.exit_reason = "END"
        size_mult = pos.total_size if full_size_dca else pos.total_size / max_entries
        if pos.direction == "LONG":
            pos.pnl_pct = (pos.exit_price - pos.avg_price) / pos.avg_price * 100 * size_mult
        else:
            pos.pnl_pct = (pos.avg_price - pos.exit_price) / pos.avg_price * 100 * size_mult
        trades.append(pos)

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


def run_filter15m(
    total_days: int = 365,
    train_days: int = 30,
    test_days: int = 5,
    sl_mult: float = 2.0,
    tp_mult: float = 2.0,
    long_mom: float = 0.30,
    short_mom: float = 0.10,
    conf_threshold: float = 0.55,
    pairs: list = None,
    use_dca: bool = False,
):
    """
    15m ML FILTER strategy: momentum for direction, XGBoost filters bad entries.

    Label: for each momentum signal, did the trade hit TP (1) or SL (0)?
    Model learns: "is this a good momentum setup?" not "which direction?"
    """
    import tempfile, os

    BARS_15M = 96
    total_candles = total_days * BARS_15M
    train_candles = train_days * BARS_15M
    test_candles = test_days * BARS_15M

    _all_pairs = (pairs if pairs else config.TRADING_PAIRS)
    # Exclude consistently worst pair
    trading_pairs_list = list(_all_pairs)  # all 11 pairs
    pair_to_id = {sym: i for i, sym in enumerate(sorted(trading_pairs_list))}

    console.print(Panel(
        f"[bold cyan]15m ML Filter Strategy[/bold cyan]\n"
        f"[bold]ML filters bad momentum entries (not predicting direction)[/bold]\n"
        f"Total: {total_days}d | Train: {train_days}d | Test: {test_days}d\n"
        f"SL={sl_mult}x TP={tp_mult}x | Conf>{conf_threshold}\n"
        f"LONG>{long_mom}% SHORT>{short_mom}%",
        box=box.DOUBLE,
    ))

    fetcher = BinanceDataFetcher(config)
    indicators = TechnicalIndicators()
    fe = FeatureEngineer()
    sig_gen = SignalGenerator(config)

    trading_pairs = trading_pairs_list
    console.print(f"\n[cyan]Fetching 15m data for {len(trading_pairs)} pairs...[/cyan]")
    all_pair_data = {}
    for symbol in trading_pairs:
        df = fetcher.fetch_ohlcv_extended(symbol, "15m", total_candles=total_candles)
        if df.empty or len(df) < 200:
            continue
        df = indicators.calculate_all(df)
        all_pair_data[symbol] = df
        console.print(f"  {symbol}: {len(df)} bars")

    if not all_pair_data:
        console.print("[red]No data![/red]")
        return

    # Walk-forward
    all_trades: List[SimTrade] = []
    fold = 0
    start = 0

    while start + train_candles + test_candles <= total_candles:
        fold += 1
        train_end = start + train_candles
        test_end = train_end + test_candles

        console.print(f"\n[bold]Fold {fold}[/bold]: train [{start}:{train_end}], test [{train_end}:{test_end}]")

        # === TRAIN: create filter labels from momentum trades ===
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
            feat = fe.add_rolling_htf_features(feat)
            cols_trend = fe.get_feature_columns(feat, model_type="trend")
            cols_mr = fe.get_feature_columns(feat, model_type="mean_reversion")
            X_trend = feat[cols_trend].replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            X_mr = feat[cols_mr].replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            X_trend["pair_id"] = pair_to_id.get(symbol, 0)
            X_mr["pair_id"] = pair_to_id.get(symbol, 0)
            feat_names_refs["trend"] = list(cols_trend) + ["pair_id"]
            feat_names_refs["mr"] = list(cols_mr) + ["pair_id"]

            # Create filter labels: for each bar with momentum signal,
            # did the momentum trade hit TP (1) or SL/timeout (0)?
            close = df_train["close"].values
            high = df_train["high"].values
            low = df_train["low"].values
            atr_vals = df_train["atr"].values
            roc_12 = df_train["roc_12"].values if "roc_12" in df_train.columns else np.zeros(len(df_train))

            labels = pd.Series(-99, index=df_train.index)  # -99 = no momentum signal
            max_bars = 12

            for i in range(len(df_train) - max_bars):
                a = atr_vals[i]
                if np.isnan(a) or a <= 0:
                    continue
                r = roc_12[i]
                if np.isnan(r):
                    continue

                # Only label bars where momentum gives a signal
                if r > long_mom:
                    direction = "LONG"
                elif r < -short_mom:
                    direction = "SHORT"
                else:
                    continue  # no signal, skip

                # Did this trade hit TP or SL?
                tp_price = close[i] + a * tp_mult if direction == "LONG" else close[i] - a * tp_mult
                sl_price = close[i] - a * sl_mult if direction == "LONG" else close[i] + a * sl_mult

                result = -99  # default: TIMEOUT = skip (don't train on unclear outcomes)
                for j in range(1, max_bars + 1):
                    idx = i + j
                    if idx >= len(df_train):
                        break
                    if direction == "LONG":
                        if high[idx] >= tp_price:
                            result = 1  # TP hit = good trade
                            break
                        if low[idx] <= sl_price:
                            result = 0  # SL hit = bad trade
                            break
                    else:
                        if low[idx] <= tp_price:
                            result = 1
                            break
                        if high[idx] >= sl_price:
                            result = 0
                            break

                labels.iloc[i] = result

            # Split by direction AND trend alignment
            ema50 = df_train["ema_50"].values if "ema_50" in df_train.columns else close
            is_uptrend = close > ema50

            for i_row in range(len(labels)):
                if labels.iloc[i_row] == -99:
                    continue
                r = roc_12[i_row]
                if np.isnan(r):
                    continue
                up = is_uptrend[i_row]
                y_val = {0: -1, 1: 1}[labels.iloc[i_row]]

                if r > long_mom and up:
                    key, X_src = "long_wt", X_trend
                elif r > long_mom and not up:
                    key, X_src = "long_at", X_mr
                elif r < -short_mom and not up:
                    key, X_src = "short_wt", X_trend
                elif r < -short_mom and up:
                    key, X_src = "short_at", X_mr
                else:
                    continue

                if X_src.index[i_row] in X_src.index:
                    train_data[key]["X"].append(X_src.iloc[[i_row]])
                    train_data[key]["y"].append(y_val)

        # Train FOUR models (with/against trend × long/short)
        models = {}
        for direction_key in ["long_wt", "long_at", "short_wt", "short_at"]:
            x_list = train_data[direction_key]["X"]
            y_list = train_data[direction_key]["y"]
            if not x_list or len(x_list) < 20:
                continue

            X_tr = pd.concat(x_list, ignore_index=True)
            y_tr = pd.Series(y_list)
            if len(X_tr) < 50:
                continue

            fn = feat_names_refs["mr"] if "_at" in direction_key else feat_names_refs["trend"]
            n_good = int((y_tr == 1).sum())
            good_pct = n_good * 100 // len(y_tr)

            tmp_path = os.path.join(tempfile.gettempdir(), f"filter15m_{direction_key}_fold{fold}.json")
            m = MLSignalModel(tmp_path)
            sw = fe.compute_sample_weights(y_tr)
            m.train(X_tr, y_tr, fn, sample_weights=sw)
            models[direction_key] = m

            label = "MR" if "_at" in direction_key else "MOM"
            console.print(f"  {direction_key.upper()} [{label}]: {len(X_tr)} signals (GOOD:{n_good} [{good_pct}%])")

        if not models:
            start += test_candles
            continue

        # === TEST: momentum + ML filter ===
        fold_trades = []
        for symbol, df_full in all_pair_data.items():
            if len(df_full) < test_end:
                continue

            df_with_lookback = df_full.iloc[:test_end].copy()
            feat_all = fe.create_features(df_with_lookback)
            feat_all = fe.add_rolling_htf_features(feat_all)
            cols_trend = fe.get_feature_columns(feat_all, model_type="trend")
            cols_mr = fe.get_feature_columns(feat_all, model_type="mean_reversion")
            X_trend = feat_all[cols_trend].replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            X_mr = feat_all[cols_mr].replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            X_trend["pair_id"] = pair_to_id.get(symbol, 0)
            X_mr["pair_id"] = pair_to_id.get(symbol, 0)

            X_t = X_trend.iloc[train_end:test_end]
            X_m = X_mr.iloc[train_end:test_end]
            df_test = df_with_lookback.iloc[train_end:test_end]
            if len(X_t) < 20:
                continue

            ema50_test = df_test["ema_50"].values if "ema_50" in df_test.columns else np.full(len(df_test), 0)
            close_test = df_test["close"].values

            # Volatility regime: ATR expansion ratio
            atr_test = df_test["atr"].values if "atr" in df_test.columns else np.ones(len(df_test))
            atr_ma20 = pd.Series(atr_test).rolling(20, min_periods=1).mean().values

            signals = []
            for j in range(len(X_t)):
                roc = float(df_test["roc_12"].iloc[j]) if "roc_12" in df_test.columns else 0

                if roc > long_mom:
                    direction = "LONG"
                elif roc < -short_mom:
                    direction = "SHORT"
                else:
                    direction = "NEUTRAL"

                conf = 0
                if direction != "NEUTRAL":
                    # Volatility regime filter: high ATR expansion = whipsaw risk
                    atr_exp = atr_test[j] / atr_ma20[j] if atr_ma20[j] > 0 else 1.0
                    if atr_exp > 1.5:
                        direction = "NEUTRAL"
                        continue

                    # Universal momentum-alive check (rsi_slope<0 = dead)
                    # But for SHORT: flip logic (rsi falling = SHORT momentum alive)
                    rsi_sl_6 = float(df_test["rsi"].diff(6).iloc[j]) if "rsi" in df_test.columns else 0
                    if direction == "LONG" and rsi_sl_6 < 0:
                        direction = "NEUTRAL"
                        continue
                    elif direction == "SHORT" and rsi_sl_6 > 0:
                        direction = "NEUTRAL"
                        continue

                    # Volume guard for LONG
                    if direction == "LONG":
                        vol_r = float(df_test["volume_ratio"].iloc[j]) if "volume_ratio" in df_test.columns else 1
                        if vol_r > 1.3:
                            direction = "NEUTRAL"
                            continue

                    # Determine with/against trend
                    e50 = ema50_test[j] if j < len(ema50_test) else 0
                    trend_up = close_test[j] > e50 if e50 > 0 else True
                    with_trend = (direction == "LONG" and trend_up) or (direction == "SHORT" and not trend_up)

                    if with_trend:
                        model_key = "long_wt" if direction == "LONG" else "short_wt"
                        X_row = X_t.iloc[[j]]
                    else:
                        model_key = "long_at" if direction == "LONG" else "short_at"
                        X_row = X_m.iloc[[j]]

                    # OBV confirmation only for LONG
                    if direction == "LONG":
                        obv_chg = float(df_test["obv"].pct_change(6).iloc[j] * 100) if "obv" in df_test.columns else 5
                        if obv_chg < 1.0:
                            min_conf_adj = 0.70
                        else:
                            min_conf_adj = 0.65 if "_at" in model_key else conf_threshold
                    else:
                        min_conf_adj = 0.65 if "_at" in model_key else conf_threshold

                    if model_key in models:
                        pred = models[model_key].predict(X_row)
                        ml_signal = pred.get("signal", 0)
                        ml_conf = pred.get("confidence", 0)
                        if ml_signal == 1 and ml_conf >= min_conf_adj:
                            conf = ml_conf
                        else:
                            direction = "NEUTRAL"
                    else:
                        direction = "NEUTRAL"

                # Dead hours
                bar_time = df_test.index[j]
                bar_hour = bar_time.hour if hasattr(bar_time, 'hour') else 0
                dead_hours = config.FILTER_DEAD_HOURS
                if dead_hours and bar_hour in dead_hours:
                    direction = "NEUTRAL"
                    conf = 0

                # Exhaustion guard
                if direction != "NEUTRAL":
                    row = X_t.iloc[j]
                    guard_feats = {k: float(row.get(k, 0)) for k in [
                        "dist_from_high_96", "dist_from_low_96",
                        "dist_from_high_288", "dist_from_low_288",
                        "max_drawdown_48h", "roc_deceleration"]}
                    if sig_gen._exhaustion_guard(direction, guard_feats):
                        direction = "NEUTRAL"
                        conf = 0

                signals.append({
                    "bar_idx": j, "symbol": symbol,
                    "direction": direction, "confidence": conf,
                })

            if use_dca:
                # 3-level trend detection
                ema9_vals = df_test["ema_9"].values if "ema_9" in df_test.columns else np.full(len(df_test), 0)
                ema21_vals = df_test["ema_21"].values if "ema_21" in df_test.columns else np.full(len(df_test), 0)
                ema50_vals = df_test["ema_50"].values if "ema_50" in df_test.columns else np.full(len(df_test), 0)
                rsi_slope_vals = df_test["rsi"].diff(6).values if "rsi" in df_test.columns else np.zeros(len(df_test))

                with_trend_sigs = []
                weak_trend_sigs = []
                against_trend_sigs = []
                for s in signals:
                    idx = s["bar_idx"]
                    e9 = ema9_vals[idx] if idx < len(ema9_vals) else 0
                    e21 = ema21_vals[idx] if idx < len(ema21_vals) else 0
                    e50 = ema50_vals[idx] if idx < len(ema50_vals) else 0
                    d = s["direction"]
                    rs = rsi_slope_vals[idx] if idx < len(rsi_slope_vals) else 0

                    # 3 levels of trend
                    if e9 > 0 and e21 > 0 and e50 > 0:
                        ema_stack_bull = e9 > e21 > e50
                        ema_stack_bear = e9 < e21 < e50
                        ema_short_bull = e9 > e21
                        ema_short_bear = e9 < e21
                    else:
                        ema_stack_bull = ema_stack_bear = False
                        ema_short_bull = ema_short_bear = False

                    # STRONG trend: EMA9 > EMA21 > EMA50 aligned with direction
                    if (d == "LONG" and ema_stack_bull) or (d == "SHORT" and ema_stack_bear):
                        if (d == "LONG" and rs > 1.5) or (d == "SHORT" and rs < -1.5):
                            weak_trend_sigs.append(s)
                            with_trend_sigs.append({**s, "direction": "NEUTRAL"})
                            against_trend_sigs.append({**s, "direction": "NEUTRAL"})
                        else:
                            with_trend_sigs.append(s)
                            weak_trend_sigs.append({**s, "direction": "NEUTRAL"})
                            against_trend_sigs.append({**s, "direction": "NEUTRAL"})

                    # WEAK trend: EMA9 > EMA21 but NOT fully stacked (pullback)
                    elif (d == "LONG" and ema_short_bull) or (d == "SHORT" and ema_short_bear):
                        weak_trend_sigs.append(s)
                        with_trend_sigs.append({**s, "direction": "NEUTRAL"})
                        against_trend_sigs.append({**s, "direction": "NEUTRAL"})

                    # AGAINST trend: EMAs against direction
                    else:
                        # Against-trend: 4 filters to reduce AT losses (85% of all losses)
                        obv_vals = df_test["obv"].pct_change(6).values * 100 if "obv" in df_test.columns else np.zeros(len(df_test))
                        obv_v = obv_vals[idx] if idx < len(obv_vals) else 0

                        # #1 rsi_slope filter: TO has -0.37, TP has +1.04
                        at_ok = True
                        if rs < 0:
                            at_ok = False  # RSI falling = momentum dying

                        # #2 OBV filter: TO has -0.79, TP has +0.87
                        if obv_v < 0:
                            at_ok = False  # OBV not confirming

                        # #3 Higher conf for AT without confirmation
                        min_at_conf = 0.65
                        if not at_ok:
                            min_at_conf = 0.80  # very strict if signals don't confirm

                        # #4 HARD_SL volume filter: HSL has obv +5.42 (volume spike)
                        # Already handled by guard #2 for LONG, here just block AT on spike
                        vol_v = float(df_test["volume_ratio"].iloc[idx]) if "volume_ratio" in df_test.columns and idx < len(df_test) else 1
                        if vol_v > 1.5:
                            at_ok = False
                            min_at_conf = 0.80

                        if s["confidence"] >= min_at_conf:
                            against_trend_sigs.append(s)
                        else:
                            against_trend_sigs.append({**s, "direction": "NEUTRAL"})
                        with_trend_sigs.append({**s, "direction": "NEUTRAL"})
                        weak_trend_sigs.append({**s, "direction": "NEUTRAL"})

                # STRONG trend: DCA with max 3 entries
                dca_trades = simulate_dca_trades(
                    df_test, with_trend_sigs,
                    tp_mult=tp_mult, dca_step_mult=1.0,
                    max_entries=3, hard_sl_mult=sl_mult,
                    max_hold=36, max_open=config.MAX_OPEN_TRADES,
                    cooldown=3, threshold=conf_threshold,
                )
                # WEAK trend: split LONG (TP=1.5) and SHORT (TP=1.0 scalp)
                weak_long_sigs = [s if s["direction"]=="LONG" else {**s,"direction":"NEUTRAL"} for s in weak_trend_sigs]
                weak_short_sigs = [s if s["direction"]=="SHORT" else {**s,"direction":"NEUTRAL"} for s in weak_trend_sigs]
                weak_long = simulate_trades(
                    df_test, weak_long_sigs,
                    sl_mult=2.0, tp_mult=tp_mult,
                    max_hold=24, max_open=config.MAX_OPEN_TRADES,
                    cooldown=3, threshold=conf_threshold,
                )
                weak_short = simulate_trades(
                    df_test, weak_short_sigs,
                    sl_mult=2.0, tp_mult=1.0,
                    max_hold=12, max_open=config.MAX_OPEN_TRADES,
                    cooldown=3, threshold=conf_threshold,
                )
                weak_trades = weak_long + weak_short
                sl_trades = simulate_trades(
                    df_test, against_trend_sigs,
                    sl_mult=2.0, tp_mult=1.0,
                    max_hold=12, max_open=config.MAX_OPEN_TRADES,
                    cooldown=3, threshold=conf_threshold,
                )
                trades = dca_trades + weak_trades + sl_trades
            else:
                trades = simulate_trades(
                    df_test, signals,
                    sl_mult=sl_mult, tp_mult=tp_mult,
                    max_hold=12, max_open=config.MAX_OPEN_TRADES,
                    cooldown=3, threshold=conf_threshold,
                )
            fold_trades.extend(trades)

        if fold_trades:
            pnl = sum(t.pnl_pct for t in fold_trades)
            wins = sum(1 for t in fold_trades if t.pnl_pct > 0)
            wr = wins / len(fold_trades) * 100
            dca_info = ""
            if use_dca:
                n_dca = sum(1 for t in fold_trades if hasattr(t, 'total_size') and t.total_size > 1)
                dca_info = f" ({n_dca} DCA'd)"
            console.print(f"  [bold]Fold {fold}: {len(fold_trades)} trades{dca_info}, WR {wr:.0f}%, PnL {pnl:+.2f}%[/bold]")
        else:
            console.print(f"  [dim]No trades[/dim]")

        all_trades.extend(fold_trades)
        start += test_candles

    # Summary
    console.print("\n" + "=" * 60)
    if not all_trades:
        console.print("[red]No trades![/red]")
        return

    total_pnl = sum(t.pnl_pct for t in all_trades)
    total_wins = sum(1 for t in all_trades if t.pnl_pct > 0)
    total_count = len(all_trades)
    total_wr = total_wins / total_count * 100

    long_trades = [t for t in all_trades if t.direction == "LONG"]
    short_trades = [t for t in all_trades if t.direction == "SHORT"]
    long_pnl = sum(t.pnl_pct for t in long_trades)
    short_pnl = sum(t.pnl_pct for t in short_trades)
    long_wr = sum(1 for t in long_trades if t.pnl_pct > 0) / max(len(long_trades), 1) * 100
    short_wr = sum(1 for t in short_trades if t.pnl_pct > 0) / max(len(short_trades), 1) * 100

    tp_count = sum(1 for t in all_trades if t.exit_reason == "TP")
    sl_count = sum(1 for t in all_trades if t.exit_reason == "SL")
    to_count = sum(1 for t in all_trades if t.exit_reason == "TIMEOUT")

    table = Table(title="15m ML Filter Results", box=box.ROUNDED)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Total trades", str(total_count))
    table.add_row("Win rate", f"{total_wr:.1f}%")
    table.add_row("Total PnL", f"{total_pnl:+.2f}%")
    table.add_row("", "")
    table.add_row("LONG", f"{len(long_trades)} (WR {long_wr:.0f}%, PnL {long_pnl:+.2f}%)")
    table.add_row("SHORT", f"{len(short_trades)} (WR {short_wr:.0f}%, PnL {short_pnl:+.2f}%)")
    table.add_row("", "")
    table.add_row("TP / SL / Timeout", f"{tp_count} / {sl_count} / {to_count}")
    table.add_row("Avg PnL/trade", f"{total_pnl / total_count:+.3f}%")
    table.add_row("Days", str(total_days))
    table.add_row("PnL/month", f"{total_pnl / total_days * 30:+.2f}%")
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
    parser = argparse.ArgumentParser(description="Walk-Forward Backtest")
    parser.add_argument("--mode", default="5m", choices=["5m", "filter15m"],
                        help="5m ML direction or 15m ML filter")
    parser.add_argument("--days", type=int, default=45, help="Total days of data")
    parser.add_argument("--train-days", type=int, default=20, help="Training window days")
    parser.add_argument("--test-days", type=int, default=5, help="Test window days")
    parser.add_argument("--sl", type=float, default=1.5, help="SL ATR multiplier")
    parser.add_argument("--tp", type=float, default=3.0, help="TP ATR multiplier")
    parser.add_argument("--long-mom", type=float, default=0.30, help="LONG momentum threshold")
    parser.add_argument("--short-mom", type=float, default=0.10, help="SHORT momentum threshold")
    parser.add_argument("--conf", type=float, default=0.55, help="ML filter confidence threshold")
    parser.add_argument("--pairs", type=str, default=None, help="Comma-separated pairs")
    parser.add_argument("--dca", action="store_true", help="Use Smart DCA instead of fixed SL")
    args = parser.parse_args()

    selected_pairs = args.pairs.split(",") if args.pairs else None

    if args.mode == "filter15m":
        run_filter15m(
            total_days=args.days,
            train_days=args.train_days,
            test_days=args.test_days,
            sl_mult=args.sl,
            tp_mult=args.tp,
            long_mom=args.long_mom,
            short_mom=args.short_mom,
            conf_threshold=args.conf,
            pairs=selected_pairs,
            use_dca=args.dca,
        )
    else:
        run_walk_forward(
            total_days=args.days,
            train_days=args.train_days,
            test_days=args.test_days,
            sl_mult=args.sl,
            tp_mult=args.tp,
        )
