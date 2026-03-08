#!/usr/bin/env python3
"""
Full 187-day deep analysis: Sep 1 2025 - Mar 7 2026.
Runs walk-forward backtest with feature capture, MFE/MAE tracking,
and produces comprehensive multi-level report.
"""

import argparse
import logging
import os
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

import functools
print = functools.partial(print, flush=True)

from src.config import config
from src.data_fetcher import BinanceDataFetcher
from src.features import FeatureEngineer
from src.indicators import TechnicalIndicators
from src.ml_model import MLSignalModel
from src.signal_generator import SignalGenerator
from backtest_wf import simulate_trades, simulate_dca_trades, SimTrade, DcaTrade


# ── Enhanced trade record with features + MFE/MAE ──────────────

@dataclass
class AnalyzedTrade:
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    exit_reason: str
    pnl_pct: float
    entry_bar: int
    exit_bar: int
    fold: int
    bars_held: int
    trend_level: str       # strong / weak / against
    confidence: float
    dca_entries: int
    mfe_atr: float         # max favorable excursion in ATR units
    mae_atr: float         # max adverse excursion in ATR units
    entry_features: Dict   # snapshot of key features at entry


TOP_FEATURES = [
    "rsi", "rsi_slope", "ADX_14", "adx_slope", "roc_12", "roc_48",
    "roc_3", "volume_ratio", "volume_climax", "atr_expansion",
    "ema_cross_9_21", "ema_cross_21_50", "price_vs_ema50",
    "momentum_consistency", "body_size", "lower_wick_ratio",
    "dist_from_high_20", "dist_from_low_20", "rsi_divergence",
    "bullish_volume", "bearish_volume", "BBP_20_2.0",
    "regime_adx", "regime_bb_width", "consecutive_candles",
    "obv_change_6", "rsi_slope_6", "vol_trend_ratio",
    "drop_speed_4", "atr_pct",
]


def compute_mfe_mae(df, entry_bar_abs, exit_bar_abs, direction, entry_price, atr_at_entry):
    """Compute MFE and MAE in ATR units between entry and exit bars."""
    if atr_at_entry <= 0 or entry_bar_abs >= len(df) or exit_bar_abs >= len(df):
        return 0.0, 0.0

    high = df["high"].values
    low = df["low"].values
    start = entry_bar_abs + 1
    end = min(exit_bar_abs + 1, len(df))
    if start >= end:
        return 0.0, 0.0

    if direction == "LONG":
        mfe = (high[start:end].max() - entry_price) / atr_at_entry
        mae = (entry_price - low[start:end].min()) / atr_at_entry
    else:
        mfe = (entry_price - low[start:end].min()) / atr_at_entry
        mae = (high[start:end].max() - entry_price) / atr_at_entry

    return max(0, mfe), max(0, mae)


def consecutive_streaks(results):
    """Compute max consecutive wins and losses from list of booleans."""
    if not results:
        return 0, 0
    max_win = max_loss = cur_win = cur_loss = 0
    for win in results:
        if win:
            cur_win += 1
            cur_loss = 0
        else:
            cur_loss += 1
            cur_win = 0
        max_win = max(max_win, cur_win)
        max_loss = max(max_loss, cur_loss)
    return max_win, max_loss


def run_analysis(total_days=187, train_days=20, test_days=5,
                 sl_mult=3.5, tp_mult=1.5, long_mom=0.30, short_mom=0.10,
                 conf_threshold=0.55):

    BARS_15M = 96
    total_candles = total_days * BARS_15M
    train_candles = train_days * BARS_15M
    test_candles = test_days * BARS_15M

    trading_pairs = list(config.TRADING_PAIRS)
    pair_to_id = {sym: i for i, sym in enumerate(sorted(trading_pairs))}

    fetcher = BinanceDataFetcher(config)
    indicators = TechnicalIndicators()
    fe = FeatureEngineer()
    sig_gen = SignalGenerator(config)

    print(f"=== FULL ANALYSIS: {total_days}d | Train: {train_days}d | Test: {test_days}d ===")
    print(f"SL={sl_mult}x TP={tp_mult}x | Conf>{conf_threshold}")
    print(f"Pairs: {len(trading_pairs)}\n")

    print("Fetching 15m data...")
    all_pair_data = {}
    for symbol in trading_pairs:
        df = fetcher.fetch_ohlcv_extended(symbol, "15m", total_candles=total_candles)
        if df.empty or len(df) < 200:
            continue
        df = indicators.calculate_all(df)
        all_pair_data[symbol] = df
        print(f"  {symbol}: {len(df)} bars")

    if not all_pair_data:
        print("No data!")
        return

    # ── Walk-forward with enhanced tracking ──
    all_analyzed: List[AnalyzedTrade] = []
    fold_results = []
    fold = 0
    start = 0

    while start + train_candles + test_candles <= total_candles:
        fold += 1
        train_end = start + train_candles
        test_end = train_end + test_candles

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
            feat = fe.add_rolling_htf_features(feat)
            cols_trend = fe.get_feature_columns(feat, model_type="trend")
            cols_mr = fe.get_feature_columns(feat, model_type="mean_reversion")
            X_trend = feat[cols_trend].replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            X_mr = feat[cols_mr].replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            X_trend["pair_id"] = pair_to_id.get(symbol, 0)
            X_mr["pair_id"] = pair_to_id.get(symbol, 0)
            feat_names_refs["trend"] = list(cols_trend) + ["pair_id"]
            feat_names_refs["mr"] = list(cols_mr) + ["pair_id"]

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
                train_data[key]["X"].append(X_src.iloc[[i_row]])
                train_data[key]["y"].append(y_val)

        models = {}
        for dk in ["long_wt", "long_at", "short_wt", "short_at"]:
            xl = train_data[dk]["X"]
            yl = train_data[dk]["y"]
            if not xl or len(xl) < 50:
                continue
            X_tr = pd.concat(xl, ignore_index=True)
            y_tr = pd.Series(yl)
            fn = feat_names_refs["mr"] if "_at" in dk else feat_names_refs["trend"]
            tmp_path = os.path.join(tempfile.gettempdir(), f"af_{dk}_f{fold}.json")
            m = MLSignalModel(tmp_path)
            sw = fe.compute_sample_weights(y_tr)
            m.train(X_tr, y_tr, fn, sample_weights=sw)
            models[dk] = m

        if not models:
            start += test_candles
            continue

        # === TEST with enhanced tracking ===
        fold_all_trades = []

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

            ema9_vals = df_test["ema_9"].values if "ema_9" in df_test.columns else np.full(len(df_test), 0)
            ema21_vals = df_test["ema_21"].values if "ema_21" in df_test.columns else np.full(len(df_test), 0)
            ema50_test = df_test["ema_50"].values if "ema_50" in df_test.columns else np.full(len(df_test), 0)
            close_test = df_test["close"].values
            atr_test = df_test["atr"].values if "atr" in df_test.columns else np.ones(len(df_test))
            atr_ma20 = pd.Series(atr_test).rolling(20, min_periods=1).mean().values
            rsi_slope_vals = df_test["rsi"].diff(6).values if "rsi" in df_test.columns else np.zeros(len(df_test))

            # Track signal metadata for each bar
            signal_meta = {}

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
                trend_level = "unknown"
                if direction != "NEUTRAL":
                    atr_exp = atr_test[j] / atr_ma20[j] if atr_ma20[j] > 0 else 1.0
                    if atr_exp > 1.5:
                        direction = "NEUTRAL"; continue

                    rsi_sl_6 = rsi_slope_vals[j] if j < len(rsi_slope_vals) else 0
                    if direction == "LONG" and rsi_sl_6 < 0:
                        direction = "NEUTRAL"; continue
                    elif direction == "SHORT" and rsi_sl_6 > 0:
                        direction = "NEUTRAL"; continue

                    if direction == "LONG":
                        vol_r = float(df_test["volume_ratio"].iloc[j]) if "volume_ratio" in df_test.columns else 1
                        if vol_r > 1.3:
                            direction = "NEUTRAL"; continue

                    e50 = ema50_test[j] if j < len(ema50_test) else 0
                    e9 = ema9_vals[j] if j < len(ema9_vals) else 0
                    e21 = ema21_vals[j] if j < len(ema21_vals) else 0
                    trend_up = close_test[j] > e50 if e50 > 0 else True
                    with_trend = (direction == "LONG" and trend_up) or (direction == "SHORT" and not trend_up)

                    if e9 > 0 and e21 > 0 and e50 > 0:
                        ema_stack_bull = e9 > e21 > e50
                        ema_stack_bear = e9 < e21 < e50
                        ema_short_bull = e9 > e21
                        ema_short_bear = e9 < e21
                    else:
                        ema_stack_bull = ema_stack_bear = False
                        ema_short_bull = ema_short_bear = False

                    if (direction == "LONG" and ema_stack_bull) or (direction == "SHORT" and ema_stack_bear):
                        rs = rsi_slope_vals[j] if j < len(rsi_slope_vals) else 0
                        if (direction == "LONG" and rs > 1.5) or (direction == "SHORT" and rs < -1.5):
                            trend_level = "weak"
                        else:
                            trend_level = "strong"
                    elif (direction == "LONG" and ema_short_bull) or (direction == "SHORT" and ema_short_bear):
                        trend_level = "weak"
                    else:
                        trend_level = "against"

                    model_key = ("long_wt" if direction == "LONG" else "short_wt") if with_trend else ("long_at" if direction == "LONG" else "short_at")
                    X_row = X_t.iloc[[j]] if with_trend else X_m.iloc[[j]]

                    if direction == "LONG":
                        obv_chg = float(df_test["obv"].pct_change(6).iloc[j] * 100) if "obv" in df_test.columns else 5
                        min_conf_adj = 0.70 if obv_chg < 1.0 else (0.65 if "_at" in model_key else conf_threshold)
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

                bar_time = df_test.index[j]
                bar_hour = bar_time.hour if hasattr(bar_time, 'hour') else 0
                if config.FILTER_DEAD_HOURS and bar_hour in config.FILTER_DEAD_HOURS:
                    direction = "NEUTRAL"; conf = 0

                if direction != "NEUTRAL":
                    row = X_t.iloc[j]
                    guard_feats = {k: float(row.get(k, 0)) for k in [
                        "dist_from_high_96", "dist_from_low_96",
                        "dist_from_high_288", "dist_from_low_288",
                        "max_drawdown_48h", "roc_deceleration"]}
                    if sig_gen._exhaustion_guard(direction, guard_feats):
                        direction = "NEUTRAL"; conf = 0

                if direction != "NEUTRAL":
                    feat_snap = {}
                    for f in TOP_FEATURES:
                        if f in df_test.columns:
                            feat_snap[f] = float(df_test[f].iloc[j])
                    signal_meta[j] = {
                        "trend_level": trend_level,
                        "confidence": conf,
                        "features": feat_snap,
                    }

                signals.append({
                    "bar_idx": j, "symbol": symbol,
                    "direction": direction, "confidence": conf,
                })

            # Run both simulators (same as run_filter15m with use_dca=True)
            with_trend_sigs = []
            weak_trend_sigs = []
            against_trend_sigs = []
            for s in signals:
                idx = s["bar_idx"]
                e9 = ema9_vals[idx] if idx < len(ema9_vals) else 0
                e21 = ema21_vals[idx] if idx < len(ema21_vals) else 0
                e50 = ema50_test[idx] if idx < len(ema50_test) else 0
                d = s["direction"]
                rs = rsi_slope_vals[idx] if idx < len(rsi_slope_vals) else 0
                if e9 > 0 and e21 > 0 and e50 > 0:
                    esb = e9 > e21 > e50
                    esbe = e9 < e21 < e50
                    esb2 = e9 > e21
                    esbe2 = e9 < e21
                else:
                    esb = esbe = esb2 = esbe2 = False

                if (d == "LONG" and esb) or (d == "SHORT" and esbe):
                    if (d == "LONG" and rs > 1.5) or (d == "SHORT" and rs < -1.5):
                        weak_trend_sigs.append(s)
                        with_trend_sigs.append({**s, "direction": "NEUTRAL"})
                        against_trend_sigs.append({**s, "direction": "NEUTRAL"})
                    else:
                        with_trend_sigs.append(s)
                        weak_trend_sigs.append({**s, "direction": "NEUTRAL"})
                        against_trend_sigs.append({**s, "direction": "NEUTRAL"})
                elif (d == "LONG" and esb2) or (d == "SHORT" and esbe2):
                    weak_trend_sigs.append(s)
                    with_trend_sigs.append({**s, "direction": "NEUTRAL"})
                    against_trend_sigs.append({**s, "direction": "NEUTRAL"})
                else:
                    obv_vals = df_test["obv"].pct_change(6).values * 100 if "obv" in df_test.columns else np.zeros(len(df_test))
                    obv_v = obv_vals[idx] if idx < len(obv_vals) else 0
                    at_ok = True
                    if rs < 0:
                        at_ok = False
                    if obv_v < 0:
                        at_ok = False
                    min_at_conf = 0.65
                    if not at_ok:
                        min_at_conf = 0.80
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

            dca_trades = simulate_dca_trades(
                df_test, with_trend_sigs, tp_mult=tp_mult, dca_step_mult=1.0,
                max_entries=3, hard_sl_mult=sl_mult, max_hold=36,
                max_open=config.MAX_OPEN_TRADES, cooldown=3, threshold=conf_threshold,
            )
            weak_long_sigs = [s if s["direction"] == "LONG" else {**s, "direction": "NEUTRAL"} for s in weak_trend_sigs]
            weak_short_sigs = [s if s["direction"] == "SHORT" else {**s, "direction": "NEUTRAL"} for s in weak_trend_sigs]
            weak_long = simulate_trades(df_test, weak_long_sigs, sl_mult=2.0, tp_mult=tp_mult, max_hold=24, max_open=config.MAX_OPEN_TRADES, cooldown=3, threshold=conf_threshold)
            weak_short = simulate_trades(df_test, weak_short_sigs, sl_mult=2.0, tp_mult=1.0, max_hold=12, max_open=config.MAX_OPEN_TRADES, cooldown=3, threshold=conf_threshold)
            sl_at_trades = simulate_trades(df_test, against_trend_sigs, sl_mult=2.0, tp_mult=1.0, max_hold=12, max_open=config.MAX_OPEN_TRADES, cooldown=3, threshold=conf_threshold)

            raw_trades = dca_trades + weak_long + weak_short + sl_at_trades

            # Enrich with MFE/MAE and features
            for t in raw_trades:
                is_dca = hasattr(t, "total_size")
                e_bar = t.first_bar if is_dca else t.entry_bar
                x_bar = t.exit_bar
                e_price = t.avg_price if is_dca else t.entry_price
                atr_e = atr_test[e_bar] if e_bar < len(atr_test) and not np.isnan(atr_test[e_bar]) else 1
                mfe, mae = compute_mfe_mae(df_test, e_bar, x_bar, t.direction, e_price, atr_e)

                meta = signal_meta.get(e_bar, {})
                tl = meta.get("trend_level", "unknown")
                if is_dca:
                    tl = "strong"

                # Determine trend level for non-DCA
                if not is_dca and tl == "unknown":
                    if t in weak_long or t in weak_short:
                        tl = "weak"
                    elif t in sl_at_trades:
                        tl = "against"

                at = AnalyzedTrade(
                    symbol=symbol, direction=t.direction,
                    entry_price=e_price, exit_price=t.exit_price,
                    exit_reason=t.exit_reason, pnl_pct=t.pnl_pct,
                    entry_bar=e_bar, exit_bar=x_bar, fold=fold,
                    bars_held=x_bar - e_bar,
                    trend_level=tl,
                    confidence=meta.get("confidence", t.confidence if hasattr(t, "confidence") else 0),
                    dca_entries=t.total_size if is_dca else 1,
                    mfe_atr=mfe, mae_atr=mae,
                    entry_features=meta.get("features", {}),
                )
                all_analyzed.append(at)
                fold_all_trades.append(at)

        # Fold summary
        f_pnl = sum(t.pnl_pct for t in fold_all_trades)
        f_cnt = len(fold_all_trades)
        f_wr = sum(1 for t in fold_all_trades if t.pnl_pct > 0) / max(f_cnt, 1) * 100
        f_dca = sum(1 for t in fold_all_trades if t.dca_entries > 1)
        fold_results.append({"fold": fold, "trades": f_cnt, "dca": f_dca, "wr": f_wr, "pnl": f_pnl})
        print(f"  Fold {fold}: {f_cnt} trades ({f_dca} DCA'd), WR {f_wr:.0f}%, PnL {f_pnl:+.2f}%")

        start += test_candles

    # ══════════════════════════════════════════════════════════
    #  ANALYSIS
    # ══════════════════════════════════════════════════════════

    trades = all_analyzed
    if not trades:
        print("No trades!")
        return

    total_pnl = sum(t.pnl_pct for t in trades)
    total_cnt = len(trades)
    total_wr = sum(1 for t in trades if t.pnl_pct > 0) / total_cnt * 100

    # ── LEVEL 1: Global Summary ──────────────────────────────
    print("\n" + "=" * 80)
    print("LEVEL 1: GLOBAL SUMMARY")
    print("=" * 80)
    print(f"\n  Trades:     {total_cnt}")
    print(f"  Win Rate:   {total_wr:.1f}%")
    print(f"  PnL:        {total_pnl:+.2f}%")
    print(f"  PnL/month:  {total_pnl / total_days * 30:+.2f}%")
    print(f"  Avg/trade:  {total_pnl / total_cnt:+.4f}%")
    print(f"  Days:       {total_days}")

    for dir_name in ["LONG", "SHORT"]:
        dt = [t for t in trades if t.direction == dir_name]
        if not dt:
            continue
        d_pnl = sum(t.pnl_pct for t in dt)
        d_wr = sum(1 for t in dt if t.pnl_pct > 0) / len(dt) * 100
        d_tp = sum(1 for t in dt if t.exit_reason == "TP")
        d_sl = sum(1 for t in dt if t.exit_reason == "SL")
        d_to = sum(1 for t in dt if t.exit_reason == "TIMEOUT")
        d_hsl = sum(1 for t in dt if t.exit_reason == "HARD_SL")
        d_dca = sum(1 for t in dt if t.dca_entries > 1)
        wins = [t.pnl_pct > 0 for t in dt]
        mw, ml = consecutive_streaks(wins)
        print(f"\n  {dir_name}:")
        print(f"    Count: {len(dt)} | WR: {d_wr:.1f}% | PnL: {d_pnl:+.2f}%")
        print(f"    TP: {d_tp} | SL: {d_sl} | TO: {d_to} | HARD_SL: {d_hsl} | DCA: {d_dca}")
        print(f"    Max consecutive: Win={mw} Loss={ml}")

    # Exit reason breakdown
    print("\n  Exit reasons:")
    for reason in ["TP", "SL", "TIMEOUT", "HARD_SL", "DCA_TIMEOUT", "END"]:
        rt = [t for t in trades if t.exit_reason == reason]
        if not rt:
            continue
        r_pnl = sum(t.pnl_pct for t in rt)
        r_wr = sum(1 for t in rt if t.pnl_pct > 0) / max(len(rt), 1) * 100
        avg_bars = np.mean([t.bars_held for t in rt])
        print(f"    {reason:>12}: {len(rt):>4} trades | PnL {r_pnl:+.2f}% | WR {r_wr:.0f}% | Avg bars: {avg_bars:.1f}")

    # Per-fold table
    print("\n  Per-fold:")
    for fr in fold_results:
        print(f"    Fold {fr['fold']:>2}: {fr['trades']:>3} trades ({fr['dca']:>2} DCA), WR {fr['wr']:.0f}%, PnL {fr['pnl']:+.2f}%")

    # Trend level breakdown
    print("\n  Trend levels:")
    for tl in ["strong", "weak", "against", "unknown"]:
        tt = [t for t in trades if t.trend_level == tl]
        if not tt:
            continue
        t_pnl = sum(t.pnl_pct for t in tt)
        t_wr = sum(1 for t in tt if t.pnl_pct > 0) / max(len(tt), 1) * 100
        print(f"    {tl:>8}: {len(tt):>4} trades | PnL {t_pnl:+.2f}% | WR {t_wr:.0f}%")

    # ── LEVEL 2: Per-Pair Deep Dive ──────────────────────────
    print("\n" + "=" * 80)
    print("LEVEL 2: PER-PAIR DEEP DIVE")
    print("=" * 80)

    symbols = sorted(set(t.symbol for t in trades))
    for sym in symbols:
        st = [t for t in trades if t.symbol == sym]
        s_pnl = sum(t.pnl_pct for t in st)
        s_wr = sum(1 for t in st if t.pnl_pct > 0) / max(len(st), 1) * 100
        name = sym.split("/")[0]

        print(f"\n  {name}: {len(st)} trades | WR {s_wr:.0f}% | PnL {s_pnl:+.2f}% | PnL/m {s_pnl / total_days * 30:+.2f}%")

        for dir_name in ["LONG", "SHORT"]:
            dt = [t for t in st if t.direction == dir_name]
            if not dt:
                continue
            d_pnl = sum(t.pnl_pct for t in dt)
            d_wr = sum(1 for t in dt if t.pnl_pct > 0) / max(len(dt), 1) * 100
            d_tp = sum(1 for t in dt if t.exit_reason == "TP")
            d_sl = sum(1 for t in dt if t.exit_reason == "SL")
            d_to = sum(1 for t in dt if t.exit_reason == "TIMEOUT")
            d_hsl = sum(1 for t in dt if t.exit_reason == "HARD_SL")
            d_dca = sum(1 for t in dt if t.dca_entries > 1)
            wins = [t.pnl_pct > 0 for t in dt]
            mw, ml = consecutive_streaks(wins)
            print(f"    {dir_name:>5}: {len(dt):>3}t WR {d_wr:.0f}% PnL {d_pnl:+.2f}% | TP:{d_tp} SL:{d_sl} TO:{d_to} HSL:{d_hsl} DCA:{d_dca} | Streak W:{mw} L:{ml}")

    # ── LEVEL 3: Feature Analysis ────────────────────────────
    print("\n" + "=" * 80)
    print("LEVEL 3: FEATURE ANALYSIS (TP vs SL)")
    print("=" * 80)

    tp_trades = [t for t in trades if t.exit_reason == "TP"]
    sl_trades = [t for t in trades if t.exit_reason in ("SL", "HARD_SL")]
    to_trades = [t for t in trades if t.exit_reason in ("TIMEOUT", "DCA_TIMEOUT")]

    # MFE / MAE analysis
    print("\n  MFE/MAE Analysis (in ATR units):")
    for label, group in [("TP", tp_trades), ("SL", sl_trades), ("TIMEOUT", to_trades)]:
        if not group:
            continue
        mfes = [t.mfe_atr for t in group]
        maes = [t.mae_atr for t in group]
        bars = [t.bars_held for t in group]
        print(f"    {label:>8}: MFE avg {np.mean(mfes):.3f}x | MAE avg {np.mean(maes):.3f}x | Bars avg {np.mean(bars):.1f}")

    # MFE distribution for SL trades
    if sl_trades:
        mfes_sl = [t.mfe_atr for t in sl_trades]
        print(f"\n  SL trades MFE distribution (how far price went RIGHT before reversing):")
        for thresh in [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]:
            cnt = sum(1 for m in mfes_sl if m >= thresh)
            print(f"    MFE >= {thresh:.1f}x ATR: {cnt} ({cnt * 100 // len(mfes_sl)}%)")

    # Feature comparison: TP vs SL
    feat_comparison = []
    for feat_name in TOP_FEATURES:
        tp_vals = [t.entry_features.get(feat_name, np.nan) for t in tp_trades if feat_name in t.entry_features]
        sl_vals = [t.entry_features.get(feat_name, np.nan) for t in sl_trades if feat_name in t.entry_features]
        if len(tp_vals) < 10 or len(sl_vals) < 10:
            continue
        tp_mean = np.nanmean(tp_vals)
        sl_mean = np.nanmean(sl_vals)
        avg = (abs(tp_mean) + abs(sl_mean)) / 2
        if avg > 0:
            diff_pct = abs(tp_mean - sl_mean) / avg * 100
        else:
            diff_pct = 0
        feat_comparison.append((feat_name, tp_mean, sl_mean, diff_pct))

    feat_comparison.sort(key=lambda x: x[3], reverse=True)
    print(f"\n  Feature comparison (TP vs SL, sorted by difference):")
    print(f"    {'Feature':>25} | {'TP avg':>10} | {'SL avg':>10} | {'Diff%':>6}")
    print(f"    {'-'*25}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}")
    for fn, tp_m, sl_m, diff in feat_comparison[:25]:
        print(f"    {fn:>25} | {tp_m:>10.4f} | {sl_m:>10.4f} | {diff:>5.1f}%")

    # Feature comparison for LONG TP vs LONG SL
    print(f"\n  LONG: Feature comparison (TP vs SL):")
    long_tp = [t for t in tp_trades if t.direction == "LONG"]
    long_sl = [t for t in sl_trades if t.direction == "LONG"]
    long_fc = []
    for feat_name in TOP_FEATURES:
        tp_v = [t.entry_features.get(feat_name, np.nan) for t in long_tp if feat_name in t.entry_features]
        sl_v = [t.entry_features.get(feat_name, np.nan) for t in long_sl if feat_name in t.entry_features]
        if len(tp_v) < 5 or len(sl_v) < 5:
            continue
        tp_m = np.nanmean(tp_v)
        sl_m = np.nanmean(sl_v)
        avg = (abs(tp_m) + abs(sl_m)) / 2
        diff = abs(tp_m - sl_m) / avg * 100 if avg > 0 else 0
        long_fc.append((feat_name, tp_m, sl_m, diff))
    long_fc.sort(key=lambda x: x[3], reverse=True)
    print(f"    {'Feature':>25} | {'TP avg':>10} | {'SL avg':>10} | {'Diff%':>6}")
    print(f"    {'-'*25}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}")
    for fn, tp_m, sl_m, diff in long_fc[:15]:
        print(f"    {fn:>25} | {tp_m:>10.4f} | {sl_m:>10.4f} | {diff:>5.1f}%")

    # Feature comparison for SHORT TP vs SHORT SL
    print(f"\n  SHORT: Feature comparison (TP vs SL):")
    short_tp = [t for t in tp_trades if t.direction == "SHORT"]
    short_sl = [t for t in sl_trades if t.direction == "SHORT"]
    short_fc = []
    for feat_name in TOP_FEATURES:
        tp_v = [t.entry_features.get(feat_name, np.nan) for t in short_tp if feat_name in t.entry_features]
        sl_v = [t.entry_features.get(feat_name, np.nan) for t in short_sl if feat_name in t.entry_features]
        if len(tp_v) < 5 or len(sl_v) < 5:
            continue
        tp_m = np.nanmean(tp_v)
        sl_m = np.nanmean(sl_v)
        avg = (abs(tp_m) + abs(sl_m)) / 2
        diff = abs(tp_m - sl_m) / avg * 100 if avg > 0 else 0
        short_fc.append((feat_name, tp_m, sl_m, diff))
    short_fc.sort(key=lambda x: x[3], reverse=True)
    print(f"    {'Feature':>25} | {'TP avg':>10} | {'SL avg':>10} | {'Diff%':>6}")
    print(f"    {'-'*25}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}")
    for fn, tp_m, sl_m, diff in short_fc[:15]:
        print(f"    {fn:>25} | {tp_m:>10.4f} | {sl_m:>10.4f} | {diff:>5.1f}%")

    # ── Per-pair feature analysis (worst pairs) ──
    print(f"\n  Per-pair feature analysis (problem pairs):")
    for sym in symbols:
        st = [t for t in trades if t.symbol == sym]
        s_pnl = sum(t.pnl_pct for t in st)
        if s_pnl > 0:
            continue
        name = sym.split("/")[0]
        s_tp = [t for t in st if t.exit_reason == "TP"]
        s_sl = [t for t in st if t.exit_reason in ("SL", "HARD_SL")]
        if len(s_tp) < 3 or len(s_sl) < 3:
            continue
        print(f"\n    {name} (PnL {s_pnl:+.2f}%):")
        pair_fc = []
        for feat_name in TOP_FEATURES:
            tp_v = [t.entry_features.get(feat_name, np.nan) for t in s_tp if feat_name in t.entry_features]
            sl_v = [t.entry_features.get(feat_name, np.nan) for t in s_sl if feat_name in t.entry_features]
            if len(tp_v) < 2 or len(sl_v) < 2:
                continue
            tp_m = np.nanmean(tp_v)
            sl_m = np.nanmean(sl_v)
            avg = (abs(tp_m) + abs(sl_m)) / 2
            diff = abs(tp_m - sl_m) / avg * 100 if avg > 0 else 0
            pair_fc.append((feat_name, tp_m, sl_m, diff))
        pair_fc.sort(key=lambda x: x[3], reverse=True)
        for fn, tp_m, sl_m, diff in pair_fc[:5]:
            print(f"      {fn:>25}: TP={tp_m:.4f} SL={sl_m:.4f} diff={diff:.0f}%")

    # ── LEVEL 4: Recommendations ─────────────────────────────
    print("\n" + "=" * 80)
    print("LEVEL 4: RECOMMENDATIONS")
    print("=" * 80)

    sl_pnl_total = sum(t.pnl_pct for t in sl_trades)
    to_pnl_total = sum(t.pnl_pct for t in to_trades)
    tp_pnl_total = sum(t.pnl_pct for t in tp_trades)
    short_t = [t for t in trades if t.direction == "SHORT"]
    short_pnl = sum(t.pnl_pct for t in short_t)
    at_trades = [t for t in trades if t.trend_level == "against"]
    at_pnl = sum(t.pnl_pct for t in at_trades)
    at_sl = [t for t in at_trades if t.exit_reason in ("SL", "HARD_SL")]
    at_sl_pnl = sum(t.pnl_pct for t in at_sl)

    # Compute potential savings from various fixes
    recommendations = []

    # 1. SL reduction
    if sl_trades:
        avg_sl_mfe = np.mean([t.mfe_atr for t in sl_trades])
        high_mfe_sl = [t for t in sl_trades if t.mfe_atr >= 0.5]
        high_mfe_pnl = sum(t.pnl_pct for t in high_mfe_sl)
        recommendations.append({
            "name": "Improve SL trailing (MFE-based)",
            "detail": f"{len(high_mfe_sl)} SL trades had MFE>=0.5x ATR (went right then reversed). Current trailing saves some but not all.",
            "potential": f"Save up to {abs(high_mfe_pnl):.1f}% from {len(high_mfe_sl)} trades",
            "impact": abs(high_mfe_pnl),
        })

    # 2. SHORT improvement
    if short_t:
        recommendations.append({
            "name": "Fix SHORT direction",
            "detail": f"SHORT: {len(short_t)} trades, PnL {short_pnl:+.2f}%. LONG is profitable, SHORT bleeds.",
            "potential": f"If SHORT breakeven → save {abs(short_pnl):.1f}%",
            "impact": abs(short_pnl) * 0.5,
        })

    # 3. Against-trend losses
    if at_sl:
        recommendations.append({
            "name": "Reduce against-trend SL losses",
            "detail": f"AT trades: {len(at_trades)} total, AT SL: {len(at_sl)} trades losing {at_sl_pnl:+.2f}%",
            "potential": f"Halving AT SL → save {abs(at_sl_pnl) * 0.5:.1f}%",
            "impact": abs(at_sl_pnl) * 0.3,
        })

    # 4. TIMEOUT drag
    if to_trades:
        recommendations.append({
            "name": "Reduce TIMEOUT losses",
            "detail": f"{len(to_trades)} timeouts losing {to_pnl_total:+.2f}%. Trades that go nowhere.",
            "potential": f"If TIMEOUT halved → save {abs(to_pnl_total) * 0.5:.1f}%",
            "impact": abs(to_pnl_total) * 0.3,
        })

    # 5. Adaptive confidence
    worst_pairs = [(sym, sum(t.pnl_pct for t in trades if t.symbol == sym))
                   for sym in symbols]
    worst_pairs.sort(key=lambda x: x[1])
    neg_pairs = [(s, p) for s, p in worst_pairs if p < 0]
    neg_pnl = sum(p for _, p in neg_pairs)
    if neg_pairs:
        recommendations.append({
            "name": "Adaptive per-pair confidence",
            "detail": f"{len(neg_pairs)} pairs negative (total {neg_pnl:+.2f}%). Model fails temporarily on some pairs.",
            "potential": f"Track WR in real-time, raise conf when WR<50% → save ~{abs(neg_pnl) * 0.4:.1f}%",
            "impact": abs(neg_pnl) * 0.3,
        })

    # 6. Rolling pair features
    recommendations.append({
        "name": "Rolling pair performance features",
        "detail": "Add rolling TP/SL ratio per pair as feature. Model sees 'this pair is in bad streak → skip'.",
        "potential": "Adaptive to any pair. No hardcoding.",
        "impact": abs(neg_pnl) * 0.2 if neg_pairs else 5,
    })

    recommendations.sort(key=lambda x: x["impact"], reverse=True)
    print()
    for i, r in enumerate(recommendations, 1):
        print(f"  {i}. [{r['impact']:.1f}% est] {r['name']}")
        print(f"     {r['detail']}")
        print(f"     → {r['potential']}")
        print()

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=187)
    parser.add_argument("--train-days", type=int, default=20)
    parser.add_argument("--test-days", type=int, default=5)
    args = parser.parse_args()
    run_analysis(total_days=args.days, train_days=args.train_days, test_days=args.test_days)
