#!/usr/bin/env python3
"""
Replay: simulate trading on recent data using the ALREADY TRAINED model.
Unlike backtest.py which trains its own model, this uses the same model
that runs on the server — giving the most realistic estimate.

Usage:
    python replay.py                     # last 3 days, all pairs
    python replay.py --days 7            # last 7 days
    python replay.py --pair BTC/USDT     # single pair
    python replay.py --days 5 --pair SOL/USDT
"""

import argparse
import logging
import sys
import time as _time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.config import config
from src.data_fetcher import BinanceDataFetcher
from src.features import FeatureEngineer
from src.indicators import TechnicalIndicators
from src.ml_model import MLSignalModel
from src.trade_monitor import TradeMonitor, TradeState

from rich import box
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

logging.basicConfig(level=logging.WARNING, handlers=[logging.StreamHandler(sys.stdout)])
console = Console()

COMMISSION_PCT = 0.04
SLIPPAGE_PCT = 0.01


@dataclass
class Trade:
    symbol: str
    direction: str
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    confidence: float
    disagreement: float
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    exit_reason: str = ""
    pnl_pct: float = 0.0
    pnl_net_pct: float = 0.0
    commission_pct: float = 0.0
    bars_held: int = 0


def replay_pair(
    symbol: str,
    ml_model: MLSignalModel,
    days: int = 3,
    _status_dict: dict = None,
) -> list[Trade]:
    """Replay trading on recent candles using the pre-trained model."""

    def _status(msg):
        if _status_dict is not None:
            _status_dict[symbol] = msg

    fetcher = BinanceDataFetcher(config)
    indicators = TechnicalIndicators()
    features = FeatureEngineer()
    monitor = TradeMonitor(config)

    candles = days * 288  # 288 = 24h of 5m candles

    _status("[yellow]Fetching data …[/yellow]")
    df_5m = fetcher.fetch_ohlcv_extended(symbol, "5m", total_candles=candles + 200)
    df_15m = fetcher.fetch_ohlcv_extended(symbol, "15m", total_candles=max(200, candles // 3))
    df_1h = fetcher.fetch_ohlcv_extended(symbol, "1h", total_candles=max(200, candles // 12))

    if df_5m.empty or len(df_5m) < 300:
        _status("[red]No data[/red]")
        return []

    df_5m = indicators.calculate_all(df_5m)
    if not df_15m.empty:
        df_15m = indicators.calculate_all(df_15m)
    if not df_1h.empty:
        df_1h = indicators.calculate_all(df_1h)

    X_full, _ = features.get_feature_matrix_mtf(df_5m, df_15m, df_1h)

    # only simulate on the last N days
    sim_start = len(df_5m) - candles
    if sim_start < 50:
        sim_start = 50

    _status("[yellow]Simulating …[/yellow]")

    trades: list[Trade] = []
    in_trade = False
    current_trade: Optional[Trade] = None
    trade_state: Optional[TradeState] = None
    cooldown = 0
    pending_signal: Optional[dict] = None
    consecutive_sl = 0
    global_pause = 0

    for i in range(sim_start, len(df_5m)):
        ts = df_5m.index[i]
        row = df_5m.iloc[i]
        price_open = float(row["open"])
        price_close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        atr = float(row["atr"]) if pd.notna(row.get("atr")) else 0

        # tick cooldowns
        if cooldown > 0:
            cooldown -= 1
        if global_pause > 0:
            global_pause -= 1

        # execute pending signal at this candle's OPEN
        if pending_signal is not None and not in_trade:
            sig = pending_signal
            pending_signal = None

            if sig["direction"] == "LONG":
                entry = price_open * (1 + SLIPPAGE_PCT / 100)
            else:
                entry = price_open * (1 - SLIPPAGE_PCT / 100)

            sl_dist = sig["atr"] * config.SL_ATR_MULTIPLIER
            tp_dist = sig["atr"] * config.TP_ATR_MULTIPLIER
            min_sl = entry * config.MIN_SL_PCT / 100
            if sl_dist < min_sl:
                sl_dist = min_sl

            if sig["direction"] == "LONG":
                sl, tp = entry - sl_dist, entry + tp_dist
            else:
                sl, tp = entry + sl_dist, entry - tp_dist

            current_trade = Trade(
                symbol=symbol, direction=sig["direction"],
                entry_price=entry, entry_time=ts,
                stop_loss=sl, take_profit=tp,
                confidence=sig["confidence"],
                disagreement=sig["disagreement"],
            )
            trade_state = monitor.create_state(symbol, sig["direction"], entry, sl, tp)
            in_trade = True

        # dead hours: close trade
        if in_trade and current_trade and config.FILTER_DEAD_HOURS:
            candle_hour = ts.hour if hasattr(ts, 'hour') else pd.Timestamp(ts).hour
            if candle_hour in config.FILTER_DEAD_HOURS:
                exit_price = price_close
                raw_pnl = _calc_pnl(current_trade.direction, current_trade.entry_price, exit_price)
                comm = 2 * COMMISSION_PCT
                _close_trade(current_trade, exit_price, ts, "DEAD_HOUR", raw_pnl, comm)
                trades.append(current_trade)
                in_trade, current_trade, trade_state = False, None, None
                cooldown = int(config.MAX_HOLD_BARS * 0.2)
                continue

        # check open trade
        if in_trade and current_trade and trade_state:
            rsi = float(row["rsi"]) if pd.notna(row.get("rsi")) else 50
            adx_val = float(row["ADX_14"]) if pd.notna(row.get("ADX_14")) else 25
            vol_r = float(row["volume_ratio"]) if pd.notna(row.get("volume_ratio")) else 1.0

            ml_sig, ml_conf = 0, 0.0
            if ts in X_full.index:
                loc = X_full.index.get_loc(ts)
                pred = ml_model.predict(X_full.iloc[[loc]])
                ml_sig = pred["signal"]
                ml_conf = pred["confidence"]

            trade_state = monitor.update_trade(
                trade_state, high, low, price_close, atr,
                rsi=rsi, adx=adx_val, volume_ratio=vol_r,
                ml_signal=ml_sig, ml_confidence=ml_conf,
            )
            current_trade.stop_loss = trade_state.current_sl

            should_exit, reason, exit_price = monitor.check_exit(trade_state, high, low, price_close)

            if should_exit:
                if current_trade.direction == "LONG":
                    exit_price *= (1 - SLIPPAGE_PCT / 100)
                else:
                    exit_price *= (1 + SLIPPAGE_PCT / 100)

                raw_pnl = _calc_pnl(current_trade.direction, current_trade.entry_price, exit_price)
                comm = 2 * COMMISSION_PCT
                _close_trade(current_trade, exit_price, ts, reason, raw_pnl, comm)
                trades.append(current_trade)
                in_trade, current_trade, trade_state = False, None, None
                cooldown = int(config.MAX_HOLD_BARS * 0.2)

                if reason in ("SL", "STOP_LOSS") or raw_pnl < -0.2:
                    consecutive_sl += 1
                    if consecutive_sl >= 3:
                        global_pause = 24
                else:
                    consecutive_sl = 0
                continue

        # generate signal
        if not in_trade and pending_signal is None and cooldown <= 0 and global_pause <= 0:
            if atr == 0 or ts not in X_full.index:
                continue

            loc = X_full.index.get_loc(ts)
            pred = ml_model.predict(X_full.iloc[[loc]])
            signal = pred["signal"]
            confidence = pred["confidence"]
            disagreement = pred.get("disagreement", 0.0)

            # disagreement penalty
            if disagreement >= 0.5:
                confidence *= 0.60
            elif disagreement > 0:
                confidence *= 0.85

            if signal == 0 or confidence < config.PREDICTION_THRESHOLD:
                continue

            direction = "LONG" if signal == 1 else "SHORT"

            # dead hours filter
            if config.FILTER_DEAD_HOURS:
                candle_hour = ts.hour if hasattr(ts, 'hour') else pd.Timestamp(ts).hour
                if candle_hour in config.FILTER_DEAD_HOURS:
                    continue

            pending_signal = {
                "direction": direction,
                "confidence": confidence,
                "disagreement": disagreement,
                "atr": atr,
            }

    # close remaining
    if in_trade and current_trade:
        last_close = float(df_5m.iloc[-1]["close"])
        raw_pnl = _calc_pnl(current_trade.direction, current_trade.entry_price, last_close)
        comm = 2 * COMMISSION_PCT
        _close_trade(current_trade, last_close, df_5m.index[-1], "END", raw_pnl, comm)
        trades.append(current_trade)

    _status(f"[green]Done — {len(trades)} trades[/green]")
    return trades


def _calc_pnl(direction, entry, exit_price):
    if direction == "LONG":
        return (exit_price - entry) / entry * 100
    return (entry - exit_price) / entry * 100


def _close_trade(trade, exit_price, ts, reason, raw_pnl, comm):
    trade.exit_price = exit_price
    trade.exit_time = ts
    trade.exit_reason = reason
    trade.pnl_pct = round(raw_pnl, 4)
    trade.pnl_net_pct = round(raw_pnl - comm, 4)
    trade.commission_pct = round(comm, 4)
    trade.bars_held = 0


def _train_fresh_model(pairs, test_days):
    """Train a model on data BEFORE the test period (no leakage)."""
    from src.ml_model import MLSignalModel as _MLModel

    fetcher = BinanceDataFetcher(config)
    indicators = TechnicalIndicators()
    features = FeatureEngineer()

    trend_candles = config.TREND_TRAIN_CANDLES
    test_candles = test_days * 288
    total_need = trend_candles + test_candles + 200  # train + test + buffer

    console.print(f"  Fetching {total_need} candles per pair (train: {trend_candles}, test: {test_candles}) …")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    all_X, all_y = [], []

    def _fetch_pair(sym):
        df_5m = fetcher.fetch_ohlcv_extended(sym, "5m", total_candles=total_need)
        df_15m = fetcher.fetch_ohlcv_extended(sym, "15m", total_candles=max(200, total_need // 3))
        df_1h = fetcher.fetch_ohlcv_extended(sym, "1h", total_candles=max(200, total_need // 12))
        if df_5m.empty:
            return None
        df_5m = indicators.calculate_all(df_5m)
        if not df_15m.empty:
            df_15m = indicators.calculate_all(df_15m)
        if not df_1h.empty:
            df_1h = indicators.calculate_all(df_1h)
        X, fn = features.get_feature_matrix_mtf(df_5m, df_15m, df_1h)

        # only use data BEFORE test period for training
        train_end = len(X) - test_candles
        X_train = X.iloc[:train_end].iloc[-trend_candles:]
        df_train = df_5m.iloc[:train_end].iloc[-trend_candles:]

        y = features.create_labels(
            df_train,
            tp_multiplier=config.LABEL_TP_MULTIPLIER,
            sl_multiplier=config.LABEL_SL_MULTIPLIER,
            max_bars=config.LABEL_MAX_BARS,
            ternary=True,
            uncertain_threshold_pct=config.UNCERTAIN_THRESHOLD_PCT,
        )
        common = X_train.index.intersection(y.index)
        X_c = X_train.loc[common].iloc[:-config.LABEL_MAX_BARS]
        y_c = y.loc[common].iloc[:-config.LABEL_MAX_BARS]
        return X_c, y_c, fn

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_fetch_pair, sym): sym for sym in pairs}
        feat_names = None
        for future in as_completed(futures):
            result = future.result()
            if result:
                X_c, y_c, fn = result
                all_X.append(X_c)
                all_y.append(y_c)
                feat_names = fn

    X_all = pd.concat(all_X, ignore_index=True)
    y_all = pd.concat(all_y, ignore_index=True)
    sw = FeatureEngineer.compute_sample_weights(y_all)

    n_buy = int((y_all == 1).sum())
    n_sell = int((y_all == -1).sum())
    n_unc = int((y_all == 0).sum())
    console.print(f"  Training on {len(X_all):,} samples (BUY:{n_buy} SELL:{n_sell} UNC:{n_unc}) …")

    model = _MLModel.__new__(_MLModel)
    model.model_path = ""
    model.models = {}
    model.feature_names = None
    model.n_classes = 3
    model.is_trained = False

    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier

    label_map = {-1: 0, 0: 1, 1: 2}
    y_mapped = y_all.map(label_map)

    feat_cols = [c for c in feat_names if c in X_all.columns]

    models = {}
    models["xgboost"] = xgb.XGBClassifier(
        n_estimators=500, max_depth=7, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        random_state=42, n_jobs=-1, verbosity=0,
    )
    models["xgboost"].fit(X_all[feat_cols], y_mapped, sample_weight=sw, verbose=False)

    models["lightgbm"] = lgb.LGBMClassifier(
        n_estimators=500, max_depth=7, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        objective="multiclass", num_class=3,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    models["lightgbm"].fit(X_all[feat_cols], y_mapped, sample_weight=sw)

    models["catboost"] = CatBoostClassifier(
        iterations=500, depth=7, learning_rate=0.03,
        l2_leaf_reg=1.0, random_seed=42, verbose=0,
        loss_function="MultiClass", classes_count=3,
    )
    models["catboost"].fit(X_all[feat_cols], y_mapped, sample_weight=sw, verbose=0)

    model.models = models
    model.feature_names = feat_cols
    model.n_classes = 3
    model.is_trained = True

    console.print(f"  [green]Model trained (no data leakage)[/green]")
    return model


def main():
    p = argparse.ArgumentParser(description="Replay with trained model")
    p.add_argument("--days", type=int, default=3)
    p.add_argument("--pair", type=str, default=None)
    args = p.parse_args()

    pairs = [args.pair] if args.pair else config.TRADING_PAIRS

    console.print(Panel(
        f"[bold cyan]Replay — Out-of-Sample Simulation[/bold cyan]\n\n"
        f"Pairs: {len(pairs)} | Test: last {args.days} days\n"
        f"Train: {config.TREND_TRAIN_CANDLES} candles BEFORE test period (no leakage)\n"
        f"SL: {config.SL_ATR_MULTIPLIER}x ATR | TP: {config.TP_ATR_MULTIPLIER}x ATR | "
        f"R:R 1:{config.TP_ATR_MULTIPLIER/config.SL_ATR_MULTIPLIER:.1f}\n"
        f"Threshold: {config.PREDICTION_THRESHOLD:.0%} | "
        f"Max hold: {config.MAX_HOLD_BARS} bars | "
        f"Dead hours: {config.FILTER_DEAD_HOURS or 'off'}\n"
        f"Commission: 0.04% x2 | Slippage: {SLIPPAGE_PCT:.2f}%",
        box=box.DOUBLE,
    ))

    console.print("\n[bold]Step 1: Training model on pre-test data …[/bold]")
    ml_model = _train_fresh_model(pairs, args.days)

    console.print(f"\n  Model: {ml_model.n_classes}-class, "
                  f"{len(ml_model.feature_names)} features, "
                  f"ensemble: {', '.join(ml_model.models.keys())}")

    console.print(f"\n[bold]Step 2: Replaying on last {args.days} days (out-of-sample) …[/bold]")

    t0 = _time.time()
    all_trades = []

    import threading
    from rich.live import Live
    from concurrent.futures import ThreadPoolExecutor, as_completed

    status_dict = {s: "Waiting …" for s in pairs}

    def _build_table():
        t = Table(box=box.SIMPLE, expand=True)
        t.add_column("Pair", style="cyan", width=12)
        t.add_column("Status", width=50)
        for sym in pairs:
            t.add_row(sym, status_dict.get(sym, ""))
        return t

    with Live(_build_table(), console=console, refresh_per_second=4) as live:
        with ThreadPoolExecutor(max_workers=min(6, len(pairs))) as pool:
            futures = {
                pool.submit(replay_pair, sym, ml_model, args.days, status_dict): sym
                for sym in pairs
            }
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    trades = future.result()
                    all_trades.extend(trades)
                except Exception as exc:
                    status_dict[sym] = f"[red]Error: {exc}[/red]"
                live.update(_build_table())

    elapsed = _time.time() - t0

    # report
    if not all_trades:
        console.print("[dim]No trades.[/dim]")
        return

    # per-pair table
    pair_trades = {}
    for t in all_trades:
        pair_trades.setdefault(t.symbol, []).append(t)

    table = Table(title=f"Replay Results — {args.days} days ({elapsed:.0f}s)", box=box.ROUNDED)
    table.add_column("Symbol", style="cyan")
    table.add_column("Trades", justify="center")
    table.add_column("WR", justify="center")
    table.add_column("Net PnL%", justify="right")
    table.add_column("Gross PnL%", justify="right")
    table.add_column("Comm%", justify="right")

    for sym in pairs:
        tt = pair_trades.get(sym, [])
        if not tt:
            table.add_row(sym, "0", "—", "—", "—", "—")
            continue
        wins = sum(1 for t in tt if t.pnl_net_pct > 0)
        wr = wins / len(tt) * 100
        net = sum(t.pnl_net_pct for t in tt)
        gross = sum(t.pnl_pct for t in tt)
        comm = sum(t.commission_pct for t in tt)
        wr_s = "green" if wr >= 50 else "red"
        pnl_s = "green" if net > 0 else "red"
        table.add_row(
            sym, str(len(tt)),
            f"[{wr_s}]{wr:.1f}%[/{wr_s}]",
            f"[{pnl_s}]{net:+.2f}%[/{pnl_s}]",
            f"{gross:+.2f}%", f"[red]-{comm:.2f}%[/red]",
        )

    console.print(table)

    # aggregate
    total = len(all_trades)
    wins = sum(1 for t in all_trades if t.pnl_net_pct > 0)
    wr = wins / total * 100
    longs = [t for t in all_trades if t.direction == "LONG"]
    shorts = [t for t in all_trades if t.direction == "SHORT"]
    l_wr = sum(1 for t in longs if t.pnl_net_pct > 0) / len(longs) * 100 if longs else 0
    s_wr = sum(1 for t in shorts if t.pnl_net_pct > 0) / len(shorts) * 100 if shorts else 0
    net = sum(t.pnl_net_pct for t in all_trades)
    gross = sum(t.pnl_pct for t in all_trades)
    comm = sum(t.commission_pct for t in all_trades)
    l_pnl = sum(t.pnl_net_pct for t in longs)
    s_pnl = sum(t.pnl_net_pct for t in shorts)

    tp_n = sum(1 for t in all_trades if t.exit_reason == "TP")
    sl_n = sum(1 for t in all_trades if t.exit_reason in ("SL", "STOP_LOSS"))
    trail_n = sum(1 for t in all_trades if t.exit_reason == "TRAIL_SL")
    early_n = sum(1 for t in all_trades if t.exit_reason == "EARLY_EXIT")
    to_n = sum(1 for t in all_trades if t.exit_reason == "TIMEOUT")
    dead_n = sum(1 for t in all_trades if t.exit_reason == "DEAD_HOUR")

    gp = sum(t.pnl_net_pct for t in all_trades if t.pnl_net_pct > 0)
    gl = abs(sum(t.pnl_net_pct for t in all_trades if t.pnl_net_pct <= 0))
    pf = gp / gl if gl > 0 else float("inf")

    equity = np.cumsum([t.pnl_net_pct for t in all_trades])
    peak = np.maximum.accumulate(equity)
    max_dd = float((peak - equity).max()) if len(equity) > 0 else 0

    c = "green" if net > 0 else "red"
    console.print(Panel(
        f"[bold]Trades:[/bold] {total}  (LONG: {len(longs)} | SHORT: {len(shorts)})\n"
        f"[bold]Win Rate:[/bold] {wr:.1f}%  (LONG: {l_wr:.1f}% | SHORT: {s_wr:.1f}%)\n"
        f"[bold]Net PnL:[/bold] [{c}]{net:+.2f}%[/{c}]  "
        f"(LONG: {l_pnl:+.2f}% | SHORT: {s_pnl:+.2f}%)\n"
        f"[bold]Gross PnL:[/bold] {gross:+.2f}%  "
        f"[bold]Commissions:[/bold] [red]-{comm:.2f}%[/red]\n"
        f"[bold]Profit Factor:[/bold] {pf:.2f}  "
        f"[bold]Max DD:[/bold] [red]{max_dd:.2f}%[/red]\n"
        f"\n[bold]Exit Reasons:[/bold] "
        f"TP: {tp_n} | SL: {sl_n} | Trail: {trail_n} | "
        f"Early: {early_n} | Timeout: {to_n} | Dead hour: {dead_n}",
        title=f"REPLAY — Pre-trained Model ({args.days} days)",
        border_style="cyan",
    ))


if __name__ == "__main__":
    main()
