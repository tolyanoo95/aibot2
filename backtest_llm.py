#!/usr/bin/env python3
"""
LLM-enhanced backtest — small sample test with full ML + LLM pipeline.
Shows detailed per-trade breakdown with LLM reasoning.

Usage:
    python backtest_llm.py                              # 10 trades, BTC/USDT
    python backtest_llm.py --pair ETH/USDT --trades 15  # 15 trades, ETH
    python backtest_llm.py --pair DOT/USDT --trades 5   # 5 trades, DOT
"""

import argparse
import logging
import sys
import time as _time
from typing import Optional

import numpy as np
import pandas as pd

from src.config import config
from src.data_fetcher import BinanceDataFetcher
from src.features import FeatureEngineer
from src.indicators import TechnicalIndicators
from src.llm_analyzer import LLMAnalyzer
from src.ml_model import MLSignalModel
from src.signal_generator import SignalGenerator

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
console = Console()

COMMISSION_PCT = 0.04
SLIPPAGE_PCT = 0.01


def run_llm_backtest(
    symbol: str = "BTC/USDT",
    max_trades: int = 10,
    min_confidence: float = 0.30,
    tp_mult: float = 3.0,
    sl_mult: float = 1.0,
    max_bars: int = 12,
):
    console.print(Panel(
        f"[bold cyan]LLM-Enhanced Backtest[/bold cyan]\n\n"
        f"Pair: {symbol} | Max trades: {max_trades}\n"
        f"SL: {sl_mult}x ATR | TP: {tp_mult}x ATR | Max hold: {max_bars} bars\n"
        f"Min combined confidence: {min_confidence:.0%}\n"
        f"Commission: 0.04% x2 | Slippage: {SLIPPAGE_PCT}%\n"
        f"Each trade calls OpenAI for LLM analysis\n"
        f"[dim]Note: LLM on historical data lacks live context (OI, dom, orderbook)\n"
        f"so it usually says NEUTRAL. Threshold is lower than live to show trades.[/dim]",
        box=box.DOUBLE,
    ))

    # ── init components ──────────────────────────────────────
    fetcher = BinanceDataFetcher(config)
    indicators = TechnicalIndicators()
    features = FeatureEngineer()
    ml_model = MLSignalModel(config.ML_MODEL_PATH)
    llm = LLMAnalyzer(config)
    signal_gen = SignalGenerator(config)

    if not ml_model.is_trained:
        console.print("[bold red]No trained model! Run train.py first.[/bold red]")
        return

    if not config.OPENAI_API_KEY:
        console.print("[bold red]No OPENAI_API_KEY in .env![/bold red]")
        return

    # ── fetch data ───────────────────────────────────────────
    console.print(f"\n  Fetching {symbol} data …")
    df_5m = fetcher.fetch_ohlcv_extended(symbol, "5m", total_candles=2000)
    df_15m = fetcher.fetch_ohlcv_extended(symbol, "15m", total_candles=700)
    df_1h = fetcher.fetch_ohlcv_extended(symbol, "1h", total_candles=200)

    if df_5m.empty or len(df_5m) < 200:
        console.print("[red]  Not enough data[/red]")
        return

    console.print(f"  5m: {len(df_5m)} | 15m: {len(df_15m)} | 1h: {len(df_1h)}")

    # ── indicators ───────────────────────────────────────────
    df_5m = indicators.calculate_all(df_5m)
    df_15m = indicators.calculate_all(df_15m) if not df_15m.empty else df_15m
    df_1h = indicators.calculate_all(df_1h) if not df_1h.empty else df_1h

    # ── features ─────────────────────────────────────────────
    X_full, feat_names = features.get_feature_matrix_mtf(df_5m, df_15m, df_1h)

    # ── walk through last 500 candles (recent data) ──────────
    console.print(f"  Walking through recent candles, looking for {max_trades} trades …\n")

    trades = []
    in_trade = False
    current_trade = None
    bars_since_exit = 5
    pending = None
    start_idx = max(0, len(df_5m) - 500)

    for i in range(start_idx, len(df_5m)):
        if len(trades) >= max_trades:
            break

        ts = df_5m.index[i]
        row = df_5m.iloc[i]
        price_open = float(row["open"])
        price_close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        atr = float(row["atr"]) if pd.notna(row.get("atr")) else 0

        # ── execute pending ──────────────────────────────────
        if pending is not None and not in_trade:
            sig = pending
            pending = None

            if sig["direction"] == "LONG":
                entry = price_open * (1 + SLIPPAGE_PCT / 100)
            else:
                entry = price_open * (1 - SLIPPAGE_PCT / 100)

            sl_d = sig["atr"] * sl_mult
            tp_d = sig["atr"] * tp_mult
            if sig["direction"] == "LONG":
                sl, tp = entry - sl_d, entry + tp_d
            else:
                sl, tp = entry + sl_d, entry - tp_d

            current_trade = {
                "direction": sig["direction"],
                "entry_price": entry,
                "entry_time": ts,
                "sl": sl, "tp": tp,
                "confidence": sig["confidence"],
                "ml_signal": sig["ml_signal"],
                "ml_conf": sig["ml_conf"],
                "llm_dir": sig["llm_dir"],
                "llm_conf": sig["llm_conf"],
                "llm_reason": sig["llm_reason"],
                "regime": sig["regime"],
                "bars": 0,
            }
            in_trade = True

        # ── check trade ──────────────────────────────────────
        if in_trade and current_trade:
            current_trade["bars"] += 1
            hit_tp = hit_sl = False

            if current_trade["direction"] == "LONG":
                hit_tp = high >= current_trade["tp"]
                hit_sl = low <= current_trade["sl"]
            else:
                hit_tp = low <= current_trade["tp"]
                hit_sl = high >= current_trade["sl"]

            if hit_tp and hit_sl:
                hit_tp = False  # conservative

            reason = ""
            exit_p = 0
            if hit_sl:
                exit_p, reason = current_trade["sl"], "SL"
            elif hit_tp:
                exit_p, reason = current_trade["tp"], "TP"
            elif current_trade["bars"] >= max_bars:
                exit_p, reason = price_close, "TIMEOUT"

            if reason:
                if current_trade["direction"] == "LONG":
                    exit_p *= (1 - SLIPPAGE_PCT / 100)
                    raw = (exit_p - current_trade["entry_price"]) / current_trade["entry_price"] * 100
                else:
                    exit_p *= (1 + SLIPPAGE_PCT / 100)
                    raw = (current_trade["entry_price"] - exit_p) / current_trade["entry_price"] * 100

                comm = 2 * COMMISSION_PCT
                net = raw - comm

                current_trade["exit_price"] = exit_p
                current_trade["exit_time"] = ts
                current_trade["exit_reason"] = reason
                current_trade["pnl_raw"] = round(raw, 4)
                current_trade["pnl_net"] = round(net, 4)
                trades.append(current_trade)

                # print trade
                _print_trade(len(trades), current_trade)

                in_trade = False
                current_trade = None
                bars_since_exit = 0
                continue

        # ── new signal ───────────────────────────────────────
        if not in_trade and pending is None:
            bars_since_exit += 1
            if bars_since_exit < 5 or atr == 0 or ts not in X_full.index:
                continue

            loc = X_full.index.get_loc(ts)
            X_row = X_full.iloc[[loc]]
            pred = ml_model.predict(X_row)

            if pred["signal"] == 0 or pred["confidence"] < 0.50:
                continue

            # ── volume + ADX filters ────────────────────────
            vol_ratio = float(row.get("volume_ratio", 1.0)) if pd.notna(row.get("volume_ratio")) else 1.0
            adx_val = float(row.get("ADX_14", 25.0)) if pd.notna(row.get("ADX_14")) else 25.0
            direction_pre = "LONG" if pred["signal"] == 1 else "SHORT"

            from src.config import config as _cfg
            if _cfg.FILTER_MIN_VOLUME_RATIO > 0 and vol_ratio < _cfg.FILTER_MIN_VOLUME_RATIO:
                console.print(f"  [dim]Candle {ts} — {direction_pre} filtered: low volume ({vol_ratio:.2f})[/dim]")
                continue
            if _cfg.FILTER_MIN_ADX > 0 and adx_val < _cfg.FILTER_MIN_ADX:
                console.print(f"  [dim]Candle {ts} — {direction_pre} filtered: weak ADX ({adx_val:.1f})[/dim]")
                continue

            # ── call LLM ─────────────────────────────────────
            market_data = {
                "primary": df_5m.iloc[max(0, i - 200): i + 1],
                "secondary": df_15m,
                "trend": df_1h,
                "funding_rate": 0,
            }

            console.print(
                f"  [dim]Candle {ts} — ML: "
                f"{'BUY' if pred['signal'] == 1 else 'SELL'} "
                f"({pred['confidence']:.1%}) → calling LLM …[/dim]",
                end=" ",
            )

            llm_result = llm.analyze_market(symbol, market_data)
            console.print(f"[dim]LLM: {llm_result.get('direction', '?')} "
                          f"({llm_result.get('confidence', 0)}/10)[/dim]")

            # ── combine ──────────────────────────────────────
            signal = signal_gen.generate(
                symbol=symbol,
                ml_result=pred,
                llm_result=llm_result,
                current_price=price_close,
                atr=atr,
            )

            if signal.direction == "NEUTRAL" or signal.confidence < min_confidence:
                console.print(f"    [dim]→ Combined: NEUTRAL (filtered out)[/dim]")
                continue

            console.print(
                f"    [bold]→ Combined: {signal.direction} "
                f"({signal.confidence:.1%})[/bold]"
            )

            pending = {
                "direction": signal.direction,
                "confidence": signal.confidence,
                "atr": atr,
                "signal_price": price_close,
                "ml_signal": pred["signal"],
                "ml_conf": pred["confidence"],
                "llm_dir": llm_result.get("direction", "?"),
                "llm_conf": llm_result.get("confidence", 0),
                "llm_reason": llm_result.get("reasoning", ""),
                "regime": llm_result.get("market_regime", "?"),
            }

    # ── summary ──────────────────────────────────────────────
    _print_summary(trades, symbol)


def _print_trade(num: int, t: dict):
    pnl_c = "green" if t["pnl_net"] > 0 else "red"
    dir_c = "green" if t["direction"] == "LONG" else "red"

    console.print(Panel(
        f"[bold {dir_c}]{t['direction']}[/bold {dir_c}]  "
        f"Confidence: {t['confidence']:.1%}\n"
        f"ML: {'BUY' if t['ml_signal'] == 1 else 'SELL'} ({t['ml_conf']:.1%})  |  "
        f"LLM: {t['llm_dir']} ({t['llm_conf']}/10)\n"
        f"Regime: {t['regime']}\n\n"
        f"Entry: {t['entry_price']:.6g} ({t['entry_time']:%H:%M})\n"
        f"Exit:  {t['exit_price']:.6g} ({t['exit_time']:%H:%M}) — "
        f"[bold]{t['exit_reason']}[/bold] after {t['bars']} bars "
        f"(~{t['bars'] * 5} min)\n\n"
        f"PnL: [{pnl_c}]{t['pnl_net']:+.3f}%[/{pnl_c}] net "
        f"({t['pnl_raw']:+.3f}% gross − 0.08% commission)\n\n"
        f"[dim italic]{t['llm_reason']}[/dim italic]",
        title=f"Trade #{num}  ({t['exit_reason']})",
        border_style=pnl_c,
    ))


def _print_summary(trades: list, symbol: str):
    if not trades:
        console.print("\n[dim]No trades generated.[/dim]")
        return

    wins = sum(1 for t in trades if t["pnl_net"] > 0)
    total_pnl = sum(t["pnl_net"] for t in trades)
    gross_pnl = sum(t["pnl_raw"] for t in trades)
    tp_n = sum(1 for t in trades if t["exit_reason"] == "TP")
    sl_n = sum(1 for t in trades if t["exit_reason"] == "SL")
    to_n = sum(1 for t in trades if t["exit_reason"] == "TIMEOUT")

    # how many times LLM agreed/disagreed/filtered
    ml_only_signals = len(trades)  # all passed combined filter
    agree = sum(1 for t in trades
                if (t["ml_signal"] == 1 and t["llm_dir"] == "LONG")
                or (t["ml_signal"] == -1 and t["llm_dir"] == "SHORT"))

    c = "green" if total_pnl > 0 else "red"

    console.print(Panel(
        f"[bold]Symbol:[/bold] {symbol}\n"
        f"[bold]Trades:[/bold] {len(trades)}  |  "
        f"Wins: {wins}  |  Losses: {len(trades) - wins}\n"
        f"[bold]Win Rate:[/bold] {wins / len(trades) * 100:.1f}%\n"
        f"[bold]Net PnL:[/bold] [{c}]{total_pnl:+.2f}%[/{c}]  "
        f"(Gross: {gross_pnl:+.2f}%)\n"
        f"[bold]Exits:[/bold] TP: {tp_n} | SL: {sl_n} | Timeout: {to_n}\n"
        f"[bold]ML+LLM Agreement:[/bold] {agree}/{len(trades)} trades",
        title="Summary (with LLM)",
        border_style="cyan",
    ))


def main():
    p = argparse.ArgumentParser(description="LLM-Enhanced Backtest")
    from src.config import config as _cfg
    # NOTE: LLM on historical data almost always says NEUTRAL (no live context),
    # so combined confidence rarely exceeds 0.55. Default threshold is 0.30 here
    # to allow ML-driven trades with LLM analysis attached.
    p.add_argument("--pair", type=str, default="DOT/USDT", help="Pair (default DOT/USDT)")
    p.add_argument("--trades", type=int, default=10, help="Max trades (default 10)")
    p.add_argument("--confidence", type=float, default=0.30, help="Min combined confidence (default 0.30 — lower than live because LLM lacks context)")
    p.add_argument("--tp", type=float, default=_cfg.TP_ATR_MULTIPLIER, help=f"TP in ATR (default {_cfg.TP_ATR_MULTIPLIER})")
    p.add_argument("--sl", type=float, default=_cfg.SL_ATR_MULTIPLIER, help=f"SL in ATR (default {_cfg.SL_ATR_MULTIPLIER})")
    args = p.parse_args()

    run_llm_backtest(
        symbol=args.pair,
        max_trades=args.trades,
        min_confidence=args.confidence,
        tp_mult=args.tp,
        sl_mult=args.sl,
    )


if __name__ == "__main__":
    main()
