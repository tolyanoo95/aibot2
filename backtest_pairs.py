#!/usr/bin/env python3
"""
Pairs Trading (Statistical Arbitrage) Backtest.

Trades the spread/ratio between correlated crypto pairs.
When ratio deviates from rolling mean → bet on reversion.
Market-neutral: doesn't depend on direction, only relative movement.
"""

import argparse
import logging
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple

import functools
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

import ccxt
import time as _time

from src.config import config
from src.data_fetcher import BinanceDataFetcher
from src.indicators import TechnicalIndicators


def fetch_spot_ohlcv(symbol: str, timeframe: str = "15m", total_candles: int = 18000) -> pd.DataFrame:
    """Fetch spot OHLCV data from Binance (not futures)."""
    exchange = ccxt.binance({"enableRateLimit": True})
    tf_ms = {"1m": 60000, "5m": 300000, "15m": 900000, "1h": 3600000}[timeframe]
    now_ms = int(_time.time() * 1000)
    since = now_ms - total_candles * tf_ms
    frames = []
    while since < now_ms:
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not data:
                break
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            frames.append(df)
            since = int(df.index[-1].timestamp() * 1000) + tf_ms
            _time.sleep(0.05)
        except Exception:
            break
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames)
    return result[~result.index.duplicated(keep="last")]

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
print = functools.partial(print, flush=True)
console = Console()


PAIR_COMBOS = [
    ("BTC/USDT", "ETH/USDT"),
    ("ETH/USDT", "SOL/USDT"),
    ("BTC/USDT", "BNB/USDT"),
    ("LINK/USDT", "DOT/USDT"),
    ("ADA/USDT", "DOT/USDT"),
    ("LTC/USDT", "BTC/USDT"),
    ("SOL/USDT", "AVAX/USDT"),
]


@dataclass
class PairTrade:
    pair_a: str
    pair_b: str
    direction: str           # "LONG_A_SHORT_B" or "SHORT_A_LONG_B"
    entry_ratio: float
    mean_ratio: float
    entry_bar: int
    entry_price_a: float
    entry_price_b: float
    entries: int = 1
    exit_bar: int = -1
    exit_price_a: float = 0.0
    exit_price_b: float = 0.0
    exit_reason: str = ""
    pnl_pct: float = 0.0


def compute_pair_pnl(trade: PairTrade) -> float:
    """Compute PnL from both legs of the pair trade."""
    if trade.direction == "LONG_A_SHORT_B":
        pnl_a = (trade.exit_price_a - trade.entry_price_a) / trade.entry_price_a * 100
        pnl_b = (trade.entry_price_b - trade.exit_price_b) / trade.entry_price_b * 100
    else:
        pnl_a = (trade.entry_price_a - trade.exit_price_a) / trade.entry_price_a * 100
        pnl_b = (trade.exit_price_b - trade.entry_price_b) / trade.entry_price_b * 100
    return (pnl_a + pnl_b) / 2 * trade.entries / 3


def run_pairs_backtest(
    total_days: int = 187,
    lookback: int = 96,        # rolling mean window (96 bars = 1 day on 15m)
    entry_z: float = 2.0,      # z-score to enter (2 std devs from mean)
    exit_z: float = 0.5,       # z-score to exit (close to mean)
    hard_sl_z: float = 6.0,    # z-score for hard SL
    max_hold: int = 96,        # max bars to hold (24 hours)
    max_dca: int = 3,          # DCA entries on deeper deviation
    dca_step_z: float = 0.5,   # additional z per DCA level
    max_open: int = 3,         # max simultaneous pair trades
    use_spot: bool = False,    # use spot basis filter + spot ratio confirmation
    basis_max: float = 0.3,    # max basis % to allow trading
):
    BARS_15M = 96
    total_candles = total_days * BARS_15M

    console.print(f"\n[bold cyan]Pairs Trading Backtest[/bold cyan]")
    console.print(f"Days: {total_days} | Lookback: {lookback} bars")
    console.print(f"Entry: {entry_z}σ | Exit: {exit_z}σ | SL: {hard_sl_z}σ")
    console.print(f"DCA: max {max_dca} (step {dca_step_z}σ) | max_hold: {max_hold}\n")

    fetcher = BinanceDataFetcher(config)
    indicators = TechnicalIndicators()

    # Fetch data for all symbols we need
    needed_symbols = set()
    for a, b in PAIR_COMBOS:
        needed_symbols.add(a)
        needed_symbols.add(b)

    all_data = {}
    for symbol in sorted(needed_symbols):
        df = fetcher.fetch_ohlcv_extended(symbol, "15m", total_candles=total_candles)
        if df.empty or len(df) < 200:
            console.print(f"  [red]{symbol}: not enough data[/red]")
            continue
        all_data[symbol] = df
        console.print(f"  {symbol}: {len(df)} bars")

    # Fetch spot data if enabled
    spot_data = {}
    if use_spot:
        console.print(f"\n[cyan]Fetching SPOT data for basis filter...[/cyan]")
        for symbol in sorted(needed_symbols):
            spot_df = fetch_spot_ohlcv(symbol, "15m", total_candles=total_candles)
            if not spot_df.empty:
                spot_data[symbol] = spot_df
                console.print(f"  {symbol} spot: {len(spot_df)} bars")

    # Run backtest per pair combo
    all_trades: List[PairTrade] = []
    combo_results = {}

    for sym_a, sym_b in PAIR_COMBOS:
        if sym_a not in all_data or sym_b not in all_data:
            continue

        df_a = all_data[sym_a]
        df_b = all_data[sym_b]

        # Align indices
        common_idx = df_a.index.intersection(df_b.index)
        if len(common_idx) < lookback + 100:
            continue

        close_a = df_a.loc[common_idx, "close"].values
        close_b = df_b.loc[common_idx, "close"].values

        # Spot data: basis and spot ratio
        has_spot = use_spot and sym_a in spot_data and sym_b in spot_data
        if has_spot:
            sp_a = spot_data[sym_a].reindex(common_idx, method="ffill")["close"].values
            sp_b = spot_data[sym_b].reindex(common_idx, method="ffill")["close"].values
            basis_a = (close_a - sp_a) / sp_a * 100
            basis_b = (close_b - sp_b) / sp_b * 100
            spot_ratio = sp_a / sp_b
            spot_ratio_s = pd.Series(spot_ratio)
            spot_ratio_mean = spot_ratio_s.rolling(lookback, min_periods=lookback // 2).mean().values
            spot_ratio_std = spot_ratio_s.rolling(lookback, min_periods=lookback // 2).std().values
        else:
            basis_a = basis_b = np.zeros(len(common_idx))
            spot_ratio_mean = spot_ratio_std = None

        # Compute ratio and z-score
        ratio = close_a / close_b
        ratio_series = pd.Series(ratio)
        ratio_mean = ratio_series.rolling(lookback, min_periods=lookback // 2).mean().values
        ratio_std = ratio_series.rolling(lookback, min_periods=lookback // 2).std().values

        name_a = sym_a.split("/")[0]
        name_b = sym_b.split("/")[0]
        combo_name = f"{name_a}/{name_b}"

        open_trades: List[PairTrade] = []
        combo_trades: List[PairTrade] = []
        cooldown_until = 0

        for i in range(lookback, len(ratio)):
            if np.isnan(ratio_mean[i]) or np.isnan(ratio_std[i]) or ratio_std[i] <= 0:
                continue

            z = (ratio[i] - ratio_mean[i]) / ratio_std[i]

            # Close existing trades
            for t in list(open_trades):
                bars_held = i - t.entry_bar
                cur_z = z

                # TP: z-score reverted to exit_z
                if t.direction == "SHORT_A_LONG_B":
                    hit_tp = cur_z <= exit_z
                    hit_sl = cur_z >= hard_sl_z
                else:
                    hit_tp = cur_z >= -exit_z
                    hit_sl = cur_z <= -hard_sl_z

                closed = False
                if hit_tp:
                    t.exit_reason = "TP"
                    closed = True
                elif hit_sl:
                    t.exit_reason = "HARD_SL"
                    closed = True
                elif bars_held >= max_hold:
                    t.exit_reason = "TIMEOUT"
                    closed = True

                if closed:
                    t.exit_bar = i
                    t.exit_price_a = close_a[i]
                    t.exit_price_b = close_b[i]
                    t.pnl_pct = compute_pair_pnl(t)
                    open_trades.remove(t)
                    combo_trades.append(t)
                    cooldown_until = i + 4
                    continue

                # DCA: add on deeper deviation
                if t.entries < max_dca:
                    next_z = entry_z + t.entries * dca_step_z
                    if t.direction == "SHORT_A_LONG_B" and cur_z >= next_z:
                        t.entries += 1
                        t.entry_price_a = (t.entry_price_a * (t.entries - 1) + close_a[i]) / t.entries
                        t.entry_price_b = (t.entry_price_b * (t.entries - 1) + close_b[i]) / t.entries
                        t.entry_ratio = (t.entry_ratio * (t.entries - 1) + ratio[i]) / t.entries
                    elif t.direction == "LONG_A_SHORT_B" and cur_z <= -next_z:
                        t.entries += 1
                        t.entry_price_a = (t.entry_price_a * (t.entries - 1) + close_a[i]) / t.entries
                        t.entry_price_b = (t.entry_price_b * (t.entries - 1) + close_b[i]) / t.entries
                        t.entry_ratio = (t.entry_ratio * (t.entries - 1) + ratio[i]) / t.entries

            # Open new trade
            if len(open_trades) >= max_open:
                continue
            if i < cooldown_until:
                continue
            if any(t.pair_a == sym_a and t.pair_b == sym_b for t in open_trades):
                continue

            # Spot basis filter: skip when futures premium is extreme
            if has_spot:
                if abs(basis_a[i]) > basis_max or abs(basis_b[i]) > basis_max:
                    continue
                # Spot ratio confirmation: spot ratio must also deviate
                if spot_ratio_std is not None and not np.isnan(spot_ratio_std[i]) and spot_ratio_std[i] > 0:
                    spot_z = (spot_ratio[i] - spot_ratio_mean[i]) / spot_ratio_std[i]
                    if z >= entry_z and spot_z < entry_z * 0.5:
                        continue  # futures deviates but spot doesn't → manipulation
                    if z <= -entry_z and spot_z > -entry_z * 0.5:
                        continue

            if z >= entry_z:
                # Ratio too high → SHORT A, LONG B (bet ratio goes down)
                trade = PairTrade(
                    pair_a=sym_a, pair_b=sym_b,
                    direction="SHORT_A_LONG_B",
                    entry_ratio=ratio[i], mean_ratio=ratio_mean[i],
                    entry_bar=i,
                    entry_price_a=close_a[i], entry_price_b=close_b[i],
                )
                open_trades.append(trade)

            elif z <= -entry_z:
                # Ratio too low → LONG A, SHORT B (bet ratio goes up)
                trade = PairTrade(
                    pair_a=sym_a, pair_b=sym_b,
                    direction="LONG_A_SHORT_B",
                    entry_ratio=ratio[i], mean_ratio=ratio_mean[i],
                    entry_bar=i,
                    entry_price_a=close_a[i], entry_price_b=close_b[i],
                )
                open_trades.append(trade)

        # Close remaining
        for t in open_trades:
            t.exit_bar = len(ratio) - 1
            t.exit_price_a = close_a[-1]
            t.exit_price_b = close_b[-1]
            t.exit_reason = "END"
            t.pnl_pct = compute_pair_pnl(t)
            combo_trades.append(t)

        if combo_trades:
            c_pnl = sum(t.pnl_pct for t in combo_trades)
            c_wr = sum(1 for t in combo_trades if t.pnl_pct > 0) / len(combo_trades) * 100
            c_dca = sum(1 for t in combo_trades if t.entries > 1)
            combo_results[combo_name] = {
                "trades": len(combo_trades), "wr": c_wr, "pnl": c_pnl, "dca": c_dca,
            }
            console.print(f"  {combo_name}: {len(combo_trades)} trades ({c_dca} DCA'd), WR {c_wr:.0f}%, PnL {c_pnl:+.2f}%")

        all_trades.extend(combo_trades)

    # ── Summary ──
    console.print("\n" + "=" * 60)
    if not all_trades:
        console.print("[red]No trades![/red]")
        return

    total_pnl = sum(t.pnl_pct for t in all_trades)
    total_wins = sum(1 for t in all_trades if t.pnl_pct > 0)
    total_count = len(all_trades)
    total_wr = total_wins / total_count * 100

    tp_count = sum(1 for t in all_trades if t.exit_reason == "TP")
    sl_count = sum(1 for t in all_trades if t.exit_reason == "HARD_SL")
    to_count = sum(1 for t in all_trades if t.exit_reason in ("TIMEOUT", "END"))
    sl_pnl = sum(t.pnl_pct for t in all_trades if t.exit_reason == "HARD_SL")
    to_pnl = sum(t.pnl_pct for t in all_trades if t.exit_reason in ("TIMEOUT", "END"))
    dca_total = sum(1 for t in all_trades if t.entries > 1)

    # Average bars held
    avg_bars = np.mean([t.exit_bar - t.entry_bar for t in all_trades if t.exit_bar > 0])

    table = Table(title="Pairs Trading Results", box=box.ROUNDED)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Total trades", str(total_count))
    table.add_row("Win rate", f"{total_wr:.1f}%")
    table.add_row("Total PnL", f"{total_pnl:+.2f}%")
    table.add_row("PnL/month", f"{total_pnl / total_days * 30:+.2f}%")
    table.add_row("", "")
    table.add_row("TP", str(tp_count))
    table.add_row("HARD_SL", f"{sl_count} (PnL {sl_pnl:+.2f}%)")
    table.add_row("TIMEOUT/END", f"{to_count} (PnL {to_pnl:+.2f}%)")
    table.add_row("DCA'd", str(dca_total))
    table.add_row("Avg bars held", f"{avg_bars:.1f}")
    table.add_row("", "")
    table.add_row("Avg PnL/trade", f"{total_pnl / total_count:+.4f}%")
    table.add_row("Days", str(total_days))
    table.add_row("Pairs", str(len(combo_results)))
    console.print(table)

    # Per-combo breakdown
    sym_table = Table(title="Per-Pair Combo", box=box.SIMPLE)
    sym_table.add_column("Combo")
    sym_table.add_column("Trades", justify="right")
    sym_table.add_column("WR", justify="right")
    sym_table.add_column("PnL", justify="right")
    sym_table.add_column("DCA", justify="right")
    for name in sorted(combo_results.keys(), key=lambda x: combo_results[x]["pnl"], reverse=True):
        r = combo_results[name]
        sym_table.add_row(name, str(r["trades"]), f"{r['wr']:.0f}%", f"{r['pnl']:+.2f}%", str(r["dca"]))
    console.print(sym_table)

    # Per 5-day window (fold equivalent)
    console.print("\n  Per 5-day window:")
    BARS_15M = 96
    window = 5 * BARS_15M
    w = 0
    while w * window < max(t.exit_bar for t in all_trades):
        w_start = w * window
        w_end = (w + 1) * window
        w_trades = [t for t in all_trades if w_start <= t.entry_bar < w_end]
        if w_trades:
            w_pnl = sum(t.pnl_pct for t in w_trades)
            w_wr = sum(1 for t in w_trades if t.pnl_pct > 0) / len(w_trades) * 100
            console.print(f"    Window {w+1}: {len(w_trades)} trades, WR {w_wr:.0f}%, PnL {w_pnl:+.2f}%")
        w += 1

    return all_trades


@dataclass
class OneLegTrade:
    symbol: str
    signal_pair: str
    direction: str
    entry_price: float
    entry_bar: int
    entry_z: float
    entries: int = 1
    avg_price: float = 0.0
    exit_bar: int = -1
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl_pct: float = 0.0


def run_oneleg_backtest(
    total_days: int = 187,
    lookback: int = 96,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    sl_pct: float = 3.0,
    max_hold: int = 96,
    max_dca: int = 3,
    dca_step_pct: float = 1.0,
    max_open: int = 3,
):
    """
    1-leg pairs trading: use ratio as SIGNAL, trade only one USDT pair.
    Cheaper (half fees), simpler, but not market-neutral.
    """
    BARS_15M = 96
    total_candles = total_days * BARS_15M

    console.print(f"\n[bold cyan]1-Leg Pairs Trading (ratio signal, USDT execution)[/bold cyan]")
    console.print(f"Days: {total_days} | Lookback: {lookback} | Entry: {entry_z}σ | SL: {sl_pct}%\n")

    fetcher = BinanceDataFetcher(config)
    indicators = TechnicalIndicators()

    needed_symbols = set()
    for a, b in PAIR_COMBOS:
        needed_symbols.add(a)
        needed_symbols.add(b)

    all_data = {}
    for symbol in sorted(needed_symbols):
        df = fetcher.fetch_ohlcv_extended(symbol, "15m", total_candles=total_candles)
        if df.empty or len(df) < 200:
            continue
        all_data[symbol] = df
        console.print(f"  {symbol}: {len(df)} bars")

    all_trades: List[OneLegTrade] = []
    combo_results = {}

    for sym_a, sym_b in PAIR_COMBOS:
        if sym_a not in all_data or sym_b not in all_data:
            continue

        df_a = all_data[sym_a]
        df_b = all_data[sym_b]
        common_idx = df_a.index.intersection(df_b.index)
        if len(common_idx) < lookback + 100:
            continue

        close_a = df_a.loc[common_idx, "close"].values
        close_b = df_b.loc[common_idx, "close"].values

        ratio = close_a / close_b
        ratio_s = pd.Series(ratio)
        ratio_mean = ratio_s.rolling(lookback, min_periods=lookback // 2).mean().values
        ratio_std = ratio_s.rolling(lookback, min_periods=lookback // 2).std().values

        name_a = sym_a.split("/")[0]
        name_b = sym_b.split("/")[0]
        combo_name = f"{name_a}/{name_b}"

        open_trades: List[OneLegTrade] = []
        combo_trades: List[OneLegTrade] = []
        cooldown_until = 0

        for i in range(lookback, len(ratio)):
            if np.isnan(ratio_mean[i]) or np.isnan(ratio_std[i]) or ratio_std[i] <= 0:
                continue

            z = (ratio[i] - ratio_mean[i]) / ratio_std[i]

            for t in list(open_trades):
                bars_held = i - t.entry_bar
                cur_z = z

                # TP: ratio reverted
                if t.direction == "SHORT" and t.signal_pair == sym_a:
                    hit_tp = cur_z <= exit_z
                elif t.direction == "LONG" and t.signal_pair == sym_b:
                    hit_tp = cur_z <= exit_z
                elif t.direction == "LONG" and t.signal_pair == sym_a:
                    hit_tp = cur_z >= -exit_z
                else:
                    hit_tp = cur_z >= -exit_z

                # SL: price-based
                if t.direction == "LONG":
                    cur_price = close_a[i] if t.symbol == sym_a else close_b[i]
                    hit_sl = cur_price <= t.avg_price * (1 - sl_pct / 100)
                else:
                    cur_price = close_a[i] if t.symbol == sym_a else close_b[i]
                    hit_sl = cur_price >= t.avg_price * (1 + sl_pct / 100)

                closed = False
                if hit_tp:
                    t.exit_reason = "TP"
                    closed = True
                elif hit_sl:
                    t.exit_reason = "SL"
                    closed = True
                elif bars_held >= max_hold:
                    t.exit_reason = "TIMEOUT"
                    closed = True

                if closed:
                    t.exit_bar = i
                    t.exit_price = close_a[i] if t.symbol == sym_a else close_b[i]
                    if t.direction == "LONG":
                        t.pnl_pct = (t.exit_price - t.avg_price) / t.avg_price * 100
                    else:
                        t.pnl_pct = (t.avg_price - t.exit_price) / t.avg_price * 100
                    open_trades.remove(t)
                    combo_trades.append(t)
                    cooldown_until = i + 4
                    continue

                # DCA
                if t.entries < max_dca:
                    cur_price = close_a[i] if t.symbol == sym_a else close_b[i]
                    if t.direction == "LONG" and cur_price <= t.avg_price * (1 - dca_step_pct * t.entries / 100):
                        t.entries += 1
                        t.avg_price = (t.avg_price * (t.entries - 1) + cur_price) / t.entries
                    elif t.direction == "SHORT" and cur_price >= t.avg_price * (1 + dca_step_pct * t.entries / 100):
                        t.entries += 1
                        t.avg_price = (t.avg_price * (t.entries - 1) + cur_price) / t.entries

            if len(open_trades) >= max_open or i < cooldown_until:
                continue

            if z >= entry_z:
                # Ratio high = A overpriced vs B → SHORT A or LONG B
                # Pick the one with stronger signal: SHORT the overpriced one
                if not any(t.symbol == sym_a for t in open_trades):
                    t = OneLegTrade(
                        symbol=sym_a, signal_pair=sym_a,
                        direction="SHORT", entry_price=close_a[i],
                        entry_bar=i, entry_z=z, avg_price=close_a[i],
                    )
                    open_trades.append(t)

            elif z <= -entry_z:
                # Ratio low = A underpriced vs B → LONG A
                if not any(t.symbol == sym_a for t in open_trades):
                    t = OneLegTrade(
                        symbol=sym_a, signal_pair=sym_a,
                        direction="LONG", entry_price=close_a[i],
                        entry_bar=i, entry_z=z, avg_price=close_a[i],
                    )
                    open_trades.append(t)

        for t in open_trades:
            t.exit_bar = len(ratio) - 1
            t.exit_price = close_a[-1] if t.symbol == sym_a else close_b[-1]
            t.exit_reason = "END"
            if t.direction == "LONG":
                t.pnl_pct = (t.exit_price - t.avg_price) / t.avg_price * 100
            else:
                t.pnl_pct = (t.avg_price - t.exit_price) / t.avg_price * 100
            combo_trades.append(t)

        if combo_trades:
            c_pnl = sum(t.pnl_pct for t in combo_trades)
            c_wr = sum(1 for t in combo_trades if t.pnl_pct > 0) / len(combo_trades) * 100
            combo_results[combo_name] = {"trades": len(combo_trades), "wr": c_wr, "pnl": c_pnl}
            console.print(f"  {combo_name}: {len(combo_trades)} trades, WR {c_wr:.0f}%, PnL {c_pnl:+.2f}%")

        all_trades.extend(combo_trades)

    console.print("\n" + "=" * 60)
    if not all_trades:
        console.print("[red]No trades![/red]")
        return

    total_pnl = sum(t.pnl_pct for t in all_trades)
    total_count = len(all_trades)
    total_wr = sum(1 for t in all_trades if t.pnl_pct > 0) / total_count * 100

    tp_c = sum(1 for t in all_trades if t.exit_reason == "TP")
    sl_c = sum(1 for t in all_trades if t.exit_reason == "SL")
    to_c = sum(1 for t in all_trades if t.exit_reason in ("TIMEOUT", "END"))

    table = Table(title="1-Leg Pairs Trading Results", box=box.ROUNDED)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Total trades", str(total_count))
    table.add_row("Win rate", f"{total_wr:.1f}%")
    table.add_row("Total PnL", f"{total_pnl:+.2f}%")
    table.add_row("PnL/month", f"{total_pnl / total_days * 30:+.2f}%")
    table.add_row("", "")
    table.add_row("TP", str(tp_c))
    table.add_row("SL", str(sl_c))
    table.add_row("TIMEOUT/END", str(to_c))
    table.add_row("Avg PnL/trade", f"{total_pnl / total_count:+.4f}%")
    console.print(table)

    sym_table = Table(title="Per-Combo", box=box.SIMPLE)
    sym_table.add_column("Combo")
    sym_table.add_column("Trades", justify="right")
    sym_table.add_column("WR", justify="right")
    sym_table.add_column("PnL", justify="right")
    for n in sorted(combo_results.keys(), key=lambda x: combo_results[x]["pnl"], reverse=True):
        r = combo_results[n]
        sym_table.add_row(n, str(r["trades"]), f"{r['wr']:.0f}%", f"{r['pnl']:+.2f}%")
    console.print(sym_table)
    return all_trades


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pairs Trading Backtest")
    parser.add_argument("--mode", default="2leg", choices=["2leg", "1leg"])
    parser.add_argument("--days", type=int, default=187)
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument("--sl-z", type=float, default=6.0)
    parser.add_argument("--max-hold", type=int, default=96)
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--spot", action="store_true", help="Use spot basis filter + ratio confirmation")
    parser.add_argument("--basis-max", type=float, default=0.3, help="Max basis %% to allow trading")
    args = parser.parse_args()

    if args.mode == "1leg":
        run_oneleg_backtest(
            total_days=args.days,
            lookback=args.lookback,
            entry_z=args.entry_z,
            exit_z=args.exit_z,
            max_hold=args.max_hold,
        )
    else:
        run_pairs_backtest(
            total_days=args.days,
            entry_z=args.entry_z,
            exit_z=args.exit_z,
            hard_sl_z=args.sl_z,
            max_hold=args.max_hold,
            lookback=args.lookback,
            use_spot=args.spot,
            basis_max=args.basis_max,
        )
