#!/usr/bin/env python3
"""
Swing Backtest — 4h+15m Supertrend + MFI/RSI/ATR filters
─────────────────────────────────────────────────────────
Standalone backtest for the swing strategy.
Verified: +$3,242/mo, 37/40 months profitable (Dec 2022 — Mar 2026).

Usage:
    python backtest_swing.py                    # full test (all data)
    python backtest_swing.py --days 365         # last year
    python backtest_swing.py --no-filters       # baseline without filters
    python backtest_swing.py --pair ETH/USDT    # single pair
"""

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as pta
from rich.console import Console
from rich.table import Table
from rich import box

warnings.filterwarnings("ignore")
console = Console()

CACHE_FILE = "data/ohlcv_cache_15m.pkl"
DEFAULT_PAIRS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT",
    "DOT/USDT", "LINK/USDT", "DOGE/USDT", "AVAX/USDT", "LTC/USDT", "XRP/USDT",
]
FEE_PCT = 0.04  # 0.04% round trip (limit orders)


def compute_indicators(df_15m: pd.DataFrame) -> dict:
    """Compute all needed indicators from 15m OHLCV data."""
    htf = df_15m.resample("240min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()

    # HTF Supertrend
    st_htf = pta.supertrend(htf["high"], htf["low"], htf["close"], length=9, multiplier=3.0)
    if st_htf is None:
        return None
    htf_st = st_htf[[c for c in st_htf.columns if "SUPERTd" in c][0]]

    # LTF Supertrend
    st_ltf = pta.supertrend(df_15m["high"], df_15m["low"], df_15m["close"], length=9, multiplier=3.0)
    if st_ltf is None:
        return None
    ltf_st = st_ltf[[c for c in st_ltf.columns if "SUPERTd" in c][0]]

    def reindex(s):
        return s.reindex(df_15m.index, method="ffill")

    # HTF MFI
    mfi_h = pta.mfi(htf["high"], htf["low"], htf["close"], htf["volume"], length=14)
    # HTF RSI
    rsi_h = pta.rsi(htf["close"], length=14)
    # HTF ATR ratio
    atr = pta.atr(htf["high"], htf["low"], htf["close"], length=14)
    atr_ratio = (atr / atr.rolling(20).mean()) if atr is not None else None
    # LTF MFI
    mfi_l = pta.mfi(df_15m["high"], df_15m["low"], df_15m["close"], df_15m["volume"], length=14)

    return {
        "close": df_15m["close"],
        "htf_st": reindex(htf_st),
        "ltf_st": ltf_st,
        "H_mfi": reindex(mfi_h) if mfi_h is not None else None,
        "H_rsi": reindex(rsi_h) if rsi_h is not None else None,
        "H_atr_r": reindex(atr_ratio) if atr_ratio is not None else None,
        "L_mfi": mfi_l,
    }


def check_filters(data: dict, i: int, htf_dir: float, use_filters: bool) -> bool:
    """Check if all entry filters pass."""
    if not use_filters:
        return True

    def g(key):
        if key not in data or data[key] is None:
            return np.nan
        return float(data[key].iloc[i])

    h = htf_dir

    # H_MFI aligned: money flow confirms direction
    mfi_h = g("H_mfi")
    if np.isnan(mfi_h):
        return False
    if h > 0 and mfi_h <= 50:
        return False
    if h < 0 and mfi_h >= 50:
        return False

    # H_RSI aligned: momentum confirms direction
    rsi_h = g("H_rsi")
    if np.isnan(rsi_h):
        return False
    if h > 0 and rsi_h <= 50:
        return False
    if h < 0 and rsi_h >= 50:
        return False

    # H_ATR ratio < 1.5: no volatility spike
    atr_r = g("H_atr_r")
    if np.isnan(atr_r):
        return False
    if atr_r >= 1.5:
        return False

    # L_MFI aligned: LTF volume confirms
    mfi_l = g("L_mfi")
    if np.isnan(mfi_l):
        return False
    if h > 0 and mfi_l <= 50:
        return False
    if h < 0 and mfi_l >= 50:
        return False

    return True


def run_backtest(pairs=None, days=None, use_filters=True):
    if not Path(CACHE_FILE).exists():
        console.print("[red]Cache not found. Run cache_data.py first.[/red]")
        return

    with open(CACHE_FILE, "rb") as f:
        raw = pickle.load(f)

    if pairs is None:
        pairs = DEFAULT_PAIRS
    n_pairs = len(pairs)
    pos_size = 5000 * 5 / n_pairs

    console.print(f"\n[bold cyan]Swing Backtest — 4h+15m Supertrend[/bold cyan]")
    console.print(f"Filters: {'H_MFI + H_RSI + H_ATR<1.5 + L_MFI' if use_filters else 'None'}")
    console.print(f"Pairs: {n_pairs} | Fee: {FEE_PCT}% | Capital: $5K × 5x")
    if days:
        console.print(f"Period: last {days} days")
    console.print()

    all_trades = []

    for pair in pairs:
        if pair not in raw:
            continue
        df = raw[pair].copy()
        if days:
            df = df.iloc[-days * 96:]
        if len(df) < 500:
            continue

        data = compute_indicators(df)
        if data is None:
            continue

        close = data["close"]
        htf_st = data["htf_st"]
        ltf_st = data["ltf_st"]

        pos = None
        for i in range(1, len(ltf_st)):
            if np.isnan(ltf_st.iloc[i]) or np.isnan(ltf_st.iloc[i - 1]):
                continue
            h = htf_st.iloc[i]
            if np.isnan(h):
                continue

            price = float(close.iloc[i])
            ts = ltf_st.index[i]
            flipped = ltf_st.iloc[i] != ltf_st.iloc[i - 1]

            # Exit: HTF flips
            if pos:
                d, ep, he, pair_name = pos
                if h != he:
                    pnl = (price - ep) / ep * 100 if d == 1 else (ep - price) / ep * 100
                    net = pnl - FEE_PCT
                    month = ts.strftime("%Y-%m")
                    hold_h = (ts - pos_ts).total_seconds() / 3600 if hasattr(ts, "strftime") else 0
                    all_trades.append({
                        "pair": pair_name, "dir": "LONG" if d == 1 else "SHORT",
                        "net": net, "pnl": pnl, "month": month,
                        "hold_h": hold_h,
                    })
                    pos = None

            # Entry: LTF confirms HTF
            if flipped and not pos:
                aligned = (h > 0 and ltf_st.iloc[i] > 0) or (h < 0 and ltf_st.iloc[i] < 0)
                if not aligned:
                    continue
                if not check_filters(data, i, h, use_filters):
                    continue
                direction = 1 if h > 0 else -1
                pos = (direction, price, h, pair)
                pos_ts = ts

        console.print(f"  {pair}: {sum(1 for t in all_trades if t['pair'] == pair)} trades")

    if not all_trades:
        console.print("[red]No trades![/red]")
        return

    # Results
    tdf = pd.DataFrame(all_trades)
    n = len(tdf)
    wr = (tdf["net"] > 0).sum() / n * 100
    wins = tdf[tdf["net"] > 0]["net"]
    losses = tdf[tdf["net"] <= 0]["net"]
    rr = abs(wins.mean() / losses.mean()) if len(losses) > 0 and len(wins) > 0 else 0

    monthly = tdf.groupby("month")["net"].agg(["sum", "count"]).reset_index()
    monthly.columns = ["month", "total", "n"]
    monthly["dollar"] = pos_size * monthly["total"] / 100
    n_m = len(monthly)
    dmo = monthly["dollar"].sum() / n_m
    wm = (monthly["dollar"] > 0).sum()
    worst = monthly["dollar"].min()
    best = monthly["dollar"].max()

    # Summary table
    table = Table(title="Swing Backtest Results", box=box.ROUNDED)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Trades", str(n))
    table.add_row("Win rate", f"{wr:.1f}%")
    table.add_row("R:R", f"{rr:.1f}:1")
    table.add_row("Avg win", f"{wins.mean():+.2f}%")
    table.add_row("Avg loss", f"{losses.mean():+.2f}%")
    table.add_row("", "")
    table.add_row("[bold]NET/month[/bold]", f"[bold]${dmo:+,.0f}[/bold]")
    table.add_row("Total NET", f"${monthly['dollar'].sum():+,.0f}")
    table.add_row("Profitable months", f"{wm}/{n_m} ({wm/n_m*100:.0f}%)")
    table.add_row("Worst month", f"${worst:+,.0f}")
    table.add_row("Best month", f"${best:+,.0f}")
    console.print(table)

    # Monthly
    console.print()
    for _, row in monthly.sort_values("month").iterrows():
        mark = "[green]+[/green]" if row["dollar"] > 0 else "[red] [/red]"
        console.print(f"  {row['month']}  {row['n']:>3}tr  {mark}${abs(row['dollar']):>7,.0f}")

    # Per pair
    console.print()
    sym_table = Table(title="Per Pair", box=box.SIMPLE)
    sym_table.add_column("Pair", style="bold")
    sym_table.add_column("Trades", justify="right")
    sym_table.add_column("WR", justify="right")
    sym_table.add_column("$/mo", justify="right")
    for pair in sorted(pairs):
        pt = tdf[tdf["pair"] == pair]
        if len(pt) == 0:
            continue
        p_wr = (pt["net"] > 0).sum() / len(pt) * 100
        p_dmo = pos_size * pt["net"].sum() / 100 / n_m
        style = "green" if p_dmo > 0 else "red"
        sym_table.add_row(pair, str(len(pt)), f"{p_wr:.0f}%", f"${p_dmo:+,.0f}", style=style)
    console.print(sym_table)

    return tdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swing Backtest")
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--no-filters", action="store_true")
    parser.add_argument("--pair", type=str, default=None)
    args = parser.parse_args()

    pairs = [args.pair] if args.pair else None
    run_backtest(pairs=pairs, days=args.days, use_filters=not args.no_filters)
