#!/usr/bin/env python3
"""
1-minute resolution backtest — most realistic simulation possible.

Signal generation: volume bars (same as live kline_1m → volume bar → signal)
Position management: raw 1m bars (same as live miniTicker → SL/TP/Move SL)

This matches the live bot architecture:
- kline_1m completes volume bar → check signal → open position
- miniTicker (~300ms) monitors SL/TP/Move SL → close position
- Here we use 1m bars as proxy for miniTicker (~1 min accuracy)

Usage:
    python backtest_1m.py              # default 35 days
    python backtest_1m.py --days 60    # custom period
"""
import argparse, pickle, logging, warnings
import numpy as np, pandas as pd
from dataclasses import dataclass, field as dc_field
from rich.console import Console
from rich.table import Table
from rich import box

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
console = Console()

from src.indicators import TechnicalIndicators
from backtest_volbars import resample_to_volume_bars
import pandas_ta as pta

ADX_MIN = 20
LONG_MOM = 0.30
SHORT_MOM = 0.10
SL_MULT = 2.0
TP_MULT = 4.0
MAX_DCA = 3
DCA_STEP = 1.0
COOLDOWN = 3
MAX_HOLD = 24
MAX_OPEN = 11
MOVE_SL_STEPS = [(0.25, 0.30), (0.5, 0.25), (1.0, 0.5), (2.0, 1.0)]
MOVE_SL_TRAIL = 1.0
MAKER_FEE = 0.0002
TAKER_FEE = 0.0005


@dataclass
class Pos:
    symbol: str
    direction: str
    entry_price: float
    avg_price: float
    hard_sl: float
    tp: float
    entry_atr: float
    entries: list = dc_field(default_factory=list)
    total_size: int = 1
    vol_bars_held: int = 0
    sl_step: int = 0
    best_price: float = 0.0
    sl_moved: bool = False
    max_price: float = 0.0
    min_price: float = 999999.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    open_1m_idx: int = 0


def check_signal(vdf, j):
    """Check signal on volume bar j (same guards as live bot)."""
    roc = float(vdf["roc_12"].iloc[j]) if "roc_12" in vdf.columns else 0
    if roc > LONG_MOM:
        d = "LONG"
    elif roc < -SHORT_MOM:
        d = "SHORT"
    else:
        return None

    htf_st = float(vdf["htf_supertrend"].iloc[j]) if "htf_supertrend" in vdf.columns else 0
    if not np.isnan(htf_st) and htf_st != 0:
        if d == "LONG" and htf_st < 0: return None
        if d == "SHORT" and htf_st > 0: return None

    atr = vdf["atr"].values
    atr_ma20 = pd.Series(atr).rolling(20, min_periods=1).mean().values
    ae = atr[j] / atr_ma20[j] if atr_ma20[j] > 0 else 1.0
    if ae > 1.5: return None

    adx = float(vdf["ADX_14"].iloc[j]) if "ADX_14" in vdf.columns else 25
    if adx < ADX_MIN: return None

    rsi_s = float(vdf["rsi"].diff(6).iloc[j]) if "rsi" in vdf.columns else 0
    if d == "LONG" and rsi_s < 0: return None
    if d == "SHORT" and rsi_s > 0: return None

    return {"direction": d, "atr": float(atr[j]), "adx": adx, "roc": roc}


def simulate_on_1m(positions, df_1m, idx_1m, symbol, vol_bar_count, cooldowns):
    """Process one 1m bar for all positions of this symbol.
    Returns list of closed positions."""
    closed = []
    h = float(df_1m["high"].iloc[idx_1m])
    l = float(df_1m["low"].iloc[idx_1m])
    c = float(df_1m["close"].iloc[idx_1m])

    for pos in list(positions):
        if pos.symbol != symbol:
            continue

        ea = pos.entry_atr
        if ea <= 0:
            continue

        pos.max_price = max(pos.max_price, h)
        pos.min_price = min(pos.min_price, l)

        # Move SL steps (activation by current 1m close — like WebSocket price)
        if pos.sl_step < len(MOVE_SL_STEPS):
            sa, st_ = MOVE_SL_STEPS[pos.sl_step]
            trig = False
            if pos.direction == "LONG" and c - pos.avg_price >= sa * ea:
                trig = True
            elif pos.direction == "SHORT" and pos.avg_price - c >= sa * ea:
                trig = True
            if trig:
                ns = pos.avg_price + st_ * ea if pos.direction == "LONG" else pos.avg_price - st_ * ea
                if pos.direction == "LONG":
                    pos.hard_sl = max(pos.hard_sl, ns)
                else:
                    pos.hard_sl = min(pos.hard_sl, ns)
                pos.sl_step += 1
                pos.sl_moved = True
        elif MOVE_SL_TRAIL > 0:
            if pos.direction == "LONG":
                pos.best_price = max(pos.best_price, h)
                ns = pos.best_price - MOVE_SL_TRAIL * ea
                if ns > pos.hard_sl:
                    pos.hard_sl = ns
            else:
                pos.best_price = min(pos.best_price, l)
                ns = pos.best_price + MOVE_SL_TRAIL * ea
                if ns < pos.hard_sl:
                    pos.hard_sl = ns

        # SL/TP checks (using 1m high/low — like WebSocket price range)
        exit_reason = None
        exit_price = c

        if pos.direction == "LONG":
            if l <= pos.hard_sl:
                exit_reason = "SL_MOVED" if pos.sl_moved else "HARD_SL"
                exit_price = pos.hard_sl
            elif h >= pos.tp:
                exit_reason = "TP"
                exit_price = pos.tp
        else:
            if h >= pos.hard_sl:
                exit_reason = "SL_MOVED" if pos.sl_moved else "HARD_SL"
                exit_price = pos.hard_sl
            elif l <= pos.tp:
                exit_reason = "TP"
                exit_price = pos.tp

        if exit_reason:
            if pos.direction == "LONG":
                pnl = (exit_price - pos.avg_price) / pos.avg_price * 100 * pos.total_size
            else:
                pnl = (pos.avg_price - exit_price) / pos.avg_price * 100 * pos.total_size
            fee = (MAKER_FEE + TAKER_FEE) * 100 * pos.total_size
            pos.pnl_pct = pnl - fee
            pos.exit_reason = exit_reason
            closed.append(pos)
            positions.remove(pos)
            cooldowns[symbol] = vol_bar_count + COOLDOWN

    return closed


def main():
    parser = argparse.ArgumentParser(description="1m Realistic Backtest")
    parser.add_argument("--days", type=int, default=35, help="Days (default: 35)")
    args = parser.parse_args()

    console.print(f"\n[bold cyan]1m Realistic Backtest v2 ({args.days} days)[/bold cyan]")
    console.print(f"Signals: volume bars | Exits: raw 1m bars")
    console.print(f"Move SL: {MOVE_SL_STEPS} + trail {MOVE_SL_TRAIL}")
    console.print(f"Fees: maker {MAKER_FEE*100:.2f}% + taker {TAKER_FEE*100:.2f}%\n")

    with open("data/ohlcv_cache_1m.pkl", "rb") as f:
        raw_1m = pickle.load(f)

    indicators = TechnicalIndicators()

    # Build data per symbol
    pair_data = {}
    for symbol, df_1m in raw_1m.items():
        # 15m from 1m for ATR
        df_15m = df_1m.resample("15min").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna()
        df_15m = indicators.calculate_all(df_15m)

        # Threshold from this period's 15m
        warmup = min(1920, len(df_15m))
        threshold = df_15m["volume"].iloc[:warmup].median() * 2
        htf_threshold = threshold * 5

        # Volume bars from 1m (fixed threshold, no rolling update)
        vdf = resample_to_volume_bars(df_1m, initial_threshold=threshold, warmup_bars=999999)
        if len(vdf) < 200:
            continue

        # HTF Supertrend
        htf_vdf = resample_to_volume_bars(df_1m, initial_threshold=htf_threshold, warmup_bars=999999)
        if len(htf_vdf) > 20:
            htf_vdf = indicators.calculate_all(htf_vdf)
            st = pta.supertrend(htf_vdf["high"], htf_vdf["low"], htf_vdf["close"], length=9, multiplier=3.0)
            if st is not None:
                for sc in st.columns:
                    if "SUPERTd" in sc:
                        vdf["htf_supertrend"] = st[sc].reindex(vdf.index, method="ffill")

        vdf = indicators.calculate_all(vdf)
        time_atr = df_15m["atr"].reindex(vdf.index, method="ffill")
        vdf["atr"] = time_atr.values

        # Map volume bar timestamps to 1m index positions
        vb_to_1m = []
        for ts in vdf.index:
            idx = df_1m.index.searchsorted(ts)
            vb_to_1m.append(min(idx, len(df_1m) - 1))

        pair_data[symbol] = {
            "vdf": vdf, "df_1m": df_1m, "vb_to_1m": vb_to_1m, "threshold": threshold
        }
        console.print(f"  {symbol}: {len(vdf)} vol bars (threshold={threshold:.0f})")

    console.print(f"\nLoaded {len(pair_data)} pairs\n")

    # Simulation: iterate volume bars for signals, 1m bars for exits
    all_trades = []
    positions = []
    cooldowns = {}
    vol_bar_counts = {}
    warmup = 200

    # Find common vol bar range
    min_vb = min(len(d["vdf"]) for d in pair_data.values())

    for vb_idx in range(warmup, min_vb):
        # For each symbol: check exits on 1m bars since last volume bar
        for symbol, pdata in pair_data.items():
            vdf = pdata["vdf"]
            df_1m = pdata["df_1m"]
            vb_to_1m = pdata["vb_to_1m"]

            # 1m range for this volume bar
            start_1m = vb_to_1m[vb_idx - 1] if vb_idx > 0 else 0
            end_1m = vb_to_1m[vb_idx]

            vbc = vol_bar_counts.get(symbol, 0)

            # Process each 1m bar in this volume bar (position management)
            for m in range(start_1m, end_1m):
                closed = simulate_on_1m(positions, df_1m, m, symbol, vbc, cooldowns)
                all_trades.extend(closed)

            # Volume bar completed — increment count
            vol_bar_counts[symbol] = vbc + 1

            # Vol_drop check on volume bar
            sym_pos = [p for p in positions if p.symbol == symbol]
            if sym_pos and len(vdf) > 3:
                for pos in list(sym_pos):
                    pos.vol_bars_held += 1
                    if pos.vol_bars_held >= 3:
                        j = vb_idx
                        vol_vals = vdf["volume"].values
                        vol_ma20 = pd.Series(vol_vals).rolling(20, min_periods=1).mean().values
                        if j >= 2 and vol_ma20[j] > 0 and vol_vals[j-2:j+1].mean() < vol_ma20[j] * 0.5:
                            c = float(df_1m["close"].iloc[min(end_1m, len(df_1m)-1)])
                            if pos.direction == "LONG":
                                pnl = (c - pos.avg_price) / pos.avg_price * 100 * pos.total_size
                            else:
                                pnl = (pos.avg_price - c) / pos.avg_price * 100 * pos.total_size
                            fee = (MAKER_FEE + TAKER_FEE) * 100 * pos.total_size
                            pos.pnl_pct = pnl - fee
                            pos.exit_reason = "VOL_DROP"
                            all_trades.append(pos)
                            positions.remove(pos)
                            cooldowns[symbol] = vol_bar_counts[symbol] + COOLDOWN

                    if pos in positions and pos.vol_bars_held >= MAX_HOLD:
                        c = float(df_1m["close"].iloc[min(end_1m, len(df_1m)-1)])
                        if pos.direction == "LONG":
                            pnl = (c - pos.avg_price) / pos.avg_price * 100 * pos.total_size
                        else:
                            pnl = (pos.avg_price - c) / pos.avg_price * 100 * pos.total_size
                        fee = (MAKER_FEE + TAKER_FEE) * 100 * pos.total_size
                        pos.pnl_pct = pnl - fee
                        pos.exit_reason = "TIMEOUT"
                        all_trades.append(pos)
                        positions.remove(pos)
                        cooldowns[symbol] = vol_bar_counts[symbol] + COOLDOWN

            # DCA check
            for pos in [p for p in positions if p.symbol == symbol]:
                if pos.total_size >= MAX_DCA:
                    continue
                j = vb_idx
                roc_v = float(vdf["roc_12"].iloc[j]) if "roc_12" in vdf.columns else 0
                if pos.direction == "LONG" and roc_v < 0: continue
                if pos.direction == "SHORT" and roc_v > 0: continue
                adx_v = float(vdf["ADX_14"].iloc[j]) if "ADX_14" in vdf.columns else 25
                dyn_max = 3 if adx_v >= 30 else (2 if adx_v >= 20 else 1)
                if pos.total_size >= min(MAX_DCA, dyn_max): continue
                ea = pos.entry_atr
                c_now = float(vdf["close"].iloc[j])
                dca_lvl = pos.entries[0] - pos.total_size * DCA_STEP * ea if pos.direction == "LONG" \
                    else pos.entries[0] + pos.total_size * DCA_STEP * ea
                triggered = (pos.direction == "LONG" and float(vdf["low"].iloc[j]) <= dca_lvl) or \
                            (pos.direction == "SHORT" and float(vdf["high"].iloc[j]) >= dca_lvl)
                if triggered:
                    pos.entries.append(dca_lvl)
                    pos.total_size += 1
                    pos.avg_price = sum(pos.entries) / pos.total_size
                    pos.tp = pos.avg_price + TP_MULT * ea if pos.direction == "LONG" else pos.avg_price - TP_MULT * ea

            # Signal check on completed volume bar
            if cooldowns.get(symbol, 0) > vol_bar_counts[symbol]:
                continue
            if any(p.symbol == symbol for p in positions):
                # Flip check
                existing = [p for p in positions if p.symbol == symbol][0]
                sig = check_signal(vdf, vb_idx)
                if sig and sig["direction"] != existing.direction:
                    c = float(vdf["close"].iloc[vb_idx])
                    if existing.direction == "LONG":
                        pnl = (c - existing.avg_price) / existing.avg_price * 100 * existing.total_size
                    else:
                        pnl = (existing.avg_price - c) / existing.avg_price * 100 * existing.total_size
                    fee = (MAKER_FEE + TAKER_FEE) * 100 * existing.total_size
                    existing.pnl_pct = pnl - fee
                    existing.exit_reason = "FLIP"
                    all_trades.append(existing)
                    positions.remove(existing)
                    cooldowns[symbol] = vol_bar_counts[symbol] + COOLDOWN
                    # Open new (fall through to entry below)
                else:
                    continue

            if len(positions) >= MAX_OPEN:
                continue

            sig = check_signal(vdf, vb_idx)
            if not sig:
                continue

            c = float(vdf["close"].iloc[vb_idx])
            ea = sig["atr"]
            if ea <= 0 or np.isnan(ea):
                continue

            if sig["direction"] == "LONG":
                sl = c - SL_MULT * ea
                tp = c + TP_MULT * ea
            else:
                sl = c + SL_MULT * ea
                tp = c - TP_MULT * ea

            pos = Pos(
                symbol=symbol, direction=sig["direction"],
                entry_price=c, avg_price=c, hard_sl=sl, tp=tp, entry_atr=ea,
                entries=[c], best_price=c, max_price=c, min_price=c,
                open_1m_idx=vb_to_1m[vb_idx],
            )
            positions.append(pos)

    # Results
    if not all_trades:
        console.print("[red]No trades![/red]")
        return

    n = len(all_trades)
    pos_size = 1000 * 5 / 11
    total_pnl = sum(t.pnl_pct for t in all_trades)
    wr = sum(1 for t in all_trades if t.pnl_pct > 0) / n * 100
    net = pos_size * total_pnl / 100
    days = args.days

    tp_c = sum(1 for t in all_trades if t.exit_reason == "TP")
    sl_c = sum(1 for t in all_trades if t.exit_reason in ("SL", "HARD_SL"))
    sm_c = sum(1 for t in all_trades if t.exit_reason == "SL_MOVED")
    vd_c = sum(1 for t in all_trades if t.exit_reason == "VOL_DROP")
    fl_c = sum(1 for t in all_trades if t.exit_reason == "FLIP")
    to_c = sum(1 for t in all_trades if t.exit_reason in ("TIMEOUT", "END"))

    avg_win = np.mean([t.pnl_pct for t in all_trades if t.pnl_pct > 0]) if any(t.pnl_pct > 0 for t in all_trades) else 0
    avg_loss = np.mean([t.pnl_pct for t in all_trades if t.pnl_pct <= 0]) if any(t.pnl_pct <= 0 for t in all_trades) else 0

    table = Table(title=f"1m Backtest v2 — Realistic ({days} days, $1K×5x)", box=box.ROUNDED)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Period", f"{days} days")
    table.add_row("Trades", str(n))
    table.add_row("[bold]Win Rate[/bold]", f"[bold]{wr:.1f}%[/bold]")
    table.add_row("", "")
    table.add_row("TP", str(tp_c))
    table.add_row("HARD_SL", str(sl_c))
    table.add_row("SL_MOVED", str(sm_c))
    table.add_row("VOL_DROP", str(vd_c))
    table.add_row("FLIP", str(fl_c))
    table.add_row("TIMEOUT", str(to_c))
    table.add_row("", "")
    table.add_row("[bold]NET profit[/bold]", f"[bold]${net:+,.2f}[/bold]")
    table.add_row("[bold]NET/month[/bold]", f"[bold]${net/days*30:+,.2f}[/bold]")
    table.add_row("", "")
    table.add_row("Avg win", f"{avg_win:+.3f}%")
    table.add_row("Avg loss", f"{avg_loss:+.3f}%")
    console.print(table)

    # Per symbol
    sym_table = Table(title="Per Symbol", box=box.SIMPLE)
    sym_table.add_column("Symbol", style="bold")
    sym_table.add_column("Trades", justify="right")
    sym_table.add_column("WR", justify="right")
    sym_table.add_column("NET $", justify="right")
    sym_table.add_column("TP", justify="right")
    sym_table.add_column("SL", justify="right")
    sym_table.add_column("SL_MOV", justify="right")

    for s in sorted(set(t.symbol for t in all_trades)):
        st = [t for t in all_trades if t.symbol == s]
        s_pnl = sum(t.pnl_pct for t in st)
        s_wr = sum(1 for t in st if t.pnl_pct > 0) / len(st) * 100
        s_net = pos_size * s_pnl / 100
        s_tp = sum(1 for t in st if t.exit_reason == "TP")
        s_sl = sum(1 for t in st if t.exit_reason in ("SL", "HARD_SL"))
        s_sm = sum(1 for t in st if t.exit_reason == "SL_MOVED")
        style = "green" if s_net > 0 else "red"
        sym_table.add_row(
            s.replace("/USDT", ""), str(len(st)), f"{s_wr:.0f}%",
            f"${s_net:+,.2f}", str(s_tp), str(s_sl), str(s_sm), style=style,
        )
    console.print(sym_table)


if __name__ == "__main__":
    main()
