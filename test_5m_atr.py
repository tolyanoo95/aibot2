#!/usr/bin/env python3
"""Test: volume bars from 15m, ATR from 5m vs 15m."""
import pickle, numpy as np, pandas as pd
from rich.console import Console; from rich.table import Table; from rich import box
console = Console()
from src.config import config
from src.indicators import TechnicalIndicators
from backtest_volbars import resample_to_volume_bars
from backtest_wf import simulate_dca_trades

with open("data/ohlcv_cache_15m.pkl", "rb") as f:
    raw_15m = pickle.load(f)
with open("data/ohlcv_cache_5m.pkl", "rb") as f:
    raw_5m = pickle.load(f)

indicators = TechnicalIndicators()

def prepare(atr_source="15m"):
    data = {}
    for sym in config.TRADING_PAIRS:
        if sym not in raw_15m: continue
        df = raw_15m[sym]
        if len(df) > 192 * 96:
            df = df.iloc[-192 * 96:]
        tdf = indicators.calculate_all(df.copy())
        vdf = resample_to_volume_bars(df)
        if len(vdf) < 1300: continue
        w = min(1920, len(df))
        htf = resample_to_volume_bars(df, initial_threshold=df["volume"].iloc[:w].median() * 10)
        if len(htf) > 20:
            htf = indicators.calculate_all(htf)
            for c in ["ema_9", "ema_21", "ema_50"]:
                if c in htf.columns:
                    vdf[f"htf_{c}"] = htf[c].reindex(vdf.index, method="ffill")
        vdf = indicators.calculate_all(vdf)

        if atr_source == "5m" and sym in raw_5m:
            df5 = raw_5m[sym]
            tdf5 = indicators.calculate_all(df5.copy())
            time_atr = tdf5["atr"].reindex(vdf.index, method="ffill")
        else:
            time_atr = tdf["atr"].reindex(vdf.index, method="ffill")
        vdf["atr"] = time_atr.values
        data[sym] = vdf
    return data

def run_test(data, sl=2.0, tp=4.0, train=1000, test_b=300):
    all_t = []; start = 0; gld = None; gss = {"LONG": 0, "SHORT": 0}
    ml = min(len(d) for d in data.values())
    while start + train + test_b <= ml:
        te = start + train; ee = te + test_b; ft = []
        for sym, df in data.items():
            if len(df) < ee: continue
            dt = df.iloc[te:ee]
            if len(dt) < 20: continue
            ct = dt["close"].values; at = dt["atr"].values
            am = pd.Series(at).rolling(20, min_periods=1).mean().values
            sigs = []
            for j in range(len(dt)):
                r = float(dt["roc_12"].iloc[j]) if "roc_12" in dt.columns else 0
                if r > 0.30: d2 = "LONG"
                elif r < -0.10: d2 = "SHORT"
                else: d2 = "NEUTRAL"
                c = 0
                if d2 != "NEUTRAL":
                    if gld == d2: d2 = "NEUTRAL"; continue
                    e9 = float(dt["htf_ema_9"].iloc[j]) if "htf_ema_9" in dt.columns else 0
                    e21 = float(dt["htf_ema_21"].iloc[j]) if "htf_ema_21" in dt.columns else 0
                    if e9 > 0 and e21 > 0:
                        if d2 == "LONG" and e9 < e21: d2 = "NEUTRAL"; continue
                        if d2 == "SHORT" and e9 > e21: d2 = "NEUTRAL"; continue
                    ax = at[j] / am[j] if am[j] > 0 else 1.0
                    if ax > 1.5: d2 = "NEUTRAL"; continue
                    adx = float(dt["ADX_14"].iloc[j]) if "ADX_14" in dt.columns else 25
                    if adx < 25: d2 = "NEUTRAL"; continue
                    rs = float(dt["rsi"].diff(6).iloc[j]) if "rsi" in dt.columns else 0
                    if d2 == "LONG" and rs < 0: d2 = "NEUTRAL"; continue
                    if d2 == "SHORT" and rs > 0: d2 = "NEUTRAL"; continue
                    c = 0.90
                sigs.append({"bar_idx": j, "symbol": sym, "direction": d2, "confidence": c})
            trades = simulate_dca_trades(dt, sigs, tp_mult=tp, dca_step_mult=1.0,
                max_entries=3, hard_sl_mult=sl, max_hold=24, max_open=11,
                cooldown=3, threshold=0.10, full_size_dca=True)
            ft.extend(trades)
        for tr in sorted(ft, key=lambda x: getattr(x, "exit_bar", getattr(x, "first_bar", 0))):
            d = tr.direction
            dsl = tr.exit_reason in ("SL", "HARD_SL") and hasattr(tr, "total_size") and tr.total_size > 1
            if dsl: gld = d
            elif tr.exit_reason == "TP": gss[d] = 0; gld = None if gld == d else gld
            elif tr.exit_reason in ("SL", "HARD_SL"): gss[d] += 1; gld = d if gss[d] >= 2 else gld
        all_t.extend(ft); start += test_b
    if not all_t: return {}
    pnl = sum(t.pnl_pct for t in all_t); n = len(all_t)
    wr = sum(1 for t in all_t if t.pnl_pct > 0) / n * 100
    ps = 1000 * 5 / 11; net = ps * pnl / 100 - 2 * ps * 0.0002 * n
    tp_c = sum(1 for t in all_t if t.exit_reason == "TP")
    sl_c = sum(1 for t in all_t if t.exit_reason in ("SL", "HARD_SL"))
    return {"trades": n, "wr": wr, "pnl": pnl, "net": net, "tp": tp_c, "sl": sl_c}

console.print("Preparing 15m ATR...")
data_15m = prepare("15m")
console.print("Preparing 5m ATR...")
data_5m = prepare("5m")

console.print(f"\nLoaded {len(data_15m)} pairs\n")

r1 = run_test(data_15m)
r2 = run_test(data_5m)

table = Table(title="ATR source: 15m vs 5m (192d)", box=box.ROUNDED)
table.add_column("ATR", style="bold")
table.add_column("Trades", justify="right")
table.add_column("WR", justify="right")
table.add_column("NET", justify="right")
table.add_column("TP", justify="right")
table.add_column("SL", justify="right")
table.add_row("15m (current)", str(r1["trades"]), f"{r1['wr']:.1f}%", f"${r1['net']:+.0f}", str(r1["tp"]), str(r1["sl"]))
table.add_row("5m", str(r2["trades"]), f"{r2['wr']:.1f}%", f"${r2['net']:+.0f}", str(r2["tp"]), str(r2["sl"]))
console.print(table)
