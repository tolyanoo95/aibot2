#!/usr/bin/env python3
"""
Test: 5m confirmation for entry and exit.
Entry: 15m signal + 5m roc confirms direction
Exit: if 5m momentum reverses while in position → close early
"""
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

# Prepare 5m indicators per symbol
console.print("Preparing 5m indicators...")
ind_5m = {}
for sym in config.TRADING_PAIRS:
    if sym not in raw_5m: continue
    df5 = raw_5m[sym]
    tdf5 = indicators.calculate_all(df5.copy())
    ind_5m[sym] = tdf5
    console.print(f"  {sym}: {len(tdf5)} 5m bars")

# Prepare 15m vol bars (standard)
console.print("Preparing 15m vol bars...")
data = {}
for sym in config.TRADING_PAIRS:
    if sym not in raw_15m: continue
    df = raw_15m[sym]
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
    vdf["atr"] = tdf["atr"].reindex(vdf.index, method="ffill").values

    # Add 5m roc and rsi at each vol bar timestamp
    if sym in ind_5m:
        t5 = ind_5m[sym]
        if "roc_12" in t5.columns:
            vdf["roc_5m"] = t5["roc_12"].reindex(vdf.index, method="ffill").values
        if "rsi" in t5.columns:
            vdf["rsi_5m"] = t5["rsi"].reindex(vdf.index, method="ffill").values
            vdf["rsi_slope_5m"] = t5["rsi"].diff(6).reindex(vdf.index, method="ffill").values

    data[sym] = vdf

console.print(f"Loaded {len(data)} pairs\n")


def run_test(data, mode="baseline", sl=2.0, tp=4.0, train=1000, test_b=300):
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

                    # 5m ENTRY confirmation
                    if mode in ("entry_5m", "both"):
                        roc5 = float(dt["roc_5m"].iloc[j]) if "roc_5m" in dt.columns else 0
                        if d2 == "LONG" and roc5 < 0:
                            d2 = "NEUTRAL"; continue
                        if d2 == "SHORT" and roc5 > 0:
                            d2 = "NEUTRAL"; continue

                    # 5m ENTRY confirmation: RSI slope
                    if mode in ("entry_rsi5m", "both_rsi"):
                        rsi_s5 = float(dt["rsi_slope_5m"].iloc[j]) if "rsi_slope_5m" in dt.columns else 0
                        if d2 == "LONG" and rsi_s5 < 0:
                            d2 = "NEUTRAL"; continue
                        if d2 == "SHORT" and rsi_s5 > 0:
                            d2 = "NEUTRAL"; continue

                    c = 0.90
                sigs.append({"bar_idx": j, "symbol": sym, "direction": d2, "confidence": c})

            # Choose early_exit based on mode
            ee_str = ""
            if mode in ("exit_5m", "both", "both_rsi"):
                ee_str = "5m_fade"

            trades = simulate_dca_trades(dt, sigs, tp_mult=tp, dca_step_mult=1.0,
                max_entries=3, hard_sl_mult=sl, max_hold=24, max_open=11,
                cooldown=3, threshold=0.10, full_size_dca=True)

            # Post-process: 5m exit — close if 5m momentum reverses
            if mode in ("exit_5m", "both", "both_rsi"):
                for tr in trades:
                    if tr.exit_reason in ("TP", "HARD_SL", "FLIP"):
                        continue
                    fb = tr.first_bar
                    eb = tr.exit_bar if hasattr(tr, "exit_bar") else len(dt) - 1
                    for b in range(fb + 3, min(eb + 1, len(dt))):
                        roc5 = float(dt["roc_5m"].iloc[b]) if "roc_5m" in dt.columns else 0
                        rsi5 = float(dt["rsi_slope_5m"].iloc[b]) if "rsi_slope_5m" in dt.columns else 0
                        cur_pnl = (ct[b] - tr.avg_price) / tr.avg_price * 100 if tr.direction == "LONG" \
                            else (tr.avg_price - ct[b]) / tr.avg_price * 100

                        # 5m momentum reversed + position has some profit or small loss
                        fade = False
                        if tr.direction == "LONG" and roc5 < -0.20 and rsi5 < -3:
                            fade = True
                        elif tr.direction == "SHORT" and roc5 > 0.20 and rsi5 > 3:
                            fade = True

                        if fade and cur_pnl > -0.5:
                            sm = tr.total_size
                            tr.exit_price = ct[b]
                            tr.exit_bar = b
                            if tr.direction == "LONG":
                                tr.pnl_pct = (ct[b] - tr.avg_price) / tr.avg_price * 100 * sm
                            else:
                                tr.pnl_pct = (tr.avg_price - ct[b]) / tr.avg_price * 100 * sm
                            tr.exit_reason = "5M_FADE"
                            break

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
    fade_c = sum(1 for t in all_t if t.exit_reason == "5M_FADE")
    return {"trades": n, "wr": wr, "pnl": pnl, "net": net, "tp": tp_c, "sl": sl_c, "fade": fade_c}


tests = [
    ("Baseline (no 5m)", "baseline"),
    ("5m entry: roc confirms", "entry_5m"),
    ("5m entry: RSI slope confirms", "entry_rsi5m"),
    ("5m exit: fade detection", "exit_5m"),
    ("Both: 5m roc entry + fade exit", "both"),
    ("Both: 5m RSI entry + fade exit", "both_rsi"),
]

results = []
for name, mode in tests:
    console.print(f"Testing: {name}...")
    r = run_test(data, mode=mode)
    r["name"] = name
    results.append(r)

table = Table(title="5m Confirmation (192d)", box=box.ROUNDED)
table.add_column("Variant", style="bold")
table.add_column("Trades", justify="right")
table.add_column("WR", justify="right")
table.add_column("NET", justify="right")
table.add_column("vs Base", justify="right")
table.add_column("TP", justify="right")
table.add_column("SL", justify="right")
table.add_column("FADE", justify="right")
base = results[0]["net"]
for r in results:
    d = r["net"] - base
    table.add_row(r["name"], str(r["trades"]), f"{r['wr']:.1f}%", f"${r['net']:+.0f}",
        f"{d:+.0f}" if r["name"] != results[0]["name"] else "—",
        str(r["tp"]), str(r["sl"]), str(r.get("fade", 0)))
console.print(table)
