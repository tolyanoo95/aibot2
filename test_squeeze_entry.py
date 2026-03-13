#!/usr/bin/env python3
"""Test: enter only when coming out of BB squeeze (volatility compression → expansion)."""
import pickle, functools, logging, warnings
import numpy as np, pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
print = functools.partial(print, flush=True)
console = Console()

from src.config import config
from src.indicators import TechnicalIndicators
from backtest_volbars import resample_to_volume_bars
from backtest_wf import simulate_dca_trades

CACHE_FILE = "data/ohlcv_cache_15m.pkl"

ADX_MIN = 20
SL_MULT = 2.0
TP_MULT = 4.0
LONG_MOM = 0.30
SHORT_MOM = 0.10
MAX_DCA = 3
MAX_OPEN = 11


def load_and_prepare():
    with open(CACHE_FILE, "rb") as f:
        raw = pickle.load(f)
    indicators = TechnicalIndicators()
    all_pair_data = {}
    for symbol, df in raw.items():
        tdf = indicators.calculate_all(df.copy())
        vdf = resample_to_volume_bars(df)
        if len(vdf) < 1300:
            continue
        warmup = min(1920, len(df))
        htf_vdf = resample_to_volume_bars(df, initial_threshold=df["volume"].iloc[:warmup].median() * 10)
        if len(htf_vdf) > 20:
            htf_vdf = indicators.calculate_all(htf_vdf)
            for col in ["ema_9", "ema_21", "ema_50"]:
                if col in htf_vdf.columns:
                    vdf[f"htf_{col}"] = htf_vdf[col].reindex(vdf.index, method="ffill")
        vdf = indicators.calculate_all(vdf)
        time_atr = tdf["atr"].reindex(vdf.index, method="ffill")
        vdf["atr"] = time_atr.values

        bbb = vdf["BBB_20_2.0_2.0"] if "BBB_20_2.0_2.0" in vdf.columns else pd.Series(dtype=float)
        vdf["bbb_pct"] = bbb.rolling(100, min_periods=50).rank(pct=True)
        vdf["bbb_expanding"] = bbb > bbb.shift(1)
        vdf["was_squeeze"] = vdf["bbb_pct"].shift(1) < 0.20

        all_pair_data[symbol] = vdf
    console.print(f"Loaded {len(all_pair_data)} pairs")
    return all_pair_data


def run_test(data, squeeze_mode="none", train_bars=1000, test_bars=300):
    """
    squeeze_mode:
      "none"          = baseline (no squeeze filter)
      "squeeze_only"  = only enter when recently in squeeze (bbb_pct<0.20 within last N bars)
      "squeeze_exit"  = only enter when BB expanding out of recent squeeze
      "prefer_squeeze" = boost: if squeeze, accept weaker roc threshold
      "anti_squeeze"  = only enter when NOT in squeeze (for comparison)
    """
    all_trades = []
    start = 0
    min_len = min(len(df) for df in data.values())

    while start + train_bars + test_bars <= min_len:
        train_end = start + train_bars
        test_end = train_end + test_bars

        for symbol, df_full in data.items():
            if len(df_full) < test_end:
                continue
            df_test = df_full.iloc[train_end:test_end]
            if len(df_test) < 20:
                continue

            atr_test = df_test["atr"].values
            atr_ma20 = pd.Series(atr_test).rolling(20, min_periods=1).mean().values

            signals = []
            for j in range(len(df_test)):
                roc = float(df_test["roc_12"].iloc[j]) if "roc_12" in df_test.columns else 0

                roc_long = LONG_MOM
                roc_short = SHORT_MOM

                # Squeeze-based threshold adjustment
                bbb_pct = float(df_test["bbb_pct"].iloc[j]) if "bbb_pct" in df_test.columns else 0.5
                was_sq = bool(df_test["was_squeeze"].iloc[j]) if "was_squeeze" in df_test.columns else False
                bbb_exp = bool(df_test["bbb_expanding"].iloc[j]) if "bbb_expanding" in df_test.columns else False

                # Check if was in squeeze within last 5 bars
                recent_squeeze = False
                if "bbb_pct" in df_test.columns and j >= 5:
                    for k in range(1, 6):
                        if j - k >= 0:
                            bp = df_test["bbb_pct"].iloc[j - k]
                            if not np.isnan(bp) and bp < 0.20:
                                recent_squeeze = True
                                break

                if squeeze_mode == "squeeze_only" and not recent_squeeze:
                    continue
                elif squeeze_mode == "squeeze_exit" and not (recent_squeeze and bbb_exp):
                    continue
                elif squeeze_mode == "anti_squeeze" and recent_squeeze:
                    continue
                elif squeeze_mode == "prefer_squeeze" and recent_squeeze:
                    roc_long = 0.15
                    roc_short = 0.05

                if roc > roc_long:
                    direction = "LONG"
                elif roc < -roc_short:
                    direction = "SHORT"
                else:
                    continue

                htf_e9 = float(df_test["htf_ema_9"].iloc[j]) if "htf_ema_9" in df_test.columns else 0
                htf_e21 = float(df_test["htf_ema_21"].iloc[j]) if "htf_ema_21" in df_test.columns else 0
                if htf_e9 > 0 and htf_e21 > 0:
                    if direction == "LONG" and htf_e9 < htf_e21: continue
                    if direction == "SHORT" and htf_e9 > htf_e21: continue

                atr_exp = atr_test[j] / atr_ma20[j] if atr_ma20[j] > 0 else 1.0
                if atr_exp > 1.5: continue

                adx = float(df_test["ADX_14"].iloc[j]) if "ADX_14" in df_test.columns else 25
                if adx < ADX_MIN: continue

                rsi_s6 = float(df_test["rsi"].diff(6).iloc[j]) if "rsi" in df_test.columns else 0
                if direction == "LONG" and rsi_s6 < 0: continue
                if direction == "SHORT" and rsi_s6 > 0: continue

                signals.append({"bar_idx": j, "symbol": symbol, "direction": direction, "confidence": 0.90})

            trades = simulate_dca_trades(
                df_test, signals, tp_mult=TP_MULT, dca_step_mult=1.0,
                max_entries=MAX_DCA, hard_sl_mult=SL_MULT, max_hold=24,
                max_open=MAX_OPEN, cooldown=3, threshold=0.10,
                full_size_dca=True, move_sl_at=2.0, move_sl_to=1.0,
            )
            all_trades.extend(trades)

        start += test_bars

    if not all_trades:
        return {"trades": 0, "wr": 0, "net": 0, "tp": 0, "sl": 0}

    n = len(all_trades)
    total_pnl = sum(t.pnl_pct for t in all_trades)
    wr = sum(1 for t in all_trades if t.pnl_pct > 0) / n * 100
    pos_size = 1000 * 5 / 11
    net = pos_size * total_pnl / 100 - 2 * pos_size * 0.0002 * n
    tp = sum(1 for t in all_trades if t.exit_reason == "TP")
    sl = sum(1 for t in all_trades if t.exit_reason in ("SL", "HARD_SL"))

    return {"trades": n, "wr": wr, "net": net, "tp": tp, "sl": sl}


if __name__ == "__main__":
    data = load_and_prepare()

    tests = [
        ("Baseline (no squeeze)", "none"),
        ("Squeeze only (last 5 bars)", "squeeze_only"),
        ("Squeeze exit (expand)", "squeeze_exit"),
        ("Prefer squeeze (lower roc)", "prefer_squeeze"),
        ("Anti-squeeze (avoid)", "anti_squeeze"),
    ]

    results = []
    for name, mode in tests:
        console.print(f"Running: {name}...")
        r = run_test(data, squeeze_mode=mode)
        r["name"] = name
        results.append(r)
        console.print(f"  Trades={r['trades']} WR={r['wr']:.1f}% NET=${r['net']:+,.0f} TP={r['tp']} SL={r['sl']}")

    table = Table(title="BB Squeeze Entry Test (3yr)", box=box.ROUNDED)
    table.add_column("Mode", style="bold")
    table.add_column("Trades", justify="right")
    table.add_column("WR", justify="right")
    table.add_column("NET", justify="right")
    table.add_column("vs Base", justify="right")
    table.add_column("TP", justify="right")
    table.add_column("SL", justify="right")

    base_net = results[0]["net"]
    for r in results:
        delta = r["net"] - base_net
        delta_pct = (delta / abs(base_net) * 100) if base_net != 0 else 0
        vs = f"${delta:+,.0f} ({delta_pct:+.1f}%)" if r["name"] != results[0]["name"] else "—"
        table.add_row(
            r["name"], str(r["trades"]), f"{r['wr']:.1f}%",
            f"${r['net']:+,.0f}", vs,
            str(r["tp"]), str(r["sl"]),
        )
    console.print(table)
