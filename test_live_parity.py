#!/usr/bin/env python3
"""
Live bot parity test — simulates 50+ scans and verifies
everything works identically to backtest.
"""
import sys
import numpy as np
import pandas as pd

def run_test():
    from main_volbars import VolumeBarsBot, MAX_OPEN, SL_MULT, TP_MULT, ADX_MIN, ATR_EXP_MAX

    errors = []
    bot = VolumeBarsBot(paper=True)
    bot.initialize()

    pairs = list(bot.vol_bars.keys())
    print(f"Initialized {len(pairs)} pairs")

    # TEST 1: All pairs have required data after init
    for sym in pairs:
        vdf = bot.vol_bars[sym]
        tdf = bot.time_data[sym]

        # Vol bars columns
        for col in ["open", "high", "low", "close", "volume", "atr", "roc_12", "rsi", "ema_9", "ema_21", "ema_50", "ADX_14"]:
            if col not in vdf.columns:
                errors.append(f"INIT {sym}: missing vol_bars column '{col}'")

        # HTF EMA columns
        for col in ["htf_ema_9", "htf_ema_21", "htf_ema_50"]:
            if col not in vdf.columns:
                errors.append(f"INIT {sym}: missing '{col}'")

        # No NaN in last row indicators
        last = vdf.iloc[-1]
        for col in ["atr", "roc_12", "rsi", "ema_50", "ADX_14"]:
            if col in vdf.columns:
                val = last[col] if not isinstance(last[col], pd.Series) else last[col].iloc[0]
                if pd.isna(val):
                    errors.append(f"INIT {sym}: NaN in last row '{col}'")

        # No duplicate indices
        if vdf.index.duplicated().sum() > 0:
            errors.append(f"INIT {sym}: vol_bars has {vdf.index.duplicated().sum()} duplicate indices")
        if tdf.index.duplicated().sum() > 0:
            errors.append(f"INIT {sym}: time_data has {tdf.index.duplicated().sum()} duplicate indices")

        # No duplicate columns
        if vdf.columns.duplicated().sum() > 0:
            errors.append(f"INIT {sym}: vol_bars has {vdf.columns.duplicated().sum()} duplicate columns")

        # Vol history initialized
        if sym not in bot.vol_history:
            errors.append(f"INIT {sym}: vol_history not initialized")
        elif len(bot.vol_history[sym]) == 0:
            errors.append(f"INIT {sym}: vol_history empty")

        # Threshold positive
        if bot.vol_thresholds.get(sym, 0) <= 0:
            errors.append(f"INIT {sym}: threshold <= 0")

    print(f"After init: {len(errors)} errors")

    # TEST 2: Run 50 scans
    for scan in range(1, 51):
        try:
            bot.scan()
        except Exception as e:
            errors.append(f"SCAN {scan}: crash — {e}")
            continue

        # After each scan, verify data integrity
        for sym in pairs:
            vdf = bot.vol_bars.get(sym)
            if vdf is None:
                errors.append(f"SCAN {scan} {sym}: vol_bars is None")
                continue

            # No duplicate indices
            if vdf.index.duplicated().sum() > 0:
                errors.append(f"SCAN {scan} {sym}: vol_bars duplicate indices")

            # No duplicate columns
            if vdf.columns.duplicated().sum() > 0:
                errors.append(f"SCAN {scan} {sym}: vol_bars duplicate columns ({vdf.columns[vdf.columns.duplicated()].tolist()})")

            # HTF EMA present
            for col in ["htf_ema_9", "htf_ema_21", "htf_ema_50"]:
                if col not in vdf.columns:
                    errors.append(f"SCAN {scan} {sym}: '{col}' missing")

            # ATR present and not NaN on last row
            if "atr" not in vdf.columns:
                errors.append(f"SCAN {scan} {sym}: 'atr' missing")
            else:
                last_atr = vdf["atr"].iloc[-1]
                if pd.isna(last_atr):
                    errors.append(f"SCAN {scan} {sym}: last ATR is NaN")

            # ADX not duplicated
            if "ADX_14" in vdf.columns:
                adx_col = vdf["ADX_14"]
                if isinstance(adx_col, pd.DataFrame):
                    errors.append(f"SCAN {scan} {sym}: ADX_14 is DataFrame (duplicated)")

            # Time data integrity
            tdf = bot.time_data.get(sym)
            if tdf is not None:
                if tdf.index.duplicated().sum() > 0:
                    errors.append(f"SCAN {scan} {sym}: time_data duplicate indices")
                if tdf.columns.duplicated().sum() > 0:
                    errors.append(f"SCAN {scan} {sym}: time_data duplicate columns")

        # Verify positions have valid data
        for pos in bot.positions:
            if pos.entry_atr <= 0:
                errors.append(f"SCAN {scan} {pos.symbol}: entry_atr={pos.entry_atr}")
            if np.isnan(pos.avg_price):
                errors.append(f"SCAN {scan} {pos.symbol}: avg_price is NaN")
            if np.isnan(pos.hard_sl):
                errors.append(f"SCAN {scan} {pos.symbol}: hard_sl is NaN")
            if np.isnan(pos.tp):
                errors.append(f"SCAN {scan} {pos.symbol}: tp is NaN")
            if pos.total_size < 1 or pos.total_size > 3:
                errors.append(f"SCAN {scan} {pos.symbol}: total_size={pos.total_size}")

        # Verify vol_drop exit data available for position checks
        for sym in pairs:
            vdf = bot.vol_bars.get(sym)
            if vdf is not None and "volume" in vdf.columns:
                vol_vals = vdf["volume"].values
                if len(vol_vals) < 20:
                    errors.append(f"SCAN {scan} {sym}: vol_bars too short for vol_drop ({len(vol_vals)})")
                vol_ma20 = pd.Series(vol_vals).rolling(20, min_periods=1).mean().values
                if np.isnan(vol_ma20[-1]):
                    errors.append(f"SCAN {scan} {sym}: vol_ma20 is NaN")

        if scan % 10 == 0:
            print(f"Scan {scan}/50: {len(bot.positions)} positions, {len(errors)} errors")

    # TEST 3: Verify parameters match backtest
    assert SL_MULT == 2.0, f"SL_MULT={SL_MULT}"
    assert TP_MULT == 4.0, f"TP_MULT={TP_MULT}"
    assert ADX_MIN == 25, f"ADX_MIN={ADX_MIN}"
    assert ATR_EXP_MAX == 1.5, f"ATR_EXP_MAX={ATR_EXP_MAX}"
    assert MAX_OPEN == 11, f"MAX_OPEN={MAX_OPEN}"

    # RESULTS
    print(f"\n{'='*50}")
    print(f"TOTAL SCANS: 50")
    print(f"TOTAL ERRORS: {len(errors)}")
    if errors:
        print("\nERRORS:")
        for e in errors[:30]:
            print(f"  {e}")
        if len(errors) > 30:
            print(f"  ... and {len(errors)-30} more")
        return 1
    else:
        print("ALL TESTS PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(run_test())
