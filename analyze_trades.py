import pandas as pd
import numpy as np
import sys
sys.path.append('src')
from backtest_5m_swing import load_data

df = load_data()
df_march = df[(df.index >= '2026-02-01') & (df.index < '2026-03-22')]

btc_q = 0.90
btc_delta_thresh_long = df_march['volume_delta_btc'].quantile(btc_q)
btc_delta_thresh_short = df_march['volume_delta_btc'].quantile(1 - btc_q)

def analyze_symbol(symbol_suffix, oi_thresh, name):
    close_col = f'close_price{symbol_suffix}'

    # Momentum signals
    long_signal = (df_march['volume_delta_btc'] > btc_delta_thresh_long) & \
                  (df_march[f'oi_change{symbol_suffix}'] > oi_thresh) & \
                  (df_march[f'liq_buy_vol{symbol_suffix}'] < 5000)
                  
    short_signal = (df_march['volume_delta_btc'] < btc_delta_thresh_short) & \
                   (df_march[f'oi_change{symbol_suffix}'] > oi_thresh) & \
                   (df_march[f'liq_sell_vol{symbol_suffix}'] < 5000)

    signal_idx = df_march[long_signal | short_signal].index

    print(f"\n=== EXACT Analysis for {name} (N={len(signal_idx)}) ===")

    def test_exact(drop_pct, tp, sl, reverse=False):
        wins = 0
        losses = 0
        missed = 0
        
        for idx in signal_idx:
            i = df_march.index.get_loc(idx)
            if i > len(df_march) - 100: continue
            
            signal_was_long = long_signal.loc[idx]
            is_long = not signal_was_long if reverse else signal_was_long
            
            entry_price = df_march[close_col].iloc[i]
            
            # Target entry price
            if drop_pct > 0:
                target_entry = entry_price * (1 - drop_pct) if is_long else entry_price * (1 + drop_pct)
                
                filled = False
                entry_bar_idx = -1
                
                for j in range(i+1, min(i+121, len(df_march))): # look ahead 10 hours max
                    bar = df_march.iloc[j]
                    if is_long and bar[f'low_price{symbol_suffix}'] <= target_entry:
                        filled = True
                        entry_bar_idx = j
                        break
                    elif not is_long and bar[f'high_price{symbol_suffix}'] >= target_entry:
                        filled = True
                        entry_bar_idx = j
                        break
                        
                if not filled:
                    missed += 1
                    continue
            else:
                target_entry = entry_price
                entry_bar_idx = i
                
            tp_price = target_entry * (1 + tp) if is_long else target_entry * (1 - tp)
            sl_price = target_entry * (1 - sl) if is_long else target_entry * (1 + sl)
            
            trade_closed = False
            for j in range(entry_bar_idx, len(df_march)):
                bar = df_march.iloc[j]
                
                if is_long:
                    # Check SL first to be conservative
                    if bar[f'low_price{symbol_suffix}'] <= sl_price:
                        losses += 1
                        trade_closed = True
                        break
                    elif bar[f'high_price{symbol_suffix}'] >= tp_price:
                        wins += 1
                        trade_closed = True
                        break
                else:
                    if bar[f'high_price{symbol_suffix}'] >= sl_price:
                        losses += 1
                        trade_closed = True
                        break
                    elif bar[f'low_price{symbol_suffix}'] <= tp_price:
                        wins += 1
                        trade_closed = True
                        break
            
            if not trade_closed:
                losses += 1

        total = wins + losses
        if total == 0:
            return
            
        wr = wins / (total + 0.001) * 100
        pnl = (wins * tp) - (losses * sl) - (total * 0.0011)
        mode = "REVERSE" if reverse else "NORMAL "
        print(f"{mode} | Delay: {drop_pct*100:.1f}%, TP: {tp*100:.1f}%, SL: {sl*100:.1f}% -> WR: {wr:.1f}%, PnL: {pnl*100:.1f}% (W:{wins}, L:{losses})")

    print("--- Normal Trades (Market Entry) ---")
    test_exact(0.0, 0.02, 0.01, reverse=False)
    test_exact(0.0, 0.02, 0.02, reverse=False)
    
    print("\n--- Reverse Trades (Market Entry) ---")
    test_exact(0.0, 0.02, 0.01, reverse=True)
    test_exact(0.0, 0.02, 0.02, reverse=True)
    
    print("\n--- Normal Trades (Delayed Limit Entry) ---")
    test_exact(0.015, 0.015, 0.03, reverse=False)
    test_exact(0.025, 0.025, 0.025, reverse=False)
    
    print("\n--- Reverse Trades (Delayed Limit Entry) ---")
    test_exact(0.015, 0.015, 0.03, reverse=True)
    test_exact(0.025, 0.025, 0.025, reverse=True)

analyze_symbol("", 5000, "SOL")
analyze_symbol("_avax", 500, "AVAX")