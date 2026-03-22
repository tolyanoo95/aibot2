import pandas as pd
import numpy as np
import sys
sys.path.append('src')
from backtest_5m_swing import load_data

df = load_data()
df_march = df[(df.index >= '2026-03-01') & (df.index < '2026-03-22')]

btc_q = 0.90
btc_delta_thresh_long = df_march['volume_delta_btc'].quantile(btc_q)
btc_delta_thresh_short = df_march['volume_delta_btc'].quantile(1 - btc_q)

def analyze_symbol(symbol_suffix, oi_thresh, name):
    close_col = f'close_price{symbol_suffix}'
    
    sma_50 = df_march[close_col].rolling(50).mean()
    sma_200 = df_march[close_col].rolling(200).mean()
    trend_up = sma_50 > sma_200
    trend_down = sma_50 < sma_200

    delta = df_march[close_col].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    tr1 = df_march[f'high_price{symbol_suffix}'] - df_march[f'low_price{symbol_suffix}']
    tr2 = np.abs(df_march[f'high_price{symbol_suffix}'] - df_march[close_col].shift(1))
    tr3 = np.abs(df_march[f'low_price{symbol_suffix}'] - df_march[close_col].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = true_range.rolling(14).mean()
    atr_pct = (atr_14 / df_march[close_col]) * 100
    atr_mean = atr_pct.rolling(200).mean()
    not_flat = atr_pct > atr_mean

    long_signal = (df_march['volume_delta_btc'] > btc_delta_thresh_long) & \
                  (df_march[f'oi_change{symbol_suffix}'] > oi_thresh) & \
                  (df_march[f'liq_buy_vol{symbol_suffix}'] < 5000)
                  
    short_signal = (df_march['volume_delta_btc'] < btc_delta_thresh_short) & \
                   (df_march[f'oi_change{symbol_suffix}'] > oi_thresh) & \
                   (df_march[f'liq_sell_vol{symbol_suffix}'] < 5000)

    signal_idx = df_march[long_signal | short_signal].index

    trades = []
    for idx in signal_idx:
        i = df_march.index.get_loc(idx)
        if i > len(df_march) - 100: continue
        
        is_long = long_signal.loc[idx]
        entry_price = df_march[close_col].iloc[i]
        
        future_bars = df_march.iloc[i+1:i+100]
        
        trades.append({
            'time': idx,
            'type': 'long' if is_long else 'short',
            'entry': entry_price,
            'future_high': future_bars[f'high_price{symbol_suffix}'].max(),
            'future_low': future_bars[f'low_price{symbol_suffix}'].min()
        })

    tdf = pd.DataFrame(trades)

    print(f"\n=== Analyzing {name} filtered signals (N={len(tdf)}) ===")

    def test_rr(tp, sl):
        wins = 0
        losses = 0
        for idx, row in tdf.iterrows():
            entry = row['entry']
            is_long = row['type'] == 'long'
            
            tp_price = entry * (1 + tp) if is_long else entry * (1 - tp)
            sl_price = entry * (1 - sl) if is_long else entry * (1 + sl)
            
            if is_long:
                if row['future_high'] >= tp_price and row['future_low'] > sl_price:
                    wins += 1
                elif row['future_low'] <= sl_price:
                    losses += 1
            else:
                if row['future_low'] <= tp_price and row['future_high'] < sl_price:
                    wins += 1
                elif row['future_high'] >= sl_price:
                    losses += 1
        
        wr = wins / (wins + losses + 0.001) * 100
        pnl = (wins * tp) - (losses * sl) - ((wins+losses)*0.0011)
        print(f"TP {tp*100:.1f}%, SL {sl*100:.1f}% -> WR: {wr:.1f}%, PnL: {pnl*100:.1f}% (W:{wins}, L:{losses})")

    test_rr(0.025, 0.01)
    def test_delayed_entry(drop_pct, tp, sl):
        wins = 0
        losses = 0
        missed = 0
        for idx, row in tdf.iterrows():
            entry = row['entry']
            is_long = row['type'] == 'long'
            
            # We only enter if price drops (for long) or rises (for short) by drop_pct from the signal bar
            target_entry = entry * (1 - drop_pct) if is_long else entry * (1 + drop_pct)
            
            # Check if our delayed entry order would be filled
            filled = False
            if is_long and row['future_low'] <= target_entry:
                filled = True
            elif not is_long and row['future_high'] >= target_entry:
                filled = True
                
            if not filled:
                missed += 1
                continue
                
            # Now calculate TP and SL from our NEW entry price
            tp_price = target_entry * (1 + tp) if is_long else target_entry * (1 - tp)
            sl_price = target_entry * (1 - sl) if is_long else target_entry * (1 + sl)
            
            # Very simple approximation for outcome after fill
            # (Assuming the low/high of the remaining period hits our targets)
            if is_long:
                if row['future_high'] >= tp_price and row['future_low'] > sl_price:
                    wins += 1
                elif row['future_low'] <= sl_price:
                    losses += 1
            else:
                if row['future_low'] <= tp_price and row['future_high'] < sl_price:
                    wins += 1
                elif row['future_high'] >= sl_price:
                    losses += 1
                    
        total = wins + losses
        if total == 0:
            return
            
        wr = wins / (total + 0.001) * 100
        pnl = (wins * tp) - (losses * sl) - (total * 0.0011) # fee on trades that executed
        print(f"Wait for {drop_pct*100:.1f}% drop -> Enter -> TP {tp*100:.1f}%, SL {sl*100:.1f}% | WR: {wr:.1f}%, PnL: {pnl*100:.1f}% (W:{wins}, L:{losses}, Missed:{missed})")

    print("\n--- Testing Delayed Limit Entries (Buy the blood) ---")
    test_delayed_entry(0.015, 0.02, 0.02)
    test_delayed_entry(0.015, 0.025, 0.02)
    test_delayed_entry(0.015, 0.015, 0.015)
    test_delayed_entry(0.02, 0.02, 0.02)
    test_delayed_entry(0.02, 0.03, 0.02)
    test_delayed_entry(0.03, 0.03, 0.02)
    test_delayed_entry(0.03, 0.04, 0.02)
    test_delayed_entry(0.04, 0.04, 0.02)
    test_delayed_entry(0.04, 0.04, 0.04)

analyze_symbol("", 5000, "SOL")
analyze_symbol("_avax", 500, "AVAX")
