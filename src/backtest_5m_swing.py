import pandas as pd
import numpy as np
from tqdm import tqdm

def load_data():
    df_btc = pd.read_pickle("data/processed/BTCUSDT_swing_5m.pkl")
    df_sol = pd.read_pickle("data/processed/SOLUSDT_swing_5m.pkl")
    df_avax = pd.read_pickle("data/processed/AVAXUSDT_swing_5m.pkl")
    
    # Merge on index
    df = df_sol.join(df_btc, lsuffix='', rsuffix='_btc', how='inner')
    df = df.join(df_avax, rsuffix='_avax', how='inner')
    return df.dropna()

def run_5m_backtest(df, symbol_suffix="_avax", fee_pct=0.0011, tp_pct=0.015, sl_pct=0.005, btc_delta_q=0.90, oi_thresh=1000, use_filters=True):
    """
    Swing strategy based on BTC extreme Delta + OI Confirmation + Trend/RSI filters
    """
    btc_delta_thresh_long = df['volume_delta_btc'].quantile(btc_delta_q)
    btc_delta_thresh_short = df['volume_delta_btc'].quantile(1 - btc_delta_q)
    
    close_col = f'close_price{symbol_suffix}'
    high_col = f'high_price{symbol_suffix}'
    low_col = f'low_price{symbol_suffix}'
    
    # 1. Trend: SMA 50 and SMA 200
    sma_50 = df[close_col].rolling(50).mean()
    sma_200 = df[close_col].rolling(200).mean()
    
    # 2. RSI (14)
    delta = df[close_col].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    rsi_14 = 100 - (100 / (1 + rs))
    
    # 3. ATR percent (14)
    tr1 = df[high_col] - df[low_col]
    tr2 = np.abs(df[high_col] - df[close_col].shift(1))
    tr3 = np.abs(df[low_col] - df[close_col].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = true_range.rolling(14).mean()
    atr_val_pct = atr_14 / df[close_col]
    atr_pct = atr_val_pct * 100
    atr_pct_mean = atr_pct.rolling(200).mean() # Baseline volatility
    
    in_position = False
    position_type = None
    entry_price = 0
    trades = []
    
    for i in range(200, len(df)):
        curr_row = df.iloc[i]
        
        # Filter Logic
        trend_is_up = sma_50.iloc[i] > sma_200.iloc[i]
        trend_is_down = sma_50.iloc[i] < sma_200.iloc[i]
        
        not_flat = atr_pct.iloc[i] > atr_pct_mean.iloc[i]
        rsi_val = rsi_14.iloc[i]
        
        if not in_position:
            # Base conditions
            base_long = (curr_row['volume_delta_btc'] > btc_delta_thresh_long and 
                         curr_row[f'oi_change{symbol_suffix}'] > oi_thresh and
                         curr_row[f'liq_buy_vol{symbol_suffix}'] < 5000 and
                         curr_row[f'volume_delta{symbol_suffix}'] > 0) # Asset's own delta must confirm
                         
            base_short = (curr_row['volume_delta_btc'] < btc_delta_thresh_short and 
                          curr_row[f'oi_change{symbol_suffix}'] > oi_thresh and
                          curr_row[f'liq_sell_vol{symbol_suffix}'] < 5000 and
                          curr_row[f'volume_delta{symbol_suffix}'] < 0) # Asset's own delta must confirm
            
            # Apply Filters
            if use_filters:
                # To catch a dip (long), global trend should be UP, and RSI should be < 50
                long_cond = base_long and trend_is_up and not_flat and (rsi_val < 50)
                # To catch a top (short), global trend should be DOWN, and RSI should be > 50
                short_cond = base_short and trend_is_down and not_flat and (rsi_val > 50)
            else:
                long_cond = base_long
                short_cond = base_short

            if long_cond:
                entry_price = curr_row[close_col]
                in_position = True
                position_type = "long"
                entry_time = df.index[i]
                
            elif short_cond:
                entry_price = curr_row[close_col]
                in_position = True
                position_type = "short"
                entry_time = df.index[i]
                
        else:
            if position_type == "long":
                pnl_pct = (curr_row[high_col] - entry_price) / entry_price
                loss_pct = (curr_row[low_col] - entry_price) / entry_price
                
                if pnl_pct >= tp_pct:
                    pnl = tp_pct - fee_pct
                    trades.append({'entry_time': entry_time, 'pnl': pnl, 'type': position_type, 'exit_reason': 'TP'})
                    in_position = False
                elif loss_pct <= -sl_pct:
                    pnl = -sl_pct - fee_pct
                    trades.append({'entry_time': entry_time, 'pnl': pnl, 'type': position_type, 'exit_reason': 'SL'})
                    in_position = False
                    
            elif position_type == "short":
                pnl_pct = (entry_price - curr_row[low_col]) / entry_price
                loss_pct = (entry_price - curr_row[high_col]) / entry_price
                
                if pnl_pct >= tp_pct:
                    pnl = tp_pct - fee_pct
                    trades.append({'entry_time': entry_time, 'pnl': pnl, 'type': position_type, 'exit_reason': 'TP'})
                    in_position = False
                elif loss_pct <= -sl_pct:
                    pnl = -sl_pct - fee_pct
                    trades.append({'entry_time': entry_time, 'pnl': pnl, 'type': position_type, 'exit_reason': 'SL'})
                    in_position = False
                    
    trades_df = pd.DataFrame(trades)
    
    if len(trades_df) == 0:
        return {'trades': 0, 'win_rate': 0, 'pnl': 0, 'avg_pnl': 0, 'tps': 0, 'sls': 0}
        
    win_rate = (trades_df['pnl'] > 0).mean() * 100
    total_pnl = trades_df['pnl'].sum() * 100
    avg_pnl = trades_df['pnl'].mean() * 100
    tps = (trades_df['exit_reason'] == 'TP').sum()
    sls = (trades_df['exit_reason'] == 'SL').sum()
    
    return {
        'trades': len(trades_df),
        'win_rate': win_rate,
        'pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'tps': tps,
        'sls': sls
    }

if __name__ == "__main__":
    df = load_data()
    print(f"Data loaded: {len(df)} 5-min bars")
    
    results = []
    
    # Try different SL settings for SOL to see if we can save the winrate
    # Keep TP fixed at 0.025 to give trades room
    tp_sl_pairs = [(0.025, 0.010), (0.03, 0.010), (0.02, 0.008)]
    oi_thresholds = [1000]
    btc_quantiles = [0.90]
    
    # Let's try Pyramiding / Adding to position
    for tp, sl in tp_sl_pairs:
        for oi in oi_thresholds:
            for btc_q in btc_quantiles:
                # Filter data for March 1 to March 21
                df_march_week = df[(df.index >= '2026-03-01') & (df.index < '2026-03-22')]
                
                if len(df_march_week) == 0:
                    continue
                    
                # Test with x3 leverage implicitly by multiplying PnL (since margin trades allow leverage)
                res_sol = run_5m_backtest(df_march_week, symbol_suffix="", fee_pct=0.0011, tp_pct=tp, sl_pct=sl, btc_delta_q=btc_q, oi_thresh=oi, use_filters=True)
                res_sol['tp'] = tp
                res_sol['sl'] = sl
                res_sol['coin'] = 'SOL'
                res_sol['pnl_x3'] = res_sol['pnl'] * 3
                results.append(res_sol)
                
                res_avax = run_5m_backtest(df_march_week, symbol_suffix="_avax", fee_pct=0.0011, tp_pct=tp, sl_pct=sl, btc_delta_q=btc_q, oi_thresh=oi/10, use_filters=True)
                res_avax['tp'] = tp
                res_avax['sl'] = sl
                res_avax['coin'] = 'AVAX'
                res_avax['pnl_x3'] = res_avax['pnl'] * 3
                results.append(res_avax)
                
    res_df = pd.DataFrame(results)
    
    print("\n--- March 1-21, 2026 Results (Leverage x3) ---")
    
    combined_res = []
    for tp, sl in tp_sl_pairs:
        sol_row = res_df[(res_df['coin'] == 'SOL') & (res_df['tp'] == tp) & (res_df['sl'] == sl)]
        avax_row = res_df[(res_df['coin'] == 'AVAX') & (res_df['tp'] == tp) & (res_df['sl'] == sl)]
        
        if not sol_row.empty and not avax_row.empty:
            sol_pnl = sol_row.iloc[0]['pnl_x3']
            avax_pnl = avax_row.iloc[0]['pnl_x3']
            
            combined_res.append({
                'tp': tp,
                'sl': sl,
                'sol_t': sol_row.iloc[0]['trades'],
                'sol_tp': sol_row.iloc[0]['tps'],
                'sol_sl': sol_row.iloc[0]['sls'],
                'sol_wr': sol_row.iloc[0]['win_rate'],
                'avax_t': avax_row.iloc[0]['trades'],
                'avax_tp': avax_row.iloc[0]['tps'],
                'avax_sl': avax_row.iloc[0]['sls'],
                'avax_wr': avax_row.iloc[0]['win_rate'],
                'sol_pnl_x3': sol_pnl,
                'avax_pnl_x3': avax_pnl,
                'tot_pnl_x3': sol_pnl + avax_pnl
            })
            
    comb_df = pd.DataFrame(combined_res)
    print(comb_df.sort_values('tot_pnl_x3', ascending=False).to_string(index=False))
