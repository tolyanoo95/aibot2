import pandas as pd
import numpy as np

def load_data():
    df_btc = pd.read_pickle("data/processed/BTCUSDT_swing_5m.pkl")
    df_sol = pd.read_pickle("data/processed/SOLUSDT_swing_5m.pkl")
    df_avax = pd.read_pickle("data/processed/AVAXUSDT_swing_5m.pkl")
    
    # Merge on index
    df = df_sol.join(df_btc, lsuffix='', rsuffix='_btc', how='inner')
    df = df.join(df_avax, rsuffix='_avax', how='inner')
    return df.dropna()

def run_smart_exit_backtest(df, symbol_suffix="_avax", fee_pct=0.0011, btc_delta_q=0.95, oi_thresh=1000, exit_rule="sma9"):
    """
    Smart Exit Backtest:
    Uses dynamic exits instead of fixed TP/SL.
    Hard SL of 2% is kept for safety.
    """
    btc_delta_thresh_long = df['volume_delta_btc'].quantile(btc_delta_q)
    btc_delta_thresh_short = df['volume_delta_btc'].quantile(1 - btc_delta_q)
    
    close_col = f'close_price{symbol_suffix}'
    high_col = f'high_price{symbol_suffix}'
    low_col = f'low_price{symbol_suffix}'
    
    # Calculate indicators for exit
    sma_9 = df[close_col].rolling(9).mean()
    sma_20 = df[close_col].rolling(20).mean()
    
    in_position = False
    position_type = None
    entry_price = 0
    trades = []
    
    hard_sl_pct = 0.02 # 2% hard stop to prevent disasters
    
    for i in range(20, len(df)):
        curr_row = df.iloc[i]
        
        if not in_position:
            # LONG SIGNAL
            if (curr_row['volume_delta_btc'] > btc_delta_thresh_long and 
                curr_row[f'oi_change{symbol_suffix}'] > oi_thresh and
                curr_row[f'liq_buy_vol{symbol_suffix}'] < 5000): 
                
                entry_price = curr_row[close_col]
                in_position = True
                position_type = "long"
                entry_time = df.index[i]
                
            # SHORT SIGNAL
            elif (curr_row['volume_delta_btc'] < btc_delta_thresh_short and 
                  curr_row[f'oi_change{symbol_suffix}'] > oi_thresh and 
                  curr_row[f'liq_sell_vol{symbol_suffix}'] < 5000): 
                
                entry_price = curr_row[close_col]
                in_position = True
                position_type = "short"
                entry_time = df.index[i]
                
        else:
            loss_pct = 0
            if position_type == "long":
                loss_pct = (curr_row[low_col] - entry_price) / entry_price
                
                # Exit conditions
                exit_signal = False
                if exit_rule == "sma9" and curr_row[close_col] < sma_9.iloc[i]:
                    exit_signal = True
                elif exit_rule == "sma20" and curr_row[close_col] < sma_20.iloc[i]:
                    exit_signal = True
                elif exit_rule == "delta_rev" and curr_row['volume_delta_btc'] < 0:
                    exit_signal = True
                
                if loss_pct <= -hard_sl_pct:
                    pnl = -hard_sl_pct - fee_pct
                    trades.append({'entry_time': entry_time, 'pnl': pnl, 'type': position_type, 'exit': 'hard_sl'})
                    in_position = False
                elif exit_signal:
                    pnl = (curr_row[close_col] - entry_price) / entry_price - fee_pct
                    trades.append({'entry_time': entry_time, 'pnl': pnl, 'type': position_type, 'exit': 'smart'})
                    in_position = False
                    
            elif position_type == "short":
                loss_pct = (entry_price - curr_row[high_col]) / entry_price
                
                # Exit conditions
                exit_signal = False
                if exit_rule == "sma9" and curr_row[close_col] > sma_9.iloc[i]:
                    exit_signal = True
                elif exit_rule == "sma20" and curr_row[close_col] > sma_20.iloc[i]:
                    exit_signal = True
                elif exit_rule == "delta_rev" and curr_row['volume_delta_btc'] > 0:
                    exit_signal = True
                
                if loss_pct <= -hard_sl_pct:
                    pnl = -hard_sl_pct - fee_pct
                    trades.append({'entry_time': entry_time, 'pnl': pnl, 'type': position_type, 'exit': 'hard_sl'})
                    in_position = False
                elif exit_signal:
                    pnl = (entry_price - curr_row[close_col]) / entry_price - fee_pct
                    trades.append({'entry_time': entry_time, 'pnl': pnl, 'type': position_type, 'exit': 'smart'})
                    in_position = False
                    
    trades_df = pd.DataFrame(trades)
    
    if len(trades_df) == 0:
        return {'trades': 0, 'win_rate': 0, 'pnl': 0, 'hard_sls': 0}
        
    win_rate = (trades_df['pnl'] > 0).mean() * 100
    total_pnl = trades_df['pnl'].sum() * 100
    hard_sls = (trades_df['exit'] == 'hard_sl').sum()
    
    return {
        'trades': len(trades_df),
        'win_rate': win_rate,
        'pnl': total_pnl,
        'hard_sls': hard_sls
    }

if __name__ == "__main__":
    df = load_data()
    # March 1 to 21
    df_march = df[(df.index >= '2026-03-01') & (df.index < '2026-03-22')]
    print(f"Data loaded: {len(df_march)} bars")
    
    results = []
    
    exit_rules = ["sma9", "sma20", "delta_rev"]
    oi_thresholds = [1000, 5000, 10000]
    btc_quantiles = [0.90, 0.95, 0.98]
    
    for rule in exit_rules:
        for oi in oi_thresholds:
            for btc_q in btc_quantiles:
                
                res_sol = run_smart_exit_backtest(df_march, symbol_suffix="", btc_delta_q=btc_q, oi_thresh=oi, exit_rule=rule)
                res_sol['rule'] = rule
                res_sol['oi'] = oi
                res_sol['btc_q'] = btc_q
                res_sol['coin'] = 'SOL'
                results.append(res_sol)
                
                res_avax = run_smart_exit_backtest(df_march, symbol_suffix="_avax", btc_delta_q=btc_q, oi_thresh=oi/10, exit_rule=rule)
                res_avax['rule'] = rule
                res_avax['oi'] = oi / 10
                res_avax['btc_q'] = btc_q
                res_avax['coin'] = 'AVAX'
                results.append(res_avax)
                
    res_df = pd.DataFrame(results)
    
    # Calculate combined
    combined_res = []
    for rule in exit_rules:
        for oi in oi_thresholds:
            for btc_q in btc_quantiles:
                sol_row = res_df[(res_df['coin'] == 'SOL') & (res_df['rule'] == rule) & (res_df['oi'] == oi) & (res_df['btc_q'] == btc_q)]
                avax_row = res_df[(res_df['coin'] == 'AVAX') & (res_df['rule'] == rule) & (res_df['oi'] == oi/10) & (res_df['btc_q'] == btc_q)]
                
                if not sol_row.empty and not avax_row.empty:
                    sol_pnl = sol_row.iloc[0]['pnl']
                    avax_pnl = avax_row.iloc[0]['pnl']
                    
                    combined_res.append({
                        'rule': rule,
                        'oi_sol': oi,
                        'btc_q': btc_q,
                        'sol_t': sol_row.iloc[0]['trades'],
                        'sol_wr': sol_row.iloc[0]['win_rate'],
                        'sol_pnl': sol_pnl,
                        'avax_t': avax_row.iloc[0]['trades'],
                        'avax_wr': avax_row.iloc[0]['win_rate'],
                        'avax_pnl': avax_pnl,
                        'tot_pnl': sol_pnl + avax_pnl
                    })
                    
    comb_df = pd.DataFrame(combined_res)
    print("\n--- Smart Exit Results (SOL vs AVAX) ---")
    print(comb_df.sort_values('tot_pnl', ascending=False).head(20).to_string(index=False))
