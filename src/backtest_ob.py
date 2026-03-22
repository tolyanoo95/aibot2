import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def prepare_data(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    df = pd.read_pickle(file_path)
    
    # Forward fill NaNs for VWAP where there were no trades
    df['vwap'] = df['vwap'].ffill()
    df['close_price'] = df['close_price'].ffill()
    
    # Drop spread since it's all NaNs based on the script logic earlier
    df = df.drop(columns=['spread'])
    
    # Fill remaining NaNs for first few rows
    df = df.fillna(0)
    
    # Calculate some additional rolling features
    df['delta_rolling_5'] = df['volume_delta'].rolling(5).sum() # 50 seconds rolling delta
    df['delta_rolling_10'] = df['volume_delta'].rolling(10).sum() # 100 seconds rolling delta
    
    # Z-score for walls to identify "abnormal" walls
    # Adding a small epsilon to avoid division by zero
    df['bid_wall_z'] = (df['bid_wall_size'] - df['bid_wall_size'].rolling(60).mean()) / (df['bid_wall_size'].rolling(60).std() + 1e-9)
    df['ask_wall_z'] = (df['ask_wall_size'] - df['ask_wall_size'].rolling(60).mean()) / (df['ask_wall_size'].rolling(60).std() + 1e-9)
    
    # Drop initial rows with NaNs from rolling
    df = df.dropna()
    
    return df

def run_backtest_hyp_b(df, fee_pct=0, tp_pct=0.004, sl_pct=0.002, obi_thresh=0.2, btc_delta_q=0.95, exit_on_obi_rev=True):
    """
    Hypothesis B: BTC Lead-Lag + SOL Imbalance
    Enter Long on SOL when BTC has strong positive Delta AND SOL has positive OBI.
    """
    in_position = False
    entry_price = 0
    trades = []
    
    # Calculate thresholds for BTC delta
    btc_delta_thresh_long = df['volume_delta_btc'].quantile(btc_delta_q)
    btc_delta_thresh_short = df['volume_delta_btc'].quantile(1 - btc_delta_q)
    
    for i in range(1, len(df)):
        curr_row = df.iloc[i]
        
        if not in_position:
            # Strong buying on BTC AND SOL orderbook is bullish
            if curr_row['volume_delta_btc'] > btc_delta_thresh_long and curr_row['obi_10_sol'] > obi_thresh:
                entry_price = curr_row['close_price_sol']
                in_position = True
                position_type = "long"
                entry_time = df.index[i]
            # Strong selling on BTC AND SOL orderbook is bearish
            elif curr_row['volume_delta_btc'] < btc_delta_thresh_short and curr_row['obi_10_sol'] < -obi_thresh:
                entry_price = curr_row['close_price_sol']
                in_position = True
                position_type = "short"
                entry_time = df.index[i]
                
        else:
            if position_type == "long":
                pnl_pct = (curr_row['close_price_sol'] - entry_price) / entry_price
                
                # Exit conditions: TP, SL, BTC delta reversal, or OBI reversal
                exit_cond = pnl_pct >= tp_pct or pnl_pct <= -sl_pct or curr_row['delta_rolling_5_btc'] < (-1 * btc_delta_thresh_long / 2)
                if exit_on_obi_rev:
                    exit_cond = exit_cond or curr_row['obi_10_sol'] < -0.1
                    
                if exit_cond:
                    exit_price = curr_row['close_price_sol']
                    pnl = pnl_pct - (2 * fee_pct)
                    trades.append({'entry_time': entry_time, 'pnl': pnl, 'type': position_type, 'hold_time_sec': (df.index[i] - entry_time).total_seconds()})
                    in_position = False
            else:
                pnl_pct = (entry_price - curr_row['close_price_sol']) / entry_price
                
                exit_cond = pnl_pct >= tp_pct or pnl_pct <= -sl_pct or curr_row['delta_rolling_5_btc'] > (-1 * btc_delta_thresh_short / 2)
                if exit_on_obi_rev:
                    exit_cond = exit_cond or curr_row['obi_10_sol'] > 0.1
                    
                if exit_cond:
                    exit_price = curr_row['close_price_sol']
                    pnl = pnl_pct - (2 * fee_pct)
                    trades.append({'entry_time': entry_time, 'pnl': pnl, 'type': position_type, 'hold_time_sec': (df.index[i] - entry_time).total_seconds()})
                    in_position = False
                
    trades_df = pd.DataFrame(trades)
    
    if len(trades_df) == 0:
        return {'trades': 0, 'win_rate': 0, 'pnl': 0, 'avg_pnl': 0, 'avg_hold': 0}
        
    win_rate = (trades_df['pnl'] > 0).mean() * 100
    total_pnl = trades_df['pnl'].sum() * 100
    avg_pnl = trades_df['pnl'].mean() * 100
    avg_hold = trades_df['hold_time_sec'].mean()
    
    return {
        'trades': len(trades_df),
        'win_rate': win_rate,
        'pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'avg_hold': avg_hold
    }

def run_backtest_hyp_c(df, fee_pct=0, tp_pct=0.006, sl_pct=0.003, obi_thresh=0.1, btc_delta_q=0.95, wall_z_thresh=1.0, timeout_bars=3):
    """
    Hypothesis C: BTC Lead-Lag + SOL Imbalance + Wall Support + Dynamic Timeout
    """
    in_position = False
    entry_price = 0
    trades = []
    
    btc_delta_thresh_long = df['volume_delta_btc'].quantile(btc_delta_q)
    btc_delta_thresh_short = df['volume_delta_btc'].quantile(1 - btc_delta_q)
    
    entry_i = 0
    
    for i in range(1, len(df)):
        curr_row = df.iloc[i]
        
        if not in_position:
            # LONG SIGNAL
            if (curr_row['volume_delta_btc'] > btc_delta_thresh_long and 
                curr_row['obi_10_sol'] > obi_thresh and 
                curr_row['bid_wall_z_sol'] > wall_z_thresh):
                entry_price = curr_row['close_price_sol']
                in_position = True
                position_type = "long"
                entry_time = df.index[i]
                entry_i = i
                
            # SHORT SIGNAL
            elif (curr_row['volume_delta_btc'] < btc_delta_thresh_short and 
                  curr_row['obi_10_sol'] < -obi_thresh and 
                  curr_row['ask_wall_z_sol'] > wall_z_thresh):
                entry_price = curr_row['close_price_sol']
                in_position = True
                position_type = "short"
                entry_time = df.index[i]
                entry_i = i
                
        else:
            bars_held = i - entry_i
            
            if position_type == "long":
                pnl_pct = (curr_row['close_price_sol'] - entry_price) / entry_price
                
                exit_cond = pnl_pct >= tp_pct or pnl_pct <= -sl_pct or curr_row['delta_rolling_5_btc'] < (-1 * btc_delta_thresh_long / 2)
                
                # Micro-structure Stop Loss (Wall disappears)
                # If the wall we leaned on is gone (z-score drops below 0), get out early
                if curr_row['bid_wall_z_sol'] < 0:
                    exit_cond = True
                    
                # Dynamic Timeout Exit (e.g. 3 bars = 30 seconds)
                if bars_held >= timeout_bars and curr_row['obi_10_sol'] < 0:
                    exit_cond = True
                    
                if exit_cond:
                    exit_price = curr_row['close_price_sol']
                    pnl = pnl_pct - (2 * fee_pct)
                    trades.append({'entry_time': entry_time, 'pnl': pnl, 'type': position_type, 'hold_time_sec': bars_held * 10})
                    in_position = False
            else:
                pnl_pct = (entry_price - curr_row['close_price_sol']) / entry_price
                
                exit_cond = pnl_pct >= tp_pct or pnl_pct <= -sl_pct or curr_row['delta_rolling_5_btc'] > (-1 * btc_delta_thresh_short / 2)
                
                # Micro-structure Stop Loss (Wall disappears)
                if curr_row['ask_wall_z_sol'] < 0:
                    exit_cond = True
                
                # Dynamic Timeout Exit
                if bars_held >= timeout_bars and curr_row['obi_10_sol'] > 0:
                    exit_cond = True
                    
                if exit_cond:
                    exit_price = curr_row['close_price_sol']
                    pnl = pnl_pct - (2 * fee_pct)
                    trades.append({'entry_time': entry_time, 'pnl': pnl, 'type': position_type, 'hold_time_sec': bars_held * 10})
                    in_position = False
                
    trades_df = pd.DataFrame(trades)
    
    if len(trades_df) == 0:
        return {'trades': 0, 'win_rate': 0, 'pnl': 0, 'avg_pnl': 0, 'avg_hold': 0}
        
    win_rate = (trades_df['pnl'] > 0).mean() * 100
    total_pnl = trades_df['pnl'].sum() * 100
    avg_pnl = trades_df['pnl'].mean() * 100
    avg_hold = trades_df['hold_time_sec'].mean()
    
    return {
        'trades': len(trades_df),
        'win_rate': win_rate,
        'pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'avg_hold': avg_hold
    }

def optimize_hyp_c():
    sol_file = "data/processed/SOLUSDT_features_10s_full.pkl"
    btc_file = "data/processed/BTCUSDT_features_10s_full.pkl"
    
    df_sol = prepare_data(sol_file)
    df_btc = prepare_data(btc_file)
    
    if df_sol is None or df_btc is None:
        return
        
    df_merged = df_sol.join(df_btc, lsuffix='_sol', rsuffix='_btc', how='inner')
    df = df_merged.dropna()
    
    print(f"Optimizing Hypothesis C on {len(df)} rows...")
    
    # We will test new parameters: Walls and Timeouts
    tp_sl_pairs = [(0.005, 0.0025), (0.006, 0.003)]
    obi_thresholds = [0.1, 0.2]
    wall_z_thresholds = [1.0, 2.0] # Require big wall (z-score > 1 or 2)
    timeout_bars_list = [2, 3] # 20 seconds, 30 seconds
    
    results = []
    
    total_iters = len(tp_sl_pairs) * len(obi_thresholds) * len(wall_z_thresholds) * len(timeout_bars_list)
    
    with tqdm(total=total_iters) as pbar:
        for tp, sl in tp_sl_pairs:
            for obi in obi_thresholds:
                for wall_z in wall_z_thresholds:
                    for t_bars in timeout_bars_list:
                        res = run_backtest_hyp_c(
                            df, 
                            fee_pct=0.0004, # Included Maker fee 0.04% for round-trip!
                            tp_pct=tp, 
                            sl_pct=sl, 
                            obi_thresh=obi, 
                            btc_delta_q=0.95,
                            wall_z_thresh=wall_z,
                            timeout_bars=t_bars
                        )
                        res['tp'] = tp
                        res['sl'] = sl
                        res['obi'] = obi
                        res['wall_z'] = wall_z
                        res['t_bars'] = t_bars
                        results.append(res)
                        pbar.update(1)
                        
    res_df = pd.DataFrame(results)
    res_df = res_df[res_df['trades'] >= 20] # Filter out noise
    
    if len(res_df) > 0:
        # Sort by Win Rate
        best_wr = res_df.sort_values('win_rate', ascending=False).head(5)
        print("\n--- Top 5 by Win Rate ---")
        print(best_wr[['tp', 'sl', 'obi', 'wall_z', 't_bars', 'trades', 'win_rate', 'pnl', 'avg_hold']].to_string(index=False))
        
        # Sort by total PnL
        best_pnl = res_df.sort_values('pnl', ascending=False).head(5)
        print("\n--- Top 5 by Total PnL ---")
        print(best_pnl[['tp', 'sl', 'obi', 'wall_z', 't_bars', 'trades', 'win_rate', 'pnl', 'avg_hold']].to_string(index=False))
    else:
        print("No valid results found.")

if __name__ == "__main__":
    optimize_hyp_c()