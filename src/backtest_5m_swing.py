import pandas as pd
import numpy as np
from tqdm import tqdm

def load_data():
    df_btc = pd.read_pickle("data/processed/BTCUSDT_swing_5m.pkl")
    df_sol = pd.read_pickle("data/processed/SOLUSDT_swing_5m.pkl")
    
    # Merge on index
    df = df_sol.join(df_btc, lsuffix='_sol', rsuffix='_btc', how='inner')
    return df.dropna()

def run_5m_backtest(df, fee_pct=0.0011, tp_pct=0.015, sl_pct=0.005, btc_delta_q=0.90, oi_thresh=1000):
    """
    Swing strategy based on BTC extreme Delta + OI Confirmation
    fee_pct=0.0011 means 0.055% taker fee x 2 (round-trip)
    """
    btc_delta_thresh_long = df['volume_delta_btc'].quantile(btc_delta_q)
    btc_delta_thresh_short = df['volume_delta_btc'].quantile(1 - btc_delta_q)
    
    in_position = False
    position_type = None
    entry_price = 0
    trades = []
    
    for i in range(1, len(df)):
        curr_row = df.iloc[i]
        
        if not in_position:
            # LONG SIGNAL: Extreme BTC buying + New money entering SOL (OI growing) + No extreme long liquidations
            if (curr_row['volume_delta_btc'] > btc_delta_thresh_long and 
                curr_row['oi_change_sol'] > oi_thresh and
                curr_row['liq_sell_vol_sol'] < 5000): # filter out fake squeezes
                
                entry_price = curr_row['close_price_sol']
                in_position = True
                position_type = "long"
                entry_time = df.index[i]
                
            # SHORT SIGNAL
            elif (curr_row['volume_delta_btc'] < btc_delta_thresh_short and 
                  curr_row['oi_change_sol'] > oi_thresh and # OI must grow for short too! (new shorts)
                  curr_row['liq_buy_vol_sol'] < 5000): 
                
                entry_price = curr_row['close_price_sol']
                in_position = True
                position_type = "short"
                entry_time = df.index[i]
                
        else:
            if position_type == "long":
                pnl_pct = (curr_row['high_price_sol'] - entry_price) / entry_price
                loss_pct = (curr_row['low_price_sol'] - entry_price) / entry_price
                
                if pnl_pct >= tp_pct:
                    pnl = tp_pct - fee_pct
                    trades.append({'entry_time': entry_time, 'pnl': pnl, 'type': position_type, 'exit_reason': 'TP'})
                    in_position = False
                elif loss_pct <= -sl_pct:
                    pnl = -sl_pct - fee_pct
                    trades.append({'entry_time': entry_time, 'pnl': pnl, 'type': position_type, 'exit_reason': 'SL'})
                    in_position = False
                    
            elif position_type == "short":
                pnl_pct = (entry_price - curr_row['low_price_sol']) / entry_price
                loss_pct = (entry_price - curr_row['high_price_sol']) / entry_price
                
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
        return {'trades': 0, 'win_rate': 0, 'pnl': 0, 'avg_pnl': 0}
        
    win_rate = (trades_df['pnl'] > 0).mean() * 100
    total_pnl = trades_df['pnl'].sum() * 100
    avg_pnl = trades_df['pnl'].mean() * 100
    
    return {
        'trades': len(trades_df),
        'win_rate': win_rate,
        'pnl': total_pnl,
        'avg_pnl': avg_pnl
    }

if __name__ == "__main__":
    df = load_data()
    print(f"Data loaded: {len(df)} 5-min bars")
    
    results = []
    
    tp_sl_pairs = [(0.015, 0.005), (0.02, 0.008), (0.01, 0.005)]
    oi_thresholds = [1000, 5000, 10000]
    btc_quantiles = [0.90, 0.95]
    
    for tp, sl in tp_sl_pairs:
        for oi in oi_thresholds:
            for btc_q in btc_quantiles:
                # Filter data for March 15 to March 21
                df_march_week = df[(df.index >= '2026-03-15') & (df.index < '2026-03-22')]
                
                if len(df_march_week) == 0:
                    continue
                    
                res = run_5m_backtest(df_march_week, fee_pct=0.0011, tp_pct=tp, sl_pct=sl, btc_delta_q=btc_q, oi_thresh=oi)
                res['tp'] = tp
                res['sl'] = sl
                res['oi'] = oi
                res['btc_q'] = btc_q
                results.append(res)
                
    res_df = pd.DataFrame(results)
    
    print("\n--- March 15-21, 2026 Specific Results ---")
    print(res_df.sort_values('pnl', ascending=False).to_string(index=False))
