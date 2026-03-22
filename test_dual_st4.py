import pandas as pd
import numpy as np

def calculate_supertrend(df, period=10, multiplier=3.0):
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    hl2 = (high + low) / 2
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    
    # tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1).values
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0.0
    
    atr = pd.Series(tr).rolling(period).mean().values
    
    basic_upperband = hl2 + (multiplier * atr)
    basic_lowerband = hl2 - (multiplier * atr)
    
    final_upperband = np.zeros_like(basic_upperband)
    final_lowerband = np.zeros_like(basic_lowerband)
    
    st_dir = np.ones_like(close)
    st = np.zeros_like(close)
    
    for i in range(1, len(close)):
        if close[i-1] <= final_upperband[i-1]:
            final_upperband[i] = min(basic_upperband[i], final_upperband[i-1])
        else:
            final_upperband[i] = basic_upperband[i]
            
        if close[i-1] >= final_lowerband[i-1]:
            final_lowerband[i] = max(basic_lowerband[i], final_lowerband[i-1])
        else:
            final_lowerband[i] = basic_lowerband[i]
            
        if st_dir[i-1] == 1:
            if close[i] < final_lowerband[i]:
                st_dir[i] = -1
                st[i] = final_upperband[i]
            else:
                st_dir[i] = 1
                st[i] = final_lowerband[i]
        else:
            if close[i] > final_upperband[i]:
                st_dir[i] = 1
                st[i] = final_lowerband[i]
            else:
                st_dir[i] = -1
                st[i] = final_upperband[i]
                
    return st_dir

def test_dual_st():
    print("Loading 15m data...")
    # Load 15m data from the cache mentioned in project-context.mdc
    df_cache = pd.read_pickle('data/ohlcv_cache_15m.pkl')
    
    results = []
    
    for symbol in ["BTC/USDT", "SOL/USDT", "AVAX/USDT", "ETH/USDT"]:
        if symbol not in df_cache:
            continue
            
        df = df_cache[symbol].copy()
        if len(df) < 1000:
            continue
            
        print(f"Testing {symbol} ({len(df)} candles)...")
        
        # Calculate Supertrends
        # Slow ST(10, 5)
        df['st_slow_dir'] = calculate_supertrend(df, period=10, multiplier=5.0)
        
        # Fast ST(10, 3)
        df['st_fast_dir'] = calculate_supertrend(df, period=10, multiplier=3.0)
        
        df = df.dropna()
        
        trades = []
        in_position = False
        pos_type = None
        entry_price = 0
        entry_time = None
        
        for i in range(1, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Fast flipped THIS candle
            fast_flipped_short = curr['st_fast_dir'] == -1 and prev['st_fast_dir'] == 1
            fast_flipped_long = curr['st_fast_dir'] == 1 and prev['st_fast_dir'] == -1
            
            if not in_position:
                # ENTRY LOGIC
                # Slow = LONG, Fast flips SHORT -> Enter LONG (buy dip in uptrend)
                if curr['st_slow_dir'] == 1 and fast_flipped_short:
                    in_position = True
                    pos_type = 'long'
                    entry_price = curr['close']
                    entry_time = df.index[i]
                    
                # Slow = SHORT, Fast flips LONG -> Enter SHORT (sell pump in downtrend)
                elif curr['st_slow_dir'] == -1 and fast_flipped_long:
                    in_position = True
                    pos_type = 'short'
                    entry_price = curr['close']
                    entry_time = df.index[i]
                    
            else:
                # EXIT LOGIC
                # Exit when fast flips BACK to slow direction
                if pos_type == 'long' and curr['st_fast_dir'] == 1 and prev['st_fast_dir'] == -1:
                    pnl = (curr['close'] - entry_price) / entry_price - 0.0011 # Taker fee
                    trades.append({'pnl': pnl, 'type': 'long', 'entry_time': entry_time, 'exit_time': df.index[i]})
                    in_position = False
                    
                elif pos_type == 'short' and curr['st_fast_dir'] == -1 and prev['st_fast_dir'] == 1:
                    pnl = (entry_price - curr['close']) / entry_price - 0.0011
                    trades.append({'pnl': pnl, 'type': 'short', 'entry_time': entry_time, 'exit_time': df.index[i]})
                    in_position = False
                    
                # Stop loss if Slow ST flips against us (trend reversed)
                elif pos_type == 'long' and curr['st_slow_dir'] == -1:
                    pnl = (curr['close'] - entry_price) / entry_price - 0.0011
                    trades.append({'pnl': pnl, 'type': 'long', 'entry_time': entry_time, 'exit_time': df.index[i]})
                    in_position = False
                    
                elif pos_type == 'short' and curr['st_slow_dir'] == 1:
                    pnl = (entry_price - curr['close']) / entry_price - 0.0011
                    trades.append({'pnl': pnl, 'type': 'short', 'entry_time': entry_time, 'exit_time': df.index[i]})
                    in_position = False
                    
        tdf = pd.DataFrame(trades)
        if len(tdf) > 0:
            wr = (tdf['pnl'] > 0).mean() * 100
            tot_pnl = tdf['pnl'].sum() * 100
            
            results.append({
                'symbol': symbol,
                'trades': len(tdf),
                'win_rate': wr,
                'total_pnl': tot_pnl
            })
            
    res_df = pd.DataFrame(results)
    print("\n=== Dual Supertrend 15m Results ===")
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    test_dual_st()
