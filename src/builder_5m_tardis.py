import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

def process_tardis_trades(file_path):
    """Processes Tardis trades file and resamples to 5m bars with Volume Delta"""
    if not os.path.exists(file_path):
        return None
        
    df = pd.read_csv(file_path)
    
    # Tardis format usually has: timestamp, local_timestamp, id, side, price, amount
    # Convert timestamp (microseconds) to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='us')
    df.set_index('datetime', inplace=True)
    
    # Signed volume
    df['signed_volume'] = np.where(df['side'] == 'buy', df['amount'], -df['amount'])
    df['price_vol'] = df['price'] * df['amount']
    
    # Resample to 5m
    resampled = df.resample('5min').agg(
        open_price=('price', 'first'),
        high_price=('price', 'max'),
        low_price=('price', 'min'),
        close_price=('price', 'last'),
        volume_total=('amount', 'sum'),
        volume_delta=('signed_volume', 'sum'),
        trade_count=('id', 'count'),
        price_vol_sum=('price_vol', 'sum')
    )
    
    resampled['vwap'] = resampled['price_vol_sum'] / resampled['volume_total']
    resampled.drop(columns=['price_vol_sum'], inplace=True)
    
    return resampled

def process_tardis_liquidations(file_path, symbol):
    """Processes Liquidations to 5m aggregated sums"""
    if not os.path.exists(file_path):
        return None
        
    df = pd.read_csv(file_path)
    if len(df) == 0:
        return None
        
    # Tardis liquidations for Bybit come in a single PERPETUALS file now
    # We must filter by symbol
    df = df[df['symbol'] == symbol]
    if len(df) == 0:
        return None
        
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='us')
    df.set_index('datetime', inplace=True)
    
    # Pre-calculate signed amounts before resample
    df['liq_buy'] = np.where(df['side'] == 'buy', df['amount'], 0)
    df['liq_sell'] = np.where(df['side'] == 'sell', df['amount'], 0)
    
    # 'buy' liquidations mean shorts were liquidated. 'sell' means longs were liquidated.
    resampled = df.resample('5min').agg(
        liq_buy_vol=('liq_buy', 'sum'),
        liq_sell_vol=('liq_sell', 'sum')
    )
    return resampled

def process_tardis_oi(file_path):
    """Processes Derivative Tickers for Open Interest changes"""
    if not os.path.exists(file_path):
        return None
        
    df = pd.read_csv(file_path)
    if len(df) == 0:
        return None
        
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='us')
    df.set_index('datetime', inplace=True)
    
    # Get the last OI reading per 5m bar
    resampled = df.resample('5min').agg(
        oi_close=('open_interest', 'last')
    )
    
    # Calculate difference from previous bar
    resampled['oi_change'] = resampled['oi_close'].diff()
    return resampled

def build_5m_features():
    print("Building 5m Swing Trading features from Tardis data...")
    
    # We have 12 days of data from 2025 (1st of each month)
    months = [f"2025-{str(i).zfill(2)}-01" for i in range(1, 13)]
    
    # Also we now have contiguous data from Mar 15 to Mar 21 2026 in the new nested directory format
    months.extend(["2026-03-15", "2026-03-16", "2026-03-17", "2026-03-18", "2026-03-19", "2026-03-20", "2026-03-21"])
    
    for symbol in ["BTCUSDT", "SOLUSDT", "AVAXUSDT"]:
        all_bars = []
        for date_str in tqdm(months, desc=f"Processing {symbol}"):
            y, m, d = date_str.split("-")
            
            # File paths matching the old and new directory structure
            trades_file = f"data/tardis/bybit_trades_{date_str}_{symbol}.csv.gz"
            # Fallback if trades are also in nested folders
            if not os.path.exists(trades_file):
                trades_file = f"data/tardis/bybit/trades/{y}/{m}/{d}/{symbol}.csv.gz"
                
            liqs_file = f"data/tardis/bybit_liquidations_{date_str}_{symbol}.csv.gz"
            if not os.path.exists(liqs_file):
                liqs_file = f"data/tardis/bybit/liquidations/{y}/{m}/{d}/PERPETUALS.csv.gz"
                
            oi_file = f"data/tardis/bybit_derivative_ticker_{date_str}_{symbol}.csv.gz"
            if not os.path.exists(oi_file):
                oi_file = f"data/tardis/bybit/derivative_ticker/{y}/{m}/{d}/{symbol}.csv.gz"
            
            # Process
            df_trades = process_tardis_trades(trades_file)
            if df_trades is None:
                continue
                
            df_liqs = process_tardis_liquidations(liqs_file, symbol)
            df_oi = process_tardis_oi(oi_file)
            
            # Merge
            df_merged = df_trades
            if df_liqs is not None:
                df_merged = df_merged.join(df_liqs, how='left')
            else:
                df_merged['liq_buy_vol'] = 0
                df_merged['liq_sell_vol'] = 0
                
            if df_oi is not None:
                df_merged = df_merged.join(df_oi, how='left')
            else:
                df_merged['oi_close'] = np.nan
                df_merged['oi_change'] = 0
                
            # Fill NaNs for liqs (0 if no liquidations happened)
            df_merged['liq_buy_vol'] = df_merged['liq_buy_vol'].fillna(0)
            df_merged['liq_sell_vol'] = df_merged['liq_sell_vol'].fillna(0)
            
            # Forward fill prices for empty bars
            df_merged['close_price'] = df_merged['close_price'].ffill()
            df_merged['open_price'] = df_merged['open_price'].fillna(df_merged['close_price'])
            df_merged['high_price'] = df_merged['high_price'].fillna(df_merged['close_price'])
            df_merged['low_price'] = df_merged['low_price'].fillna(df_merged['close_price'])
            
            df_merged['volume_total'] = df_merged['volume_total'].fillna(0)
            df_merged['volume_delta'] = df_merged['volume_delta'].fillna(0)
            df_merged['oi_change'] = df_merged['oi_change'].fillna(0)
            
            all_bars.append(df_merged)
            
        if all_bars:
            final_df = pd.concat(all_bars)
            out_file = f"data/processed/{symbol}_swing_5m.pkl"
            final_df.to_pickle(out_file)
            print(f"Saved {len(final_df)} 5m bars to {out_file}")

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    build_5m_features()