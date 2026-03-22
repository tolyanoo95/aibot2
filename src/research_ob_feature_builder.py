import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
import pyarrow.parquet as pq

def process_trades_for_day(symbol: str, date_str: str, base_dir: str = "data/bybit_trades") -> pd.DataFrame:
    """
    Reads trade data for a specific day and symbol, resamples to 10-second bars.
    Calculates Volume Delta (buy volume - sell volume).
    """
    file_path = f"{base_dir}/{symbol}/{symbol}_{date_str}.csv.gz"
    if not os.path.exists(file_path):
        print(f"Trade file not found: {file_path}")
        return None

    # trades schema: id, timestamp, price, volume, side, rpi
    df = pd.read_csv(file_path, compression='gzip')
    
    # Convert timestamp (ms) to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Set index for resampling
    df.set_index('datetime', inplace=True)
    
    # Calculate signed volume (positive for buy, negative for sell)
    # On Bybit, 'side' usually indicates the aggressor side (Buy/Sell)
    # Wait, in the preview the side was 'sell' (string). Let's check lowercase/uppercase.
    df['signed_volume'] = np.where(df['side'].str.lower() == 'buy', df['volume'], -df['volume'])
    
    # We need a different approach for VWAP in resample
    # First, calculate price * volume
    df['price_vol'] = df['price'] * df['volume']
    
    # Resample to 10 seconds
    resampled = df.resample('10s').agg(
        trade_count=('id', 'count'),
        volume_total=('volume', 'sum'),
        volume_delta=('signed_volume', 'sum'),
        price_vol_sum=('price_vol', 'sum'),
        close_price=('price', 'last')
    )
    
    # Calculate VWAP
    resampled['vwap'] = resampled['price_vol_sum'] / resampled['volume_total']
    resampled.drop(columns=['price_vol_sum'], inplace=True)
    
    
    return resampled

def process_orderbook_for_day(symbol: str, date_str: str, base_dir: str = "data/bybit_ob") -> pd.DataFrame:
    """
    Reads orderbook parquet data for a specific day and symbol, resamples to 10-second bars.
    Calculates Order Book Imbalance (OBI) and detects walls.
    """
    file_path = f"{base_dir}/{symbol}/{date_str}.parquet"
    if not os.path.exists(file_path):
        print(f"Orderbook file not found: {file_path}")
        return None

    df = pd.read_parquet(file_path)
    
    # ts is timestamp in ms
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('datetime', inplace=True)
    
    # We only need the last snapshot per 10-second window
    resampled = df.resample('10s').last()
    
    # Calculate features from bids/asks
    # In Parquet, bids/asks are likely stored as arrays/lists of [price, size] strings
    
    def calc_ob_features(row):
        if pd.isna(row['ts']):
            return pd.Series({'obi_5': np.nan, 'obi_10': np.nan, 'bid_wall_size': np.nan, 'ask_wall_size': np.nan, 'spread': np.nan})
            
        try:
            # Handle possible string representations if not native lists
            bids = row['bids']
            asks = row['asks']
            
            if isinstance(bids, np.ndarray):
                bids = bids.tolist()
            if isinstance(asks, np.ndarray):
                asks = asks.tolist()
                
            if isinstance(bids, str):
                import json
                try:
                    bids = json.loads(bids)
                    asks = json.loads(asks)
                except:
                    bids = []
                    asks = []
                    
            top_bids = bids[:20] if isinstance(bids, (list, np.ndarray)) else []
            top_asks = asks[:20] if isinstance(asks, (list, np.ndarray)) else []
            
            # If the inner elements are also ndarrays, convert them
            if top_bids and isinstance(top_bids[0], np.ndarray):
                top_bids = [b.tolist() for b in top_bids]
            if top_asks and isinstance(top_asks[0], np.ndarray):
                top_asks = [a.tolist() for a in top_asks]
            
            # Convert to floats: [ [price, size], ... ]
            try:
                bid_vols = [float(b[1]) for b in top_bids] if top_bids else [0]
                ask_vols = [float(a[1]) for a in top_asks] if top_asks else [0]
                best_bid = float(top_bids[0][0]) if top_bids else np.nan
                best_ask = float(top_asks[0][0]) if top_asks else np.nan
            except Exception as e:
                bid_vols = [0]
                ask_vols = [0]
                best_bid = np.nan
                best_ask = np.nan
            
            bid_vol_5 = sum(bid_vols[:5])
            ask_vol_5 = sum(ask_vols[:5])
            
            bid_vol_10 = sum(bid_vols[:10])
            ask_vol_10 = sum(ask_vols[:10])
            
            obi_5 = (bid_vol_5 - ask_vol_5) / (bid_vol_5 + ask_vol_5) if (bid_vol_5 + ask_vol_5) > 0 else 0
            obi_10 = (bid_vol_10 - ask_vol_10) / (bid_vol_10 + ask_vol_10) if (bid_vol_10 + ask_vol_10) > 0 else 0
            
            return pd.Series({
                'obi_5': obi_5,
                'obi_10': obi_10,
                'bid_wall_size': max(bid_vols) if bid_vols else 0, # largest order in top 10
                'ask_wall_size': max(ask_vols) if ask_vols else 0,
                'spread': best_ask - best_bid
            })
        except Exception as e:
            return pd.Series({'obi_5': np.nan, 'obi_10': np.nan, 'bid_wall_size': np.nan, 'ask_wall_size': np.nan, 'spread': np.nan})

    # Apply function to calculate features
    # Since apply on rows is slow, we might want to optimize this later if it's too slow
    ob_features = resampled.apply(calc_ob_features, axis=1)
    
    # Merge back
    return pd.concat([resampled[['ts', 'seq']], ob_features], axis=1)

def build_features_for_symbol(symbol: str, dates: list, output_file: str):
    """
    Combines trades and orderbook data for multiple days and saves to a single file.
    """
    print(f"Building features for {symbol}...")
    all_data = []
    
    for date_str in tqdm(dates, desc="Processing days"):
        trades_df = process_trades_for_day(symbol, date_str)
        ob_df = process_orderbook_for_day(symbol, date_str)
        
        if trades_df is not None and ob_df is not None:
            # Merge on the 10-second datetime index
            merged = pd.merge(trades_df, ob_df, left_index=True, right_index=True, how='outer')
            # Forward fill missing orderbook data (if no update in that 10s window)
            merged['obi_5'] = merged['obi_5'].ffill()
            merged['obi_10'] = merged['obi_10'].ffill()
            merged['spread'] = merged['spread'].ffill()
            
            # Fill NaNs for trades (if no trades in that 10s window)
            merged['volume_total'] = merged['volume_total'].fillna(0)
            merged['volume_delta'] = merged['volume_delta'].fillna(0)
            merged['trade_count'] = merged['trade_count'].fillna(0)
            merged['close_price'] = merged['close_price'].ffill()
            
            all_data.append(merged)
            
    if all_data:
        final_df = pd.concat(all_data)
        final_df.to_pickle(output_file)
        print(f"Saved {len(final_df)} feature rows to {output_file}")
    else:
        print("No data processed.")

if __name__ == "__main__":
    import re
    
    # Get all available dates from SOLUSDT orderbook directory
    ob_dir = "data/bybit_ob/SOLUSDT"
    if os.path.exists(ob_dir):
        files = os.listdir(ob_dir)
        all_dates = []
        for f in files:
            match = re.match(r"(\d{4}-\d{2}-\d{2})\.parquet", f)
            if match:
                all_dates.append(match.group(1))
        all_dates.sort()
    else:
        all_dates = ["2025-06-01", "2025-06-02", "2025-06-03"]
        
    print(f"Found {len(all_dates)} days of data to process.")
    
    # Ensure data directory exists
    os.makedirs("data/processed", exist_ok=True)
    
    build_features_for_symbol("SOLUSDT", all_dates, "data/processed/SOLUSDT_features_10s_full.pkl")
    build_features_for_symbol("BTCUSDT", all_dates, "data/processed/BTCUSDT_features_10s_full.pkl")
