import os
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta

def download_bybit_trades(symbol="AVAXUSDT", start_date="2026-03-15", end_date="2026-03-21"):
    base_url = f"https://public.bybit.com/trading/{symbol}/"
    out_dir = "data/bybit_public"
    os.makedirs(out_dir, exist_ok=True)
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    curr = start
    downloaded_files = []
    
    while curr <= end:
        date_str = curr.strftime("%Y-%m-%d")
        file_name = f"{symbol}{date_str}.csv.gz"
        url = base_url + file_name
        out_path = os.path.join(out_dir, file_name)
        
        print(f"Downloading {file_name}...")
        
        if not os.path.exists(out_path):
            try:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(out_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    downloaded_files.append(out_path)
                    print(f"Successfully downloaded {file_name}")
                else:
                    print(f"Failed to download {file_name}, status code: {response.status_code}")
            except Exception as e:
                print(f"Error downloading {file_name}: {e}")
        else:
            print(f"{file_name} already exists. Skipping.")
            downloaded_files.append(out_path)
            
        curr += timedelta(days=1)
        
    return downloaded_files

def process_bybit_trades(files, symbol="AVAXUSDT"):
    print(f"Processing {len(files)} files to 5m bars...")
    all_bars = []
    
    for f in tqdm(files):
        try:
            # Bybit public trades CSV format:
            # timestamp,symbol,side,size,price,tickDirection,trdMatchID,grossValue,homeNotional,foreignNotional
            df = pd.read_csv(f)
            
            # Use timestamp in seconds to convert to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('datetime', inplace=True)
            
            # Format to match Tardis structure
            df['signed_volume'] = np.where(df['side'] == 'Buy', df['size'], -df['size'])
            df['price_vol'] = df['price'] * df['size']
            
            resampled = df.resample('5min').agg(
                open_price=('price', 'first'),
                high_price=('price', 'max'),
                low_price=('price', 'min'),
                close_price=('price', 'last'),
                volume_total=('size', 'sum'),
                volume_delta=('signed_volume', 'sum'),
                trade_count=('size', 'count'),
                price_vol_sum=('price_vol', 'sum')
            )
            
            resampled['vwap'] = resampled['price_vol_sum'] / resampled['volume_total']
            resampled.drop(columns=['price_vol_sum'], inplace=True)
            
            all_bars.append(resampled)
            
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    if all_bars:
        final_df = pd.concat(all_bars)
        
        # Merge with liquidations and OI from Tardis if we have them
        print("Merging with Tardis liquidations and OI...")
        
        # We need to extract the exact tardis code for AVAX
        import sys
        sys.path.append('src')
        from builder_5m_tardis import process_tardis_liquidations, process_tardis_oi
        import glob
        
        # Find liquidations and OI for this symbol
        for date_str in pd.date_range(start="2026-02-01", end="2026-03-21").strftime("%Y-%m-%d"):
            y, m, d = date_str.split("-")
            
            liqs_file = f"data/tardis/bybit/liquidations/{y}/{m}/{d}/PERPETUALS.csv.gz"
            oi_file = f"data/tardis/bybit/derivative_ticker/{y}/{m}/{d}/{symbol}.csv.gz"
            
            if os.path.exists(liqs_file):
                df_liqs = process_tardis_liquidations(liqs_file, symbol)
                if df_liqs is not None:
                    final_df.update(df_liqs)
                    # For non-overlapping index, use combine_first
                    final_df = final_df.combine_first(df_liqs)
            
            if os.path.exists(oi_file):
                df_oi = process_tardis_oi(oi_file)
                if df_oi is not None:
                    final_df.update(df_oi)
                    final_df = final_df.combine_first(df_oi)
                    
        final_df = final_df.fillna(0)
        
        out_file = f"data/processed/{symbol}_swing_5m.pkl"
        final_df.to_pickle(out_file)
        print(f"Saved {len(final_df)} 5m bars to {out_file}")

def download_all_pairs():
    # Downloader wrapper for multiple pairs for full march
    import time
    pairs = ["BTCUSDT", "SOLUSDT", "AVAXUSDT"]
    all_downloaded = {p: [] for p in pairs}
    
    for pair in pairs:
        print(f"--- Downloading {pair} ---")
        files = download_bybit_trades(symbol=pair, start_date="2026-02-01", end_date="2026-02-28")
        all_downloaded[pair] = files
        
    for pair in pairs:
        if all_downloaded[pair]:
            process_bybit_trades(all_downloaded[pair], symbol=pair)

if __name__ == "__main__":
    import numpy as np
    download_all_pairs()
