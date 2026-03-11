#!/usr/bin/env python3
"""
Fetch and cache OHLCV data. Only fetches new bars since last cache.

Usage:
    python cache_data.py                # update 15m (default)
    python cache_data.py --tf 5m        # update 5m
    python cache_data.py --tf 1h        # update 1h
    python cache_data.py --tf 15m --full  # force full refetch
"""
import argparse, os, pickle, functools, logging, warnings
import pandas as pd
from src.config import config
from src.data_fetcher import BinanceDataFetcher

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
print = functools.partial(print, flush=True)

CACHE_DIR = "data"
BARS_PER_DAY = {"1m": 1440, "5m": 288, "15m": 96, "30m": 48, "1h": 24, "4h": 6, "1d": 1}


def cache_file(tf):
    return os.path.join(CACHE_DIR, f"ohlcv_cache_{tf}.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf", type=str, default="15m", help="Timeframe: 1m,5m,15m,30m,1h,4h,1d")
    parser.add_argument("--days", type=int, default=435, help="Days of history for first fetch")
    parser.add_argument("--full", action="store_true", help="Force full refetch")
    args = parser.parse_args()

    tf = args.tf
    bpd = BARS_PER_DAY.get(tf, 96)
    path = cache_file(tf)

    os.makedirs(CACHE_DIR, exist_ok=True)
    fetcher = BinanceDataFetcher(config)

    existing = {}
    if os.path.exists(path) and not args.full:
        with open(path, "rb") as f:
            existing = pickle.load(f)
        print(f"Cache {tf}: {len(existing)} pairs")

    for symbol in config.TRADING_PAIRS:
        if symbol in existing and len(existing[symbol]) > 0 and not args.full:
            df_new = fetcher.fetch_ohlcv(symbol, tf, limit=bpd * 2)
            if df_new.empty:
                print(f"  {symbol}: no new data")
                continue
            df = pd.concat([existing[symbol], df_new])
            df = df[~df.index.duplicated(keep='last')]
            df = df.sort_index()
            new_bars = len(df) - len(existing[symbol])
            existing[symbol] = df
            print(f"  {symbol}: +{new_bars} bars (total {len(df)})")
        else:
            total_candles = args.days * bpd
            print(f"  {symbol}: fetching {args.days}d {tf}...")
            df = fetcher.fetch_ohlcv_extended(symbol, tf, total_candles=total_candles)
            if df.empty:
                continue
            existing[symbol] = df
            print(f"  {symbol}: {len(df)} bars")

    with open(path, "wb") as f:
        pickle.dump(existing, f)
    print(f"\nSaved {len(existing)} pairs to {path}")


if __name__ == "__main__":
    main()
