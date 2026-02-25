#!/usr/bin/env python3
"""
Train the Range Model — specialized for sideways/ranging markets.
Mean reversion: buy near support, sell near resistance.

Usage:
    python train_range.py
"""

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.config import config
from src.data_fetcher import BinanceDataFetcher
from src.features import FeatureEngineer
from src.indicators import TechnicalIndicators
from src.range_model import RangeModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
console = Console()


def create_range_labels(df: pd.DataFrame, max_bars: int = 6) -> pd.Series:
    """
    Label range trading opportunities (mean reversion):
      BUY (1): price near range low and bounces up >0.5%
      SELL (-1): price near range high and drops >0.5%
      HOLD (0): price in middle or breakout
    """
    labels = pd.Series(0, index=df.index, dtype=int)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)

    range_period = 24  # bars to define range

    for i in range(range_period, n - max_bars):
        range_high = np.max(high[i - range_period:i])
        range_low = np.min(low[i - range_period:i])
        range_width = (range_high - range_low) / close[i] * 100

        # Skip if range too wide (not really ranging) or too narrow
        if range_width > 2.0 or range_width < 0.2:
            continue

        price = close[i]
        position_in_range = (price - range_low) / (range_high - range_low) if range_high > range_low else 0.5

        future_close = close[min(i + max_bars, n - 1)]
        future_move = (future_close - price) / price * 100

        # Near bottom of range (< 25%) and bounces up
        if position_in_range < 0.25 and future_move > 0.3:
            labels.iloc[i] = 1  # BUY

        # Near top of range (> 75%) and drops
        elif position_in_range > 0.75 and future_move < -0.3:
            labels.iloc[i] = -1  # SELL

    return labels


def fetch_pair_data(pair, fetcher, indicators, features):
    try:
        data = {}
        for tf, limit in [("5m", config.TRAIN_CANDLES), ("15m", config.TRAIN_CANDLES // 3), ("1h", config.TRAIN_CANDLES // 12)]:
            raw = fetcher.fetch_ohlcv(pair, tf, limit=limit)
            if raw.empty:
                return pair, None, None
            data[tf] = indicators.calculate_all(raw)
            time.sleep(0.2)

        X, _ = features.get_feature_matrix_mtf(
            data["5m"], data["15m"], data["1h"],
        )
        y = create_range_labels(data["5m"], max_bars=6)

        common = X.index.intersection(y.index)
        X = X.loc[common].iloc[:-6]
        y = y.loc[common].iloc[:-6]

        return pair, X, y
    except Exception as exc:
        logger.error("Error processing %s: %s", pair, exc)
        return pair, None, None


def train_range():
    console.print("\n[bold]Range Model Training Pipeline[/bold]")
    console.print(f"Pairs: {len(config.TRADING_PAIRS)} | Candles: {config.TRAIN_CANDLES}")
    console.print("Strategy: Mean reversion (buy low, sell high in range)")

    fetcher = BinanceDataFetcher(config)
    indicators = TechnicalIndicators()
    features = FeatureEngineer()

    all_X = []
    all_y = []

    console.print("\n[bold]Step 1/3 — Fetching & Processing[/bold]")

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(fetch_pair_data, pair, fetcher, indicators, features): pair
            for pair in config.TRADING_PAIRS
        }
        for future in as_completed(futures):
            pair, X, y = future.result()
            if X is not None and y is not None and len(X) > 100:
                all_X.append(X)
                all_y.append(y)
                counts = y.value_counts()
                console.print(
                    f"  {pair:<12s} {len(X):>6d} samples "
                    f"(BUY:{counts.get(1, 0)} SELL:{counts.get(-1, 0)} HOLD:{counts.get(0, 0)})"
                )
            else:
                console.print(f"  {pair:<12s} [red]SKIP[/red]")

    if not all_X:
        console.print("[red]No data![/red]")
        return

    X_all = pd.concat(all_X, ignore_index=True)
    y_all = pd.concat(all_y, ignore_index=True)

    # Filter: only keep samples where label != 0 (HOLD) + subsample of HOLD
    buy_sell = y_all != 0
    hold_sample = y_all[y_all == 0].sample(frac=0.3, random_state=42)
    keep_idx = buy_sell | y_all.index.isin(hold_sample.index)
    X_filtered = X_all[keep_idx]
    y_filtered = y_all[keep_idx]

    console.print(f"\n[bold]Step 2/3 — Dataset Summary[/bold]")
    console.print(f"Total samples (filtered): {len(X_filtered):,}")
    for label, name in {1: "BUY (near support)", -1: "SELL (near resistance)", 0: "HOLD"}.items():
        count = (y_filtered == label).sum()
        pct = count / len(y_filtered) * 100 if len(y_filtered) > 0 else 0
        console.print(f"  {name}: {count:,} ({pct:.1f}%)")

    range_cols = features.get_feature_columns(X_filtered, model_type="range")
    available = [c for c in range_cols if c in X_filtered.columns]
    console.print(f"Features: {len(available)}")

    # Map labels: -1 → 0, 0 → 1, 1 → 2
    y_mapped = y_filtered.map({-1: 0, 0: 1, 1: 2})

    console.print(f"\n[bold]Step 3/3 — Training XGBoost + LightGBM[/bold]")

    model = RangeModel()
    metrics = model.train(X_filtered, y_mapped, feature_names=available)

    table = Table(title="Range Model Results")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("CV Accuracy", f"{metrics['cv_accuracy']:.4f}")
    table.add_row("CV Std", f"±{metrics['cv_std']:.4f}")
    console.print(table)

    console.print("\nTop Feature Importances:")
    for i, (name, imp) in enumerate(model.feature_importance(15), 1):
        bar = "█" * max(1, int(imp / 20))
        console.print(f"  {i:>2d}. {name:<30s} {bar} {imp:.1f}")

    console.print(f"\n[green]Range Model saved → models/range_model.pkl[/green]")
    return metrics


if __name__ == "__main__":
    train_range()
