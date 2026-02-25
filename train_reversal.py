#!/usr/bin/env python3
"""
Train the Reversal Model — specialized for catching market reversals.
Filters training data to only reversal points for focused learning.

Usage:
    python train_reversal.py
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
from src.reversal_model import ReversalModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
console = Console()


def create_reversal_labels(df: pd.DataFrame, max_bars: int = 12) -> pd.Series:
    """
    Label reversal points:
      BUY (1): price was falling, then reverses up >1%
      SELL (-1): price was rising, then reverses down >1%
      HOLD (0): no reversal
    """
    labels = pd.Series(0, index=df.index, dtype=int)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)

    lookback = 12  # bars to check past trend

    for i in range(lookback, n - max_bars):
        past_move = (close[i] - close[i - lookback]) / close[i - lookback] * 100
        future_close = close[min(i + max_bars, n - 1)]
        future_move = (future_close - close[i]) / close[i] * 100

        # Was falling, now reverses up
        if past_move < -0.5 and future_move > 1.0:
            labels.iloc[i] = 1  # BUY reversal

        # Was rising, now reverses down
        elif past_move > 0.5 and future_move < -1.0:
            labels.iloc[i] = -1  # SELL reversal

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

        X, feat_names = features.get_feature_matrix_mtf(
            data["5m"], data["15m"], data["1h"],
        )
        y = create_reversal_labels(data["5m"], max_bars=config.LABEL_MAX_BARS)

        common = X.index.intersection(y.index)
        X = X.loc[common].iloc[:-config.LABEL_MAX_BARS]
        y = y.loc[common].iloc[:-config.LABEL_MAX_BARS]

        return pair, X, y
    except Exception as exc:
        logger.error("Error processing %s: %s", pair, exc)
        return pair, None, None


def train_reversal():
    console.print("\n[bold]Reversal Model Training Pipeline[/bold]")
    console.print(f"Pairs: {len(config.TRADING_PAIRS)} | Candles: {config.TRAIN_CANDLES}")

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

    console.print(f"\n[bold]Step 2/3 — Dataset Summary[/bold]")
    console.print(f"Total samples: {len(X_all):,}")
    for label, name in {1: "BUY reversal", -1: "SELL reversal", 0: "HOLD"}.items():
        count = (y_all == label).sum()
        pct = count / len(y_all) * 100
        console.print(f"  {name}: {count:,} ({pct:.1f}%)")

    # Use reversal-specific features (base + reversal extras)
    reversal_cols = features.get_feature_columns(X_all, model_type="reversal")
    available = [c for c in reversal_cols if c in X_all.columns]
    console.print(f"Features: {len(available)}")

    # Map labels: -1 → 0, 0 → 1, 1 → 2 (for sklearn)
    y_mapped = y_all.map({-1: 0, 0: 1, 1: 2})

    console.print(f"\n[bold]Step 3/3 — Training XGBoost + LightGBM[/bold]")

    model = ReversalModel()
    metrics = model.train(X_all, y_mapped, feature_names=available)

    table = Table(title="Reversal Model Results")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("CV Accuracy", f"{metrics['cv_accuracy']:.4f}")
    table.add_row("CV Std", f"±{metrics['cv_std']:.4f}")
    console.print(table)

    console.print("\nTop Feature Importances:")
    for i, (name, imp) in enumerate(model.feature_importance(15), 1):
        bar = "█" * max(1, int(imp / 20))
        console.print(f"  {i:>2d}. {name:<30s} {bar} {imp:.1f}")

    console.print(f"\n[green]Reversal Model saved → models/reversal_model.pkl[/green]")
    return metrics


if __name__ == "__main__":
    train_reversal()
