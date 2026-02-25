#!/usr/bin/env python3
"""
Train the Regime Classifier (TREND / REVERSAL / RANGE).
Uses the same data pipeline as train.py but with regime labels.

Usage:
    python train_regime.py
"""

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from rich.console import Console
from rich.table import Table

from src.config import config
from src.data_fetcher import BinanceDataFetcher
from src.features import FeatureEngineer
from src.indicators import TechnicalIndicators
from src.regime_classifier import RegimeClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
console = Console()


def fetch_pair_data(pair: str, fetcher, indicators, features):
    """Fetch and process data for one pair."""
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

        y = features.create_regime_labels(
            data["5m"],
            trend_threshold=1.5,
            range_threshold=0.5,
            max_bars=config.LABEL_MAX_BARS,
        )

        common = X.index.intersection(y.index)
        X = X.loc[common].iloc[:-config.LABEL_MAX_BARS]
        y = y.loc[common].iloc[:-config.LABEL_MAX_BARS]

        return pair, X, y
    except Exception as exc:
        logger.error("Error processing %s: %s", pair, exc)
        return pair, None, None


def train_regime():
    console.print("\n[bold]Regime Classifier Training Pipeline[/bold]")
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
                    f"(T:{counts.get(1, 0)} R:{counts.get(2, 0)} Rng:{counts.get(0, 0)})"
                )
            else:
                console.print(f"  {pair:<12s} [red]SKIP[/red]")

    if not all_X:
        console.print("[red]No data collected![/red]")
        return

    X_all = pd.concat(all_X, ignore_index=True)
    y_all = pd.concat(all_y, ignore_index=True)

    console.print(f"\n[bold]Step 2/3 — Dataset Summary[/bold]")
    console.print(f"Total samples: {len(X_all):,}")
    for label_id, label_name in {0: "RANGE", 1: "TREND", 2: "REVERSAL"}.items():
        count = (y_all == label_id).sum()
        pct = count / len(y_all) * 100
        console.print(f"  {label_name}: {count:,} ({pct:.1f}%)")

    regime_features = features.REGIME_FEATURES
    available = [c for c in regime_features if c in X_all.columns]
    console.print(f"Regime features: {len(available)}")

    console.print(f"\n[bold]Step 3/3 — Training LightGBM[/bold]")

    classifier = RegimeClassifier()
    metrics = classifier.train(X_all, y_all, feature_names=available)

    table = Table(title="Regime Classifier Results")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("CV Accuracy", f"{metrics['cv_accuracy']:.4f}")
    table.add_row("CV Std", f"±{metrics['cv_std']:.4f}")
    console.print(table)

    console.print("\nTop Feature Importances:")
    for i, (name, imp) in enumerate(classifier.feature_importance(12), 1):
        bar = "█" * max(1, int(imp / 50))
        console.print(f"  {i:>2d}. {name:<25s} {bar} {imp:.1f}")

    console.print(f"\n[green]Regime Classifier saved → models/regime_model.pkl[/green]")

    return metrics


if __name__ == "__main__":
    train_regime()
