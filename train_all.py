#!/usr/bin/env python3
"""
Train ALL models in parallel: Trend + Regime + Reversal + Range.
Fetches data ONCE, then trains all 4 models simultaneously.

Usage:
    python train_all.py
"""

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, MofNCompleteColumn

from src.config import config
from src.data_fetcher import BinanceDataFetcher
from src.features import FeatureEngineer
from src.indicators import TechnicalIndicators
from src.ml_model import MLSignalModel
from src.regime_classifier import RegimeClassifier
from src.reversal_model import ReversalModel
from src.range_model import RangeModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler("train_all.log")],
)
logger = logging.getLogger(__name__)
console = Console()


def fetch_pair_data(pair, fetcher, indicators, features):
    """Fetch and process data for one pair (all timeframes, with pagination)."""
    try:
        data = {}
        for tf, total in [("5m", config.TRAIN_CANDLES), ("15m", config.TRAIN_CANDLES // 3), ("1h", config.TRAIN_CANDLES // 12)]:
            raw = fetcher.fetch_ohlcv_extended(pair, tf, total_candles=total)
            if raw.empty:
                return pair, None
            data[tf] = indicators.calculate_all(raw)

        X, feat_names = features.get_feature_matrix_mtf(
            data["5m"], data["15m"], data["1h"],
        )
        return pair, {"X": X, "df_5m": data["5m"], "feat_names": feat_names}
    except Exception as exc:
        logger.error("Error fetching %s: %s", pair, exc)
        return pair, None


# ── Individual model trainers (called in parallel) ─────────

def _train_trend(pair_data, features):
    """Train Trend Model on shared data."""
    all_X, all_y = [], []
    for pair, d in pair_data.items():
        X = d["X"]
        y = features.create_labels(
            d["df_5m"],
            tp_multiplier=config.LABEL_TP_MULTIPLIER,
            sl_multiplier=config.LABEL_SL_MULTIPLIER,
            max_bars=config.LABEL_MAX_BARS,
            binary=True,
        )
        common = X.index.intersection(y.index)
        X_c = X.loc[common].iloc[:-config.LABEL_MAX_BARS]
        y_c = y.loc[common].iloc[:-config.LABEL_MAX_BARS]
        keep = y_c != 0
        all_X.append(X_c.loc[keep])
        all_y.append(y_c.loc[keep])

    X_all = pd.concat(all_X, ignore_index=True)
    y_all = pd.concat(all_y, ignore_index=True)

    trend_model = MLSignalModel(config.ML_MODEL_PATH)
    feat_cols = features.get_feature_columns(X_all, model_type="trend")
    available = [c for c in feat_cols if c in X_all.columns]
    # Pass original labels (-1, 1) — MLSignalModel maps them internally
    metrics = trend_model.train(X_all[available], y_all, feature_names=available)

    samples = f"BUY:{(y_all==1).sum()} SELL:{(y_all==-1).sum()}"
    return "Trend", metrics, len(X_all), samples


def _train_regime(pair_data, features):
    """Train Regime Classifier on shared data."""
    all_X, all_y = [], []
    for pair, d in pair_data.items():
        X = d["X"]
        y = features.create_regime_labels(d["df_5m"], max_bars=config.LABEL_MAX_BARS)
        common = X.index.intersection(y.index)
        all_X.append(X.loc[common].iloc[:-config.LABEL_MAX_BARS])
        all_y.append(y.loc[common].iloc[:-config.LABEL_MAX_BARS])

    X_all = pd.concat(all_X, ignore_index=True)
    y_all = pd.concat(all_y, ignore_index=True)

    regime_clf = RegimeClassifier()
    regime_cols = [c for c in features.REGIME_FEATURES if c in X_all.columns]
    metrics = regime_clf.train(X_all, y_all, feature_names=regime_cols)

    samples = f"T:{(y_all==1).sum()} R:{(y_all==2).sum()} Rng:{(y_all==0).sum()}"
    return "Regime", metrics, len(X_all), samples


def _train_reversal(pair_data, features):
    """Train Reversal Model on shared data."""
    from train_reversal import create_reversal_labels

    all_X, all_y = [], []
    for pair, d in pair_data.items():
        X = d["X"]
        y = create_reversal_labels(d["df_5m"], max_bars=config.LABEL_MAX_BARS)
        common = X.index.intersection(y.index)
        all_X.append(X.loc[common].iloc[:-config.LABEL_MAX_BARS])
        all_y.append(y.loc[common].iloc[:-config.LABEL_MAX_BARS])

    X_all = pd.concat(all_X, ignore_index=True)
    y_all = pd.concat(all_y, ignore_index=True)

    rev_cols = features.get_feature_columns(X_all, model_type="reversal")
    available = [c for c in rev_cols if c in X_all.columns]
    y_mapped = y_all.map({-1: 0, 0: 1, 1: 2})

    rev_model = ReversalModel()
    metrics = rev_model.train(X_all, y_mapped, feature_names=available)

    samples = f"BUY:{(y_all==1).sum()} SELL:{(y_all==-1).sum()} HOLD:{(y_all==0).sum()}"
    return "Reversal", metrics, len(X_all), samples


def _train_range(pair_data, features):
    """Train Range Model on shared data."""
    from train_range import create_range_labels

    all_X, all_y = [], []
    for pair, d in pair_data.items():
        X = d["X"]
        y = create_range_labels(d["df_5m"], max_bars=6)
        common = X.index.intersection(y.index)
        all_X.append(X.loc[common].iloc[:-6])
        all_y.append(y.loc[common].iloc[:-6])

    X_all = pd.concat(all_X, ignore_index=True)
    y_all = pd.concat(all_y, ignore_index=True)

    buy_sell = y_all != 0
    hold_sample = y_all[y_all == 0].sample(frac=0.3, random_state=42)
    keep_idx = buy_sell | y_all.index.isin(hold_sample.index)
    X_filt = X_all[keep_idx]
    y_filt = y_all[keep_idx]

    rng_cols = features.get_feature_columns(X_filt, model_type="range")
    available = [c for c in rng_cols if c in X_filt.columns]
    y_mapped = y_filt.map({-1: 0, 0: 1, 1: 2})

    rng_model = RangeModel()
    metrics = rng_model.train(X_filt, y_mapped, feature_names=available)

    samples = f"BUY:{(y_filt==1).sum()} SELL:{(y_filt==-1).sum()} HOLD:{(y_filt==0).sum()}"
    return "Range", metrics, len(X_filt), samples


# ── Main ──────────────────────────────────────────────────

def train_all():
    t0 = time.time()
    console.print(Panel.fit(
        f"[bold]Multi-Model Training Pipeline[/bold]\n"
        f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Models: Trend + Regime + Reversal + Range (parallel)\n"
        f"Pairs: {len(config.TRADING_PAIRS)} | Candles: {config.TRAIN_CANDLES}",
        border_style="green",
    ))

    fetcher = BinanceDataFetcher(config)
    indicators = TechnicalIndicators()
    features = FeatureEngineer()

    # ═══════════════════════════════════════════════════════
    # STEP 1: Fetch data ONCE
    # ═══════════════════════════════════════════════════════
    console.print("\n[bold cyan]═══ STEP 1: Fetching Data ═══[/bold cyan]")

    pair_data = {}
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30), MofNCompleteColumn(), TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching …", total=len(config.TRADING_PAIRS))
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {
                pool.submit(fetch_pair_data, pair, fetcher, indicators, features): pair
                for pair in config.TRADING_PAIRS
            }
            for future in as_completed(futures):
                pair, data = future.result()
                if data is not None:
                    pair_data[pair] = data
                    progress.update(task, advance=1, description=f"Fetched {pair} ({len(data['X'])} bars)")
                else:
                    progress.update(task, advance=1, description=f"[red]{pair} SKIP[/red]")

    if not pair_data:
        console.print("[red]No data![/red]")
        return

    total_bars = sum(len(d["X"]) for d in pair_data.values())
    console.print(f"\n  ✓ {total_bars:,} bars from {len(pair_data)} pairs")

    # ═══════════════════════════════════════════════════════
    # STEP 2: Train all 4 models in PARALLEL
    # ═══════════════════════════════════════════════════════
    console.print("\n[bold cyan]═══ STEP 2: Training 4 Models (parallel) ═══[/bold cyan]")

    results = {}
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30), MofNCompleteColumn(), TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Training …", total=4)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(_train_trend, pair_data, features): "Trend",
                pool.submit(_train_regime, pair_data, features): "Regime",
                pool.submit(_train_reversal, pair_data, features): "Reversal",
                pool.submit(_train_range, pair_data, features): "Range",
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    model_name, metrics, n_samples, samples_str = future.result()
                    results[model_name] = {
                        "accuracy": metrics.get("cv_accuracy", 0),
                        "std": metrics.get("cv_std", 0),
                        "samples": n_samples,
                        "detail": samples_str,
                        "status": "OK",
                    }
                    acc = metrics.get("cv_accuracy", 0)
                    progress.update(task, advance=1, description=f"[green]{model_name}: {acc:.4f}[/green]")
                except Exception as exc:
                    results[name] = {"accuracy": 0, "std": 0, "samples": 0, "detail": "", "status": f"FAIL: {exc}"}
                    progress.update(task, advance=1, description=f"[red]{name}: FAILED[/red]")

    # ═══════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════
    elapsed = time.time() - t0
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)

    table = Table(title=f"Training Complete ({mins}m {secs}s)")
    table.add_column("Model", style="bold")
    table.add_column("Accuracy", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Samples", justify="right")
    table.add_column("Distribution")
    table.add_column("Status")

    for name in ["Trend", "Regime", "Reversal", "Range"]:
        r = results.get(name, {"accuracy": 0, "std": 0, "samples": 0, "detail": "", "status": "SKIP"})
        style = "green" if r["status"] == "OK" else "red"
        table.add_row(
            name,
            f"{r['accuracy']:.4f}" if r["accuracy"] > 0 else "—",
            f"±{r['std']:.4f}" if r["std"] > 0 else "—",
            f"{r['samples']:,}" if r["samples"] > 0 else "—",
            r.get("detail", ""),
            f"[{style}]{r['status']}[/{style}]",
        )

    console.print()
    console.print(table)

    ok_count = sum(1 for r in results.values() if r["status"] == "OK")
    console.print(f"\n[bold]{ok_count}/4 models trained successfully[/bold]")

    if ok_count == 4:
        console.print("[green]All models ready! Start bot: systemctl start aibot[/green]")
    else:
        console.print("[yellow]Some models failed — check logs above[/yellow]")

    return results


if __name__ == "__main__":
    train_all()
