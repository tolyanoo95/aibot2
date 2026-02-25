#!/usr/bin/env python3
"""
Train ALL models at once: Trend + Regime + Reversal + Range.
Fetches data ONCE, then trains all models in parallel on shared data.

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
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
console = Console()


def fetch_pair_data(pair, fetcher, indicators, features):
    """Fetch and process data for one pair (all timeframes)."""
    try:
        data = {}
        for tf, limit in [("5m", config.TRAIN_CANDLES), ("15m", config.TRAIN_CANDLES // 3), ("1h", config.TRAIN_CANDLES // 12)]:
            raw = fetcher.fetch_ohlcv(pair, tf, limit=limit)
            if raw.empty:
                return pair, None
            data[tf] = indicators.calculate_all(raw)
            time.sleep(0.2)

        X, feat_names = features.get_feature_matrix_mtf(
            data["5m"], data["15m"], data["1h"],
        )
        return pair, {"X": X, "df_5m": data["5m"], "feat_names": feat_names}
    except Exception as exc:
        logger.error("Error fetching %s: %s", pair, exc)
        return pair, None


def train_all():
    t0 = time.time()
    console.print(Panel.fit(
        f"[bold]Multi-Model Training Pipeline[/bold]\n"
        f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Models: Trend + Regime + Reversal + Range\n"
        f"Pairs: {len(config.TRADING_PAIRS)} | Candles: {config.TRAIN_CANDLES}",
        border_style="green",
    ))

    fetcher = BinanceDataFetcher(config)
    indicators = TechnicalIndicators()
    features = FeatureEngineer()

    # ═══════════════════════════════════════════════════════
    # STEP 1: Fetch data ONCE for all models
    # ═══════════════════════════════════════════════════════
    console.print("\n[bold cyan]═══ STEP 1: Fetching Data (once for all models) ═══[/bold cyan]")

    pair_data = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(fetch_pair_data, pair, fetcher, indicators, features): pair
            for pair in config.TRADING_PAIRS
        }
        for future in as_completed(futures):
            pair, data = future.result()
            if data is not None:
                pair_data[pair] = data
                console.print(f"  {pair:<12s} {len(data['X']):>6d} bars ✓")
            else:
                console.print(f"  {pair:<12s} [red]SKIP[/red]")

    if not pair_data:
        console.print("[red]No data![/red]")
        return

    console.print(f"\n  Total: {sum(len(d['X']) for d in pair_data.values()):,} bars from {len(pair_data)} pairs")

    results = {}

    # ═══════════════════════════════════════════════════════
    # STEP 2: Train all models on shared data
    # ═══════════════════════════════════════════════════════

    # ── 2a. Trend Model ────────────────────────────────────
    console.print("\n[bold cyan]═══ 2a/4 TREND MODEL ═══[/bold cyan]")
    try:
        all_X = []
        all_y = []
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
        y_mapped = y_all.map({-1: 0, 1: 1})

        console.print(f"  Samples: {len(X_all):,} | BUY: {(y_all==1).sum()} SELL: {(y_all==-1).sum()}")

        trend_model = MLSignalModel(config.ML_MODEL_PATH)
        feat_cols = features.get_feature_columns(X_all, model_type="trend")
        available = [c for c in feat_cols if c in X_all.columns]
        metrics = trend_model.train(X_all[available], y_mapped, feature_names=available)

        results["Trend"] = {"accuracy": metrics.get("cv_accuracy", 0), "std": metrics.get("cv_std", 0), "status": "OK"}
        console.print(f"  [green]Trend: {metrics.get('cv_accuracy', 0):.4f} (±{metrics.get('cv_std', 0):.4f})[/green]")
    except Exception as exc:
        results["Trend"] = {"accuracy": 0, "std": 0, "status": f"FAIL: {exc}"}
        console.print(f"  [red]FAILED: {exc}[/red]")

    # ── 2b. Regime Classifier ──────────────────────────────
    console.print("\n[bold cyan]═══ 2b/4 REGIME CLASSIFIER ═══[/bold cyan]")
    try:
        all_X = []
        all_y = []
        for pair, d in pair_data.items():
            X = d["X"]
            y = features.create_regime_labels(d["df_5m"], max_bars=config.LABEL_MAX_BARS)
            common = X.index.intersection(y.index)
            all_X.append(X.loc[common].iloc[:-config.LABEL_MAX_BARS])
            all_y.append(y.loc[common].iloc[:-config.LABEL_MAX_BARS])

        X_all = pd.concat(all_X, ignore_index=True)
        y_all = pd.concat(all_y, ignore_index=True)

        console.print(f"  Samples: {len(X_all):,} | TREND: {(y_all==1).sum()} REV: {(y_all==2).sum()} RANGE: {(y_all==0).sum()}")

        regime_clf = RegimeClassifier()
        regime_cols = [c for c in features.REGIME_FEATURES if c in X_all.columns]
        metrics = regime_clf.train(X_all, y_all, feature_names=regime_cols)

        results["Regime"] = {"accuracy": metrics.get("cv_accuracy", 0), "std": metrics.get("cv_std", 0), "status": "OK"}
        console.print(f"  [green]Regime: {metrics.get('cv_accuracy', 0):.4f} (±{metrics.get('cv_std', 0):.4f})[/green]")
    except Exception as exc:
        results["Regime"] = {"accuracy": 0, "std": 0, "status": f"FAIL: {exc}"}
        console.print(f"  [red]FAILED: {exc}[/red]")

    # ── 2c. Reversal Model ─────────────────────────────────
    console.print("\n[bold cyan]═══ 2c/4 REVERSAL MODEL ═══[/bold cyan]")
    try:
        from train_reversal import create_reversal_labels

        all_X = []
        all_y = []
        for pair, d in pair_data.items():
            X = d["X"]
            y = create_reversal_labels(d["df_5m"], max_bars=config.LABEL_MAX_BARS)
            common = X.index.intersection(y.index)
            all_X.append(X.loc[common].iloc[:-config.LABEL_MAX_BARS])
            all_y.append(y.loc[common].iloc[:-config.LABEL_MAX_BARS])

        X_all = pd.concat(all_X, ignore_index=True)
        y_all = pd.concat(all_y, ignore_index=True)

        console.print(f"  Samples: {len(X_all):,} | BUY_rev: {(y_all==1).sum()} SELL_rev: {(y_all==-1).sum()} HOLD: {(y_all==0).sum()}")

        rev_cols = features.get_feature_columns(X_all, model_type="reversal")
        available = [c for c in rev_cols if c in X_all.columns]
        y_mapped = y_all.map({-1: 0, 0: 1, 1: 2})

        rev_model = ReversalModel()
        metrics = rev_model.train(X_all, y_mapped, feature_names=available)

        results["Reversal"] = {"accuracy": metrics.get("cv_accuracy", 0), "std": metrics.get("cv_std", 0), "status": "OK"}
        console.print(f"  [green]Reversal: {metrics.get('cv_accuracy', 0):.4f} (±{metrics.get('cv_std', 0):.4f})[/green]")
    except Exception as exc:
        results["Reversal"] = {"accuracy": 0, "std": 0, "status": f"FAIL: {exc}"}
        console.print(f"  [red]FAILED: {exc}[/red]")

    # ── 2d. Range Model ────────────────────────────────────
    console.print("\n[bold cyan]═══ 2d/4 RANGE MODEL ═══[/bold cyan]")
    try:
        from train_range import create_range_labels

        all_X = []
        all_y = []
        for pair, d in pair_data.items():
            X = d["X"]
            y = create_range_labels(d["df_5m"], max_bars=6)
            common = X.index.intersection(y.index)
            all_X.append(X.loc[common].iloc[:-6])
            all_y.append(y.loc[common].iloc[:-6])

        X_all = pd.concat(all_X, ignore_index=True)
        y_all = pd.concat(all_y, ignore_index=True)

        # Downsample HOLD
        buy_sell = y_all != 0
        hold_sample = y_all[y_all == 0].sample(frac=0.3, random_state=42)
        keep_idx = buy_sell | y_all.index.isin(hold_sample.index)
        X_filt = X_all[keep_idx]
        y_filt = y_all[keep_idx]

        console.print(f"  Samples: {len(X_filt):,} | BUY: {(y_filt==1).sum()} SELL: {(y_filt==-1).sum()} HOLD: {(y_filt==0).sum()}")

        rng_cols = features.get_feature_columns(X_filt, model_type="range")
        available = [c for c in rng_cols if c in X_filt.columns]
        y_mapped = y_filt.map({-1: 0, 0: 1, 1: 2})

        rng_model = RangeModel()
        metrics = rng_model.train(X_filt, y_mapped, feature_names=available)

        results["Range"] = {"accuracy": metrics.get("cv_accuracy", 0), "std": metrics.get("cv_std", 0), "status": "OK"}
        console.print(f"  [green]Range: {metrics.get('cv_accuracy', 0):.4f} (±{metrics.get('cv_std', 0):.4f})[/green]")
    except Exception as exc:
        results["Range"] = {"accuracy": 0, "std": 0, "status": f"FAIL: {exc}"}
        console.print(f"  [red]FAILED: {exc}[/red]")

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
    table.add_column("Status")

    for name, r in results.items():
        status_style = "green" if r["status"] == "OK" else "red"
        table.add_row(
            name,
            f"{r['accuracy']:.4f}" if r["accuracy"] > 0 else "—",
            f"±{r['std']:.4f}" if r["std"] > 0 else "—",
            f"[{status_style}]{r['status']}[/{status_style}]",
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
