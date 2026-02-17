#!/usr/bin/env python3
"""
Training pipeline for the ML signal model.
"""

import logging
import sys
import time

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from rich import box

from src.config import config
from src.data_fetcher import BinanceDataFetcher
from src.display import Display
from src.features import FeatureEngineer
from src.indicators import TechnicalIndicators
from src.market_context import MarketContext
from src.ml_model import MLSignalModel

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

console = Console()

_HTF_RATIO = {"15m": 3, "1h": 12}


def _htf_candles(primary_count: int, tf: str) -> int:
    return max(200, primary_count // _HTF_RATIO.get(tf, 1))


def train_model():
    console.print(Panel(
        "[bold cyan]ML Training Pipeline[/bold cyan]\n"
        f"Pairs: {len(config.TRADING_PAIRS)} | "
        f"Candles: {config.TRAIN_CANDLES} | "
        f"Model: Ensemble (XGBoost + LightGBM + CatBoost)",
        box=box.DOUBLE,
    ))

    fetcher = BinanceDataFetcher(config)
    indicators = TechnicalIndicators()
    feature_eng = FeatureEngineer()
    mkt_ctx = MarketContext(config)

    all_X: list[pd.DataFrame] = []
    all_y: list[pd.Series] = []
    feat_names = None

    # ── Step 1: Fetch & process data ─────────────────────────
    pair_table = Table(title="Step 1/3 — Data Collection", box=box.SIMPLE)
    pair_table.add_column("Pair", style="cyan", width=12)
    pair_table.add_column("5m", justify="right", width=8)
    pair_table.add_column("15m", justify="right", width=8)
    pair_table.add_column("1h", justify="right", width=8)
    pair_table.add_column("Context", justify="right", width=10)
    pair_table.add_column("Samples", justify="right", width=10)
    pair_table.add_column("BUY", justify="right", width=8, style="green")
    pair_table.add_column("SELL", justify="right", width=8, style="red")
    pair_table.add_column("HOLD", justify="right", width=8, style="yellow")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching pairs …", total=len(config.TRADING_PAIRS))

        for symbol in config.TRADING_PAIRS:
            progress.update(task, description=f"Fetching {symbol} …")
            try:
                df_5m = fetcher.fetch_ohlcv_extended(
                    symbol, config.PRIMARY_TIMEFRAME,
                    total_candles=config.TRAIN_CANDLES,
                )
                if df_5m.empty or len(df_5m) < 200:
                    pair_table.add_row(symbol, "[red]—[/red]", "—", "—", "—", "—", "—", "—", "—")
                    progress.advance(task)
                    continue

                n_15m = _htf_candles(len(df_5m), config.SECONDARY_TIMEFRAME)
                df_15m = fetcher.fetch_ohlcv_extended(
                    symbol, config.SECONDARY_TIMEFRAME, total_candles=n_15m,
                )
                n_1h = _htf_candles(len(df_5m), config.TREND_TIMEFRAME)
                df_1h = fetcher.fetch_ohlcv_extended(
                    symbol, config.TREND_TIMEFRAME, total_candles=n_1h,
                )
                ctx_df = mkt_ctx.fetch_training_context(symbol)

                df_5m = indicators.calculate_all(df_5m)
                df_15m = indicators.calculate_all(df_15m) if not df_15m.empty else df_15m
                df_1h = indicators.calculate_all(df_1h) if not df_1h.empty else df_1h

                if not ctx_df.empty:
                    df_5m = feature_eng.add_context_features(df_5m, ctx_df)

                for col in ("funding_rate", "liq_pressure_enc",
                            "dist_to_short_liq_pct", "dist_to_long_liq_pct",
                            "bid_ask_imbalance"):
                    if col not in df_5m.columns:
                        df_5m[col] = 0.0

                X, fn = feature_eng.get_feature_matrix_mtf(df_5m, df_15m, df_1h)
                feat_names = fn

                y = feature_eng.create_labels(
                    df_5m,
                    tp_multiplier=config.LABEL_TP_MULTIPLIER,
                    sl_multiplier=config.LABEL_SL_MULTIPLIER,
                    max_bars=config.LABEL_MAX_BARS,
                )

                common = X.index.intersection(y.index)
                X = X.loc[common].iloc[: -config.LABEL_MAX_BARS]
                y = y.loc[common].iloc[: -config.LABEL_MAX_BARS]

                all_X.append(X)
                all_y.append(y)

                n_buy = int((y == 1).sum())
                n_sell = int((y == -1).sum())
                n_hold = int((y == 0).sum())

                pair_table.add_row(
                    symbol,
                    str(len(df_5m)), str(len(df_15m)), str(len(df_1h)),
                    str(len(ctx_df)) if not ctx_df.empty else "0",
                    str(len(X)),
                    str(n_buy), str(n_sell), str(n_hold),
                )
            except Exception as exc:
                pair_table.add_row(symbol, f"[red]ERR[/red]", "—", "—", "—", "—", "—", "—", "—")

            progress.advance(task)

    console.print(pair_table)

    if not all_X:
        console.print("[bold red]No training data collected — aborting.[/bold red]")
        return

    X_all = pd.concat(all_X, ignore_index=True)
    y_all = pd.concat(all_y, ignore_index=True)

    # ── Step 2: Summary ──────────────────────────────────────
    console.print(Panel(
        f"[bold]Step 2/3 — Dataset Summary[/bold]\n\n"
        f"Total samples: [cyan]{len(X_all):,}[/cyan]\n"
        f"Features: [cyan]{len(X_all.columns)}[/cyan]\n"
        f"BUY:  [green]{int((y_all == 1).sum()):>6,}[/green]  "
        f"({(y_all == 1).mean() * 100:.1f}%)\n"
        f"SELL: [red]{int((y_all == -1).sum()):>6,}[/red]  "
        f"({(y_all == -1).mean() * 100:.1f}%)\n"
        f"HOLD: [yellow]{int((y_all == 0).sum()):>6,}[/yellow]  "
        f"({(y_all == 0).mean() * 100:.1f}%)",
        border_style="dim",
    ))

    # ── Step 3: Train ensemble ───────────────────────────────
    sw = feature_eng.compute_sample_weights(y_all)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Step 3/3 — Training Ensemble …", total=None)
        model = MLSignalModel(config.ML_MODEL_PATH)
        metrics = model.train(X_all, y_all, feat_names, sample_weights=sw)
        progress.update(task, description="Step 3/3 — Training complete!")

    # ── Results ──────────────────────────────────────────────
    per_model = metrics.get("per_model", {})

    results_table = Table(title="Training Results", box=box.ROUNDED)
    results_table.add_column("Model", style="cyan", width=15)
    results_table.add_column("CV Accuracy", justify="center", width=14)
    results_table.add_column("Std", justify="center", width=10)

    for name, m in per_model.items():
        results_table.add_row(
            name,
            f"{m['cv_accuracy']:.4f}",
            f"±{m['cv_std']:.4f}",
        )
    results_table.add_row(
        "[bold]Ensemble[/bold]",
        f"[bold]{metrics['cv_accuracy']:.4f}[/bold]",
        f"[bold]±{metrics['cv_std']:.4f}[/bold]",
    )

    console.print(results_table)

    # feature importance
    console.print("\n[bold]Top-10 Feature Importances:[/bold]")
    for rank, (feat, imp) in enumerate(
        list(model.get_feature_importance().items())[:10], 1
    ):
        bar = "█" * int(imp / 100)
        console.print(f"  {rank:>2}. {feat:<28s} {bar} {imp:.1f}")

    console.print(Panel(
        f"Model saved → [cyan]{config.ML_MODEL_PATH}[/cyan]",
        border_style="green",
    ))

    return metrics


def train_model_and_return_metrics() -> dict:
    return train_model()


if __name__ == "__main__":
    train_model()
