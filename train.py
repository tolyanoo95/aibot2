#!/usr/bin/env python3
"""
Training pipeline for the ML signal model.
"""

import logging
import sys
import time
import threading

import pandas as pd
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich import box

from src.config import config
from src.data_fetcher import BinanceDataFetcher
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


def _htf_candles(n: int, tf: str) -> int:
    return max(200, n // _HTF_RATIO.get(tf, 1))


def train_model():
    console.print(Panel(
        "[bold cyan]ML Training Pipeline[/bold cyan]\n"
        f"Pairs: {len(config.TRADING_PAIRS)} | "
        f"Candles: {config.TRAIN_CANDLES} | "
        f"Ensemble (XGBoost + LightGBM + CatBoost)",
        box=box.DOUBLE,
    ))

    fetcher = BinanceDataFetcher(config)
    indicators = TechnicalIndicators()
    feature_eng = FeatureEngineer()
    mkt_ctx = MarketContext(config)

    # shared state for live display
    pair_status: dict[str, str] = {s: "Waiting …" for s in config.TRADING_PAIRS}
    pair_results: dict[str, dict] = {}
    lock = threading.Lock()

    all_X: list[pd.DataFrame] = []
    all_y: list[pd.Series] = []
    feat_names_ref = [None]

    def _build_status_table() -> Table:
        t = Table(title="Step 1/3 — Fetching & Processing (parallel)", box=box.SIMPLE, expand=True)
        t.add_column("Pair", style="cyan", width=12)
        t.add_column("Status", width=40)
        t.add_column("Samples", justify="right", width=10)
        for sym in config.TRADING_PAIRS:
            status = pair_status.get(sym, "")
            res = pair_results.get(sym, {})
            samples = str(res.get("samples", "")) if res else ""
            t.add_row(sym, status, samples)
        return t

    def _process_pair(symbol: str):
        try:
            with lock:
                pair_status[symbol] = "[yellow]Fetching 5m …[/yellow]"
            df_5m = fetcher.fetch_ohlcv_extended(
                symbol, config.PRIMARY_TIMEFRAME, total_candles=config.TRAIN_CANDLES,
            )
            if df_5m.empty or len(df_5m) < 200:
                with lock:
                    pair_status[symbol] = "[red]No data[/red]"
                return

            with lock:
                pair_status[symbol] = f"[yellow]Fetching 15m + 1h …[/yellow]"
            n_15m = _htf_candles(len(df_5m), config.SECONDARY_TIMEFRAME)
            df_15m = fetcher.fetch_ohlcv_extended(symbol, config.SECONDARY_TIMEFRAME, total_candles=n_15m)
            n_1h = _htf_candles(len(df_5m), config.TREND_TIMEFRAME)
            df_1h = fetcher.fetch_ohlcv_extended(symbol, config.TREND_TIMEFRAME, total_candles=n_1h)

            with lock:
                pair_status[symbol] = "[yellow]Fetching OI + L/S …[/yellow]"
            ctx_df = mkt_ctx.fetch_training_context(symbol)

            with lock:
                pair_status[symbol] = "[yellow]Computing indicators …[/yellow]"
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

            with lock:
                pair_status[symbol] = "[yellow]Building features …[/yellow]"
            X, fn = feature_eng.get_feature_matrix_mtf(df_5m, df_15m, df_1h)
            y = feature_eng.create_labels(
                df_5m,
                tp_multiplier=config.LABEL_TP_MULTIPLIER,
                sl_multiplier=config.LABEL_SL_MULTIPLIER,
                max_bars=config.LABEL_MAX_BARS,
            )
            common = X.index.intersection(y.index)
            X = X.loc[common].iloc[: -config.LABEL_MAX_BARS]
            y = y.loc[common].iloc[: -config.LABEL_MAX_BARS]

            n_buy = int((y == 1).sum())
            n_sell = int((y == -1).sum())
            n_hold = int((y == 0).sum())

            with lock:
                all_X.append(X)
                all_y.append(y)
                feat_names_ref[0] = fn
                pair_results[symbol] = {
                    "samples": len(X),
                    "buy": n_buy, "sell": n_sell, "hold": n_hold,
                    "candles_5m": len(df_5m),
                }
                pair_status[symbol] = (
                    f"[green]Done[/green] — {len(X)} samples "
                    f"(B:{n_buy} S:{n_sell} H:{n_hold})"
                )
        except Exception as exc:
            with lock:
                pair_status[symbol] = f"[red]Error: {exc}[/red]"

    # ── run all pairs in parallel with live display ───────────
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with Live(_build_status_table(), console=console, refresh_per_second=4) as live:
        with ThreadPoolExecutor(max_workers=len(config.TRADING_PAIRS)) as pool:
            futures = [pool.submit(_process_pair, sym) for sym in config.TRADING_PAIRS]
            while not all(f.done() for f in futures):
                live.update(_build_status_table())
                time.sleep(0.25)
            live.update(_build_status_table())

    if not all_X:
        console.print("[bold red]No training data — aborting.[/bold red]")
        return

    X_all = pd.concat(all_X, ignore_index=True)
    y_all = pd.concat(all_y, ignore_index=True)
    feat_names = feat_names_ref[0]

    # ── Step 2: Summary ──────────────────────────────────────
    console.print(Panel(
        f"[bold]Step 2/3 — Dataset Summary[/bold]\n\n"
        f"Total samples: [cyan]{len(X_all):,}[/cyan]  |  "
        f"Features: [cyan]{len(X_all.columns)}[/cyan]\n"
        f"BUY:  [green]{int((y_all == 1).sum()):>6,}[/green]  "
        f"({(y_all == 1).mean() * 100:.1f}%)  |  "
        f"SELL: [red]{int((y_all == -1).sum()):>6,}[/red]  "
        f"({(y_all == -1).mean() * 100:.1f}%)  |  "
        f"HOLD: [yellow]{int((y_all == 0).sum()):>6,}[/yellow]  "
        f"({(y_all == 0).mean() * 100:.1f}%)",
        border_style="dim",
    ))

    # ── Step 3: Train ensemble ───────────────────────────────
    sw = feature_eng.compute_sample_weights(y_all)

    train_status = {"text": "Training XGBoost …"}

    def _train_table() -> Panel:
        return Panel(f"[bold]Step 3/3[/bold] — {train_status['text']}", border_style="yellow")

    with Live(_train_table(), console=console, refresh_per_second=2) as live:
        model = MLSignalModel(config.ML_MODEL_PATH)
        metrics = model.train(X_all, y_all, feat_names, sample_weights=sw)
        train_status["text"] = "[green]Training complete![/green]"
        live.update(_train_table())

    # ── Results ──────────────────────────────────────────────
    per_model = metrics.get("per_model", {})

    results_table = Table(title="Training Results", box=box.ROUNDED)
    results_table.add_column("Model", style="cyan", width=15)
    results_table.add_column("CV Accuracy", justify="center", width=14)
    results_table.add_column("Std", justify="center", width=10)

    for name, m in per_model.items():
        results_table.add_row(name, f"{m['cv_accuracy']:.4f}", f"±{m['cv_std']:.4f}")
    results_table.add_row(
        "[bold]Ensemble[/bold]",
        f"[bold]{metrics['cv_accuracy']:.4f}[/bold]",
        f"[bold]±{metrics['cv_std']:.4f}[/bold]",
    )
    console.print(results_table)

    console.print("\n[bold]Top-10 Feature Importances:[/bold]")
    max_imp = max(model.get_feature_importance().values()) if model.get_feature_importance() else 1
    for rank, (feat, imp) in enumerate(list(model.get_feature_importance().items())[:10], 1):
        bar_len = int(imp / max_imp * 20)
        console.print(f"  {rank:>2}. {feat:<28s} {'█' * bar_len} {imp:.1f}")

    console.print(Panel(
        f"Model saved → [cyan]{config.ML_MODEL_PATH}[/cyan]",
        border_style="green",
    ))
    return metrics


def train_model_and_return_metrics() -> dict:
    return train_model()


if __name__ == "__main__":
    train_model()
