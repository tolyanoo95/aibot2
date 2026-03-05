#!/usr/bin/env python3
"""
Training pipeline for the Trend Health model.
Binary: HEALTHY (trend continues) vs ENDING (trend about to reverse).
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
from src.trend_health_model import TrendHealthModel

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

console = Console()

_HTF_RATIO = {"15m": 3, "1h": 12}


def _htf_candles(n: int, tf: str) -> int:
    return max(200, n // _HTF_RATIO.get(tf, 1))


def train_health():
    console.print(Panel(
        "[bold cyan]Trend Health Training Pipeline[/bold cyan]\n"
        f"Pairs: {len(config.TRADING_PAIRS)} | "
        f"Candles: {config.TREND_TRAIN_CANDLES} (~{config.TREND_TRAIN_CANDLES // 288} days) | "
        f"Binary (HEALTHY / ENDING)",
        box=box.DOUBLE,
    ))

    fetcher = BinanceDataFetcher(config)
    indicators = TechnicalIndicators()
    feature_eng = FeatureEngineer()
    mkt_ctx = MarketContext(config)

    pair_status: dict[str, str] = {s: "Waiting ..." for s in config.TRADING_PAIRS}
    pair_results: dict[str, dict] = {}
    lock = threading.Lock()

    all_X: list[pd.DataFrame] = []
    all_y: list[pd.Series] = []
    feat_names_ref = [None]

    def _build_status_table() -> Table:
        t = Table(title="Step 1/3 — Fetching & Processing", box=box.SIMPLE, expand=True)
        t.add_column("Pair", style="cyan", width=12)
        t.add_column("Status", width=50)
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
                pair_status[symbol] = "[yellow]Fetching 5m ...[/yellow]"
            df_5m = fetcher.fetch_ohlcv_extended(
                symbol, config.PRIMARY_TIMEFRAME, total_candles=config.TREND_TRAIN_CANDLES,
            )
            if df_5m.empty or len(df_5m) < 200:
                with lock:
                    pair_status[symbol] = "[red]No data[/red]"
                return

            with lock:
                pair_status[symbol] = "[yellow]Fetching 15m + 1h ...[/yellow]"
            n_15m = _htf_candles(len(df_5m), config.SECONDARY_TIMEFRAME)
            df_15m = fetcher.fetch_ohlcv_extended(symbol, config.SECONDARY_TIMEFRAME, total_candles=n_15m)
            n_1h = _htf_candles(len(df_5m), config.TREND_TIMEFRAME)
            df_1h = fetcher.fetch_ohlcv_extended(symbol, config.TREND_TIMEFRAME, total_candles=n_1h)

            with lock:
                pair_status[symbol] = "[yellow]Fetching OI + L/S ...[/yellow]"
            ctx_df = mkt_ctx.fetch_training_context(symbol)

            with lock:
                pair_status[symbol] = "[yellow]Computing indicators ...[/yellow]"
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
                pair_status[symbol] = "[yellow]Building features ...[/yellow]"

            # Use ALL features (trend + reversal + range) for max context
            feat = feature_eng.create_features(df_5m)
            if df_15m is not None and not df_15m.empty:
                feat = feature_eng.add_htf_features(feat, df_15m, prefix="htf_15m")
            if df_1h is not None and not df_1h.empty:
                feat = feature_eng.add_htf_features(feat, df_1h, prefix="htf_1h")
            feat = feature_eng.add_rolling_htf_features(feat)

            fn = list(dict.fromkeys(feature_eng.get_feature_columns(feat, model_type="all")))
            X = feat[fn].replace([float("inf"), float("-inf")], float("nan")).ffill().fillna(0)
            X = X.loc[:, ~X.columns.duplicated()]

            y = feature_eng.create_health_labels(
                df_5m,
                max_bars=config.LABEL_MAX_BARS,
                reversal_pct=1.0,
            )

            common = X.index.intersection(y.index)
            X = X.loc[common].iloc[:-config.LABEL_MAX_BARS]
            y = y.loc[common].iloc[:-config.LABEL_MAX_BARS]

            n_healthy = int((y == 1).sum())
            n_ending = int((y == 0).sum())

            with lock:
                all_X.append(X)
                all_y.append(y)
                feat_names_ref[0] = fn
                pair_results[symbol] = {
                    "samples": len(X),
                    "healthy": n_healthy, "ending": n_ending,
                }
                pair_status[symbol] = (
                    f"[green]Done[/green] -- {len(X)} samples "
                    f"(H:{n_healthy} E:{n_ending})"
                )
        except Exception as exc:
            with lock:
                pair_status[symbol] = f"[red]Error: {exc}[/red]"

    from concurrent.futures import ThreadPoolExecutor

    with Live(_build_status_table(), console=console, refresh_per_second=4) as live:
        with ThreadPoolExecutor(max_workers=len(config.TRADING_PAIRS)) as pool:
            futures = [pool.submit(_process_pair, sym) for sym in config.TRADING_PAIRS]
            while not all(f.done() for f in futures):
                live.update(_build_status_table())
                time.sleep(0.25)
            live.update(_build_status_table())

    if not all_X:
        console.print("[bold red]No training data -- aborting.[/bold red]")
        return

    X_all = pd.concat(all_X, ignore_index=True)
    y_all = pd.concat(all_y, ignore_index=True)
    feat_names = feat_names_ref[0]

    n_healthy = int((y_all == 1).sum())
    n_ending = int((y_all == 0).sum())
    console.print(Panel(
        f"[bold]Step 2/3 -- Dataset Summary[/bold]\n\n"
        f"Total samples: [cyan]{len(X_all):,}[/cyan]  |  "
        f"Features: [cyan]{len(X_all.columns)}[/cyan]  |  "
        f"Mode: [cyan]Binary (HEALTHY/ENDING)[/cyan]\n"
        f"HEALTHY: [green]{n_healthy:>6,}[/green]  "
        f"({n_healthy / len(y_all) * 100:.1f}%)  |  "
        f"ENDING:  [red]{n_ending:>6,}[/red]  "
        f"({n_ending / len(y_all) * 100:.1f}%)",
        border_style="dim",
    ))

    train_status = {"text": "Training XGBoost ..."}

    def _train_table() -> Panel:
        return Panel(f"[bold]Step 3/3[/bold] -- {train_status['text']}", border_style="yellow")

    with Live(_train_table(), console=console, refresh_per_second=2) as live:
        model = TrendHealthModel("models/trend_health_model.pkl")
        metrics = model.train(X_all, y_all, feat_names)
        train_status["text"] = "[green]Training complete![/green]"
        live.update(_train_table())

    console.print(Panel(
        f"[bold]Trend Health Model[/bold]\n"
        f"CV Accuracy: [cyan]{metrics['cv_accuracy']:.4f}[/cyan] +/- {metrics['cv_std']:.4f}\n"
        f"Model saved -> [cyan]models/trend_health_model.pkl[/cyan]",
        border_style="green",
    ))

    return metrics


if __name__ == "__main__":
    train_health()
