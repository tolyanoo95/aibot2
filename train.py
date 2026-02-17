#!/usr/bin/env python3
"""
Training pipeline for the ML signal model (v3).
────────────────────────────────────────────────
Now fetches multi-timeframe data + market context (OI, L/S ratio)
for richer feature engineering.

Usage:
    python train.py
"""

import logging
import sys
import time

import pandas as pd

from src.config import config
from src.data_fetcher import BinanceDataFetcher
from src.display import Display
from src.features import FeatureEngineer
from src.indicators import TechnicalIndicators
from src.market_context import MarketContext
from src.ml_model import MLSignalModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

_HTF_RATIO = {"15m": 3, "1h": 12}


def _htf_candles(primary_count: int, tf: str) -> int:
    return max(200, primary_count // _HTF_RATIO.get(tf, 1))


def train_model():
    Display.show_info("=== ML Training Pipeline (v3 — multi-TF + context) ===\n")

    fetcher = BinanceDataFetcher(config)
    indicators = TechnicalIndicators()
    feature_eng = FeatureEngineer()
    mkt_ctx = MarketContext(config)

    all_X: list[pd.DataFrame] = []
    all_y: list[pd.Series] = []

    def _process_pair(symbol: str):
        """Fetch data, compute features & labels for one pair."""
        df_5m = fetcher.fetch_ohlcv_extended(
            symbol, config.PRIMARY_TIMEFRAME,
            total_candles=config.TRAIN_CANDLES,
        )
        if df_5m.empty or len(df_5m) < 200:
            return None, None, None

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

        X, feat_names = feature_eng.get_feature_matrix_mtf(df_5m, df_15m, df_1h)

        y = feature_eng.create_labels(
            df_5m,
            tp_multiplier=config.LABEL_TP_MULTIPLIER,
            sl_multiplier=config.LABEL_SL_MULTIPLIER,
            max_bars=config.LABEL_MAX_BARS,
        )

        common = X.index.intersection(y.index)
        X = X.loc[common].iloc[: -config.LABEL_MAX_BARS]
        y = y.loc[common].iloc[: -config.LABEL_MAX_BARS]

        return X, y, feat_names

    # ── parallel data fetch + feature engineering ────────────
    from concurrent.futures import ThreadPoolExecutor, as_completed

    Display.show_info(f"Fetching {len(config.TRADING_PAIRS)} pairs in parallel …\n")
    feat_names = None

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_process_pair, sym): sym
            for sym in config.TRADING_PAIRS
        }
        for future in as_completed(futures):
            sym = futures[future]
            try:
                X, y, fn = future.result()
                if X is None:
                    Display.show_error(f"  {sym}: not enough data — skipping")
                    continue
                feat_names = fn
                all_X.append(X)
                all_y.append(y)

                n_buy = int((y == 1).sum())
                n_sell = int((y == -1).sum())
                n_hold = int((y == 0).sum())
                Display.show_info(
                    f"  {sym}: {len(X)} samples | BUY {n_buy} | SELL {n_sell} | HOLD {n_hold}"
                )
            except Exception as exc:
                Display.show_error(f"  {sym}: {exc}")

    if not all_X:
        Display.show_error("No training data collected — aborting.")
        return

    X_all = pd.concat(all_X, ignore_index=True)
    y_all = pd.concat(all_y, ignore_index=True)

    Display.show_info(f"\n{'─' * 50}")
    Display.show_info(f"Total samples : {len(X_all)}")
    Display.show_info(f"Features      : {len(X_all.columns)}")
    Display.show_info(
        f"  BUY  : {int((y_all == 1).sum()):>6}  "
        f"({(y_all == 1).mean() * 100:.1f}%)"
    )
    Display.show_info(
        f"  SELL : {int((y_all == -1).sum()):>6}  "
        f"({(y_all == -1).mean() * 100:.1f}%)"
    )
    Display.show_info(
        f"  HOLD : {int((y_all == 0).sum()):>6}  "
        f"({(y_all == 0).mean() * 100:.1f}%)"
    )

    # ── balanced sample weights ──────────────────────────────
    sw = feature_eng.compute_sample_weights(y_all)
    Display.show_info("\nClass-balanced sample weights applied.")

    # ── train ────────────────────────────────────────────────
    Display.show_info("Training Ensemble (XGBoost + LightGBM + CatBoost) …")
    model = MLSignalModel(config.ML_MODEL_PATH)
    metrics = model.train(X_all, y_all, feat_names, sample_weights=sw)

    # per-model results
    per_model = metrics.get("per_model", {})
    for name, m in per_model.items():
        Display.show_info(
            f"  {name:<12s} CV: {m['cv_accuracy']:.4f} (±{m['cv_std']:.4f})"
        )

    Display.show_info(f"\nEnsemble CV : {metrics['cv_accuracy']:.4f} "
                      f"(±{metrics['cv_std']:.4f})")
    Display.show_info(
        f"Fold scores : {[f'{s:.4f}' for s in metrics['scores']]}"
    )

    Display.show_info("\nTop-10 feature importances:")
    for rank, (feat, imp) in enumerate(
        list(model.get_feature_importance().items())[:10], 1
    ):
        Display.show_info(f"  {rank:>2}. {feat:<30s} {imp:.4f}")

    Display.show_info(f"\nModel saved → {config.ML_MODEL_PATH}")
    Display.show_info("Done.")

    return metrics


def train_model_and_return_metrics() -> dict:
    """Entry point for auto_retrain.py — returns metrics dict."""
    return train_model()


if __name__ == "__main__":
    train_model()
