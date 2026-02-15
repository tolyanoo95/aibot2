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

    for symbol in config.TRADING_PAIRS:
        Display.show_info(f"Fetching {symbol} …")
        try:
            # ── primary (5m) ─────────────────────────────────
            df_5m = fetcher.fetch_ohlcv_extended(
                symbol, config.PRIMARY_TIMEFRAME,
                total_candles=config.TRAIN_CANDLES,
            )
            if df_5m.empty or len(df_5m) < 200:
                Display.show_error(f"  Not enough 5m data — skipping")
                continue
            Display.show_info(f"  5m : {len(df_5m)} candles")

            # ── secondary (15m) ──────────────────────────────
            n_15m = _htf_candles(len(df_5m), config.SECONDARY_TIMEFRAME)
            df_15m = fetcher.fetch_ohlcv_extended(
                symbol, config.SECONDARY_TIMEFRAME, total_candles=n_15m,
            )
            Display.show_info(f"  15m: {len(df_15m)} candles")

            # ── trend (1h) ───────────────────────────────────
            n_1h = _htf_candles(len(df_5m), config.TREND_TIMEFRAME)
            df_1h = fetcher.fetch_ohlcv_extended(
                symbol, config.TREND_TIMEFRAME, total_candles=n_1h,
            )
            Display.show_info(f"  1h : {len(df_1h)} candles")

            # ── market context (OI + L/S at 1h) ─────────────
            Display.show_info("  Fetching OI + L/S history …")
            ctx_df = mkt_ctx.fetch_training_context(symbol)
            if not ctx_df.empty:
                Display.show_info(f"  Context: {len(ctx_df)} rows (1h)")
            else:
                Display.show_info("  Context: no data (will use zeros)")
            time.sleep(0.5)  # gentle on rate limits

            # ── indicators ───────────────────────────────────
            df_5m = indicators.calculate_all(df_5m)
            df_15m = indicators.calculate_all(df_15m) if not df_15m.empty else df_15m
            df_1h = indicators.calculate_all(df_1h) if not df_1h.empty else df_1h

            # ── merge context into primary ───────────────────
            if not ctx_df.empty:
                df_5m = feature_eng.add_context_features(df_5m, ctx_df)

            # Add columns that realtime scanner provides so the
            # model is trained with the full feature set:
            #   funding_rate, liq_pressure_enc,
            #   dist_to_short_liq_pct, dist_to_long_liq_pct
            # During training we set them to 0 (neutral) because
            # historical per-bar values are not available cheaply.
            for col in ("funding_rate", "liq_pressure_enc",
                        "dist_to_short_liq_pct", "dist_to_long_liq_pct",
                        "bid_ask_imbalance"):
                if col not in df_5m.columns:
                    df_5m[col] = 0.0

            # ── multi-TF features ────────────────────────────
            X, feat_names = feature_eng.get_feature_matrix_mtf(
                df_5m, df_15m, df_1h,
            )

            # ── labels ───────────────────────────────────────
            y = feature_eng.create_labels(
                df_5m,
                tp_multiplier=config.LABEL_TP_MULTIPLIER,
                sl_multiplier=config.LABEL_SL_MULTIPLIER,
                max_bars=config.LABEL_MAX_BARS,
            )

            # align & trim tail
            common = X.index.intersection(y.index)
            X = X.loc[common].iloc[: -config.LABEL_MAX_BARS]
            y = y.loc[common].iloc[: -config.LABEL_MAX_BARS]

            all_X.append(X)
            all_y.append(y)

            n_buy = int((y == 1).sum())
            n_sell = int((y == -1).sum())
            n_hold = int((y == 0).sum())
            Display.show_info(
                f"  Samples: {len(X)} | BUY {n_buy} | SELL {n_sell} | HOLD {n_hold}"
            )

        except Exception as exc:
            Display.show_error(f"  {symbol}: {exc}")

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
    Display.show_info("Training XGBoost …")
    model = MLSignalModel(config.ML_MODEL_PATH)
    metrics = model.train(X_all, y_all, feat_names, sample_weights=sw)

    Display.show_info(f"\nCV Accuracy : {metrics['cv_accuracy']:.4f} "
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
