"""
Feature engineering and triple-barrier labeling for the ML pipeline.
Supports multi-timeframe feature alignment.
"""

from typing import Tuple, List

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Build the feature matrix consumed by the XGBoost model."""

    # Base features from the primary (5m) timeframe
    BASE_FEATURES: List[str] = [
        # momentum
        "rsi",
        "MACD_12_26_9",
        "MACDh_12_26_9",
        "STOCHRSIk_14_14_3_3",
        "STOCHRSId_14_14_3_3",
        # volatility
        "BBP_20_2.0",
        "atr",
        # trend strength
        "ADX_14",
        "DMP_14",
        "DMN_14",
        # volume
        "volume_ratio",
        # price changes
        "pct_change_1",
        "pct_change_5",
        "pct_change_15",
        # candle structure
        "body_size",
        "upper_wick",
        "lower_wick",
        # derived
        "ema_cross_9_21",
        "ema_cross_21_50",
        "price_vs_ema50",
        "rsi_divergence",
    ]

    # Higher-TF features (added with a prefix like "htf_15m_", "htf_1h_")
    HTF_FEATURES: List[str] = [
        "rsi",
        "MACDh_12_26_9",
        "ADX_14",
        "ema_cross_9_21",
        "volume_ratio",
        "BBP_20_2.0",
    ]

    # ── public API ───────────────────────────────────────────

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived feature columns to an indicator-enriched DataFrame."""
        feat = df.copy()

        # EMA crossover distances (normalised by price)
        feat["ema_cross_9_21"] = (feat["ema_9"] - feat["ema_21"]) / feat["close"] * 100
        feat["ema_cross_21_50"] = (feat["ema_21"] - feat["ema_50"]) / feat["close"] * 100

        # Price position relative to EMA-50
        feat["price_vs_ema50"] = (feat["close"] - feat["ema_50"]) / feat["close"] * 100

        # Simplified RSI divergence
        feat["rsi_divergence"] = feat["rsi"].diff(5) - feat["pct_change_5"] * 100

        # Ensure BB %B exists
        if "BBP_20_2.0" not in feat.columns:
            if "BBU_20_2.0" in feat.columns and "BBL_20_2.0" in feat.columns:
                bw = feat["BBU_20_2.0"] - feat["BBL_20_2.0"]
                feat["BBP_20_2.0"] = np.where(
                    bw != 0,
                    (feat["close"] - feat["BBL_20_2.0"]) / bw,
                    0.5,
                )

        return feat

    def add_htf_features(
        self,
        primary_df: pd.DataFrame,
        htf_df: pd.DataFrame,
        prefix: str,
    ) -> pd.DataFrame:
        """
        Merge higher-timeframe indicators into the primary DataFrame.

        Uses forward-fill alignment: for each 5m bar, the most recent
        completed HTF bar's indicators are used.
        """
        htf_feat = self.create_features(htf_df)

        # pick available HTF columns
        available = [c for c in self.HTF_FEATURES if c in htf_feat.columns]
        htf_subset = htf_feat[available].rename(
            columns={c: f"{prefix}_{c}" for c in available},
        )

        # reindex to primary timestamps with forward-fill
        merged = primary_df.join(htf_subset, how="left")
        for col in htf_subset.columns:
            merged[col] = merged[col].ffill()

        return merged

    # Market context features (from OI, L/S, liq zones, orderbook)
    CONTEXT_FEATURES: List[str] = [
        "oi_change_pct",
        "long_short_ratio",
        "funding_rate",
        "liq_pressure_enc",        # -1 / 0 / +1
        "dist_to_short_liq_pct",
        "dist_to_long_liq_pct",
        "bid_ask_imbalance",       # -1.0 to +1.0
    ]

    # ── context merging ──────────────────────────────────────

    def add_context_features(
        self,
        df: pd.DataFrame,
        context_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge slow-changing context (OI, L/S) into the primary 5 m DataFrame.
        context_df is indexed by timestamp at 1 h resolution; forward-filled.
        """
        if context_df.empty:
            return df
        merged = df.join(context_df, how="left")
        for col in context_df.columns:
            merged[col] = merged[col].ffill().fillna(0)
        return merged

    @staticmethod
    def add_realtime_context(
        df: pd.DataFrame,
        context: dict,
    ) -> pd.DataFrame:
        """
        Stamp live context values onto the last row(s) of the DataFrame.
        Used during real-time scanning (not training).
        """
        df = df.copy()
        mapping = {
            "long_short_ratio": context.get("long_short_ratio", 1.0),
            "oi_change_pct": 0.0,           # can't compute diff from single value
            "funding_rate": context.get("funding_rate", 0.0),
            "liq_pressure_enc": {
                "SHORT_SQUEEZE": 1, "LONG_CASCADE": -1, "NEUTRAL": 0,
            }.get(context.get("liq_pressure", "NEUTRAL"), 0),
            "dist_to_short_liq_pct": context.get("dist_to_short_liq_pct", 0),
            "dist_to_long_liq_pct": context.get("dist_to_long_liq_pct", 0),
            "bid_ask_imbalance": context.get("bid_ask_imbalance", 0),
        }
        for col, val in mapping.items():
            df[col] = val
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Return ordered list of all feature columns present in df."""
        # base features
        cols = [c for c in self.BASE_FEATURES if c in df.columns]
        # HTF features (any column starting with "htf_")
        htf_cols = sorted([c for c in df.columns if c.startswith("htf_")])
        # context features
        ctx_cols = [c for c in self.CONTEXT_FEATURES if c in df.columns]
        return cols + htf_cols + ctx_cols

    def get_feature_matrix(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Return a clean (X, column_names) tuple ready for the model."""
        features = self.create_features(df)
        available = self.get_feature_columns(features)
        X = features[available].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.ffill().fillna(0)
        return X, available

    def get_feature_matrix_mtf(
        self,
        primary_df: pd.DataFrame,
        secondary_df: pd.DataFrame = None,
        trend_df: pd.DataFrame = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Build a feature matrix with multi-timeframe inputs.

        primary_df  : 5m  candles with indicators
        secondary_df: 15m candles with indicators (optional)
        trend_df    : 1h  candles with indicators (optional)
        """
        feat = self.create_features(primary_df)

        if secondary_df is not None and not secondary_df.empty:
            feat = self.add_htf_features(feat, secondary_df, prefix="htf_15m")

        if trend_df is not None and not trend_df.empty:
            feat = self.add_htf_features(feat, trend_df, prefix="htf_1h")

        available = self.get_feature_columns(feat)
        X = feat[available].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.ffill().fillna(0)
        return X, available

    # ── labeling ─────────────────────────────────────────────

    @staticmethod
    def create_labels(
        df: pd.DataFrame,
        tp_multiplier: float = 1.5,
        sl_multiplier: float = 1.5,
        max_bars: int = 12,
    ) -> pd.Series:
        """
        Triple-barrier labeling for intraday trading.

        For each bar *i* we look forward up to *max_bars* candles:
          - If price hits  entry + ATR * tp_multiplier  first  →  label =  1 (BUY)
          - If price hits  entry − ATR * sl_multiplier  first  →  label = −1 (SELL)
          - If neither within the window                        →  label =  0 (HOLD)
        """
        labels = pd.Series(0, index=df.index, dtype=int)
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        atr = df["atr"].values

        n = len(df)
        for i in range(n - max_bars):
            a = atr[i]
            if np.isnan(a) or a == 0:
                continue

            tp_up = close[i] + a * tp_multiplier
            tp_down = close[i] - a * sl_multiplier

            for j in range(1, max_bars + 1):
                idx = i + j
                if idx >= n:
                    break
                hit_up = high[idx] >= tp_up
                hit_down = low[idx] <= tp_down
                if hit_up and hit_down:
                    # both hit in same candle — use close direction
                    if close[idx] > close[i]:
                        labels.iloc[i] = 1
                    else:
                        labels.iloc[i] = -1
                    break
                if hit_up:
                    labels.iloc[i] = 1
                    break
                if hit_down:
                    labels.iloc[i] = -1
                    break

        return labels

    # ── class balancing ──────────────────────────────────────

    @staticmethod
    def compute_sample_weights(y: pd.Series) -> np.ndarray:
        """
        Compute per-sample weights inversely proportional to class frequency.
        This balances the contribution of BUY / SELL / HOLD during training.
        """
        counts = y.value_counts()
        total = len(y)
        n_classes = len(counts)
        class_weight = {cls: total / (n_classes * cnt) for cls, cnt in counts.items()}
        return np.array([class_weight[v] for v in y])
