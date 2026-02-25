"""
Feature engineering and triple-barrier labeling for the ML pipeline.
Supports multi-timeframe feature alignment and rolling HTF features.
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
        # rate of change (longer windows)
        "roc_12",
        "roc_48",
        # candle structure
        "body_size",
        "upper_wick",
        "lower_wick",
        # derived
        "ema_cross_9_21",
        "ema_cross_21_50",
        "price_vs_ema50",
        "price_vs_ema100",
        "price_vs_ema200",
        "rsi_divergence",
        # acceleration / slope
        "rsi_slope",
        "adx_slope",
        # volume
        "volume_spike",
        "atr_pct",
        # regime detection
        "regime_adx",
        "regime_volatility",
        "regime_bb_width",
        "regime_range_score",
    ]

    # Reversal-specific features
    REVERSAL_FEATURES: List[str] = [
        "rsi_div_5",
        "rsi_div_10",
        "rsi_div_20",
        "consecutive_candles",
        "atr_change",
        "dist_from_high_12",
        "dist_from_low_12",
        "dist_from_high_48",
        "dist_from_low_48",
        "volume_delta",
        "price_vs_vwap",
        "funding_velocity",
        "oi_momentum",
    ]

    # Range-specific features
    RANGE_FEATURES: List[str] = [
        "bb_squeeze",
        "bb_upper_dist",
        "bb_lower_dist",
        "range_mid_dist",
        "touches_upper",
        "touches_lower",
        "mean_revert_speed",
    ]

    # Regime Classifier features
    REGIME_FEATURES: List[str] = [
        "ADX_14",
        "adx_slope",
        "regime_bb_width",
        "regime_volatility",
        "regime_range_score",
        "rsi",
        "rsi_div_10",
        "volume_spike",
        "volume_ratio",
        "price_vs_ema200",
        "atr_change",
        "consecutive_candles",
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

        # Price position relative to EMAs
        feat["price_vs_ema50"] = (feat["close"] - feat["ema_50"]) / feat["close"] * 100
        if "ema_100" in feat.columns:
            feat["price_vs_ema100"] = (feat["close"] - feat["ema_100"]) / feat["close"] * 100
        if "ema_200" in feat.columns:
            feat["price_vs_ema200"] = (feat["close"] - feat["ema_200"]) / feat["close"] * 100

        # Simplified RSI divergence
        feat["rsi_divergence"] = feat["rsi"].diff(5) - feat["pct_change_5"] * 100

        # Acceleration / slope (how fast indicators change)
        feat["rsi_slope"] = feat["rsi"].diff(3)
        if "ADX_14" in feat.columns:
            feat["adx_slope"] = feat["ADX_14"].diff(3)

        # Volume spike (binary)
        feat["volume_spike"] = (feat["volume_ratio"] > 2.0).astype(float)

        # Normalized ATR (comparable across pairs)
        feat["atr_pct"] = feat["atr"] / feat["close"] * 100

        # ── Regime detection features ────────────────────────
        if "ADX_14" in feat.columns:
            feat["regime_adx"] = (feat["ADX_14"] > 25).astype(float)

        feat["regime_volatility"] = feat["atr"].rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False,
        )

        if "BBU_20_2.0" in feat.columns and "BBL_20_2.0" in feat.columns:
            bbm = feat.get("BBM_20_2.0", feat["close"])
            feat["regime_bb_width"] = (feat["BBU_20_2.0"] - feat["BBL_20_2.0"]) / bbm * 100

        feat["regime_range_score"] = (
            (feat["high"].rolling(24).max() - feat["low"].rolling(24).min())
            / feat["close"] * 100
        )

        # Ensure BB %B exists
        if "BBP_20_2.0" not in feat.columns:
            if "BBU_20_2.0" in feat.columns and "BBL_20_2.0" in feat.columns:
                bw = feat["BBU_20_2.0"] - feat["BBL_20_2.0"]
                feat["BBP_20_2.0"] = np.where(
                    bw != 0,
                    (feat["close"] - feat["BBL_20_2.0"]) / bw,
                    0.5,
                )

        # ── Reversal detection features ────────────────────────
        # Multi-timeframe RSI divergence (price vs RSI direction mismatch)
        feat["rsi_div_5"] = feat["rsi"].diff(5) - feat["pct_change_5"] * 100
        feat["rsi_div_10"] = feat["rsi"].diff(10) - feat["close"].pct_change(10) * 100
        feat["rsi_div_20"] = feat["rsi"].diff(20) - feat["close"].pct_change(20) * 100

        # Consecutive candle count (how many bars in same direction)
        direction = (feat["close"] > feat["open"]).astype(int) * 2 - 1  # +1 green, -1 red
        streak = direction.copy()
        for i in range(1, len(streak)):
            if direction.iloc[i] == direction.iloc[i - 1]:
                streak.iloc[i] = streak.iloc[i - 1] + direction.iloc[i]
            else:
                streak.iloc[i] = direction.iloc[i]
        feat["consecutive_candles"] = streak

        # ATR expansion/contraction (is the move exhausting?)
        feat["atr_change"] = feat["atr"].pct_change(3) * 100

        # Distance from recent high/low (how far from extreme)
        feat["dist_from_high_12"] = (feat["close"] - feat["high"].rolling(12).max()) / feat["close"] * 100
        feat["dist_from_low_12"] = (feat["close"] - feat["low"].rolling(12).min()) / feat["close"] * 100
        feat["dist_from_high_48"] = (feat["close"] - feat["high"].rolling(48).max()) / feat["close"] * 100
        feat["dist_from_low_48"] = (feat["close"] - feat["low"].rolling(48).min()) / feat["close"] * 100

        # Volume delta proxy (buy vs sell pressure via candle position)
        candle_range = feat["high"] - feat["low"]
        candle_pos = np.where(candle_range > 0, (feat["close"] - feat["low"]) / candle_range, 0.5)
        feat["volume_delta"] = (candle_pos * 2 - 1) * feat["volume_ratio"]

        # Price vs VWAP proxy (cumulative volume-weighted price)
        cum_vol = feat["volume"].rolling(24).sum()
        cum_vp = (feat["close"] * feat["volume"]).rolling(24).sum()
        vwap = np.where(cum_vol > 0, cum_vp / cum_vol, feat["close"])
        feat["price_vs_vwap"] = (feat["close"] - vwap) / feat["close"] * 100

        # ── Range detection features ───────────────────────────
        # BB squeeze (narrowing = breakout coming)
        if "BBU_20_2.0" in feat.columns and "BBL_20_2.0" in feat.columns:
            bb_w = (feat["BBU_20_2.0"] - feat["BBL_20_2.0"]) / feat["close"] * 100
            feat["bb_squeeze"] = bb_w / bb_w.rolling(50).mean()

        # BB upper/lower distance (how close to band edges)
        if "BBU_20_2.0" in feat.columns and "BBL_20_2.0" in feat.columns:
            feat["bb_upper_dist"] = (feat["BBU_20_2.0"] - feat["close"]) / feat["close"] * 100
            feat["bb_lower_dist"] = (feat["close"] - feat["BBL_20_2.0"]) / feat["close"] * 100

        # Range midpoint distance
        range_high = feat["high"].rolling(24).max()
        range_low = feat["low"].rolling(24).min()
        range_mid = (range_high + range_low) / 2
        feat["range_mid_dist"] = (feat["close"] - range_mid) / feat["close"] * 100

        # Support/resistance touches (how many times price touched edges)
        near_high = (feat["high"] >= range_high * 0.998).rolling(24).sum()
        near_low = (feat["low"] <= range_low * 1.002).rolling(24).sum()
        feat["touches_upper"] = near_high
        feat["touches_lower"] = near_low

        # Mean reversion speed (how fast price returns to mean)
        ema20 = feat["close"].ewm(span=20).mean()
        deviation = (feat["close"] - ema20).abs()
        feat["mean_revert_speed"] = deviation.diff(3) * -1  # negative = reverting

        # Funding rate velocity (speed of change)
        if "funding_rate" in feat.columns:
            feat["funding_velocity"] = feat["funding_rate"].diff(3)

        # OI change momentum
        if "oi_change_pct" in feat.columns:
            feat["oi_momentum"] = feat["oi_change_pct"].rolling(3).sum()

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

    # ── rolling HTF (computed from 5m data, no lag) ──────────

    ROLLING_HTF_PERIODS = {"15m": 3, "60m": 12}

    def add_rolling_htf_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute rolling higher-timeframe indicators directly from 5m data.
        For each window (3 bars = 15m, 12 bars = 60m), resample OHLCV and
        compute indicators — updated every 5m bar with zero lag.
        """
        import pandas_ta as _ta

        feat = df.copy()
        close = feat["close"]
        high = feat["high"]
        low = feat["low"]
        volume = feat["volume"]

        for label, window in self.ROLLING_HTF_PERIODS.items():
            prefix = f"rhtf_{label}"

            rolling_high = high.rolling(window).max()
            rolling_low = low.rolling(window).min()
            rolling_close = close  # use latest close
            rolling_vol = volume.rolling(window).sum()

            rsi_len = max(5, 14 * window // 3)
            rsi_val = _ta.rsi(close, length=rsi_len)
            feat[f"{prefix}_rsi"] = rsi_val

            ema_fast = _ta.ema(close, length=9 * window)
            ema_slow = _ta.ema(close, length=21 * window)
            if ema_fast is not None and ema_slow is not None:
                feat[f"{prefix}_ema_cross"] = (ema_fast - ema_slow) / close * 100

            atr_val = _ta.atr(rolling_high, rolling_low, rolling_close, length=14)
            if atr_val is not None:
                feat[f"{prefix}_atr_pct"] = atr_val / close * 100

            vol_sma = rolling_vol.rolling(20).mean()
            feat[f"{prefix}_vol_ratio"] = np.where(vol_sma > 0, rolling_vol / vol_sma, 1.0)

            adx_val = _ta.adx(rolling_high, rolling_low, rolling_close, length=14)
            if adx_val is not None and "ADX_14" in adx_val.columns:
                feat[f"{prefix}_adx"] = adx_val["ADX_14"]

        return feat

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

    def get_feature_columns(self, df: pd.DataFrame, model_type: str = "trend") -> List[str]:
        """Return ordered list of feature columns for a specific model type.

        model_type: "trend" (default), "reversal", "range", "regime", or "all"
        """
        cols = [c for c in self.BASE_FEATURES if c in df.columns]
        htf_cols = sorted([c for c in df.columns if c.startswith("htf_")])
        rhtf_cols = sorted([c for c in df.columns if c.startswith("rhtf_")])
        ctx_cols = [c for c in self.CONTEXT_FEATURES if c in df.columns]

        base = cols + htf_cols + rhtf_cols + ctx_cols

        if model_type == "trend":
            return base
        elif model_type == "reversal":
            extra = [c for c in self.REVERSAL_FEATURES if c in df.columns]
            return base + extra
        elif model_type == "range":
            extra = [c for c in self.RANGE_FEATURES if c in df.columns]
            return base + extra
        elif model_type == "regime":
            return [c for c in self.REGIME_FEATURES if c in df.columns]
        else:  # "all"
            rev = [c for c in self.REVERSAL_FEATURES if c in df.columns]
            rng = [c for c in self.RANGE_FEATURES if c in df.columns]
            return base + rev + rng

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

        # legacy HTF (lagging, kept for backward compat)
        if secondary_df is not None and not secondary_df.empty:
            feat = self.add_htf_features(feat, secondary_df, prefix="htf_15m")

        if trend_df is not None and not trend_df.empty:
            feat = self.add_htf_features(feat, trend_df, prefix="htf_1h")

        # rolling HTF (computed from 5m data, no lag)
        feat = self.add_rolling_htf_features(feat)

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
        binary: bool = False,
    ) -> pd.Series:
        """
        Triple-barrier labeling for intraday trading.

        For each bar *i* we look forward up to *max_bars* candles:
          - If price hits  entry + ATR * tp_multiplier  first  →  label =  1 (BUY)
          - If price hits  entry − ATR * sl_multiplier  first  →  label = −1 (SELL)
          - If neither within the window:
              binary=False → label = 0 (HOLD)
              binary=True  → label based on close direction at max_bars
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

            resolved = False
            for j in range(1, max_bars + 1):
                idx = i + j
                if idx >= n:
                    break
                hit_up = high[idx] >= tp_up
                hit_down = low[idx] <= tp_down
                if hit_up and hit_down:
                    if close[idx] > close[i]:
                        labels.iloc[i] = 1
                    else:
                        labels.iloc[i] = -1
                    resolved = True
                    break
                if hit_up:
                    labels.iloc[i] = 1
                    resolved = True
                    break
                if hit_down:
                    labels.iloc[i] = -1
                    resolved = True
                    break

            if not resolved and binary:
                end_idx = min(i + max_bars, n - 1)
                if close[end_idx] > close[i]:
                    labels.iloc[i] = 1
                else:
                    labels.iloc[i] = -1

        return labels

    # ── regime labeling ─────────────────────────────────────

    @staticmethod
    def create_regime_labels(
        df: pd.DataFrame,
        trend_threshold: float = 1.5,
        range_threshold: float = 0.5,
        max_bars: int = 12,
    ) -> pd.Series:
        """
        Label each bar with market regime: TREND (1), REVERSAL (2), RANGE (0).

        - TREND: price moves >trend_threshold% in one direction over max_bars
        - REVERSAL: price was trending then reverses >1% (direction change)
        - RANGE: price stays within range_threshold% over max_bars
        """
        labels = pd.Series(0, index=df.index, dtype=int)  # default RANGE
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        atr = df["atr"].values if "atr" in df.columns else np.zeros(len(df))
        adx = df["ADX_14"].values if "ADX_14" in df.columns else np.zeros(len(df))

        n = len(df)
        for i in range(n - max_bars):
            future_high = np.max(high[i + 1:i + max_bars + 1])
            future_low = np.min(low[i + 1:i + max_bars + 1])
            future_close = close[min(i + max_bars, n - 1)]
            price = close[i]
            if price == 0:
                continue

            up_move = (future_high - price) / price * 100
            down_move = (price - future_low) / price * 100
            net_move = (future_close - price) / price * 100
            total_range = (future_high - future_low) / price * 100

            # TREND: strong directional move
            if (abs(net_move) > trend_threshold and
                    (adx[i] > 20 or abs(net_move) > trend_threshold * 1.5)):
                labels.iloc[i] = 1  # TREND

            # REVERSAL: was trending, now reversing
            elif i >= 12:
                past_move = (close[i] - close[i - 12]) / close[i - 12] * 100
                if abs(past_move) > 1.0 and np.sign(past_move) != np.sign(net_move) and abs(net_move) > 0.5:
                    labels.iloc[i] = 2  # REVERSAL

            # RANGE: small total range
            elif total_range < range_threshold:
                labels.iloc[i] = 0  # RANGE (default)

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
