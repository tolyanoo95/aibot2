"""
Technical indicator calculations using pandas_ta.
Returns a DataFrame enriched with all indicators needed for the ML pipeline.
"""

import pandas as pd
import pandas_ta as ta


class TechnicalIndicators:
    """Compute a full suite of intraday-relevant technical indicators."""

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """Add indicator columns to an OHLCV DataFrame (in-place-safe copy)."""
        if df.empty or len(df) < 50:
            return df

        df = df.copy()

        # ── Trend: EMAs ──────────────────────────────────────
        df["ema_9"] = ta.ema(df["close"], length=9)
        df["ema_21"] = ta.ema(df["close"], length=21)
        df["ema_50"] = ta.ema(df["close"], length=50)

        # ── Momentum ─────────────────────────────────────────
        df["rsi"] = ta.rsi(df["close"], length=14)

        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd is not None:
            df = pd.concat([df, macd], axis=1)

        stoch_rsi = ta.stochrsi(df["close"], length=14)
        if stoch_rsi is not None:
            df = pd.concat([df, stoch_rsi], axis=1)

        # ── Volatility ───────────────────────────────────────
        bb = ta.bbands(df["close"], length=20, std=2)
        if bb is not None:
            df = pd.concat([df, bb], axis=1)

        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

        # ── Trend strength ───────────────────────────────────
        adx = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx is not None:
            df = pd.concat([df, adx], axis=1)

        # ── Volume ───────────────────────────────────────────
        df["obv"] = ta.obv(df["close"], df["volume"])
        df["volume_sma"] = ta.sma(df["volume"], length=20)
        df["volume_ratio"] = df["volume"] / df["volume_sma"]

        # ── Derived price features ───────────────────────────
        df["pct_change_1"] = df["close"].pct_change(1)
        df["pct_change_5"] = df["close"].pct_change(5)
        df["pct_change_15"] = df["close"].pct_change(15)

        # ── Candle anatomy ───────────────────────────────────
        df["body_size"] = abs(df["close"] - df["open"]) / df["open"] * 100
        df["upper_wick"] = (
            (df["high"] - df[["close", "open"]].max(axis=1)) / df["open"] * 100
        )
        df["lower_wick"] = (
            (df[["close", "open"]].min(axis=1) - df["low"]) / df["open"] * 100
        )

        return df
