"""
1-minute entry refinement.
──────────────────────────
After the 5m scanner generates a signal, this module fetches 1m candles
and looks for an optimal entry point — a pullback towards a short-term
support/resistance on the 1m timeframe.

For LONG signals:  waits for a dip (RSI oversold, touch EMA-9, lower BB)
For SHORT signals: waits for a bounce (RSI overbought, touch EMA-9, upper BB)

If no pullback within ENTRY_WAIT_BARS minutes → enters at market.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)


@dataclass
class RefinedEntry:
    """Result of 1m entry refinement."""
    entry_price: float
    stop_loss: float
    take_profit: float
    method: str             # "PULLBACK", "EMA_TOUCH", "BB_TOUCH", "MARKET" (no pullback)
    wait_bars: int          # how many 1m bars we waited
    improvement_pct: float  # entry improvement vs original signal price


class EntryRefiner:
    """Watch 1m candles for a better entry after a 5m signal fires."""

    def __init__(self, config, fetcher):
        self.config = config
        self.fetcher = fetcher

    def refine(
        self,
        symbol: str,
        direction: str,
        signal_price: float,
        atr_5m: float,
    ) -> RefinedEntry:
        """
        Fetch recent 1m candles and find the best entry.

        Returns a RefinedEntry with potentially better price and tighter SL.
        """
        if not self.config.USE_1M_ENTRY:
            return self._market_entry(signal_price, direction, atr_5m)

        try:
            df_1m = self.fetcher.fetch_ohlcv(symbol, "1m", limit=50)
            if df_1m.empty or len(df_1m) < 20:
                logger.warning("Not enough 1m data for %s", symbol)
                return self._market_entry(signal_price, direction, atr_5m)

            return self._find_entry(df_1m, direction, signal_price, atr_5m)

        except Exception as exc:
            logger.error("1m refine error %s: %s", symbol, exc)
            return self._market_entry(signal_price, direction, atr_5m)

    def _find_entry(
        self,
        df: pd.DataFrame,
        direction: str,
        signal_price: float,
        atr_5m: float,
    ) -> RefinedEntry:
        """Analyze 1m candles for a pullback entry."""

        # calculate 1m indicators
        df = df.copy()
        df["ema_9"] = ta.ema(df["close"], length=9)
        df["rsi"] = ta.rsi(df["close"], length=14)
        bb = ta.bbands(df["close"], length=20, std=2)
        if bb is not None:
            df = pd.concat([df, bb], axis=1)
        df["atr_1m"] = ta.atr(df["high"], df["low"], df["close"], length=14)

        # look at last N bars for a pullback
        lookback = min(self.config.ENTRY_WAIT_BARS, len(df) - 1)
        recent = df.iloc[-lookback:]

        best_price = signal_price
        best_method = "MARKET"
        best_bar = 0

        for i, (ts, row) in enumerate(recent.iterrows()):
            price = float(row["close"])
            rsi = float(row["rsi"]) if pd.notna(row.get("rsi")) else 50
            ema = float(row["ema_9"]) if pd.notna(row.get("ema_9")) else price

            if direction == "LONG":
                # for LONG: look for a dip — lower price is better
                is_pullback = False

                # RSI oversold on 1m
                if rsi <= self.config.ENTRY_RSI_OVERSOLD:
                    is_pullback = True
                    method = "RSI_OVERSOLD"

                # price touched or dipped below EMA-9
                elif price <= ema * 1.001:
                    is_pullback = True
                    method = "EMA_TOUCH"

                # price touched lower Bollinger Band
                elif "BBL_20_2.0" in row.index and pd.notna(row.get("BBL_20_2.0")):
                    if price <= float(row["BBL_20_2.0"]) * 1.002:
                        is_pullback = True
                        method = "BB_TOUCH"

                if is_pullback and price < best_price:
                    best_price = price
                    best_method = method
                    best_bar = i + 1

            else:  # SHORT
                # for SHORT: look for a bounce — higher price is better
                is_pullback = False

                if rsi >= self.config.ENTRY_RSI_OVERBOUGHT:
                    is_pullback = True
                    method = "RSI_OVERBOUGHT"

                elif price >= ema * 0.999:
                    is_pullback = True
                    method = "EMA_TOUCH"

                elif "BBU_20_2.0" in row.index and pd.notna(row.get("BBU_20_2.0")):
                    if price >= float(row["BBU_20_2.0"]) * 0.998:
                        is_pullback = True
                        method = "BB_TOUCH"

                if is_pullback and price > best_price:
                    best_price = price
                    best_method = method
                    best_bar = i + 1

        # calculate tighter SL from 1m ATR
        last_row = df.iloc[-1]
        atr_1m = float(last_row["atr_1m"]) if pd.notna(last_row.get("atr_1m")) else 0

        # SL: use tighter of 1m-based or 5m-based
        sl_1m = atr_1m * 3 if atr_1m > 0 else atr_5m * self.config.SL_ATR_MULTIPLIER
        sl_5m = atr_5m * self.config.SL_ATR_MULTIPLIER
        sl_dist = min(sl_1m, sl_5m)  # tighter SL

        # TP stays the same (5m based)
        tp_dist = atr_5m * self.config.TP_ATR_MULTIPLIER

        if direction == "LONG":
            sl = best_price - sl_dist
            tp = best_price + tp_dist
            improvement = (signal_price - best_price) / signal_price * 100
        else:
            sl = best_price + sl_dist
            tp = best_price - tp_dist
            improvement = (best_price - signal_price) / signal_price * 100

        return RefinedEntry(
            entry_price=round(best_price, 8),
            stop_loss=round(sl, 8),
            take_profit=round(tp, 8),
            method=best_method,
            wait_bars=best_bar,
            improvement_pct=round(improvement, 3),
        )

    def _market_entry(
        self, price: float, direction: str, atr_5m: float,
    ) -> RefinedEntry:
        """Fallback: enter at current price with standard SL/TP."""
        sl_dist = atr_5m * self.config.SL_ATR_MULTIPLIER
        tp_dist = atr_5m * self.config.TP_ATR_MULTIPLIER
        if direction == "LONG":
            sl, tp = price - sl_dist, price + tp_dist
        else:
            sl, tp = price + sl_dist, price - tp_dist
        return RefinedEntry(
            entry_price=round(price, 8),
            stop_loss=round(sl, 8),
            take_profit=round(tp, 8),
            method="MARKET",
            wait_bars=0,
            improvement_pct=0.0,
        )
