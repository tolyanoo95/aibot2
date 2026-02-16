"""
Trade monitoring: trailing stop, opposite signal detection, trade health analysis.
Used by both live scanner and backtester.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradeState:
    """Tracks the state of an open trade."""
    symbol: str
    direction: str          # LONG / SHORT
    entry_price: float
    original_sl: float
    original_tp: float
    current_sl: float       # moves with trailing stop
    best_price: float       # best price since entry (high for LONG, low for SHORT)
    bars_held: int = 0
    trailing_active: bool = False
    health: str = "HEALTHY"  # HEALTHY / WEAKENING / CLOSE_EARLY
    health_reason: str = ""


class TradeMonitor:
    """Monitors open trades and recommends actions."""

    def __init__(self, config):
        self.config = config
        self._trailing_activation = getattr(config, "TRAILING_ACTIVATION_ATR", 1.5)
        self._trailing_distance = getattr(config, "TRAILING_DISTANCE_ATR", 1.0)

    def update_trade(
        self,
        state: TradeState,
        high: float,
        low: float,
        close: float,
        atr: float,
        rsi: float = 50.0,
        adx: float = 25.0,
        volume_ratio: float = 1.0,
        ml_signal: int = 0,
        ml_confidence: float = 0.0,
    ) -> TradeState:
        """Update trade state with new bar data. Returns updated state."""
        state.bars_held += 1

        # ── 1. Update best price ─────────────────────────────
        if state.direction == "LONG":
            state.best_price = max(state.best_price, high)
        else:
            state.best_price = min(state.best_price, low)

        # ── 2. Trailing stop ─────────────────────────────────
        state = self._update_trailing(state, atr)

        # ── 3. Trade health ──────────────────────────────────
        state = self._check_health(
            state, close, rsi, adx, volume_ratio, ml_signal, ml_confidence,
        )

        return state

    def _update_trailing(self, state: TradeState, atr: float) -> TradeState:
        """Move SL when price moves in our favor."""
        if atr <= 0:
            return state

        activation_dist = atr * self._trailing_activation
        trail_dist = atr * self._trailing_distance

        if state.direction == "LONG":
            # activate trailing when price moved activation_dist above entry
            if state.best_price >= state.entry_price + activation_dist:
                state.trailing_active = True
                new_sl = state.best_price - trail_dist
                if new_sl > state.current_sl:
                    state.current_sl = new_sl
        else:  # SHORT
            if state.best_price <= state.entry_price - activation_dist:
                state.trailing_active = True
                new_sl = state.best_price + trail_dist
                if new_sl < state.current_sl:
                    state.current_sl = new_sl

        return state

    def _check_health(
        self,
        state: TradeState,
        close: float,
        rsi: float,
        adx: float,
        volume_ratio: float,
        ml_signal: int,
        ml_confidence: float,
    ) -> TradeState:
        """Analyze trade health and recommend action."""
        issues = []

        # ── Opposite signal detected ─────────────────────────
        if ml_confidence >= 0.55:
            if state.direction == "LONG" and ml_signal == -1:
                issues.append("ML reversed to SELL")
            elif state.direction == "SHORT" and ml_signal == 1:
                issues.append("ML reversed to BUY")

        # ── Momentum fading ──────────────────────────────────
        if adx < 15:
            issues.append(f"ADX weak ({adx:.0f})")

        # ── Volume dying ─────────────────────────────────────
        if volume_ratio < 0.5:
            issues.append(f"Volume dead ({volume_ratio:.2f})")

        # ── RSI extreme against position ─────────────────────
        if state.direction == "LONG" and rsi > 80:
            issues.append(f"RSI overbought ({rsi:.0f})")
        elif state.direction == "SHORT" and rsi < 20:
            issues.append(f"RSI oversold ({rsi:.0f})")

        # ── Determine health status ──────────────────────────
        if any("reversed" in i.lower() for i in issues):
            state.health = "CLOSE_EARLY"
            state.health_reason = " | ".join(issues)
        elif len(issues) >= 2:
            state.health = "WEAKENING"
            state.health_reason = " | ".join(issues)
        else:
            state.health = "HEALTHY"
            state.health_reason = ""

        return state

    def check_exit(
        self,
        state: TradeState,
        high: float,
        low: float,
        close: float,
    ) -> tuple[bool, str, float]:
        """
        Check if trade should exit.
        Returns: (should_exit, reason, exit_price)
        """
        # ── TP hit ───────────────────────────────────────────
        if state.direction == "LONG" and high >= state.original_tp:
            return True, "TP", state.original_tp
        if state.direction == "SHORT" and low <= state.original_tp:
            return True, "TP", state.original_tp

        # ── Trailing SL hit ──────────────────────────────────
        if state.direction == "LONG" and low <= state.current_sl:
            reason = "TRAIL_SL" if state.trailing_active else "SL"
            return True, reason, state.current_sl
        if state.direction == "SHORT" and high >= state.current_sl:
            reason = "TRAIL_SL" if state.trailing_active else "SL"
            return True, reason, state.current_sl

        # ── Max bars ─────────────────────────────────────────
        max_bars = getattr(self.config, "MAX_HOLD_BARS", 12)
        if state.bars_held >= max_bars:
            return True, "TIMEOUT", close

        # ── Early exit on CLOSE_EARLY health ─────────────────
        if state.health == "CLOSE_EARLY" and state.bars_held >= 3:
            return True, "EARLY_EXIT", close

        return False, "", 0.0

    @staticmethod
    def create_state(
        symbol: str,
        direction: str,
        entry_price: float,
        sl: float,
        tp: float,
    ) -> TradeState:
        """Create initial trade state."""
        best = entry_price  # will be updated on first bar
        return TradeState(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            original_sl=sl,
            original_tp=tp,
            current_sl=sl,
            best_price=best,
        )
