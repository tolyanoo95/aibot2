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
    worst_rsi: float = 50.0  # most extreme RSI during trade (lowest for SHORT, highest for LONG)


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
        funding_rate: float = 0.0,
        ls_ratio: float = 1.0,
    ) -> TradeState:
        """Update trade state with new bar data. Returns updated state."""
        state.bars_held += 1

        # ── 1. Update best price ─────────────────────────────
        if state.direction == "LONG":
            state.best_price = max(state.best_price, high)
        else:
            state.best_price = min(state.best_price, low)

        # ── 2. Track extreme RSI for divergence detection ───
        if state.direction == "SHORT":
            state.worst_rsi = min(state.worst_rsi, rsi)
        else:
            state.worst_rsi = max(state.worst_rsi, rsi)

        # ── 3. Trailing stop ─────────────────────────────────
        state = self._update_trailing(state, atr)

        # ── 4. Trade health ──────────────────────────────────
        state = self._check_health(
            state, close, rsi, adx, volume_ratio, ml_signal, ml_confidence,
            funding_rate, ls_ratio,
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
        funding_rate: float = 0.0,
        ls_ratio: float = 1.0,
    ) -> TradeState:
        """Analyze trade health and recommend action."""
        issues = []
        critical = False

        # ── Opposite signal detected (only strong reversal) ──
        if ml_confidence >= 0.65:
            if state.direction == "LONG" and ml_signal == -1:
                issues.append("ML reversed to SELL")
                critical = True
            elif state.direction == "SHORT" and ml_signal == 1:
                issues.append("ML reversed to BUY")
                critical = True

        # ── Momentum fading ──────────────────────────────────
        if adx < 12:
            issues.append(f"ADX weak ({adx:.0f})")

        # ── Volume dying ─────────────────────────────────────
        if volume_ratio < 0.3:
            issues.append(f"Volume dead ({volume_ratio:.2f})")
            critical = critical or (volume_ratio < 0.15)

        # ── RSI extreme against position ─────────────────────
        if state.direction == "LONG" and rsi > 80:
            issues.append(f"RSI overbought ({rsi:.0f})")
            if rsi > 85:
                critical = True
        elif state.direction == "SHORT" and rsi < 20:
            issues.append(f"RSI oversold ({rsi:.0f})")
            if rsi < 15:
                critical = True

        # ── Funding rate squeeze risk ────────────────────────
        if state.direction == "SHORT" and funding_rate < -0.03:
            issues.append(f"Funding negative ({funding_rate:.4f}) → squeeze risk")
            critical = True
        elif state.direction == "LONG" and funding_rate > 0.03:
            issues.append(f"Funding positive ({funding_rate:.4f}) → squeeze risk")
            critical = True

        # ── RSI divergence (momentum reversal) ────────────────
        if state.bars_held >= 3:
            rsi_recovery = rsi - state.worst_rsi
            if state.direction == "SHORT" and rsi_recovery > 15 and close <= state.entry_price:
                issues.append(f"RSI divergence (recovered +{rsi_recovery:.0f} from {state.worst_rsi:.0f})")
                critical = True
            elif state.direction == "LONG" and rsi_recovery < -15 and close >= state.entry_price:
                issues.append(f"RSI divergence (dropped {rsi_recovery:.0f} from {state.worst_rsi:.0f})")
                critical = True

        # ── L/S ratio squeeze risk ───────────────────────────
        if state.direction == "SHORT" and ls_ratio < 0.8:
            issues.append(f"L/S low ({ls_ratio:.2f}) → crowd short, squeeze risk")
            critical = True
        elif state.direction == "LONG" and ls_ratio > 2.5:
            issues.append(f"L/S high ({ls_ratio:.2f}) → crowd long, squeeze risk")
            critical = True

        # ── Determine health status ──────────────────────────
        if critical:
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

        # ── Early exit: only on CLOSE_EARLY + trade is in profit ──
        if state.health == "CLOSE_EARLY" and state.bars_held >= 4:
            if state.direction == "LONG" and close > state.entry_price:
                return True, "EARLY_EXIT", close
            elif state.direction == "SHORT" and close < state.entry_price:
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
        best = entry_price
        worst_rsi = 100.0 if direction == "SHORT" else 0.0
        return TradeState(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            original_sl=sl,
            original_tp=tp,
            current_sl=sl,
            best_price=best,
            worst_rsi=worst_rsi,
        )
