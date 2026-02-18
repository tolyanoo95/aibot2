"""
Hybrid signal generator — combines ML predictions with LLM analysis
into a single actionable Signal object.
Includes filters: volume, ADX, time-of-day, order book imbalance.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Represents one trading signal for a symbol."""

    symbol: str
    direction: str          # LONG | SHORT | NEUTRAL
    confidence: float       # 0.0 – 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: int

    # breakdown
    ml_signal: int          # -1 / 0 / 1
    ml_confidence: float
    llm_direction: str
    llm_confidence: int     # 1-10 from GPT
    llm_reasoning: str
    market_regime: str
    risk_level: str

    timestamp: datetime = field(default_factory=datetime.now)

    # tracking: filled by the scanner when signal first appeared
    first_seen: Optional[datetime] = None
    original_entry: float = 0.0
    original_sl: float = 0.0
    original_tp: float = 0.0
    is_new: bool = True     # True only on the first scan
    age_bars: int = 0       # how many past bars had the same signal (freshness)
    filter_reason: str = "" # why signal was filtered (empty = passed)

    # candle data for accurate paper trading
    candle_high: float = 0.0
    candle_low: float = 0.0
    candle_atr: float = 0.0
    candle_rsi: float = 50.0


class SignalGenerator:
    """Weighted voting of ML + LLM into a final signal."""

    ML_WEIGHT = 0.60
    LLM_WEIGHT = 0.40

    def __init__(self, config):
        self.config = config

    # ── public ───────────────────────────────────────────────

    def generate(
        self,
        symbol: str,
        ml_result: dict,
        llm_result: dict,
        current_price: float,
        atr: float,
        volume_ratio: float = 1.0,
        adx: float = 25.0,
        bid_ask_imbalance: float = 0.0,
        ema_trend: int = 0,  # +1 = bullish (EMA9>EMA21), -1 = bearish, 0 = unknown
    ) -> Signal:
        ml_sig = ml_result.get("signal", 0)
        ml_conf = ml_result.get("confidence", 0.0)

        llm_dir = llm_result.get("direction", "NEUTRAL")
        llm_conf = llm_result.get("confidence", 0)
        llm_reason = llm_result.get("reasoning", "")
        regime = llm_result.get("market_regime", "UNKNOWN")
        risk = llm_result.get("risk_assessment", "HIGH")

        direction, confidence = self._combine(ml_sig, ml_conf, llm_dir, llm_conf)

        # ── apply filters ────────────────────────────────────
        filter_reason = ""
        if direction != "NEUTRAL":
            filter_reason = self._apply_filters(
                direction, volume_ratio, adx, bid_ask_imbalance, ema_trend,
            )
            if filter_reason:
                logger.info(
                    "%s %s filtered: %s", symbol, direction, filter_reason,
                )
                direction = "NEUTRAL"
                confidence *= 0.3

        # SL / TP via ATR with minimum floor
        sl = tp = 0.0
        if atr > 0 and direction != "NEUTRAL":
            sl_dist = atr * self.config.SL_ATR_MULTIPLIER
            tp_dist = atr * self.config.TP_ATR_MULTIPLIER

            # enforce minimum SL/TP distance (% of price)
            min_sl = current_price * self.config.MIN_SL_PCT / 100
            if sl_dist < min_sl:
                ratio = min_sl / sl_dist if sl_dist > 0 else 1
                sl_dist = min_sl
                tp_dist = tp_dist * ratio  # scale TP proportionally

            if direction == "LONG":
                sl = current_price - sl_dist
                tp = current_price + tp_dist
            else:
                sl = current_price + sl_dist
                tp = current_price - tp_dist

        leverage = self._leverage(confidence, risk)

        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=sl,
            take_profit=tp,
            leverage=leverage,
            ml_signal=ml_sig,
            ml_confidence=ml_conf,
            llm_direction=llm_dir,
            llm_confidence=llm_conf,
            llm_reasoning=llm_reason,
            market_regime=regime,
            risk_level=risk,
            filter_reason=filter_reason,
        )

    # ── filters ──────────────────────────────────────────────

    def _apply_filters(
        self,
        direction: str,
        volume_ratio: float,
        adx: float,
        bid_ask_imbalance: float,
        ema_trend: int = 0,
    ) -> str:
        """
        Return empty string if signal passes all filters,
        or a reason string if filtered out.
        """
        # 1. Volume filter — skip dead market
        min_vol = self.config.FILTER_MIN_VOLUME_RATIO
        if min_vol > 0 and volume_ratio < min_vol:
            return f"Low volume ({volume_ratio:.2f} < {min_vol})"

        # 2. ADX filter — skip no-trend market
        min_adx = self.config.FILTER_MIN_ADX
        if min_adx > 0 and adx < min_adx:
            return f"Weak trend (ADX {adx:.1f} < {min_adx})"

        # 3. Trend EMA filter — don't trade against the short-term trend
        if self.config.FILTER_TREND_EMA and ema_trend != 0:
            if direction == "LONG" and ema_trend < 0:
                return "LONG against bearish EMA trend (EMA9 < EMA21)"
            if direction == "SHORT" and ema_trend > 0:
                return "SHORT against bullish EMA trend (EMA9 > EMA21)"

        # 4. Time-of-day filter — skip dead hours
        dead_hours = self.config.FILTER_DEAD_HOURS
        if dead_hours:
            current_hour = datetime.now(timezone.utc).hour
            if current_hour in dead_hours:
                return f"Dead hour (UTC {current_hour}:00)"

        # 5. Order book filter — don't trade against strong book pressure
        if abs(bid_ask_imbalance) > 0.5:
            if direction == "LONG" and bid_ask_imbalance < -0.5:
                return f"Orderbook against LONG (imbalance {bid_ask_imbalance:+.2f})"
            if direction == "SHORT" and bid_ask_imbalance > 0.5:
                return f"Orderbook against SHORT (imbalance {bid_ask_imbalance:+.2f})"

        return ""  # all filters passed

    # ── internals ────────────────────────────────────────────

    def _combine(
        self,
        ml_sig: int,
        ml_conf: float,
        llm_dir: str,
        llm_conf: int,
    ) -> tuple[str, float]:
        llm_num = {"LONG": 1, "SHORT": -1}.get(llm_dir, 0)
        llm_norm = llm_conf / 10.0

        score = (
            ml_sig * ml_conf * self.ML_WEIGHT
            + llm_num * llm_norm * self.LLM_WEIGHT
        )
        conf = min(abs(score), 1.0)

        threshold = self.config.PREDICTION_THRESHOLD * 0.5

        if score > threshold:
            direction = "LONG"
        elif score < -threshold:
            direction = "SHORT"
        else:
            direction = "NEUTRAL"

        # agreement bonus
        if ml_sig == llm_num and ml_sig != 0:
            conf = min(conf * 1.2, 1.0)

        # disagreement → stay out
        if ml_sig != 0 and llm_num != 0 and ml_sig != llm_num:
            direction = "NEUTRAL"
            conf *= 0.5

        return direction, round(conf, 3)

    def _leverage(self, confidence: float, risk: str) -> int:
        risk_mult = {"LOW": 1.5, "MEDIUM": 1.0, "HIGH": 0.5}.get(risk, 0.5)
        lev = int(self.config.DEFAULT_LEVERAGE * confidence * risk_mult)
        return max(1, min(lev, self.config.MAX_LEVERAGE))
