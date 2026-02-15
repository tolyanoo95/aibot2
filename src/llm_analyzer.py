"""
OpenAI-powered market analysis for the hybrid signal pipeline.
Sends a structured prompt with current market state and returns a JSON verdict.
"""

import json
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LLMAnalyzer:
    """Query OpenAI for a directional market opinion on a single symbol."""

    def __init__(self, config):
        self.config = config
        self._client = None
        self._cache: Dict[str, dict] = {}

        if config.OPENAI_API_KEY:
            from openai import OpenAI
            self._client = OpenAI(api_key=config.OPENAI_API_KEY)

    # ── public API ───────────────────────────────────────────

    def analyze_market(
        self,
        symbol: str,
        market_data: dict,
        context: dict | None = None,
    ) -> dict:
        """Return an analysis dict (direction, confidence, …)."""
        if self._client is None:
            logger.warning("OpenAI key not set — skipping LLM analysis")
            return self.default_response()

        prompt = self._build_prompt(symbol, market_data, context)

        try:
            resp = self._client.chat.completions.create(
                model=self.config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
                max_tokens=500,
            )
            result = json.loads(resp.choices[0].message.content)
            self._cache[symbol] = result
            return result

        except Exception as exc:
            logger.error("LLM error for %s: %s", symbol, exc)
            return self._cache.get(symbol, self.default_response())

    @staticmethod
    def default_response() -> dict:
        return {
            "direction": "NEUTRAL",
            "confidence": 0,
            "reasoning": "No LLM analysis available",
            "key_levels": {"support": 0, "resistance": 0},
            "risk_assessment": "HIGH",
            "market_regime": "UNKNOWN",
        }

    # ── prompts ──────────────────────────────────────────────

    @staticmethod
    def _system_prompt() -> str:
        return """You are a professional crypto futures intraday trader specializing in 5-minute scalping on Binance USDⓈ-M Futures. You manage risk aggressively and only take high-probability setups.

RESPOND WITH VALID JSON IN EXACTLY THIS SCHEMA:
{
  "direction": "LONG" | "SHORT" | "NEUTRAL",
  "confidence": <int 1-10>,
  "reasoning": "<max 2 sentences explaining the KEY factor>",
  "key_levels": {"support": <float>, "resistance": <float>},
  "risk_assessment": "LOW" | "MEDIUM" | "HIGH",
  "market_regime": "TRENDING" | "RANGING" | "VOLATILE" | "QUIET"
}

DECISION FRAMEWORK (follow in order):

1. TREND FIRST — Check 1H EMA trend direction. Never trade against it.
   - 1H EMA bullish → only LONG or NEUTRAL
   - 1H EMA bearish → only SHORT or NEUTRAL

2. MOMENTUM CONFIRMATION — Check 5m indicators:
   - RSI < 25 + ADX > 25 + bearish EMA = strong SHORT (conf 8-9)
   - RSI > 75 + ADX > 25 + bullish EMA = strong LONG (conf 8-9)
   - RSI 40-60 + ADX < 20 = no momentum → NEUTRAL

3. SMART MONEY SIGNALS — Check positioning data:
   - Long/Short ratio > 1.5 = crowd is long → SHORT bias (squeeze risk)
   - Long/Short ratio < 0.7 = crowd is short → LONG bias (squeeze risk)
   - Funding rate > 0.05% = longs overpaying → SHORT bias
   - Funding rate < -0.05% = shorts overpaying → LONG bias

4. LIQUIDATION MAGNETS — Price tends to move toward liquidation clusters:
   - Long liq zone close (< 1% below) = price likely goes DOWN to trigger them
   - Short liq zone close (< 1% above) = price likely goes UP to trigger them

5. ORDER BOOK PRESSURE — Immediate direction clue:
   - Imbalance > +0.3 = buyers dominating → supports LONG
   - Imbalance < -0.3 = sellers dominating → supports SHORT
   - Do NOT trade against strong book imbalance

6. MACRO FILTER — Stablecoin dominance context:
   - USDT.D rising + Total MCap falling = risk-off → only SHORT or NEUTRAL
   - USDT.D falling + Total MCap rising = risk-on → LONG bias OK
   - 24h MCap change > +3% = overextended → caution on LONG
   - 24h MCap change < -3% = panic → look for SHORT

CONFIDENCE SCALE:
  10 = All signals aligned, extreme setup (very rare)
  8-9 = Strong trend + momentum + smart money agree
  7 = Good setup, minor conflicting signal
  6 = Decent but not convincing → output NEUTRAL
  1-5 = Mixed or weak → MUST output NEUTRAL

CRITICAL RULES:
- Confidence < 7 → ALWAYS output "NEUTRAL"
- Never chase: if last 5+ candles moved strongly in one direction, the move may be done
- Volume ratio < 0.5 = dead market → NEUTRAL regardless
- Multiple conflicting signals = NEUTRAL (protect capital)
- Be DECISIVE: if setup is clear, give 8-9. Don't default to 5-6."""

    def _build_prompt(
        self, symbol: str, data: dict, context: dict | None = None,
    ) -> str:
        df: pd.DataFrame = data.get("primary", pd.DataFrame())
        if df.empty:
            return f"No data for {symbol}"

        ctx = context or {}
        last = df.iloc[-1]
        lines = [
            f"Symbol: {symbol}",
            f"Current Price: {last['close']:.6g}",
        ]

        # 24 h change (5 m candle → 288 bars = 24 h)
        lookback = min(288, len(df) - 1)
        if lookback > 0:
            ref = df.iloc[-lookback - 1]["close"]
            pct = (last["close"] - ref) / ref * 100
            lines.append(f"~24 h change: {pct:+.2f}%")

        # ── Technical indicators ─────────────────────────────
        lines.append("\n=== Indicators (5 m) ===")
        _ind_map = {
            "rsi": "RSI(14)",
            "MACD_12_26_9": "MACD",
            "MACDh_12_26_9": "MACD-Hist",
            "BBP_20_2.0": "BB %B",
            "atr": "ATR(14)",
            "ADX_14": "ADX(14)",
            "ema_9": "EMA-9",
            "ema_21": "EMA-21",
            "ema_50": "EMA-50",
            "volume_ratio": "Vol Ratio",
            "STOCHRSIk_14_14_3_3": "StochRSI-K",
        }
        for col, name in _ind_map.items():
            if col in last.index:
                val = last[col]
                if pd.notna(val):
                    lines.append(f"  {name}: {val:.6g}")

        # ── Last 10 candles ──────────────────────────────────
        lines.append("\n=== Last 10 candles ===")
        for _, row in df.tail(10).iterrows():
            chg = (row["close"] - row["open"]) / row["open"] * 100
            lines.append(
                f"  O:{row['open']:.6g} H:{row['high']:.6g} "
                f"L:{row['low']:.6g} C:{row['close']:.6g} "
                f"V:{row['volume']:.0f} ({chg:+.2f}%)"
            )

        # ── 1 H trend context ────────────────────────────────
        trend_df: pd.DataFrame = data.get("trend", pd.DataFrame())
        if not trend_df.empty:
            tl = trend_df.iloc[-1]
            lines.append("\n=== 1 H trend context ===")
            if "rsi" in tl.index and pd.notna(tl["rsi"]):
                lines.append(f"  1H RSI: {tl['rsi']:.2f}")
            if "ema_9" in tl.index and "ema_21" in tl.index:
                if pd.notna(tl["ema_9"]) and pd.notna(tl["ema_21"]):
                    bias = "BULLISH" if tl["ema_9"] > tl["ema_21"] else "BEARISH"
                    lines.append(f"  1H EMA trend: {bias}")

        # ── Market context (OI, L/S, dominance, liquidations) ─
        if ctx:
            lines.append("\n=== Market Context ===")

            fr = data.get("funding_rate", 0)
            lines.append(f"  Funding Rate: {fr:.6f}")

            oi = ctx.get("open_interest", 0)
            if oi:
                lines.append(f"  Open Interest: {oi:,.0f} contracts")

            ls = ctx.get("long_short_ratio", 0)
            if ls:
                side = "more longs" if ls > 1 else "more shorts"
                lines.append(f"  Long/Short Ratio: {ls:.3f} ({side})")
                lp = ctx.get("long_account_pct", 0)
                sp = ctx.get("short_account_pct", 0)
                lines.append(f"  Long accounts: {lp:.1%} | Short accounts: {sp:.1%}")

            lines.append("\n=== Stablecoin Dominance ===")
            lines.append(f"  USDT Dominance: {ctx.get('usdt_dom', 0):.2f}%")
            lines.append(f"  USDC Dominance: {ctx.get('usdc_dom', 0):.2f}%")
            lines.append(
                f"  Total Stablecoin Dom: {ctx.get('stablecoin_dom', 0):.2f}%"
            )

            lines.append("\n=== Global Market ===")
            lines.append(
                f"  Total Market Cap: ${ctx.get('total_mcap_b', 0):,.0f}B"
            )
            lines.append(
                f"  24h MCap Change: {ctx.get('mcap_change_24h', 0):+.2f}%"
            )
            lines.append(f"  BTC Dominance: {ctx.get('btc_dom', 0):.1f}%")

            lines.append("\n=== Liquidation Zones ===")
            slz = ctx.get("short_liq_zone", 0)
            llz = ctx.get("long_liq_zone", 0)
            if slz:
                lines.append(
                    f"  Short liq zone (squeeze risk): {slz:.6g} "
                    f"({ctx.get('dist_to_short_liq_pct', 0):.2f}% above)"
                )
            if llz:
                lines.append(
                    f"  Long liq zone (cascade risk):  {llz:.6g} "
                    f"({ctx.get('dist_to_long_liq_pct', 0):.2f}% below)"
                )
            lp = ctx.get("liq_pressure", "NEUTRAL")
            lines.append(f"  Liquidation Pressure: {lp}")

            ob_imb = ctx.get("bid_ask_imbalance", 0)
            if ob_imb:
                if ob_imb > 0.2:
                    ob_desc = "strong buy pressure"
                elif ob_imb > 0:
                    ob_desc = "slight buy pressure"
                elif ob_imb < -0.2:
                    ob_desc = "strong sell pressure"
                else:
                    ob_desc = "slight sell pressure"
                lines.append(f"\n=== Order Book ===")
                lines.append(f"  Bid/Ask Imbalance: {ob_imb:+.2f} ({ob_desc})")
                lines.append(
                    f"  Bid Vol: {ctx.get('bid_volume', 0):,.0f} | "
                    f"Ask Vol: {ctx.get('ask_volume', 0):,.0f}"
                )
                bw_p = ctx.get("bid_wall_price", 0)
                aw_p = ctx.get("ask_wall_price", 0)
                if bw_p:
                    lines.append(
                        f"  Bid wall: {ctx.get('bid_wall_size', 0):,.1f} @ {bw_p:.6g}"
                    )
                if aw_p:
                    lines.append(
                        f"  Ask wall: {ctx.get('ask_wall_size', 0):,.1f} @ {aw_p:.6g}"
                    )
        else:
            fr = data.get("funding_rate", 0)
            lines.append(f"\nFunding Rate: {fr:.6f}")

        return "\n".join(lines)
