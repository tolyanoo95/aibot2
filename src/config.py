"""
Configuration for AI Crypto Futures Scanner.
All settings are loaded from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # ── Binance ──────────────────────────────────────────────
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_SECRET: str = os.getenv("BINANCE_SECRET", "")
    BINANCE_TESTNET: bool = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

    # ── OpenAI ───────────────────────────────────────────────
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # ── Trading pairs (top futures by volume) ────────────────
    TRADING_PAIRS: list = field(default_factory=lambda: [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
        "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT",
    ])

    # ── Timeframes ───────────────────────────────────────────
    PRIMARY_TIMEFRAME: str = "5m"       # main signal timeframe
    SECONDARY_TIMEFRAME: str = "15m"    # confirmation
    TREND_TIMEFRAME: str = "1h"         # higher-TF trend filter

    # ── Data ─────────────────────────────────────────────────
    CANDLES_LIMIT: int = 200            # candles per fetch

    # ── ML Model ─────────────────────────────────────────────
    ML_MODEL_PATH: str = "models/signal_model.json"
    PREDICTION_THRESHOLD: float = 0.55  # min confidence for a signal (backtest-optimized)

    # ── Trade parameters (backtest-optimized) ────────────────
    SL_ATR_MULTIPLIER: float = 1.0      # stop-loss = ATR * this
    TP_ATR_MULTIPLIER: float = 3.0      # take-profit = ATR * this  (R:R = 1:3)
    MAX_HOLD_BARS: int = 12             # max holding time in candles (60 min on 5m)

    # ── Risk management ──────────────────────────────────────
    RISK_PER_TRADE: float = 0.01        # 1% risk per trade
    MAX_DAILY_LOSS: float = 0.03        # 3% max daily loss
    DEFAULT_LEVERAGE: int = 5
    MAX_LEVERAGE: int = 20

    # ── LLM ──────────────────────────────────────────────────
    LLM_ANALYSIS_INTERVAL: int = 15     # minutes between LLM calls per pair
    USE_LLM: bool = True

    # ── Signal Filters ───────────────────────────────────────
    FILTER_MIN_VOLUME_RATIO: float = 0.8    # skip if volume < 80% of avg (dead market)
    FILTER_MIN_ADX: float = 15.0            # skip if ADX < 15 (no trend at all)
    FILTER_DEAD_HOURS: list = field(         # UTC hours with low volume → skip signals
        default_factory=lambda: []           # e.g. [2,3,4,5] for 2-5 AM UTC
    )

    # ── 1m Entry Refinement ─────────────────────────────────
    USE_1M_ENTRY: bool = True           # enable 1m entry refinement
    ENTRY_WAIT_BARS: int = 6            # max 1m bars to wait for a pullback (6 min)
    ENTRY_RSI_OVERSOLD: float = 30.0    # 1m RSI below this → good long entry
    ENTRY_RSI_OVERBOUGHT: float = 70.0  # 1m RSI above this → good short entry

    # ── Scanner ──────────────────────────────────────────────
    SCAN_INTERVAL: int = 300            # seconds between scans (5 min)

    # ── Training ─────────────────────────────────────────────
    TRAIN_CANDLES: int = 10_000         # ~35 days of 5m data
    LABEL_TP_MULTIPLIER: float = 1.5    # take-profit = ATR * this
    LABEL_SL_MULTIPLIER: float = 1.5    # stop-loss   = ATR * this  (symmetric!)
    LABEL_MAX_BARS: int = 12            # max holding period in candles


config = Config()
