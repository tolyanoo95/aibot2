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
    SL_ATR_MULTIPLIER: float = 1.5      # stop-loss = ATR * this (backtest-optimized)
    TP_ATR_MULTIPLIER: float = 2.5      # take-profit = ATR * this  (R:R = 1:1.67)
    MIN_SL_PCT: float = 0.3             # minimum SL distance in % (floor for low-ATR periods)
    MAX_HOLD_BARS: int = 12             # max holding time in candles (60 min on 5m)

    # ── Risk management ──────────────────────────────────────
    RISK_PER_TRADE: float = 0.01        # 1% risk per trade
    MAX_DAILY_LOSS: float = 0.03        # 3% max daily loss
    DEFAULT_LEVERAGE: int = 5
    MAX_LEVERAGE: int = 20

    # ── LLM ──────────────────────────────────────────────────
    LLM_ANALYSIS_INTERVAL: int = 15     # minutes between LLM calls per pair
    USE_LLM: bool = True

    # ── Trailing Stop & Trade Monitoring ────────────────────
    TRAILING_ACTIVATION_ATR: float = 2.0  # activate trailing after price moves 2.0x ATR in favor (2/3 of TP)
    TRAILING_DISTANCE_ATR: float = 1.5    # trail SL at 1.5x ATR behind best price (wider = less chop)

    # ── Signal Filters ───────────────────────────────────────
    FILTER_MIN_VOLUME_RATIO: float = 0.8    # skip if volume < 80% of avg (dead market)
    FILTER_MIN_ADX: float = 15.0            # skip if ADX < 15 (no trend at all)
    FILTER_DEAD_HOURS: list = field(         # UTC hours to skip new signals
        default_factory=lambda: [14]         # 14:00-15:00 UTC = NYSE open (high reversal risk)
    )
    FILTER_TREND_EMA: bool = True           # don't LONG when EMA9<EMA21, don't SHORT when EMA9>EMA21

    # ── 1m Entry Refinement ─────────────────────────────────
    USE_1M_ENTRY: bool = True           # enable 1m entry refinement
    ENTRY_WAIT_BARS: int = 6            # max 1m bars to wait for a pullback (6 min)
    ENTRY_RSI_OVERSOLD: float = 30.0    # 1m RSI below this → good long entry
    ENTRY_RSI_OVERBOUGHT: float = 70.0  # 1m RSI above this → good short entry

    # ── Trading Mode ────────────────────────────────────────
    # "paper" = signals only, log trades, no real orders
    # "live"  = auto-execute real orders on Binance
    TRADING_MODE: str = os.getenv("TRADING_MODE", "paper")
    TRADE_BALANCE_USDT: float = float(os.getenv("TRADE_BALANCE_USDT", "100"))  # balance for position sizing
    MAX_OPEN_TRADES: int = int(os.getenv("MAX_OPEN_TRADES", "3"))              # max simultaneous positions

    # ── Scanner ──────────────────────────────────────────────
    SCAN_INTERVAL: int = 300            # seconds between scans (5 min)

    # ── Training ─────────────────────────────────────────────
    TRAIN_CANDLES: int = 10_000         # ~35 days of 5m data
    LABEL_TP_MULTIPLIER: float = 1.5    # take-profit = ATR * this
    LABEL_SL_MULTIPLIER: float = 1.5    # stop-loss   = ATR * this  (symmetric!)
    LABEL_MAX_BARS: int = 12            # max holding period in candles


config = Config()
