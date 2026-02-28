"""
Configuration for AI Crypto Futures Scanner.
All settings are loaded from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


def _compute_dead_hours() -> list:
    """Auto-detect DST for NYSE open + custom hours from EXTRA_DEAD_HOURS env var.
    Example: EXTRA_DEAD_HOURS=2,3,4
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo
    et_now = datetime.now(ZoneInfo("America/New_York"))
    utc_offset = int(et_now.utcoffset().total_seconds() // 3600)  # -5 winter, -4 summer
    nyse_open_utc = 9 - utc_offset  # 9:30 ET → 14 UTC (winter) or 13 UTC (summer)
    hours = [nyse_open_utc - 1, nyse_open_utc, nyse_open_utc + 1]

    extra = os.getenv("EXTRA_DEAD_HOURS", "")
    if extra:
        hours += [int(h.strip()) for h in extra.split(",") if h.strip().isdigit()]

    return sorted(set(hours))


@dataclass
class Config:
    # ── Binance ──────────────────────────────────────────────
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_SECRET: str = os.getenv("BINANCE_SECRET", "")
    BINANCE_TESTNET: bool = False

    # ── OpenAI ───────────────────────────────────────────────
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # ── Trading pairs (top futures by volume) ────────────────
    TRADING_PAIRS: list = field(default_factory=lambda: [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
        "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT",
        "LTC/USDT",
    ])

    # ── Timeframes ───────────────────────────────────────────
    PRIMARY_TIMEFRAME: str = "5m"       # main signal timeframe
    SECONDARY_TIMEFRAME: str = "15m"    # confirmation
    TREND_TIMEFRAME: str = "1h"         # higher-TF trend filter

    # ── Data ─────────────────────────────────────────────────
    CANDLES_LIMIT: int = 200            # candles per fetch

    # ── ML Model ─────────────────────────────────────────────
    ML_MODEL_PATH: str = "models/signal_model.json"
    PREDICTION_THRESHOLD: float = float(os.getenv("PREDICTION_THRESHOLD", "0.70"))

    # ── Trade parameters ────────────────────────────────────
    SL_ATR_MULTIPLIER: float = float(os.getenv("SL_ATR_MULTIPLIER", "1.0"))
    TP_ATR_MULTIPLIER: float = float(os.getenv("TP_ATR_MULTIPLIER", "3.0"))
    MIN_SL_PCT: float = float(os.getenv("MIN_SL_PCT", "0.3"))
    MAX_HOLD_BARS: int = int(os.getenv("MAX_HOLD_BARS", "18"))

    # ── Risk management ──────────────────────────────────────
    RISK_PER_TRADE: float = 0.01        # 1% risk per trade
    MAX_DAILY_LOSS: float = 0.03        # 3% max daily loss
    DEFAULT_LEVERAGE: int = 5
    MAX_LEVERAGE: int = 20

    # ── LLM ──────────────────────────────────────────────────
    LLM_ANALYSIS_INTERVAL: int = 15     # minutes between LLM calls per pair
    USE_LLM: bool = os.getenv("USE_LLM", "true").lower() == "true"

    # ── Trailing Stop & Trade Monitoring ────────────────────
    TRAILING_ACTIVATION_ATR: float = 2.0  # activate trailing after price moves 2.0x ATR in favor (2/3 of TP)
    TRAILING_DISTANCE_ATR: float = 1.5    # trail SL at 1.5x ATR behind best price (wider = less chop)

    # ── Signal Filters ───────────────────────────────────────
    FILTER_MIN_VOLUME_RATIO: float = 0.0    # disabled — volume filter was blocking too many valid signals
    FILTER_MIN_ADX: float = 0.0             # disabled — ADX filter was blocking valid signals
    FILTER_DEAD_HOURS: list = field(
        default_factory=lambda: _compute_dead_hours() if os.getenv("DEAD_HOURS", "true").lower() == "true" else []
    )
    FILTER_TREND_EMA: bool = False           # disabled — let ML decide direction

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
    MAX_OPEN_TRADES: int = int(os.getenv("MAX_OPEN_TRADES", "11"))             # allow all pairs to trade simultaneously
    MAX_SAME_DIRECTION: int = int(os.getenv("MAX_SAME_DIRECTION", "11"))     # allow all same direction

    # ── Scanner ──────────────────────────────────────────────
    SCAN_INTERVAL: int = 300            # seconds between scans (5 min)

    # ── Training ─────────────────────────────────────────────
    TRAIN_CANDLES: int = 30_000         # ~104 days of 5m data
    LABEL_TP_MULTIPLIER: float = 1.5    # take-profit = ATR * this
    LABEL_SL_MULTIPLIER: float = 1.5    # stop-loss   = ATR * this  (symmetric!)
    LABEL_MAX_BARS: int = 12            # max holding period in candles


config = Config()
