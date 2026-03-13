#!/usr/bin/env python3
"""
Volume Bars Trading Bot — Live/Paper
─────────────────────────────────────
Rules-based momentum strategy on volume bars.
No ML. Pure rules: momentum + guards + HTF vol trend + DCA + global lock + ADX.

Usage:
    python main_volbars.py              # run live (paper mode by default)
    python main_volbars.py --once       # single scan then exit
    python main_volbars.py --live       # real trading (needs API keys)
"""

VERSION = "1.0.0"

import argparse
import functools
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import config
from src.data_fetcher import BinanceDataFetcher
from src.indicators import TechnicalIndicators
from backtest_volbars import resample_to_volume_bars

logging.basicConfig(level=logging.WARNING)

import functools
_print = functools.partial(print, flush=True)

class _Logger:
    def __init__(self):
        self._file = open("volbars_bot.log", "a")
    def info(self, msg):
        from datetime import datetime
        line = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
        _print(line)
        self._file.write(line + "\n")
        self._file.flush()
    def warning(self, msg): self.info(f"[WARN] {msg}")
    def error(self, msg): self.info(f"[ERROR] {msg}")
    def debug(self, msg): pass

logger = _Logger()

def _pfmt(price: float) -> str:
    """Format price with correct decimal precision based on magnitude."""
    ap = abs(price)
    if ap >= 1000: return f"{price:.2f}"
    if ap >= 10: return f"{price:.2f}"
    if ap >= 1: return f"{price:.4f}"
    if ap >= 0.01: return f"{price:.5f}"
    return f"{price:.8f}"

def _prnd(price: float) -> float:
    """Round price to correct precision for JSON output."""
    ap = abs(price)
    if ap >= 1000: return round(price, 2)
    if ap >= 10: return round(price, 2)
    if ap >= 1: return round(price, 4)
    if ap >= 0.01: return round(price, 5)
    return round(price, 8)

# ── Strategy parameters ──────────────────────────────────────
SL_MULT = 2.0
TP_MULT = 4.0
LONG_MOM = 0.30
SHORT_MOM = 0.10
ADX_MIN = 20
ATR_EXP_MAX = 1.5
MAX_OPEN = 11
MAX_DCA = 3
DCA_STEP_MULT = 1.0
COOLDOWN_BARS = 3
MOVE_SL_AT = 2.0      # activate when profit >= 2.0x ATR
MOVE_SL_TO = 1.0      # move SL to entry + 1.0x ATR
MOVE_SL_CHECK = 60    # check every 60 seconds
WARMUP_DAYS = 60
SCAN_INTERVAL = 900  # 15 minutes


@dataclass
class Position:
    symbol: str
    direction: str
    entries: list = field(default_factory=list)
    avg_price: float = 0.0
    total_size: int = 0
    hard_sl: float = 0.0
    tp: float = 0.0
    entry_time: float = 0.0
    bars_held: int = 0
    entry_atr: float = 0.0
    max_price: float = 0.0
    min_price: float = float('inf')
    sl_moved: bool = False


class VolumeBarsBot:
    def __init__(self, paper: bool = True):
        import threading
        self.paper = paper
        self.fetcher = BinanceDataFetcher(config)
        self.indicators = TechnicalIndicators()
        self.pairs = list(config.TRADING_PAIRS)

        # State
        self.time_data: Dict[str, pd.DataFrame] = {}
        self.vol_bars: Dict[str, pd.DataFrame] = {}
        self.htf_vol_bars: Dict[str, pd.DataFrame] = {}
        self.vol_buffers: Dict[str, dict] = {}
        self.vol_thresholds: Dict[str, float] = {}
        self.positions: List[Position] = []
        self.cooldowns: Dict[str, int] = {}
        self.global_locked_dir: Optional[str] = None  # kept for state compat, not used
        self.global_sl_streak = {"LONG": 0, "SHORT": 0}  # kept for state compat, not used
        self.scan_count = 0
        self.pair_sl_streaks: Dict[str, int] = {}
        self.pair_dir_cooldowns: Dict[str, int] = {}
        self.PAIR_COOLDOWN_SL = 2
        self.PAIR_COOLDOWN_BARS = 8
        self.vol_bar_counts: Dict[str, int] = {}
        self.vol_history: Dict[str, list] = {}
        self.time_bar_counts: Dict[str, int] = {}

        # Thread lock for shared resources (positions, global_lock, cooldowns)
        self._lock = threading.Lock()
        self._data_dir = "data/volbars"
        os.makedirs(self._data_dir, exist_ok=True)
        self._trades_log = "trades.log"
        self._paper_trades_file = "paper_trades.json"

    def _save_state(self):
        """Save volume bars, positions, and bot state to disk."""
        import json
        try:
            import pickle
            # Save volume bars per symbol (pickle for reliability with duplicate cols)
            for symbol, vdf in self.vol_bars.items():
                fname = symbol.replace("/", "_").replace(":", "_")
                with open(os.path.join(self._data_dir, f"{fname}_volbars.pkl"), "wb") as f:
                    pickle.dump(vdf, f)

            # Save thresholds
            with open(os.path.join(self._data_dir, "thresholds.json"), "w") as f:
                json.dump(self.vol_thresholds, f)

            # Save bot state
            state = {
                "global_locked_dir": self.global_locked_dir,
                "global_sl_streak": self.global_sl_streak,
                "scan_count": self.scan_count,
                "cooldowns": self.cooldowns,
                "pair_sl_streaks": self.pair_sl_streaks,
                "pair_dir_cooldowns": self.pair_dir_cooldowns,
                "vol_bar_counts": self.vol_bar_counts,
                "positions": [
                    {
                        "symbol": p.symbol, "direction": p.direction,
                        "entries": p.entries, "avg_price": p.avg_price,
                        "total_size": p.total_size, "hard_sl": p.hard_sl,
                        "tp": p.tp, "entry_time": p.entry_time, "bars_held": p.bars_held,
                        "entry_atr": p.entry_atr, "max_price": p.max_price, "min_price": p.min_price,
                        "sl_moved": p.sl_moved,
                    } for p in self.positions
                ],
                "vol_buffers": {k: {kk: (vv if not isinstance(vv, pd.Timestamp) else str(vv))
                                    for kk, vv in v.items()} for k, v in self.vol_buffers.items()},
            }
            with open(os.path.join(self._data_dir, "state.json"), "w") as f:
                json.dump(state, f, indent=2, default=str)

            logger.info(f"  State saved to {self._data_dir}/")
        except Exception as e:
            logger.error(f"Save error: {e}")

    def _load_state(self) -> bool:
        """Load saved state from disk. Returns True if loaded successfully."""
        import json
        state_path = os.path.join(self._data_dir, "state.json")
        if not os.path.exists(state_path):
            return False

        try:
            # Load volume bars
            import pickle
            loaded = 0
            for symbol in self.pairs:
                fname = symbol.replace("/", "_").replace(":", "_")
                vb_path = os.path.join(self._data_dir, f"{fname}_volbars.pkl")
                if os.path.exists(vb_path):
                    with open(vb_path, "rb") as f:
                        vdf = pickle.load(f)
                    if len(vdf) > 50:
                        self.vol_bars[symbol] = vdf
                        loaded += 1

            # Load thresholds
            th_path = os.path.join(self._data_dir, "thresholds.json")
            if os.path.exists(th_path):
                with open(th_path) as f:
                    self.vol_thresholds = json.load(f)

            # Load state
            with open(state_path) as f:
                state = json.load(f)

            self.global_locked_dir = state.get("global_locked_dir")
            self.global_sl_streak = state.get("global_sl_streak", {"LONG": 0, "SHORT": 0})
            self.scan_count = state.get("scan_count", 0)
            self.cooldowns = state.get("cooldowns", {})
            self.pair_sl_streaks = state.get("pair_sl_streaks", {})
            self.pair_dir_cooldowns = state.get("pair_dir_cooldowns", {})
            self.vol_bar_counts = state.get("vol_bar_counts", {})

            for p_data in state.get("positions", []):
                pos = Position(
                    symbol=p_data["symbol"], direction=p_data["direction"],
                    entries=p_data["entries"], avg_price=p_data["avg_price"],
                    total_size=p_data["total_size"], hard_sl=p_data["hard_sl"],
                    tp=p_data["tp"], entry_time=p_data.get("entry_time", 0),
                    bars_held=p_data.get("bars_held", 0),
                    entry_atr=p_data.get("entry_atr", 0),
                    max_price=p_data.get("max_price", 0),
                    min_price=p_data.get("min_price", float('inf')),
                    sl_moved=p_data.get("sl_moved", False),
                )
                self.positions.append(pos)

            # Init vol buffers
            for symbol in self.vol_bars:
                self.vol_buffers[symbol] = {
                    "cum_vol": 0, "bar_open": None, "bar_high": None,
                    "bar_low": None, "bar_start": None,
                }

            logger.info(f"  Loaded state: {loaded} pairs, {len(self.positions)} positions, lock={self.global_locked_dir}")
            return loaded > 0

        except Exception as e:
            logger.error(f"Load error: {e}")
            return False

    def initialize(self):
        """Always fetch fresh data. Load only positions/lock from saved state."""
        # Load positions + global lock (but NOT volume bars)
        state_path = os.path.join(self._data_dir, "state.json")
        if os.path.exists(state_path):
            import json
            try:
                with open(state_path) as f:
                    state = json.load(f)
                self.global_locked_dir = state.get("global_locked_dir")
                self.global_sl_streak = state.get("global_sl_streak", {"LONG": 0, "SHORT": 0})
                self.scan_count = state.get("scan_count", 0)
                self.cooldowns = state.get("cooldowns", {})
                self.pair_sl_streaks = state.get("pair_sl_streaks", {})
                self.pair_dir_cooldowns = state.get("pair_dir_cooldowns", {})
                self.vol_bar_counts = state.get("vol_bar_counts", {})
                for p_data in state.get("positions", []):
                    pos = Position(
                        symbol=p_data["symbol"], direction=p_data["direction"],
                        entries=p_data["entries"], avg_price=p_data["avg_price"],
                        total_size=p_data["total_size"], hard_sl=p_data["hard_sl"],
                        tp=p_data["tp"], entry_time=p_data.get("entry_time", 0),
                        bars_held=p_data.get("bars_held", 0),
                        entry_atr=p_data.get("entry_atr", 0),
                        max_price=p_data.get("max_price", 0),
                        min_price=p_data.get("min_price", float('inf')),
                        sl_moved=p_data.get("sl_moved", False),
                    )
                    self.positions.append(pos)
                logger.info(f"  Loaded {len(self.positions)} positions, lock={self.global_locked_dir}")
            except Exception as e:
                logger.error(f"  State load error: {e}")

        logger.info(f"Fetching fresh {WARMUP_DAYS} days of data...")
        total_candles = WARMUP_DAYS * 96

        for symbol in self.pairs:
            logger.info(f"  Fetching {symbol}...")
            df = self.fetcher.fetch_ohlcv_extended(symbol, "15m", total_candles=total_candles)
            if df.empty or len(df) < 500:
                logger.warning(f"  {symbol}: not enough data, skipping")
                continue

            tdf = self.indicators.calculate_all(df.copy())
            tdf = tdf[~tdf.index.duplicated(keep='last')]
            self.time_data[symbol] = tdf

            vdf = resample_to_volume_bars(df)
            if len(vdf) < 200:
                logger.warning(f"  {symbol}: only {len(vdf)} vol bars, skipping")
                continue

            # HTF volume bars (5x threshold)
            warmup = min(1920, len(df))
            htf_threshold = df["volume"].iloc[:warmup].median() * 10
            htf_vdf = resample_to_volume_bars(df, initial_threshold=htf_threshold)
            if len(htf_vdf) > 20:
                htf_vdf = self.indicators.calculate_all(htf_vdf)
                htf_vdf = htf_vdf[~htf_vdf.index.duplicated(keep='last')]
                for col in ["ema_9", "ema_21", "ema_50"]:
                    if col in htf_vdf.columns:
                        vdf[f"htf_{col}"] = htf_vdf[col].reindex(vdf.index, method="ffill")
                import pandas_ta as pta
                st = pta.supertrend(htf_vdf["high"], htf_vdf["low"], htf_vdf["close"], length=9, multiplier=3.0)
                if st is not None:
                    for sc in st.columns:
                        if "SUPERTd" in sc:
                            vdf["htf_supertrend"] = st[sc].reindex(vdf.index, method="ffill")
                self.htf_vol_bars[symbol] = htf_vdf

            vdf = self.indicators.calculate_all(vdf)
            vdf = vdf[~vdf.index.duplicated(keep='last')]
            tdf = tdf[~tdf.index.duplicated(keep='last')]
            time_atr = tdf["atr"].reindex(vdf.index, method="ffill")
            vdf["atr"] = time_atr.values

            self.vol_bars[symbol] = vdf
            self.vol_thresholds[symbol] = df["volume"].iloc[:warmup].median() * 2

            # Init rolling threshold history (last 960 time bars, like backtest)
            self.vol_history[symbol] = list(df["volume"].values[-960:])
            self.time_bar_counts[symbol] = len(df)

            # Init volume buffer for incremental updates
            self.vol_buffers[symbol] = {
                "cum_vol": 0, "bar_open": None, "bar_high": None,
                "bar_low": None, "bar_start": None,
            }

            # Init HTF volume buffer for incremental updates
            if not hasattr(self, '_htf_buffers'):
                self._htf_buffers = {}
            self._htf_buffers[symbol] = {
                "cum_vol": 0, "bar_open": None, "bar_high": None,
                "bar_low": None, "bar_start": None,
            }

            logger.info(f"  {symbol}: {len(vdf)} vol bars, threshold={self.vol_thresholds[symbol]:.0f}")

        logger.info(f"Initialized {len(self.vol_bars)} pairs")

    def update_volume_bars(self, symbol: str) -> bool:
        """Fetch latest 15m bar and update volume bars. Returns True if new vol bar closed."""
        try:
            df_new = self.fetcher.fetch_ohlcv(symbol, "15m", limit=2)
            if df_new.empty:
                return False

            latest = df_new.iloc[-1]
            buf = self.vol_buffers[symbol]

            # Update time data (keep only OHLCV before concat to avoid duplicate column issues)
            if symbol in self.time_data:
                base = self.time_data[symbol][["open", "high", "low", "close", "volume"]]
                base = pd.concat([base, df_new.iloc[[-1]]])
                base = base[~base.index.duplicated(keep='last')]
                self.time_data[symbol] = self.indicators.calculate_all(base.tail(1000))
                self.time_data[symbol] = self.time_data[symbol][~self.time_data[symbol].index.duplicated(keep='last')]

            # Rolling threshold update (every 960 time bars, like backtest)
            if symbol in self.vol_history:
                self.vol_history[symbol].append(float(latest["volume"]))
                if len(self.vol_history[symbol]) > 960:
                    self.vol_history[symbol] = self.vol_history[symbol][-960:]
                self.time_bar_counts[symbol] = self.time_bar_counts.get(symbol, 0) + 1
                if self.time_bar_counts[symbol] % 960 == 0:
                    new_threshold = float(np.median(self.vol_history[symbol])) * 2
                    old_threshold = self.vol_thresholds.get(symbol, 0)
                    self.vol_thresholds[symbol] = new_threshold
                    if abs(new_threshold - old_threshold) / max(old_threshold, 1) > 0.05:
                        logger.info(f"  {symbol} threshold updated: {old_threshold:.0f} → {new_threshold:.0f}")

            if buf["bar_open"] is None:
                buf["bar_open"] = latest["open"]
                buf["bar_high"] = latest["high"]
                buf["bar_low"] = latest["low"]
                buf["bar_start"] = latest.name

            buf["bar_high"] = max(buf["bar_high"], latest["high"])
            buf["bar_low"] = min(buf["bar_low"], latest["low"])
            buf["cum_vol"] += latest["volume"]

            threshold = self.vol_thresholds.get(symbol, 2000)

            if buf["cum_vol"] >= threshold:
                new_bar = pd.DataFrame([{
                    "open": buf["bar_open"],
                    "high": buf["bar_high"],
                    "low": buf["bar_low"],
                    "close": latest["close"],
                    "volume": buf["cum_vol"],
                }], index=[buf["bar_start"]])

                vb_base = self.vol_bars[symbol][["open", "high", "low", "close", "volume"]]
                vb_base = pd.concat([vb_base, new_bar])
                vb_base = vb_base[~vb_base.index.duplicated(keep='last')]
                self.vol_bars[symbol] = self.indicators.calculate_all(vb_base.tail(1000))
                self.vol_bars[symbol] = self.vol_bars[symbol][~self.vol_bars[symbol].index.duplicated(keep='last')]

                # Update ATR from time bars
                if symbol in self.time_data:
                    time_atr = self.time_data[symbol]["atr"].reindex(
                        self.vol_bars[symbol].index, method="ffill"
                    )
                    self.vol_bars[symbol]["atr"] = time_atr.values

                # Re-apply HTF EMA + Supertrend values (lost after OHLCV-only recalculate)
                if symbol in self.htf_vol_bars:
                    for col in ["ema_9", "ema_21", "ema_50"]:
                        if col in self.htf_vol_bars[symbol].columns:
                            self.vol_bars[symbol][f"htf_{col}"] = self.htf_vol_bars[symbol][col].reindex(
                                self.vol_bars[symbol].index, method="ffill")
                    import pandas_ta as pta
                    htf_df = self.htf_vol_bars[symbol]
                    if "high" in htf_df.columns and "low" in htf_df.columns:
                        st = pta.supertrend(htf_df["high"], htf_df["low"], htf_df["close"], length=9, multiplier=3.0)
                        if st is not None:
                            for sc in st.columns:
                                if "SUPERTd" in sc:
                                    self.vol_bars[symbol]["htf_supertrend"] = st[sc].reindex(
                                        self.vol_bars[symbol].index, method="ffill")

                # Update HTF volume bars (5x threshold)
                if symbol in self.htf_vol_bars:
                    htf_threshold = self.vol_thresholds.get(symbol, 2000) * 5
                    htf_buf = getattr(self, '_htf_buffers', {}).get(symbol, {"cum_vol": 0, "bar_open": None, "bar_high": None, "bar_low": None, "bar_start": None})
                    if htf_buf["bar_open"] is None:
                        htf_buf["bar_open"] = buf["bar_open"]
                        htf_buf["bar_high"] = buf["bar_high"]
                        htf_buf["bar_low"] = buf["bar_low"]
                        htf_buf["bar_start"] = buf["bar_start"]
                    htf_buf["bar_high"] = max(htf_buf["bar_high"], float(latest["high"]))
                    htf_buf["bar_low"] = min(htf_buf["bar_low"], float(latest["low"]))
                    htf_buf["cum_vol"] += buf["cum_vol"]
                    if htf_buf["cum_vol"] >= htf_threshold:
                        htf_bar = pd.DataFrame([{
                            "open": htf_buf["bar_open"], "high": htf_buf["bar_high"],
                            "low": htf_buf["bar_low"], "close": float(latest["close"]),
                            "volume": htf_buf["cum_vol"],
                        }], index=[htf_buf["bar_start"]])
                        htf_base = self.htf_vol_bars[symbol][["open", "high", "low", "close", "volume"]]
                        htf_base = pd.concat([htf_base, htf_bar]).tail(200)
                        htf_base = htf_base[~htf_base.index.duplicated(keep='last')]
                        self.htf_vol_bars[symbol] = self.indicators.calculate_all(htf_base)
                        for col in ["ema_9", "ema_21", "ema_50"]:
                            if col in self.htf_vol_bars[symbol].columns:
                                self.vol_bars[symbol][f"htf_{col}"] = self.htf_vol_bars[symbol][col].reindex(self.vol_bars[symbol].index, method="ffill")
                        import pandas_ta as pta
                        htf_df = self.htf_vol_bars[symbol]
                        st = pta.supertrend(htf_df["high"], htf_df["low"], htf_df["close"], length=9, multiplier=3.0)
                        if st is not None:
                            for sc in st.columns:
                                if "SUPERTd" in sc:
                                    self.vol_bars[symbol]["htf_supertrend"] = st[sc].reindex(self.vol_bars[symbol].index, method="ffill")
                        htf_buf = {"cum_vol": 0, "bar_open": None, "bar_high": None, "bar_low": None, "bar_start": None}
                    if not hasattr(self, '_htf_buffers'):
                        self._htf_buffers = {}
                    self._htf_buffers[symbol] = htf_buf

                # Reset buffer
                buf["cum_vol"] = 0
                buf["bar_open"] = None

                logger.debug(f"{symbol}: new volume bar closed")
                return True

            return False

        except Exception as e:
            logger.error(f"Error updating {symbol}: {e}")
            return False

    def check_signal(self, symbol: str) -> Optional[dict]:
        """Check if there's a trading signal on the latest volume bar."""
        vdf = self.vol_bars.get(symbol)
        if vdf is None or len(vdf) < 50:
            return None

        j = len(vdf) - 1
        roc_12 = float(vdf["roc_12"].iloc[j]) if "roc_12" in vdf.columns else 0

        adx_col = vdf["ADX_14"] if "ADX_14" in vdf.columns else None
        if adx_col is not None and isinstance(adx_col, pd.DataFrame):
            adx_col = adx_col.iloc[:, 0]
        adx_raw = float(adx_col.iloc[j]) if adx_col is not None else 0
        rsi_s6_raw = float(vdf["rsi"].diff(6).iloc[j]) if "rsi" in vdf.columns else 0

        if roc_12 > LONG_MOM:
            direction = "LONG"
        elif roc_12 < -SHORT_MOM:
            direction = "SHORT"
        else:
            logger.debug(f"  {symbol}: roc={roc_12:+.2f}% (weak) ADX={adx_raw:.0f}")
            return None

        # HTF Supertrend trend filter (+11.2% vs EMA, stable across 2023-2025)
        htf_st = float(vdf["htf_supertrend"].iloc[j]) if "htf_supertrend" in vdf.columns else 0
        if not np.isnan(htf_st) and htf_st != 0:
            if direction == "LONG" and htf_st < 0:
                logger.debug(f"  {symbol}: {direction} blocked by HTF Supertrend (downtrend) roc={roc_12:+.2f}%")
                return None
            if direction == "SHORT" and htf_st > 0:
                logger.debug(f"  {symbol}: {direction} blocked by HTF Supertrend (uptrend) roc={roc_12:+.2f}%")
                return None

        # ATR expansion filter
        atr = vdf["atr"].values
        atr_ma20 = pd.Series(atr).rolling(20, min_periods=1).mean().values
        atr_exp = atr[j] / atr_ma20[j] if atr_ma20[j] > 0 else 1.0
        if np.isnan(atr_exp) or atr_exp > ATR_EXP_MAX:
            logger.debug(f"  {symbol}: {direction} blocked by ATR_EXP={atr_exp:.2f} roc={roc_12:+.2f}%")
            return None

        # ADX guard
        adx = adx_raw
        if np.isnan(adx) or adx < ADX_MIN:
            logger.debug(f"  {symbol}: {direction} blocked by ADX={adx:.0f} roc={roc_12:+.2f}%")
            return None

        # RSI slope guard
        rsi_s6 = rsi_s6_raw
        if np.isnan(rsi_s6):
            logger.debug(f"  {symbol}: {direction} blocked by RSI=NaN roc={roc_12:+.2f}%")
            return None
        if direction == "LONG" and rsi_s6 < 0:
            logger.debug(f"  {symbol}: {direction} blocked by RSI_slope={rsi_s6:.1f} roc={roc_12:+.2f}%")
            return None
        if direction == "SHORT" and rsi_s6 > 0:
            logger.debug(f"  {symbol}: {direction} blocked by RSI_slope={rsi_s6:+.1f} roc={roc_12:+.2f}%")
            return None

        bar_close_price = float(vdf["close"].iloc[j])
        atr_val = float(atr[j])
        if np.isnan(atr_val) or atr_val <= 0:
            return None

        # Get real-time price from exchange (not bar close)
        try:
            ticker = self.fetcher.exchange.fetch_ticker(symbol)
            live_price = float(ticker["last"])
        except Exception:
            live_price = bar_close_price

        return {
            "symbol": symbol,
            "direction": direction,
            "price": live_price,
            "bar_open": float(vdf["open"].iloc[j]),
            "bar_high": float(vdf["high"].iloc[j]),
            "bar_low": float(vdf["low"].iloc[j]),
            "bar_close": bar_close_price,
            "atr": atr_val,
            "adx": adx,
            "roc_12": roc_12,
        }

    def open_position(self, signal: dict):
        """Open a new position or add DCA entry."""
        symbol = signal["symbol"]
        direction = signal["direction"]
        price = signal["price"]
        atr_val = signal["atr"]

        # Check cooldown (in volume bar units per pair, like backtest)
        pair_vb = self.vol_bar_counts.get(symbol, 0)
        if self.cooldowns.get(symbol, 0) > pair_vb:
            return

        # Per-pair per-direction cooldown (like backtest)
        pair_dir_key = f"{symbol}_{direction}"
        if self.pair_dir_cooldowns.get(pair_dir_key, 0) > pair_vb:
            return

        # Flip: if opposite direction signal, close existing and open new
        existing = [p for p in self.positions if p.symbol == symbol]
        if existing:
            ex = existing[0]
            if ex.direction == direction:
                return  # same direction, DCA handled in _try_dca
            # Flip: close opposite position
            try:
                ticker = self.fetcher.exchange.fetch_ticker(symbol)
                flip_price = float(ticker["last"])
            except Exception:
                flip_price = price
            if ex.direction == "LONG":
                flip_pnl = (flip_price - ex.avg_price) / ex.avg_price * 100 * ex.total_size
            else:
                flip_pnl = (ex.avg_price - flip_price) / ex.avg_price * 100 * ex.total_size
            logger.info(f"  FLIP {ex.direction}→{direction} {symbol} @ {_pfmt(flip_price)} | PnL {flip_pnl:+.2f}%")

            # Update global lock / pair cooldown
            pair_vb = self.vol_bar_counts.get(symbol, 0)
            pair_dir_key_ex = f"{symbol}_{ex.direction}"
            if flip_pnl < 0:
                self.pair_sl_streaks[pair_dir_key_ex] = self.pair_sl_streaks.get(pair_dir_key_ex, 0) + 1
                if self.pair_sl_streaks[pair_dir_key_ex] >= self.PAIR_COOLDOWN_SL:
                    self.pair_dir_cooldowns[pair_dir_key_ex] = pair_vb + self.PAIR_COOLDOWN_BARS
            else:
                self.pair_sl_streaks[pair_dir_key_ex] = 0

            self._log_trade_close(ex, flip_price, "FLIP", flip_pnl)
            self.positions.remove(ex)
            self.cooldowns[symbol] = pair_vb + COOLDOWN_BARS

        # Max open check for NEW positions only
        if len(self.positions) >= MAX_OPEN:
            return

        # New position
        if direction == "LONG":
            hard_sl = price - SL_MULT * atr_val
            tp = price + TP_MULT * atr_val
        else:
            hard_sl = price + SL_MULT * atr_val
            tp = price - TP_MULT * atr_val

        pos = Position(
            symbol=symbol, direction=direction,
            entries=[(price, time.time())],
            avg_price=price, total_size=1,
            hard_sl=hard_sl, tp=tp,
            entry_time=time.time(),
            entry_atr=atr_val,
            max_price=price,
            min_price=price,
            sl_moved=False,
        )
        self.positions.append(pos)

        logger.info(f"  OPEN {direction} {symbol} @ {_pfmt(price)} | SL={_pfmt(hard_sl)} TP={_pfmt(tp)} ATR={_pfmt(atr_val)}")
        self._log_trade_open(signal, pos)

    def _log_dca_entry(self, pos: Position, price: float, signal: dict):
        """Log DCA entry to trades.log + update existing OPEN in paper_trades.json."""
        import json
        from datetime import datetime
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # trades.log
        line = f"{now} DCA #{pos.total_size} {pos.direction} {pos.symbol} @ {_pfmt(price)} avg={_pfmt(pos.avg_price)} TP={_pfmt(pos.tp)}\n"
        with open(self._trades_log, "a") as f:
            f.write(line)

        # paper_trades.json — update existing OPEN object
        trades = []
        if os.path.exists(self._paper_trades_file):
            try:
                with open(self._paper_trades_file) as f:
                    trades = json.load(f)
            except Exception:
                trades = []
        for t in reversed(trades):
            if t.get("symbol") == pos.symbol and t.get("status") == "OPEN":
                if "dca" not in t:
                    t["dca"] = []
                t["dca"].append({
                    "number": pos.total_size,
                    "price": _prnd(price),
                    "time": now,
                })
                t["avg_price"] = _prnd(pos.avg_price)
                t["tp"] = _prnd(pos.tp)
                t["dca_count"] = pos.total_size
                break
        with open(self._paper_trades_file, "w") as f:
            json.dump(trades, f, indent=2)

    def _log_trade_open(self, signal: dict, pos: Position):
        """Log trade open to trades.log + paper_trades.json."""
        import json
        from datetime import datetime
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # trades.log
        line = (
            f"{now} OPEN {pos.direction} {pos.symbol} "
            f"@ {_pfmt(pos.avg_price)} SL={_pfmt(pos.hard_sl)} TP={_pfmt(pos.tp)} "
            f"O={_pfmt(signal.get('bar_open',0))} H={_pfmt(signal.get('bar_high',0))} "
            f"L={_pfmt(signal.get('bar_low',0))} C={_pfmt(signal.get('bar_close',0))} "
            f"ATR={_pfmt(signal['atr'])} ADX={signal.get('adx',0):.0f} roc={signal.get('roc_12',0):.2f}%\n"
        )
        with open(self._trades_log, "a") as f:
            f.write(line)

        # paper_trades.json
        p = pos.avg_price
        trade_record = {
            "symbol": pos.symbol,
            "direction": pos.direction,
            "status": "OPEN",
            "entry_price": _prnd(p),
            "sl": _prnd(pos.hard_sl),
            "tp": _prnd(pos.tp),
            "atr": _prnd(signal["atr"]),
            "adx": round(signal.get("adx", 0), 1),
            "roc_12": round(signal.get("roc_12", 0), 2),
            "bar": {
                "open": _prnd(signal.get("bar_open", 0)),
                "high": _prnd(signal.get("bar_high", 0)),
                "low": _prnd(signal.get("bar_low", 0)),
                "close": _prnd(signal.get("bar_close", 0)),
            },
            "open_time": now,
            "dca_count": 1,
        }
        trades = []
        if os.path.exists(self._paper_trades_file):
            try:
                with open(self._paper_trades_file) as f:
                    trades = json.load(f)
            except Exception:
                trades = []
        trades.append(trade_record)
        with open(self._paper_trades_file, "w") as f:
            json.dump(trades, f, indent=2)

    def _log_sl_moved(self, pos: Position):
        """Log SL move event to trades.log + update OPEN in paper_trades.json."""
        import json
        from datetime import datetime
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        line = f"{now} SL_MOVED {pos.direction} {pos.symbol} new_sl={_pfmt(pos.hard_sl)} entry={_pfmt(pos.avg_price)} ATR={_pfmt(pos.entry_atr)}\n"
        with open(self._trades_log, "a") as f:
            f.write(line)

        trades = []
        if os.path.exists(self._paper_trades_file):
            try:
                with open(self._paper_trades_file) as f:
                    trades = json.load(f)
            except Exception:
                trades = []
        for t in reversed(trades):
            if t.get("symbol") == pos.symbol and t.get("status") == "OPEN":
                t["sl"] = _prnd(pos.hard_sl)
                t["sl_moved"] = True
                t["sl_moved_time"] = now
                break
        with open(self._paper_trades_file, "w") as f:
            json.dump(trades, f, indent=2)

    def _log_trade_close(self, pos: Position, exit_price: float, exit_reason: str, pnl_pct: float):
        """Log trade close to trades.log + append to paper_trades.json."""
        import json
        from datetime import datetime

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # trades.log
        if pos.direction == "LONG":
            mfe = (pos.max_price - pos.avg_price) / pos.avg_price * 100
            mae = (pos.avg_price - pos.min_price) / pos.avg_price * 100
        else:
            mfe = (pos.avg_price - pos.min_price) / pos.avg_price * 100
            mae = (pos.max_price - pos.avg_price) / pos.avg_price * 100
        line = (f"{now} CLOSE {pos.direction} {pos.symbol} @ {_pfmt(exit_price)} | {exit_reason} | "
                f"PnL {pnl_pct:+.2f}% | DCA:{pos.total_size} | Bars:{pos.bars_held} | "
                f"MFE:{mfe:+.2f}% MAE:{mae:.2f}% High:{_pfmt(pos.max_price)} Low:{_pfmt(pos.min_price)}\n")
        with open(self._trades_log, "a") as f:
            f.write(line)

        # paper_trades.json — update existing OPEN object
        trades = []
        if os.path.exists(self._paper_trades_file):
            try:
                with open(self._paper_trades_file) as f:
                    trades = json.load(f)
            except Exception:
                trades = []
        for t in reversed(trades):
            if t.get("symbol") == pos.symbol and t.get("status") == "OPEN":
                t["status"] = "CLOSED"
                t["avg_price"] = _prnd(pos.avg_price)
                t["exit_price"] = _prnd(exit_price)
                t["pnl_pct"] = round(pnl_pct, 2)
                t["exit_reason"] = exit_reason
                t["dca_count"] = pos.total_size
                t["bars_held"] = pos.bars_held
                t["max_price"] = _prnd(pos.max_price)
                t["min_price"] = _prnd(pos.min_price)
                t["mfe_pct"] = round(mfe, 2)
                t["mae_pct"] = round(mae, 2)
                t["close_time"] = now
                break
        with open(self._paper_trades_file, "w") as f:
            json.dump(trades, f, indent=2)

    def _try_dca(self, symbol: str):
        """Check DCA for existing position — independent of signal guards (like backtest).
        Backtest DCA only needs: price at level + roc >= 0 + ADX max entries.
        """
        with self._lock:
            existing = [p for p in self.positions if p.symbol == symbol]
            if not existing:
                return
            pos = existing[0]

        vdf = self.vol_bars.get(symbol)
        if vdf is None or len(vdf) < 20:
            return

        j = len(vdf) - 1
        roc_val = float(vdf["roc_12"].iloc[j]) if "roc_12" in vdf.columns else 0
        if pos.direction == "LONG" and roc_val < 0:
            return
        if pos.direction == "SHORT" and roc_val > 0:
            return

        adx_col = vdf["ADX_14"] if "ADX_14" in vdf.columns else None
        if adx_col is not None:
            if isinstance(adx_col, pd.DataFrame):
                adx_col = adx_col.iloc[:, 0]
            adx_val = float(adx_col.iloc[j])
        else:
            adx_val = 25.0

        if adx_val >= 30:
            dyn_max = 3
        elif adx_val >= 20:
            dyn_max = 2
        else:
            dyn_max = 1

        with self._lock:
            if pos.total_size >= min(MAX_DCA, dyn_max):
                return

            ea = pos.entry_atr if pos.entry_atr > 0 else float(vdf["atr"].iloc[j]) if "atr" in vdf.columns else 0
            if ea <= 0:
                return

            atr_arr = vdf["atr"].values if "atr" in vdf.columns else np.array([ea])
            atr_ma20 = pd.Series(atr_arr).rolling(20, min_periods=1).mean().values
            cur_atr = atr_arr[-1] if len(atr_arr) > 0 else ea
            atr_exp = cur_atr / atr_ma20[-1] if len(atr_ma20) > 0 and atr_ma20[-1] > 0 else 1.0
            vol_scale = max(1.0, atr_exp)
            effective_step = DCA_STEP_MULT * vol_scale

            dca_level = pos.entries[0][0] - pos.total_size * effective_step * ea if pos.direction == "LONG" \
                else pos.entries[0][0] + pos.total_size * effective_step * ea

            current_low = float(vdf["low"].iloc[j])
            current_high = float(vdf["high"].iloc[j])
            try:
                ticker = self.fetcher.exchange.fetch_ticker(symbol)
                price = float(ticker["last"])
            except Exception:
                price = float(vdf["close"].iloc[j])

            triggered = (pos.direction == "LONG" and current_low <= dca_level) or \
                        (pos.direction == "SHORT" and current_high >= dca_level)

            if triggered:
                pos.entries.append((price, time.time()))
                pos.total_size += 1
                pos.avg_price = sum(e[0] for e in pos.entries) / pos.total_size
                pos.tp = pos.avg_price + TP_MULT * ea if pos.direction == "LONG" else pos.avg_price - TP_MULT * ea
                logger.info(f"  DCA #{pos.total_size} {symbol} {pos.direction} @ {_pfmt(price)} (avg: {_pfmt(pos.avg_price)}) vol_scale={vol_scale:.2f}")
                signal = {"atr": ea, "adx": adx_val, "roc_12": roc_val}
                self._log_dca_entry(pos, price, signal)

    def _process_pair(self, symbol: str):
        """Fully independent pair processing: scan + signal + open + close. Runs in own thread."""
        try:
            # 1. Update volume bars (pair-specific data, no lock needed)
            new_bar = self.update_volume_bars(symbol)
            if not new_bar:
                # Still check positions (live SL/TP) but DON'T increment bars_held
                self._check_pair_positions(symbol, new_bar_closed=False)
                return

            # Track per-pair volume bar count (for cooldowns in vol bar units)
            self.vol_bar_counts[symbol] = self.vol_bar_counts.get(symbol, 0) + 1

            # 2. Check DCA for existing positions (independent of signal guards, like backtest)
            self._try_dca(symbol)

            # 3. Check signal for NEW positions
            signal = self.check_signal(symbol)

            if signal:
                logger.info(f"  SIGNAL: {signal['direction']} {signal['symbol']} @ {_pfmt(signal['price'])} roc={signal['roc_12']:.2f}% ADX={signal['adx']:.0f}")
                with self._lock:
                    self.open_position(signal)

            # 4. Check positions for this pair (new bar = increment bars_held)
            self._check_pair_positions(symbol, new_bar_closed=True)

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

    def _check_pair_positions(self, symbol: str, new_bar_closed: bool = True):
        """Check SL/TP/timeout for positions of this specific pair."""
        with self._lock:
            for pos in list(self.positions):
                if pos.symbol != symbol:
                    continue

                vdf = self.vol_bars.get(pos.symbol)
                if vdf is None or len(vdf) < 2:
                    continue

                try:
                    ticker = self.fetcher.exchange.fetch_ticker(pos.symbol)
                    current_price = float(ticker["last"])
                except Exception:
                    current_price = float(vdf["close"].iloc[-1])
                current_high = float(vdf["high"].iloc[-1])
                current_low = float(vdf["low"].iloc[-1])

                # Track MFE/MAE
                pos.max_price = max(pos.max_price, current_high, current_price)
                pos.min_price = min(pos.min_price, current_low, current_price)

                # Only count volume bars, not scans (like backtest)
                if new_bar_closed:
                    pos.bars_held += 1

                # Volatility-scaled dynamic hard SL (like backtest)
                ea = pos.entry_atr if pos.entry_atr > 0 else float(vdf["atr"].iloc[-1]) if "atr" in vdf.columns else 0
                atr_arr = vdf["atr"].values if "atr" in vdf.columns else np.array([ea])
                atr_ma20 = pd.Series(atr_arr).rolling(20, min_periods=1).mean().values
                cur_atr = atr_arr[-1] if len(atr_arr) > 0 else ea
                atr_exp = cur_atr / atr_ma20[-1] if len(atr_ma20) > 0 and atr_ma20[-1] > 0 else 1.0
                vol_scale = max(1.0, atr_exp)
                effective_hard_sl_dist = SL_MULT * vol_scale * ea

                if pos.direction == "LONG":
                    dynamic_hard_sl = pos.entries[0][0] - effective_hard_sl_dist
                    effective_sl = min(pos.hard_sl, dynamic_hard_sl)
                else:
                    dynamic_hard_sl = pos.entries[0][0] + effective_hard_sl_dist
                    effective_sl = max(pos.hard_sl, dynamic_hard_sl)

                hit_tp = hit_sl = False
                if pos.direction == "LONG":
                    hit_tp = current_high >= pos.tp
                    hit_sl = current_low <= effective_sl
                else:
                    hit_tp = current_low <= pos.tp
                    hit_sl = current_high >= effective_sl

                # Move SL: when profit >= 2.0 ATR → move SL to entry + 1.0 ATR
                if not pos.sl_moved and ea > 0:
                    if pos.direction == "LONG":
                        profit = current_high - pos.avg_price
                        if profit >= MOVE_SL_AT * ea:
                            new_sl = pos.avg_price + MOVE_SL_TO * ea
                            pos.hard_sl = new_sl
                            pos.sl_moved = True
                            logger.info(f"  SL_MOVED {pos.symbol} {pos.direction} → {_pfmt(new_sl)} (locked +{MOVE_SL_TO}x ATR)")
                            self._log_sl_moved(pos)
                    else:
                        profit = pos.avg_price - current_low
                        if profit >= MOVE_SL_AT * ea:
                            new_sl = pos.avg_price - MOVE_SL_TO * ea
                            pos.hard_sl = new_sl
                            pos.sl_moved = True
                            logger.info(f"  SL_MOVED {pos.symbol} {pos.direction} → {_pfmt(new_sl)} (locked +{MOVE_SL_TO}x ATR)")
                            self._log_sl_moved(pos)

                # Volume Drop early exit (like backtest)
                vol_drop_exit = False
                if pos.bars_held >= 3 and "volume" in vdf.columns:
                    j = len(vdf) - 1
                    vol_vals = vdf["volume"].values
                    vol_ma20_v = pd.Series(vol_vals).rolling(20, min_periods=1).mean().values
                    if j >= 2 and vol_ma20_v[j] > 0:
                        vol_avg3 = vol_vals[j-2:j+1].mean()
                        if vol_avg3 < vol_ma20_v[j] * 0.5:
                            vol_drop_exit = True

                exit_reason = None
                exit_price = current_price

                if hit_sl and hit_tp:
                    exit_reason = "HARD_SL"
                    exit_price = pos.hard_sl
                elif hit_sl:
                    exit_reason = "HARD_SL"
                    exit_price = pos.hard_sl
                elif hit_tp:
                    exit_reason = "TP"
                    exit_price = pos.tp
                elif vol_drop_exit:
                    exit_reason = "VOL_DROP"
                    exit_price = current_price
                elif pos.bars_held >= 24:
                    exit_reason = "TIMEOUT"
                    exit_price = current_price

                if exit_reason:
                    if pos.direction == "LONG":
                        pnl_pct = (exit_price - pos.avg_price) / pos.avg_price * 100 * pos.total_size
                    else:
                        pnl_pct = (pos.avg_price - exit_price) / pos.avg_price * 100 * pos.total_size

                    logger.info(
                        f"  CLOSE {pos.direction} {pos.symbol} @ {_pfmt(exit_price)} "
                        f"| {exit_reason} | PnL {pnl_pct:+.2f}% | Bars: {pos.bars_held} | DCA: {pos.total_size}"
                    )

                    # Per-pair per-direction SL cooldown (in volume bar units)
                    pair_vb = self.vol_bar_counts.get(pos.symbol, 0)
                    pair_dir_key = f"{pos.symbol}_{pos.direction}"
                    if exit_reason == "HARD_SL":
                        self.pair_sl_streaks[pair_dir_key] = self.pair_sl_streaks.get(pair_dir_key, 0) + 1
                        if self.pair_sl_streaks[pair_dir_key] >= self.PAIR_COOLDOWN_SL:
                            self.pair_dir_cooldowns[pair_dir_key] = pair_vb + self.PAIR_COOLDOWN_BARS
                            logger.info(f"  PAIR_COOLDOWN {pos.symbol} {pos.direction} for {self.PAIR_COOLDOWN_BARS} vol bars")
                    else:
                        self.pair_sl_streaks[pair_dir_key] = 0

                    self._log_trade_close(pos, exit_price, exit_reason, pnl_pct)
                    self.positions.remove(pos)
                    self.cooldowns[pos.symbol] = pair_vb + COOLDOWN_BARS

    def scan(self):
        """Run one scan cycle — each pair fully independent in its own thread."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        self.scan_count += 1
        logger.info(f"\n{'='*50}")
        logger.info(f"Scan #{self.scan_count} | Positions: {len(self.positions)}")

        symbols = list(self.vol_bars.keys())

        with ThreadPoolExecutor(max_workers=len(symbols)) as pool:
            futures = {pool.submit(self._process_pair, sym): sym for sym in symbols}
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error {sym}: {e}")

        # Status
        for pos in self.positions:
            vdf = self.vol_bars.get(pos.symbol)
            if vdf is not None:
                current = float(vdf["close"].iloc[-1])
                if pos.direction == "LONG":
                    unrealized = (current - pos.avg_price) / pos.avg_price * 100
                else:
                    unrealized = (pos.avg_price - current) / pos.avg_price * 100
                logger.info(
                    f"  HOLDING: {pos.direction} {pos.symbol} entry={_pfmt(pos.avg_price)} "
                    f"now={_pfmt(current)} PnL={unrealized:+.2f}% bars={pos.bars_held} dca={pos.total_size} "
                    f"high={_pfmt(pos.max_price)} low={_pfmt(pos.min_price)}"
                    f"{' SL_MOVED' if pos.sl_moved else ''}"
                )

        logger.info(f"  Positions: {len(self.positions)} | Total pairs: {len(self.vol_bars)}")

        # Scan summary: why each pair has no signal
        held_symbols = {p.symbol for p in self.positions}
        summaries = []
        for sym in sorted(self.vol_bars.keys()):
            if sym in held_symbols:
                continue
            vdf = self.vol_bars[sym]
            if len(vdf) < 50:
                continue
            j = len(vdf) - 1
            roc = float(vdf["roc_12"].iloc[j]) if "roc_12" in vdf.columns else 0
            adx_c = vdf["ADX_14"] if "ADX_14" in vdf.columns else None
            if adx_c is not None and isinstance(adx_c, pd.DataFrame):
                adx_c = adx_c.iloc[:, 0]
            adx_v = float(adx_c.iloc[j]) if adx_c is not None else 0
            rsi_s = float(vdf["rsi"].diff(6).iloc[j]) if "rsi" in vdf.columns else 0
            short_name = sym.replace("/USDT", "")

            if -SHORT_MOM <= roc <= LONG_MOM:
                summaries.append(f"{short_name}:roc={roc:+.1f}%")
            elif adx_v < ADX_MIN:
                summaries.append(f"{short_name}:ADX={adx_v:.0f}")
            elif (roc > 0 and rsi_s < 0) or (roc < 0 and rsi_s > 0):
                summaries.append(f"{short_name}:RSI={rsi_s:+.0f}")
            else:
                summaries.append(f"{short_name}:HTF")

        if summaries:
            logger.info(f"  Skip: {' | '.join(summaries)}")

        # Save state to disk after every scan
        self._save_state()

    def _sl_monitor(self):
        """Daemon thread: check move SL activation + SL hit every 60s."""
        while True:
            try:
                time.sleep(MOVE_SL_CHECK)
                with self._lock:
                    active_positions = list(self.positions)
                if not active_positions:
                    continue

                for pos in active_positions:
                    try:
                        ticker = self.fetcher.exchange.fetch_ticker(pos.symbol)
                        price = float(ticker["last"])
                    except Exception:
                        continue

                    ea = pos.entry_atr if pos.entry_atr > 0 else 0
                    if ea <= 0:
                        continue

                    with self._lock:
                        if pos not in self.positions:
                            continue

                        pos.max_price = max(pos.max_price, price)
                        pos.min_price = min(pos.min_price, price)

                        # Move SL if not yet moved (use max/min price like backtest uses high/low)
                        if not pos.sl_moved:
                            if pos.direction == "LONG" and pos.max_price - pos.avg_price >= MOVE_SL_AT * ea:
                                pos.hard_sl = pos.avg_price + MOVE_SL_TO * ea
                                pos.sl_moved = True
                                logger.info(f"  SL_MOVED {pos.symbol} {pos.direction} → {_pfmt(pos.hard_sl)} (60s check)")
                                self._log_sl_moved(pos)
                            elif pos.direction == "SHORT" and pos.avg_price - pos.min_price >= MOVE_SL_AT * ea:
                                pos.hard_sl = pos.avg_price - MOVE_SL_TO * ea
                                pos.sl_moved = True
                                logger.info(f"  SL_MOVED {pos.symbol} {pos.direction} → {_pfmt(pos.hard_sl)} (60s check)")
                                self._log_sl_moved(pos)

                        # Check if moved SL is hit
                        if pos.sl_moved:
                            sl_hit = False
                            if pos.direction == "LONG" and price <= pos.hard_sl:
                                pnl_pct = (pos.hard_sl - pos.avg_price) / pos.avg_price * 100 * pos.total_size
                                sl_hit = True
                            elif pos.direction == "SHORT" and price >= pos.hard_sl:
                                pnl_pct = (pos.avg_price - pos.hard_sl) / pos.avg_price * 100 * pos.total_size
                                sl_hit = True

                            if sl_hit:
                                logger.info(f"  SL_HIT {pos.direction} {pos.symbol} @ {_pfmt(pos.hard_sl)} | PnL {pnl_pct:+.2f}% (60s check)")
                                self._log_trade_close(pos, pos.hard_sl, "SL_MOVED", pnl_pct)
                                self.positions.remove(pos)
                                pair_vb = self.vol_bar_counts.get(pos.symbol, 0)
                                self.cooldowns[pos.symbol] = pair_vb + COOLDOWN_BARS

                                pair_dir_key = f"{pos.symbol}_{pos.direction}"
                                if pnl_pct < 0:
                                    self.pair_sl_streaks[pair_dir_key] = self.pair_sl_streaks.get(pair_dir_key, 0) + 1
                                    if self.pair_sl_streaks[pair_dir_key] >= self.PAIR_COOLDOWN_SL:
                                        self.pair_dir_cooldowns[pair_dir_key] = pair_vb + self.PAIR_COOLDOWN_BARS
                                else:
                                    self.pair_sl_streaks[pair_dir_key] = 0
            except Exception as e:
                logger.error(f"SL monitor error: {e}")
                time.sleep(10)

    @staticmethod
    def _archive_logs():
        """Move old logs to archive folder on startup."""
        import shutil
        from datetime import datetime

        archive_dir = "logs_archive"
        os.makedirs(archive_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for fname in ("volbars_bot.log", "trades.log", "paper_trades.json"):
            if os.path.exists(fname) and os.path.getsize(fname) > 0:
                dest = os.path.join(archive_dir, f"{stamp}_{fname}")
                shutil.copy2(fname, dest)
                with open(fname, "w"):
                    pass

    def run(self, once: bool = False):
        """Main loop."""
        self._archive_logs()
        logger.info(f"Volume Bars Bot v{VERSION}")
        logger.info(f"Mode: {'PAPER' if self.paper else 'LIVE'}")
        logger.info(f"Params: SL={SL_MULT}x TP={TP_MULT}x ADX>{ADX_MIN} MaxOpen={MAX_OPEN} DCA={MAX_DCA}")
        logger.info(f"Pairs: {len(self.pairs)}")

        self.initialize()

        if once:
            self.scan()
            return

        # Start SL monitor thread (checks move SL + SL hit every 60s)
        import threading
        sl_thread = threading.Thread(target=self._sl_monitor, daemon=True)
        sl_thread.start()
        logger.info(f"SL monitor started (check every {MOVE_SL_CHECK}s, move at +{MOVE_SL_AT}x ATR to +{MOVE_SL_TO}x ATR)")

        logger.info(f"Starting scan loop (aligned to 15m bar close)...")
        while True:
            try:
                self.scan()

                # Wait until next 15m bar close (:00, :15, :30, :45) + 5 sec buffer
                from datetime import datetime, timedelta
                now = datetime.utcnow()
                minutes = now.minute
                next_bar = 15 - (minutes % 15)
                if next_bar == 0:
                    next_bar = 15
                wait_until = now.replace(second=0, microsecond=0) + timedelta(minutes=next_bar, seconds=5)
                wait_secs = max(10, (wait_until - datetime.utcnow()).total_seconds())
                logger.info(f"  Next scan at {wait_until.strftime('%H:%M:%S')} UTC ({wait_secs:.0f}s)")
                time.sleep(wait_secs)
            except KeyboardInterrupt:
                logger.info("Stopping bot...")
                break
            except Exception as e:
                logger.error(f"Scan error: {e}")
                time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Volume Bars Trading Bot")
    parser.add_argument("--once", action="store_true", help="Single scan then exit")
    parser.add_argument("--live", action="store_true", help="Real trading (default: paper)")
    args = parser.parse_args()

    bot = VolumeBarsBot(paper=not args.live)
    bot.run(once=args.once)
