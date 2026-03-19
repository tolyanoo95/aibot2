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
import json
import logging
import os
import pickle
import queue
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pandas_ta as pta

from src.config import config
from src.data_fetcher import BinanceDataFetcher
from src.indicators import TechnicalIndicators
from backtest_volbars import resample_to_volume_bars

logger = logging.getLogger("volbars")
logger.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
_sh.setLevel(logging.INFO)
logger.addHandler(_sh)
_fh = logging.FileHandler("volbars_bot.log")
_fh.setFormatter(_fmt)
_fh.setLevel(logging.INFO)
logger.addHandler(_fh)

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
MOVE_SL_STEPS = [(0.25, 0.30), (0.5, 0.25), (1.0, 0.5), (2.0, 1.0)]  # 4-step: scalp at +0.25, then progressive (no Step0 BE)
MOVE_SL_TRAIL = 1.0   # after all steps: trail SL at this distance from best price (ATR)
MOVE_SL_CHECK = 5     # check every 5 seconds (132 req/min, limit 2400)
MAKER_FEE = 0.0002    # 0.02% maker (limit orders)
TAKER_FEE = 0.0005    # 0.05% taker (market orders)
WARMUP_DAYS = 60
SCAN_INTERVAL = 900  # 15 minutes


def _safe_reindex(source: pd.Series, target_idx: pd.Index) -> pd.Series:
    """Reindex with ffill, sorting both source and target to avoid monotonic errors."""
    try:
        return source.sort_index().reindex(target_idx.sort_values(), method="ffill")
    except ValueError:
        return pd.Series(np.nan, index=target_idx)


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
    sl_step: int = 0
    best_price: float = 0.0


class VolumeBarsBot:
    def __init__(self, paper: bool = True):
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

        self._lock = threading.Lock()
        self._file_lock = threading.Lock()
        self._msg_queue = queue.Queue()
        self._data_dir = "data/volbars"
        os.makedirs(self._data_dir, exist_ok=True)
        self._trades_log = "trades.log"
        self._paper_trades_file = "paper_trades.json"

    def _write_trades_log(self, line: str):
        """Thread-safe append to trades.log."""
        with self._file_lock:
            with open(self._trades_log, "a") as f:
                f.write(line)

    def _write_paper_trades(self, trades_data):
        """Thread-safe write to paper_trades.json."""
        with self._file_lock:
            with open(self._paper_trades_file, "w") as f:
                json.dump(trades_data, f, indent=2)

    def _read_paper_trades(self) -> list:
        """Thread-safe read from paper_trades.json."""
        with self._file_lock:
            if not os.path.exists(self._paper_trades_file):
                return []
            try:
                with open(self._paper_trades_file) as f:
                    return json.load(f)
            except Exception:
                return []

    def _save_state(self):
        """Save state to disk. Uses _file_lock only (no _lock — caller may hold it)."""
        try:
            pos_list = list(self.positions)
            state = {
                "global_locked_dir": self.global_locked_dir,
                "global_sl_streak": self.global_sl_streak,
                "scan_count": self.scan_count,
                "cooldowns": dict(self.cooldowns),
                "pair_sl_streaks": dict(self.pair_sl_streaks),
                "pair_dir_cooldowns": dict(self.pair_dir_cooldowns),
                "vol_bar_counts": dict(self.vol_bar_counts),
                "positions": [
                    {
                        "symbol": p.symbol, "direction": p.direction,
                        "entries": list(p.entries), "avg_price": p.avg_price,
                        "total_size": p.total_size, "hard_sl": p.hard_sl,
                        "tp": p.tp, "entry_time": p.entry_time, "bars_held": p.bars_held,
                        "entry_atr": p.entry_atr, "max_price": p.max_price, "min_price": p.min_price,
                        "sl_moved": p.sl_moved, "sl_step": p.sl_step, "best_price": p.best_price,
                    } for p in pos_list
                ],
                "vol_buffers": {k: {kk: (str(vv) if isinstance(vv, pd.Timestamp) else vv)
                                    for kk, vv in v.items()} for k, v in self.vol_buffers.items()},
            }
            with self._file_lock:
                for symbol, vdf in self.vol_bars.items():
                    fname = symbol.replace("/", "_").replace(":", "_")
                    with open(os.path.join(self._data_dir, f"{fname}_volbars.pkl"), "wb") as f:
                        pickle.dump(vdf, f)
                with open(os.path.join(self._data_dir, "thresholds.json"), "w") as f:
                    json.dump(dict(self.vol_thresholds), f)
                with open(os.path.join(self._data_dir, "state.json"), "w") as f:
                    json.dump(state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Save error: {e}", exc_info=True)

    # _load_state removed — initialize() handles state loading

    def initialize(self):
        """Always fetch fresh data. Load only positions/lock from saved state."""
        # Load positions + global lock (but NOT volume bars)
        state_path = os.path.join(self._data_dir, "state.json")
        if os.path.exists(state_path):
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
                        sl_step=p_data.get("sl_step", 0),
                        best_price=p_data.get("best_price", p_data.get("max_price", 0)),
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
                        vdf[f"htf_{col}"] = _safe_reindex(htf_vdf[col], vdf.index)
                st = pta.supertrend(htf_vdf["high"], htf_vdf["low"], htf_vdf["close"], length=9, multiplier=3.0)
                if st is not None:
                    for sc in st.columns:
                        if "SUPERTd" in sc:
                            vdf["htf_supertrend"] = _safe_reindex(st[sc], vdf.index)
                self.htf_vol_bars[symbol] = htf_vdf

            vdf = self.indicators.calculate_all(vdf)
            vdf = vdf[~vdf.index.duplicated(keep='last')]
            tdf = tdf[~tdf.index.duplicated(keep='last')]
            time_atr = _safe_reindex(tdf["atr"], vdf.index)
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

    def _complete_volume_bar(self, symbol: str, bar_open, bar_high, bar_low, bar_close, bar_vol, bar_start):
        """Single method for volume bar completion: indicators, HTF, ATR, signals.
        Called from both _process_kline_1m (WS) and update_volume_bars (fallback)."""
        new_bar = pd.DataFrame([{
            "open": bar_open, "high": bar_high, "low": bar_low,
            "close": bar_close, "volume": bar_vol,
        }], index=[bar_start])

        with self._lock:
            vb_base = self.vol_bars[symbol][["open", "high", "low", "close", "volume"]]
            vb_base = pd.concat([vb_base, new_bar])
            vb_base = vb_base[~vb_base.index.duplicated(keep='last')].sort_index()
            self.vol_bars[symbol] = self.indicators.calculate_all(vb_base.tail(1000))
            self.vol_bars[symbol] = self.vol_bars[symbol][~self.vol_bars[symbol].index.duplicated(keep='last')].sort_index()

            if symbol in self.time_data:
                vb_idx = self.vol_bars[symbol].index
                td_idx = self.time_data[symbol].index
                if len(vb_idx) > 0 and len(td_idx) > 0:
                    time_atr = _safe_reindex(self.time_data[symbol]["atr"], vb_idx)
                    self.vol_bars[symbol]["atr"] = time_atr.values

            self._apply_htf_supertrend(symbol)
            self._update_htf_bar(symbol, bar_open, bar_high, bar_low, bar_close, bar_vol, bar_start)

        self.vol_bar_counts[symbol] = self.vol_bar_counts.get(symbol, 0) + 1
        logger.info(f"  RT_BAR {symbol} vol={bar_vol:.0f} threshold={self.vol_thresholds.get(symbol,0):.0f}")

        # Log full bar data for offline simulation
        try:
            vdf = self.vol_bars.get(symbol)
            if vdf is not None and len(vdf) > 0:
                j = len(vdf) - 1
                roc = float(vdf["roc_12"].iloc[j]) if "roc_12" in vdf.columns else 0
                adx = float(vdf["ADX_14"].iloc[j]) if "ADX_14" in vdf.columns else 0
                rsi = float(vdf["rsi"].iloc[j]) if "rsi" in vdf.columns else 0
                rsi_s = float(vdf["rsi"].diff(6).iloc[j]) if "rsi" in vdf.columns else 0
                atr_v = float(vdf["atr"].iloc[j]) if "atr" in vdf.columns else 0
                atr_ma = pd.Series(vdf["atr"].values).rolling(20, min_periods=1).mean().values
                atr_exp = atr_v / atr_ma[j] if atr_ma[j] > 0 else 1.0
                htf = float(vdf["htf_supertrend"].iloc[j]) if "htf_supertrend" in vdf.columns else 0
                htf_dir = "UP" if htf > 0 else ("DN" if htf < 0 else "?")
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                line = (f"{now} BAR_DATA {symbol} O={_pfmt(bar_open)} H={_pfmt(bar_high)} L={_pfmt(bar_low)} "
                        f"C={_pfmt(bar_close)} V={bar_vol:.0f} ATR={atr_v:.6f} ROC={roc:+.2f}% ADX={adx:.0f} "
                        f"RSI={rsi:.0f} RSI_s={rsi_s:+.1f} ATR_EXP={atr_exp:.2f} HTF={htf_dir}\n")
                with self._file_lock:
                    with open("bar_data.log", "a") as f:
                        f.write(line)
        except Exception:
            pass

        self._check_vol_drop_timeout(symbol, bar_close)
        self._try_dca(symbol)

        signal = self.check_signal(symbol)
        if signal:
            logger.info(f"  RT_SIGNAL: {signal['direction']} {signal['symbol']} @ {_pfmt(signal['price'])} roc={signal['roc_12']:.2f}% ADX={signal['adx']:.0f}")
            with self._lock:
                self.open_position(signal)

        with self._lock:
            for pos in list(self.positions):
                if pos.symbol == symbol:
                    pos.bars_held += 1

        self._save_state()

    def _apply_htf_supertrend(self, symbol: str):
        """Re-apply HTF Supertrend to vol_bars. Must be called under _lock."""
        if symbol not in self.htf_vol_bars:
            return
        vb_idx = self.vol_bars[symbol].sort_index().index
        for col in ["ema_9", "ema_21", "ema_50"]:
            if col in self.htf_vol_bars[symbol].columns:
                self.vol_bars[symbol][f"htf_{col}"] = _safe_reindex(self.htf_vol_bars[symbol][col], vb_idx).values
        htf_df = self.htf_vol_bars[symbol]
        if "high" in htf_df.columns and "low" in htf_df.columns:
            st = pta.supertrend(htf_df["high"], htf_df["low"], htf_df["close"], length=9, multiplier=3.0)
            if st is not None:
                for sc in st.columns:
                    if "SUPERTd" in sc:
                        self.vol_bars[symbol]["htf_supertrend"] = _safe_reindex(st[sc], vb_idx).values

    def _update_htf_bar(self, symbol, bar_open, bar_high, bar_low, bar_close, bar_vol, bar_start):
        """Update HTF volume bar accumulation. Must be called under _lock."""
        if symbol not in self.htf_vol_bars:
            return
        htf_threshold = self.vol_thresholds.get(symbol, 2000) * 5
        if not hasattr(self, '_htf_buffers'):
            self._htf_buffers = {}
        htf_buf = self._htf_buffers.get(symbol, {"cum_vol": 0, "bar_open": None, "bar_high": None, "bar_low": None, "bar_start": None})
        if htf_buf["bar_open"] is None:
            htf_buf["bar_open"] = bar_open
            htf_buf["bar_high"] = bar_high
            htf_buf["bar_low"] = bar_low
            htf_buf["bar_start"] = bar_start
        htf_buf["bar_high"] = max(htf_buf["bar_high"], float(bar_high))
        htf_buf["bar_low"] = min(htf_buf["bar_low"], float(bar_low))
        htf_buf["cum_vol"] += bar_vol
        if htf_buf["cum_vol"] >= htf_threshold:
            htf_bar = pd.DataFrame([{
                "open": htf_buf["bar_open"], "high": htf_buf["bar_high"],
                "low": htf_buf["bar_low"], "close": float(bar_close),
                "volume": htf_buf["cum_vol"],
            }], index=[htf_buf["bar_start"]])
            htf_base = self.htf_vol_bars[symbol][["open", "high", "low", "close", "volume"]]
            htf_base = pd.concat([htf_base, htf_bar]).tail(200)
            htf_base = htf_base[~htf_base.index.duplicated(keep='last')]
            self.htf_vol_bars[symbol] = self.indicators.calculate_all(htf_base)
            self._apply_htf_supertrend(symbol)
            htf_buf = {"cum_vol": 0, "bar_open": None, "bar_high": None, "bar_low": None, "bar_start": None}
        self._htf_buffers[symbol] = htf_buf

    def _check_vol_drop_timeout(self, symbol: str, close_price: float):
        """Check vol_drop and timeout exits after volume bar completion."""
        with self._lock:
            vdf = self.vol_bars.get(symbol)
            for pos in list(self.positions):
                if pos.symbol != symbol:
                    continue
                exit_reason = None
                if vdf is not None and pos.bars_held >= 3 and len(vdf) >= 3:
                    vol_vals = vdf["volume"].values
                    vol_ma20 = pd.Series(vol_vals).rolling(20, min_periods=1).mean().values
                    j = len(vdf) - 1
                    if vol_ma20[j] > 0 and vol_vals[j-2:j+1].mean() < vol_ma20[j] * 0.5:
                        exit_reason = "VOL_DROP"
                if not exit_reason and pos.bars_held >= 24:
                    exit_reason = "TIMEOUT"
                if exit_reason:
                    if pos.direction == "LONG":
                        pnl_pct = (close_price - pos.avg_price) / pos.avg_price * 100 * pos.total_size
                    else:
                        pnl_pct = (pos.avg_price - close_price) / pos.avg_price * 100 * pos.total_size
                    logger.info(f"  {exit_reason} {pos.direction} {pos.symbol} @ {_pfmt(close_price)} | PnL {pnl_pct:+.2f}% | Bars: {pos.bars_held}")
                    self._log_trade_close(pos, close_price, exit_reason, pnl_pct)
                    self.positions.remove(pos)
                    pair_vb = self.vol_bar_counts.get(symbol, 0)
                    self.cooldowns[symbol] = pair_vb + COOLDOWN_BARS
                    self.pair_sl_streaks[f"{pos.symbol}_{pos.direction}"] = 0

    def update_volume_bars(self, symbol: str) -> bool:
        """Fetch latest 15m bar and update volume bars. Returns True if new vol bar closed."""
        try:
            df_new = self.fetcher.fetch_ohlcv(symbol, "15m", limit=2)
            if df_new.empty:
                return False

            latest = df_new.iloc[-1]
            buf = self.vol_buffers[symbol]

            if symbol in self.time_data:
                base = self.time_data[symbol][["open", "high", "low", "close", "volume"]]
                base = pd.concat([base, df_new.iloc[[-1]]])
                base = base[~base.index.duplicated(keep='last')]
                self.time_data[symbol] = self.indicators.calculate_all(base.tail(1000))
                self.time_data[symbol] = self.time_data[symbol][~self.time_data[symbol].index.duplicated(keep='last')]

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
                self._complete_volume_bar(
                    symbol, buf["bar_open"], buf["bar_high"], buf["bar_low"],
                    latest["close"], buf["cum_vol"], buf["bar_start"],
                )
                buf["cum_vol"] = 0
                buf["bar_open"] = None
                return True

            return False

        except Exception as e:
            logger.error(f"Error updating {symbol}: {e}", exc_info=True)
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

        # Compute all guard values upfront for logging
        htf_st = float(vdf["htf_supertrend"].iloc[j]) if "htf_supertrend" in vdf.columns else 0
        atr = vdf["atr"].values
        atr_ma20 = pd.Series(atr).rolling(20, min_periods=1).mean().values
        atr_exp = atr[j] / atr_ma20[j] if atr_ma20[j] > 0 else 1.0
        adx = adx_raw
        rsi_s6 = rsi_s6_raw
        bar_close_price = float(vdf["close"].iloc[j])
        atr_val = float(atr[j])

        def _log_blocked(guard_name):
            """Log blocked signal to trades.log for analysis."""
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            line = (f"{now} BLOCKED {direction} {symbol} @ {_pfmt(bar_close_price)} | {guard_name} | "
                    f"roc={roc_12:+.2f}% ADX={adx:.0f} RSI_s={rsi_s6:+.1f} ATR_EXP={atr_exp:.2f} "
                    f"HTF={'UP' if htf_st > 0 else 'DN' if htf_st < 0 else '?'} ATR={atr_val:.6f}\n")
            self._write_trades_log(line)

        # HTF Supertrend trend filter
        if not np.isnan(htf_st) and htf_st != 0:
            if direction == "LONG" and htf_st < 0:
                _log_blocked("HTF_DN")
                return None
            if direction == "SHORT" and htf_st > 0:
                _log_blocked("HTF_UP")
                return None

        # ATR expansion filter
        if np.isnan(atr_exp) or atr_exp > ATR_EXP_MAX:
            _log_blocked(f"ATR_EXP={atr_exp:.2f}")
            return None

        # ADX guard
        if np.isnan(adx) or adx < ADX_MIN:
            _log_blocked(f"ADX={adx:.0f}")
            return None

        # RSI slope guard
        if np.isnan(rsi_s6):
            _log_blocked("RSI=NaN")
            return None
        if direction == "LONG" and rsi_s6 < 0:
            _log_blocked(f"RSI_slope={rsi_s6:.1f}")
            return None
        if direction == "SHORT" and rsi_s6 > 0:
            _log_blocked(f"RSI_slope={rsi_s6:+.1f}")
            return None

        atr_val = float(atr[j])
        if np.isnan(atr_val) or atr_val <= 0:
            return None

        return {
            "symbol": symbol,
            "direction": direction,
            "price": bar_close_price,
            "bar_open": float(vdf["open"].iloc[j]),
            "bar_high": float(vdf["high"].iloc[j]),
            "bar_low": float(vdf["low"].iloc[j]),
            "bar_close": bar_close_price,
            "atr": atr_val,
            "adx": adx,
            "roc_12": roc_12,
        }

    def _log_signal_metrics(self, signal: dict):
        """Log all tick-level metrics at signal time (no filtering)."""
        import numpy as np
        symbol = signal["symbol"]
        direction = signal["direction"]
        atr_val = signal.get("atr", 1)
        trades = list(getattr(self, '_recent_trades', {}).get(symbol, []))
        if len(trades) < 5:
            return

        now = time.time()

        # 1. Tick momentum (% aligned with direction)
        recent20 = trades[-20:]
        if direction == "SHORT":
            tick_mom = sum(1 for t in recent20 if t["is_sell"]) / len(recent20) * 100
        else:
            tick_mom = sum(1 for t in recent20 if not t["is_sell"]) / len(recent20) * 100

        # 2. Price velocity (price change over last 1 sec in ATR)
        recent_1s = [t for t in trades if now - t["ts"] <= 1.0]
        if len(recent_1s) >= 2:
            velocity = (recent_1s[-1]["price"] - recent_1s[0]["price"]) / atr_val
        else:
            velocity = 0

        # 3. Volume burst (volume last 3s vs average)
        recent_3s = [t for t in trades if now - t["ts"] <= 3.0]
        recent_30s = [t for t in trades if now - t["ts"] <= 30.0]
        vol_3s = sum(t["qty"] for t in recent_3s)
        vol_30s = sum(t["qty"] for t in recent_30s)
        vol_burst = vol_3s / (vol_30s / 10) if vol_30s > 0 else 1.0

        # 4. Consecutive same-direction trades
        consec = 0
        for t in reversed(trades):
            if direction == "SHORT" and t["is_sell"]:
                consec += 1
            elif direction == "LONG" and not t["is_sell"]:
                consec += 1
            else:
                break

        # 5. Tick volatility (std of price changes over last 1 sec)
        if len(recent_1s) >= 3:
            prices = [t["price"] for t in recent_1s]
            changes = [abs(prices[i+1] - prices[i]) for i in range(len(prices)-1)]
            tick_vol = sum(changes) / len(changes) / atr_val if atr_val > 0 else 0
        else:
            tick_vol = 0

        # 6. Trade size analysis
        sizes = [t["qty"] for t in recent20]
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        median_size = sorted(sizes)[len(sizes)//2] if sizes else 0
        large_pct = sum(1 for s in sizes if s > median_size * 3) / len(sizes) * 100 if sizes else 0

        # 7. Entry delay simulation (price 3s ago vs now)
        trades_3s_ago = [t for t in trades if now - t["ts"] >= 2.5 and now - t["ts"] <= 3.5]
        if trades_3s_ago:
            price_3s = trades_3s_ago[0]["price"]
            delay_ok = (direction == "LONG" and signal["price"] >= price_3s) or \
                       (direction == "SHORT" and signal["price"] <= price_3s)
        else:
            price_3s = 0
            delay_ok = True

        # 8. Sweep detection (5+ consecutive same-side through 3+ price levels in <500ms)
        sweep_dir = None
        sweep_vol = 0
        recent_500ms = [t for t in trades if now - t["ts"] <= 0.5]
        if len(recent_500ms) >= 5:
            buy_run = 0; sell_run = 0; buy_prices = set(); sell_prices = set()
            buy_vol = 0; sell_vol = 0
            for t in recent_500ms:
                if not t["is_sell"]:
                    buy_run += 1; buy_prices.add(round(t["price"], 6)); buy_vol += t["qty"]
                else:
                    buy_run = 0; buy_prices = set(); buy_vol = 0
                if t["is_sell"]:
                    sell_run += 1; sell_prices.add(round(t["price"], 6)); sell_vol += t["qty"]
                else:
                    sell_run = 0; sell_prices = set(); sell_vol = 0
            if buy_run >= 5 and len(buy_prices) >= 3:
                sweep_dir = "BUY"
                sweep_vol = buy_vol
            elif sell_run >= 5 and len(sell_prices) >= 3:
                sweep_dir = "SELL"
                sweep_vol = sell_vol

        sweep_aligned = (sweep_dir == "BUY" and direction == "LONG") or \
                        (sweep_dir == "SELL" and direction == "SHORT")
        sweep_str = f"{sweep_dir}({sweep_vol:.2f})" if sweep_dir else "none"

        # 9. Spread
        spread_data = getattr(self, '_spreads', {}).get(symbol, {})
        spread_pct = spread_data.get("spread_pct", 0)

        # 10. Volume-weighted imbalance (buy$ vs sell$)
        buy_vol = sum(t["price"] * t["qty"] for t in recent20 if not t["is_sell"])
        sell_vol = sum(t["price"] * t["qty"] for t in recent20 if t["is_sell"])
        total_vol_dollar = buy_vol + sell_vol
        vol_imbalance = (buy_vol - sell_vol) / total_vol_dollar * 100 if total_vol_dollar > 0 else 0

        # 11. VWAP distance (entry vs volume-weighted avg price)
        vwap_vol = sum(t["qty"] for t in recent20)
        vwap = sum(t["price"] * t["qty"] for t in recent20) / vwap_vol if vwap_vol > 0 else signal["price"]
        vwap_dist = (signal["price"] - vwap) / atr_val if atr_val > 0 else 0

        # 12. Price impact (price change per unit volume over last 20 trades)
        if len(recent20) >= 2 and atr_val > 0:
            price_change = abs(recent20[-1]["price"] - recent20[0]["price"])
            total_qty = sum(t["qty"] for t in recent20)
            price_impact = (price_change / total_qty) if total_qty > 0 else 0
            price_impact_norm = price_impact / atr_val * 1000
        else:
            price_impact_norm = 0

        # 13. Trade frequency (trades per second over last 3 sec)
        recent_3s_count = len([t for t in trades if now - t["ts"] <= 3.0])
        trade_freq = recent_3s_count / 3.0

        # 14. Trade clustering CV (coefficient of variation of inter-arrival times)
        if len(recent20) >= 5:
            arrivals = [recent20[i+1]["ts"] - recent20[i]["ts"] for i in range(len(recent20)-1)]
            arrivals = [a for a in arrivals if a > 0]
            if arrivals and len(arrivals) >= 3:
                mean_iat = sum(arrivals) / len(arrivals)
                std_iat = (sum((a - mean_iat)**2 for a in arrivals) / len(arrivals)) ** 0.5
                clustering_cv = std_iat / mean_iat if mean_iat > 0 else 1.0
            else:
                clustering_cv = 1.0
        else:
            clustering_cv = 1.0

        # 15. CVD (Cumulative Volume Delta over last 50 trades)
        all_recent = list(trades)[-50:]
        cvd = sum(t["qty"] if not t["is_sell"] else -t["qty"] for t in all_recent)
        # CVD direction vs price direction
        price_dir = 1 if len(all_recent) >= 2 and all_recent[-1]["price"] > all_recent[0]["price"] else -1
        cvd_dir = 1 if cvd > 0 else -1
        cvd_divergence = cvd_dir != price_dir

        # 16. Price efficiency (net / gross movement)
        if len(recent20) >= 3:
            prices_list = [t["price"] for t in recent20]
            net_move = abs(prices_list[-1] - prices_list[0])
            gross_move = sum(abs(prices_list[i+1] - prices_list[i]) for i in range(len(prices_list)-1))
            price_efficiency = net_move / gross_move if gross_move > 0 else 0
        else:
            price_efficiency = 0

        # 17. Dollar volume per trade (avg)
        dollar_per_trade = total_vol_dollar / len(recent20) if recent20 else 0

        # 18.5 Order book depth (top 5 levels)
        depth_data = getattr(self, '_depth', {}).get(symbol, {})
        depth_imb = depth_data.get("imbalance", 0)
        depth_bid_wall = depth_data.get("bid_wall", False)
        depth_ask_wall = depth_data.get("ask_wall", False)

        # 18. Trade acceleration (freq last 1s vs freq 3-4s ago)
        freq_now = len([t for t in trades if now - t["ts"] <= 1.0])
        freq_before = len([t for t in trades if 3.0 <= now - t["ts"] <= 4.0])
        trade_accel = freq_now - freq_before

        # 22. Absorption: price stable despite high volume (someone absorbing with limit)
        if len(recent20) >= 5 and atr_val > 0:
            price_range_20 = max(t["price"] for t in recent20) - min(t["price"] for t in recent20)
            vol_sum_20 = sum(t["qty"] for t in recent20)
            absorption = (vol_sum_20 / (price_range_20 / atr_val)) if price_range_20 > 0 else 0
        else:
            absorption = 0

        # 23. Iceberg: repeated same-size trades at same price (hidden large order)
        iceberg = False
        if len(recent20) >= 5:
            sizes_rounded = [round(t["qty"], 2) for t in recent20[-10:]]
            prices_rounded = [round(t["price"], 4) for t in recent20[-10:]]
            from collections import Counter
            size_counts = Counter(sizes_rounded)
            most_common_size, most_common_count = size_counts.most_common(1)[0]
            if most_common_count >= 4 and most_common_size > 0:
                same_size_trades = [t for t in recent20[-10:] if round(t["qty"], 2) == most_common_size]
                same_price = len(set(round(t["price"], 4) for t in same_size_trades)) <= 2
                iceberg = same_price

        # 24. Delta acceleration (CVD change rate)
        if len(all_recent) >= 10:
            cvd_first = sum(t["qty"] if not t["is_sell"] else -t["qty"] for t in all_recent[:len(all_recent)//2])
            cvd_second = sum(t["qty"] if not t["is_sell"] else -t["qty"] for t in all_recent[len(all_recent)//2:])
            delta_accel = cvd_second - cvd_first
        else:
            delta_accel = 0

        # 25. Size trend (are trades getting bigger or smaller?)
        if len(recent20) >= 10:
            first_half_avg = sum(t["qty"] for t in recent20[:10]) / 10
            second_half_avg = sum(t["qty"] for t in recent20[10:]) / max(len(recent20[10:]), 1)
            size_trend = (second_half_avg - first_half_avg) / first_half_avg if first_half_avg > 0 else 0
        else:
            size_trend = 0

        # 26. Price momentum consistency (how consistently price moves in one direction)
        if len(recent20) >= 5:
            price_changes = [recent20[i+1]["price"] - recent20[i]["price"] for i in range(len(recent20)-1)]
            if direction == "SHORT":
                price_changes = [-c for c in price_changes]
            positive = sum(1 for c in price_changes if c > 0)
            mom_consistency = positive / len(price_changes) if price_changes else 0.5
        else:
            mom_consistency = 0.5

        # 27. Spread change (from bookTicker history)
        spread_change = 0
        if hasattr(self, '_spread_history') and symbol in self._spread_history:
            sh = list(self._spread_history[symbol])
            if len(sh) >= 2:
                spread_change = sh[-1] - sh[0]

        # 28. Bid/ask size ratio (from bookTicker)
        ba_ratio = 1.0
        if spread_data:
            bid_sz = spread_data.get("bid_qty", 0)
            ask_sz = spread_data.get("ask_qty", 0)
            ba_ratio = bid_sz / ask_sz if ask_sz > 0 else 10.0

        # 30. Total depth (sum all 5 levels bid + ask)
        total_depth = depth_data.get("bid_vol", 0) + depth_data.get("ask_vol", 0)

        # 31. Depth gradient (is volume concentrated at best level or spread?)
        depth_gradient = 0
        if hasattr(self, '_depth') and symbol in self._depth:
            dd = self._depth[symbol]
            bids_raw = dd.get("bids_raw", [])
            asks_raw = dd.get("asks_raw", [])
            if bids_raw:
                best_bid_vol = bids_raw[0] if bids_raw else 0
                total_bid = sum(bids_raw) if bids_raw else 1
                depth_gradient = best_bid_vol / total_bid if total_bid > 0 else 0

        # 32. Wall distance (ATR to nearest wall)
        wall_dist = 99.0
        if hasattr(self, '_depth') and symbol in self._depth:
            dd = self._depth[symbol]
            wall_price = dd.get("wall_price", 0)
            if wall_price > 0 and atr_val > 0:
                wall_dist = abs(signal["price"] - wall_price) / atr_val

        # 33. Depth velocity (is depth decreasing = orders pulled = incoming move)
        depth_vel = 0
        if hasattr(self, '_depth_history') and symbol in self._depth_history:
            dh = list(self._depth_history[symbol])
            if len(dh) >= 2:
                depth_vel = dh[-1] - dh[0]

        # 34. Whale trade (single trade > 10x median in last 20)
        if sizes:
            med_size = sorted(sizes)[len(sizes)//2]
            whale = any(s > med_size * 10 for s in sizes) if med_size > 0 else False
        else:
            whale = False

        # 35. Reversal count (how many times price changed direction in last 20)
        if len(recent20) >= 3:
            dirs = [1 if recent20[i+1]["price"] > recent20[i]["price"] else -1
                    for i in range(len(recent20)-1) if recent20[i+1]["price"] != recent20[i]["price"]]
            reversals = sum(1 for i in range(len(dirs)-1) if dirs[i] != dirs[i+1]) if len(dirs) >= 2 else 0
        else:
            reversals = 0

        # 36. Aggressor exhaustion (volume dropped >50% after burst)
        if len(trades) >= 10:
            vol_last5 = sum(t["qty"] for t in list(trades)[-5:])
            vol_prev5 = sum(t["qty"] for t in list(trades)[-10:-5])
            exhaustion = vol_last5 < vol_prev5 * 0.5 if vol_prev5 > 0 else False
        else:
            exhaustion = False

        # 37. Trade gap (seconds since last trade)
        if len(trades) >= 2:
            trade_gap = now - list(trades)[-1]["ts"]
        else:
            trade_gap = 0

        # 38. Microprice (fair price estimate from bookTicker)
        microprice_delta = 0
        if spread_data and spread_data.get("bid_qty", 0) > 0:
            bid_p = spread_data.get("bid", 0)
            ask_p = spread_data.get("ask", 0)
            bid_q = spread_data.get("bid_qty", 0)
            ask_q = spread_data.get("ask_qty", 0)
            if bid_q + ask_q > 0:
                microprice = (bid_p * ask_q + ask_p * bid_q) / (bid_q + ask_q)
                mid = (bid_p + ask_p) / 2
                microprice_delta = (microprice - mid) / atr_val if atr_val > 0 else 0

        # 39. Spread stability (std of spread over last 20 bookTicker updates)
        spread_std = 0
        if hasattr(self, '_spread_history') and symbol in self._spread_history:
            sh = list(self._spread_history[symbol])
            if len(sh) >= 3:
                mean_s = sum(sh) / len(sh)
                spread_std = (sum((s - mean_s)**2 for s in sh) / len(sh)) ** 0.5

        # 40. Quote imbalance trend (bid_qty/ask_qty changing?)
        qi_trend = 0
        if hasattr(self, '_qi_history') and symbol in self._qi_history:
            qh = list(self._qi_history[symbol])
            if len(qh) >= 2:
                qi_trend = qh[-1] - qh[0]
        if spread_data and spread_data.get("ask_qty", 0) > 0:
            qi = spread_data.get("bid_qty", 0) / spread_data.get("ask_qty", 1)
            if not hasattr(self, '_qi_history'):
                self._qi_history = {}
            import collections as _col
            if symbol not in self._qi_history:
                self._qi_history[symbol] = _col.deque(maxlen=20)
            self._qi_history[symbol].append(qi)

        # 41. Disappearing liquidity (depth dropped >50% recently)
        disappearing_liq = False
        if hasattr(self, '_depth_history') and symbol in self._depth_history:
            dh = list(self._depth_history[symbol])
            if len(dh) >= 3 and dh[0] > 0:
                disappearing_liq = dh[-1] < dh[0] * 0.5

        # 42. Book skewness (deep levels bid vs ask volume ratio)
        book_skew = 0
        if hasattr(self, '_depth') and symbol in self._depth:
            dd = self._depth[symbol]
            br = dd.get("bids_raw", [])
            ar = dd.get("asks_raw", [])
            if len(br) >= 3 and len(ar) >= 3:
                deep_bid = sum(br[2:])
                deep_ask = sum(ar[2:])
                book_skew = (deep_bid - deep_ask) / (deep_bid + deep_ask) * 100 if (deep_bid + deep_ask) > 0 else 0

        # 46. Net flow at 1s, 5s, 30s windows (signed volume)
        net_flow_1s = sum(t["qty"] if not t["is_sell"] else -t["qty"]
                         for t in trades if now - t["ts"] <= 1.0)
        net_flow_5s = sum(t["qty"] if not t["is_sell"] else -t["qty"]
                         for t in trades if now - t["ts"] <= 5.0)
        net_flow_30s = sum(t["qty"] if not t["is_sell"] else -t["qty"]
                          for t in trades if now - t["ts"] <= 30.0)

        # Early init mark_data (used by multiple metrics below)
        mark_data = getattr(self, '_mark_data', {}).get(symbol, {})

        # 47. Price-volume divergence (price up but volume down = weak)
        pv_divergence = False
        if len(trades) >= 20:
            half = len(list(trades)) // 2
            t_list = list(trades)
            first_half_vol = sum(t["qty"] for t in t_list[:half])
            second_half_vol = sum(t["qty"] for t in t_list[half:])
            price_up = t_list[-1]["price"] > t_list[0]["price"]
            vol_down = second_half_vol < first_half_vol * 0.7
            pv_divergence = (price_up and vol_down) or (not price_up and vol_down)

        # 48. Mark-index spread (liquidation pressure)
        mark_index_spread = 0
        if mark_data and atr_val > 0:
            mp = mark_data.get("mark_price", 0)
            ip = mark_data.get("index_price", 0)
            if mp > 0 and ip > 0:
                mark_index_spread = (mp - ip) / atr_val

        # 49. Funding change (tracking funding rate over time)
        funding_change = 0
        if hasattr(self, '_funding_history') and symbol in self._funding_history:
            fh = list(self._funding_history[symbol])
            if len(fh) >= 2:
                funding_change = fh[-1] - fh[0]
        if mark_data and mark_data.get("funding_rate", 0) != 0:
            if not hasattr(self, '_funding_history'):
                self._funding_history = {}
            import collections as _colx
            if symbol not in self._funding_history:
                self._funding_history[symbol] = _colx.deque(maxlen=20)
            self._funding_history[symbol].append(mark_data.get("funding_rate", 0))

        # 50. Depth-weighted midprice (weighted avg across all 5 levels)
        depth_mid = 0
        if hasattr(self, '_depth') and symbol in self._depth:
            dd = self._depth[symbol]
            br = dd.get("bids_raw", [])
            ar = dd.get("asks_raw", [])
            bids_prices = [float(b[0]) for b in data.get("b", [])] if "b" in data and isinstance(data.get("b"), list) else []
            # Use stored data instead
            if br and ar and spread_data:
                bid_p = spread_data.get("bid", 0)
                ask_p = spread_data.get("ask", 0)
                if bid_p > 0 and ask_p > 0:
                    total_w = sum(br) + sum(ar)
                    if total_w > 0:
                        depth_mid = (bid_p * sum(ar) + ask_p * sum(br)) / total_w
                        depth_mid = (depth_mid - (bid_p + ask_p) / 2) / atr_val if atr_val > 0 else 0

        # 51. Book recovery (depth change after large recent trade)
        book_recovery = 0
        if hasattr(self, '_depth_history') and symbol in self._depth_history:
            dh = list(self._depth_history[symbol])
            if len(dh) >= 5:
                mid_val = dh[len(dh)//2]
                if mid_val > 0:
                    drop = min(dh) / mid_val
                    book_recovery = dh[-1] / mid_val if mid_val > 0 else 1.0

        # 52. Cross-pair BTC signal (BTC moving but pair not = divergence)
        btc_velocity = 0
        btc_trades = getattr(self, '_recent_trades', {}).get("BTC/USDT", [])
        if len(list(btc_trades)) >= 5 and symbol != "BTC/USDT":
            btc_list = list(btc_trades)
            btc_recent = [t for t in btc_list if now - t["ts"] <= 3.0]
            if len(btc_recent) >= 2:
                btc_velocity = (btc_recent[-1]["price"] - btc_recent[0]["price"]) / btc_recent[0]["price"] * 10000

        # 53. Bid/ask momentum (bid price trend over recent bookTicker updates)
        bid_momentum = 0
        if hasattr(self, '_bid_history') and symbol in self._bid_history:
            bh = list(self._bid_history[symbol])
            if len(bh) >= 3 and atr_val > 0:
                bid_momentum = (bh[-1] - bh[0]) / atr_val
        if spread_data and spread_data.get("bid", 0) > 0:
            if not hasattr(self, '_bid_history'):
                self._bid_history = {}
            import collections as _coly
            if symbol not in self._bid_history:
                self._bid_history[symbol] = _coly.deque(maxlen=20)
            self._bid_history[symbol].append(spread_data["bid"])

        # 54. Volume concentration (is volume at one price or spread?)
        vol_concentration = 0
        if len(recent20) >= 5:
            from collections import Counter as _Ctr
            price_vols = {}
            for t in recent20:
                rp = round(t["price"], 2)
                price_vols[rp] = price_vols.get(rp, 0) + t["qty"]
            if price_vols:
                max_vol = max(price_vols.values())
                total_v = sum(price_vols.values())
                vol_concentration = max_vol / total_v if total_v > 0 else 0

        # 55. Trade-sign entropy
        import math
        if len(recent20) >= 5:
            p_buy = sum(1 for t in recent20 if not t["is_sell"]) / len(recent20)
            p_sell = 1 - p_buy
            sign_entropy = 0
            if 0 < p_buy < 1:
                sign_entropy = -(p_buy * math.log2(p_buy) + p_sell * math.log2(p_sell))
        else:
            sign_entropy = 1.0

        # 56. KL surprise (deviation from random 50/50)
        if len(recent20) >= 5 and 0 < p_buy < 1:
            kl_surprise = p_buy * math.log2(p_buy / 0.5) + p_sell * math.log2(p_sell / 0.5)
        else:
            kl_surprise = 0

        # 58. Book convexity (bid side)
        book_convexity = 0
        if hasattr(self, '_depth') and symbol in self._depth:
            dd = self._depth[symbol]
            br = dd.get("bids_raw", [])
            if len(br) >= 3:
                book_convexity = (br[0] + br[-1]) / 2 - br[len(br)//2] if br[len(br)//2] > 0 else 0

        # 61. Level-weighted imbalance (exponential decay by level)
        lw_imbalance = 0
        if hasattr(self, '_depth') and symbol in self._depth:
            dd = self._depth[symbol]
            br = dd.get("bids_raw", [])
            ar = dd.get("asks_raw", [])
            if br and ar:
                weights = [2**(-i) for i in range(min(len(br), len(ar)))]
                w_bid = sum(br[i] * weights[i] for i in range(len(weights)))
                w_ask = sum(ar[i] * weights[i] for i in range(len(weights)))
                total_w = w_bid + w_ask
                lw_imbalance = (w_bid - w_ask) / total_w * 100 if total_w > 0 else 0

        # 62. Book resilience (depth beyond best level)
        book_resilience = 0
        if hasattr(self, '_depth') and symbol in self._depth:
            dd = self._depth[symbol]
            br = dd.get("bids_raw", [])
            if len(br) >= 2 and sum(br) > 0:
                book_resilience = sum(br[1:]) / sum(br)

        # 63. VPIN (simplified: absolute imbalance / total volume over last 50 trades)
        if len(all_recent) >= 10:
            buy_v = sum(t["qty"] for t in all_recent if not t["is_sell"])
            sell_v = sum(t["qty"] for t in all_recent if t["is_sell"])
            total_v = buy_v + sell_v
            vpin = abs(buy_v - sell_v) / total_v if total_v > 0 else 0
        else:
            vpin = 0

        # 66. Jump ratio (realized vol vs bipower variation)
        jump_ratio = 0
        if len(recent20) >= 5:
            returns = [abs(recent20[i+1]["price"] / recent20[i]["price"] - 1)
                      for i in range(len(recent20)-1) if recent20[i]["price"] > 0]
            if len(returns) >= 3:
                rv = sum(r**2 for r in returns)
                bv = (math.pi / 2) * sum(abs(returns[i]) * abs(returns[i-1])
                      for i in range(1, len(returns))) / max(len(returns)-1, 1)
                jump_ratio = max(0, 1 - bv / rv) if rv > 0 else 0

        # 71. Trade-sign autocorrelation lag-1
        sign_autocorr = 0
        if len(recent20) >= 10:
            signs = [1 if not t["is_sell"] else -1 for t in recent20]
            mean_s = sum(signs) / len(signs)
            var_s = sum((s - mean_s)**2 for s in signs)
            if var_s > 0:
                cov_s = sum((signs[i] - mean_s) * (signs[i+1] - mean_s)
                           for i in range(len(signs)-1))
                sign_autocorr = cov_s / var_s

        # 79. Basis Z-score (mark-index spread vs recent history)
        basis_zscore = 0
        if mark_data and mark_data.get("mark_price", 0) > 0 and mark_data.get("index_price", 0) > 0:
            basis = (mark_data["mark_price"] - mark_data["index_price"]) / mark_data["index_price"] * 10000
            if hasattr(self, '_basis_history') and symbol in self._basis_history:
                bh = list(self._basis_history[symbol])
                if len(bh) >= 3:
                    mean_b = sum(bh) / len(bh)
                    std_b = (sum((b - mean_b)**2 for b in bh) / len(bh)) ** 0.5
                    basis_zscore = (basis - mean_b) / std_b if std_b > 0 else 0
            if not hasattr(self, '_basis_history'):
                self._basis_history = {}
            import collections as _colz
            if symbol not in self._basis_history:
                self._basis_history[symbol] = _colz.deque(maxlen=50)
            self._basis_history[symbol].append(basis)

        # 80. Price-level Gini (concentration of trades at price levels)
        price_gini = 0
        if len(recent20) >= 5:
            from collections import Counter as _Ctr2
            price_counts = _Ctr2(round(t["price"], 2) for t in recent20)
            vals = sorted(price_counts.values())
            n_g = len(vals)
            if n_g > 0 and sum(vals) > 0:
                cum = sum((2 * (i+1) - n_g - 1) * v for i, v in enumerate(vals))
                price_gini = cum / (n_g * sum(vals))

        # 81. Informed flow proxy (aggressive trades when spread wide)
        informed_flow = 0
        if spread_data and spread_data.get("spread_pct", 0) > 0:
            median_spread = 0.01  # rough default
            if hasattr(self, '_spread_history') and symbol in self._spread_history:
                sh = list(self._spread_history[symbol])
                if sh:
                    median_spread = sorted(sh)[len(sh)//2]
            wide_spread = spread_data["spread_pct"] > median_spread
            if wide_spread and len(recent20) >= 5:
                aggressive = sum(1 for t in recent20[-10:]
                               if (direction == "LONG" and not t["is_sell"]) or
                                  (direction == "SHORT" and t["is_sell"]))
                informed_flow = aggressive / min(10, len(recent20[-10:]))

        # 82. Funding-mark alignment
        fund_mark_align = 0
        if mark_data and hasattr(self, '_mark_data'):
            fr = mark_data.get("funding_rate", 0)
            mp = mark_data.get("mark_price", 0)
            if hasattr(self, '_mark_history') and symbol in self._mark_history:
                mh = list(self._mark_history[symbol])
                if mh and mp > 0:
                    mark_change = mp - mh[-1]
                    fund_mark_align = 1 if (fr > 0 and mark_change > 0) or (fr < 0 and mark_change < 0) else -1
            if not hasattr(self, '_mark_history'):
                self._mark_history = {}
            import collections as _colm
            if symbol not in self._mark_history:
                self._mark_history[symbol] = _colm.deque(maxlen=20)
            if mp > 0:
                self._mark_history[symbol].append(mp)

        # 60. Book slope (qty change per price level)
        book_slope = 0
        if hasattr(self, '_depth') and symbol in self._depth:
            dd = self._depth[symbol]
            br = dd.get("bids_raw", [])
            if len(br) >= 3:
                book_slope = (br[-1] - br[0]) / max(len(br)-1, 1)

        # 43-45. Mark price / funding rate
        mark_data = getattr(self, '_mark_data', {}).get(symbol, {})
        funding_rate = mark_data.get("funding_rate", 0) * 100  # as percentage
        mark_price = mark_data.get("mark_price", 0)
        last_price = signal["price"]
        mark_vs_last = (mark_price - last_price) / atr_val if mark_price > 0 and atr_val > 0 else 0
        # Funding aligned: SHORT + negative funding = good, LONG + positive = good
        funding_aligned = (direction == "SHORT" and funding_rate < 0) or \
                         (direction == "LONG" and funding_rate > 0)

        logger.info(
            f"  {'HOLD_DATA' if getattr(self, '_hold_log_prefix', None) else 'SIGNAL_DATA'} {direction} {symbol} "
            f"{getattr(self, '_hold_log_prefix', '') or ''}"
            f"tick_mom={tick_mom:.0f}% "
            f"velocity={velocity:+.4f} "
            f"vol_burst={vol_burst:.1f}x "
            f"consec={consec} "
            f"tick_vol={tick_vol:.4f} "
            f"avg_size={avg_size:.2f} "
            f"large_pct={large_pct:.0f}% "
            f"delay_ok={delay_ok} "
            f"sweep={sweep_str} "
            f"sweep_aligned={sweep_aligned} "
            f"spread={spread_pct:.4f}% "
            f"vol_imb={vol_imbalance:+.1f}% "
            f"vwap_dist={vwap_dist:+.3f} "
            f"p_impact={price_impact_norm:.2f} "
            f"trade_freq={trade_freq:.1f}/s "
            f"cluster_cv={clustering_cv:.2f} "
            f"cvd={cvd:+.2f} "
            f"cvd_div={cvd_divergence} "
            f"p_eff={price_efficiency:.2f} "
            f"dollar_pt={dollar_per_trade:.0f} "
            f"trade_accel={trade_accel:+d} "
            f"ob_imb={depth_imb:+.1f}% "
            f"ob_bid_wall={depth_bid_wall} "
            f"ob_ask_wall={depth_ask_wall} "
            f"absorb={absorption:.0f} "
            f"iceberg={iceberg} "
            f"d_accel={delta_accel:+.1f} "
            f"sz_trend={size_trend:+.2f} "
            f"mom_cons={mom_consistency:.2f} "
            f"spr_chg={spread_change:+.6f} "
            f"ba_ratio={ba_ratio:.2f} "
            f"tot_depth={total_depth:.1f} "
            f"d_grad={depth_gradient:.2f} "
            f"wall_dist={wall_dist:.1f} "
            f"d_vel={depth_vel:+.1f} "
            f"whale={whale} "
            f"reversals={reversals} "
            f"exhaustion={exhaustion} "
            f"trade_gap={trade_gap:.2f}s "
            f"microprice={microprice_delta:+.4f} "
            f"spr_std={spread_std:.4f} "
            f"qi_trend={qi_trend:+.2f} "
            f"disap_liq={disappearing_liq} "
            f"book_skew={book_skew:+.1f}% "
            f"funding={funding_rate:+.4f}% "
            f"mark_vs_last={mark_vs_last:+.3f} "
            f"fund_aligned={funding_aligned} "
            f"nf1={net_flow_1s:+.2f} "
            f"nf5={net_flow_5s:+.2f} "
            f"nf30={net_flow_30s:+.2f} "
            f"pv_div={pv_divergence} "
            f"mi_spr={mark_index_spread:+.4f} "
            f"fund_chg={funding_change:+.6f} "
            f"d_mid={depth_mid:+.4f} "
            f"bk_rec={book_recovery:.2f} "
            f"btc_vel={btc_velocity:+.2f} "
            f"bid_mom={bid_momentum:+.4f} "
            f"vol_conc={vol_concentration:.2f} "
            f"entropy={sign_entropy:.2f} "
            f"kl_surp={kl_surprise:.3f} "
            f"bk_conv={book_convexity:.2f} "
            f"lw_imb={lw_imbalance:+.1f}% "
            f"bk_resil={book_resilience:.2f} "
            f"vpin={vpin:.3f} "
            f"jump={jump_ratio:.3f} "
            f"sign_ac={sign_autocorr:+.2f} "
            f"basis_z={basis_zscore:+.2f} "
            f"p_gini={price_gini:.2f} "
            f"inf_flow={informed_flow:.2f} "
            f"fm_align={fund_mark_align:+d} "
            f"bk_slope={book_slope:+.2f}"
        )

    def open_position(self, signal: dict):
        """Open a new position or add DCA entry."""
        self._log_signal_metrics(signal)
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
            self._save_state()

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
            sl_step=0,
            best_price=price,
        )
        self.positions.append(pos)

        logger.info(f"  OPEN {direction} {symbol} @ {_pfmt(price)} | SL={_pfmt(hard_sl)} TP={_pfmt(tp)} ATR={_pfmt(atr_val)}")
        self._log_trade_open(signal, pos)
        self._save_state()

    def _log_dca_entry(self, pos: Position, price: float, signal: dict):
        """Log DCA entry to trades.log + update existing OPEN in paper_trades.json."""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # trades.log
        line = f"{now} DCA #{pos.total_size} {pos.direction} {pos.symbol} @ {_pfmt(price)} avg={_pfmt(pos.avg_price)} TP={_pfmt(pos.tp)}\n"
        self._write_trades_log(line)

        # paper_trades.json — update existing OPEN object
        trades = self._read_paper_trades()
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
        self._write_paper_trades(trades)

    def _log_trade_open(self, signal: dict, pos: Position):
        """Log trade open to trades.log + paper_trades.json."""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # trades.log
        line = (
            f"{now} OPEN {pos.direction} {pos.symbol} "
            f"@ {_pfmt(pos.avg_price)} SL={_pfmt(pos.hard_sl)} TP={_pfmt(pos.tp)} "
            f"O={_pfmt(signal.get('bar_open',0))} H={_pfmt(signal.get('bar_high',0))} "
            f"L={_pfmt(signal.get('bar_low',0))} C={_pfmt(signal.get('bar_close',0))} "
            f"ATR={_pfmt(signal['atr'])} ADX={signal.get('adx',0):.0f} roc={signal.get('roc_12',0):.2f}%\n"
        )
        self._write_trades_log(line)

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
        trades = self._read_paper_trades()
        trades.append(trade_record)
        self._write_paper_trades(trades)

    def _log_sl_moved(self, pos: Position):
        """Log SL move event to trades.log + update OPEN in paper_trades.json."""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        line = f"{now} SL_MOVED {pos.direction} {pos.symbol} step {pos.sl_step}/{len(MOVE_SL_STEPS)} new_sl={_pfmt(pos.hard_sl)} entry={_pfmt(pos.avg_price)} ATR={_pfmt(pos.entry_atr)}\n"
        self._write_trades_log(line)

        trades = self._read_paper_trades()
        for t in reversed(trades):
            if t.get("symbol") == pos.symbol and t.get("status") == "OPEN":
                t["sl"] = _prnd(pos.hard_sl)
                t["sl_moved"] = True
                t["sl_moved_time"] = now
                t["sl_step"] = pos.sl_step
                break
        self._write_paper_trades(trades)

    def _log_trade_close(self, pos: Position, exit_price: float, exit_reason: str, pnl_pct: float):
        """Log trade close to trades.log + append to paper_trades.json."""

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Fee: maker for entry (limit), taker for exit (market) per DCA entry
        fee_pct = (MAKER_FEE + TAKER_FEE) * 100 * pos.total_size  # % of position
        net_pnl_pct = pnl_pct - fee_pct

        # trades.log
        if pos.direction == "LONG":
            mfe = (pos.max_price - pos.avg_price) / pos.avg_price * 100
            mae = (pos.avg_price - pos.min_price) / pos.avg_price * 100
        else:
            mfe = (pos.avg_price - pos.min_price) / pos.avg_price * 100
            mae = (pos.max_price - pos.avg_price) / pos.avg_price * 100
        sl_info = f" SL_step:{pos.sl_step}/{len(MOVE_SL_STEPS)}" if pos.sl_step > 0 else ""
        line = (f"{now} CLOSE {pos.direction} {pos.symbol} @ {_pfmt(exit_price)} | {exit_reason} | "
                f"PnL {net_pnl_pct:+.2f}% (gross {pnl_pct:+.2f}% fee -{fee_pct:.2f}%) | DCA:{pos.total_size} | Bars:{pos.bars_held}{sl_info} | "
                f"MFE:{mfe:+.2f}% MAE:{mae:.2f}% High:{_pfmt(pos.max_price)} Low:{_pfmt(pos.min_price)}\n")
        self._write_trades_log(line)

        # CLOSE_DATA: log 68 metrics at exit moment
        try:
            _sig = {"symbol": pos.symbol, "direction": pos.direction, "price": exit_price, "atr": pos.entry_atr}
            self._log_signal_metrics(_sig)
        except Exception:
            pass

        # paper_trades.json — update existing OPEN object
        trades = self._read_paper_trades()
        for t in reversed(trades):
            if t.get("symbol") == pos.symbol and t.get("status") == "OPEN":
                t["status"] = "CLOSED"
                t["avg_price"] = _prnd(pos.avg_price)
                t["exit_price"] = _prnd(exit_price)
                t["pnl_pct"] = round(net_pnl_pct, 2)
                t["gross_pnl_pct"] = round(pnl_pct, 2)
                t["fee_pct"] = round(fee_pct, 2)
                t["exit_reason"] = exit_reason
                t["dca_count"] = pos.total_size
                t["bars_held"] = pos.bars_held
                t["max_price"] = _prnd(pos.max_price)
                t["min_price"] = _prnd(pos.min_price)
                t["mfe_pct"] = round(mfe, 2)
                t["mae_pct"] = round(mae, 2)
                t["close_time"] = now
                break
        self._write_paper_trades(trades)

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
        """15m scan: ATR refresh + position management. Full signal if WS is dead."""
        try:
            # 1. Refresh time data (ATR) from latest 15m candle
            try:
                df_new = self.fetcher.fetch_ohlcv(symbol, "15m", limit=2)
                if not df_new.empty and symbol in self.time_data:
                    base = self.time_data[symbol][["open", "high", "low", "close", "volume"]]
                    base = pd.concat([base, df_new.iloc[[-1]]])
                    base = base[~base.index.duplicated(keep='last')]
                    self.time_data[symbol] = self.indicators.calculate_all(base.tail(1000))
                    self.time_data[symbol] = self.time_data[symbol][~self.time_data[symbol].index.duplicated(keep='last')]

                    with self._lock:
                        if symbol in self.vol_bars and symbol in self.time_data:
                            vb_sorted = self.vol_bars[symbol].sort_index()
                            time_atr = _safe_reindex(self.time_data[symbol]["atr"], vb_sorted.index)
                            self.vol_bars[symbol]["atr"] = time_atr.values

                # Rolling threshold update (every 960 time bars)
                if not df_new.empty and symbol in self.vol_history:
                    latest = df_new.iloc[-1]
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
            except Exception as e:
                logger.debug(f"  {symbol} ATR refresh error: {e}")

            # 2. Fallback: if kline_1m dead (>10s, after 60s grace), do full volume bar update + signal check
            ws_age = time.time() - self._ws_start_time if hasattr(self, '_ws_start_time') else 0
            kline_alive = (time.time() - self._ws_last_kline) < 10 if (hasattr(self, '_ws_last_kline') and ws_age > 60) else True
            if not kline_alive:
                new_bar = self.update_volume_bars(symbol)
                if new_bar:
                    self.vol_bar_counts[symbol] = self.vol_bar_counts.get(symbol, 0) + 1
                    self._try_dca(symbol)
                    signal = self.check_signal(symbol)
                    if signal:
                        logger.info(f"  FALLBACK_SIGNAL: {signal['direction']} {signal['symbol']} @ {_pfmt(signal['price'])} roc={signal['roc_12']:.2f}%")
                        with self._lock:
                            self.open_position(signal)

            # 3. Position management (SL/TP/vol_drop/timeout)
            self._check_pair_positions(symbol, new_bar_closed=False)

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

    def _check_pair_positions(self, symbol: str, new_bar_closed: bool = True):
        """Check SL/TP/timeout for positions of this specific pair."""
        # Fetch price OUTSIDE lock
        try:
            ticker = self.fetcher.exchange.fetch_ticker(symbol)
            current_price = float(ticker["last"])
        except Exception:
            current_price = None

        with self._lock:
            for pos in list(self.positions):
                if pos.symbol != symbol:
                    continue

                vdf = self.vol_bars.get(pos.symbol)
                if vdf is None or len(vdf) < 2:
                    continue

                if current_price is None:
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

                # Trail only (Move SL steps handled by WebSocket _process_price)
                if ea > 0:
                    if pos.sl_step >= len(MOVE_SL_STEPS) and MOVE_SL_TRAIL > 0:
                        if pos.direction == "LONG":
                            pos.best_price = max(pos.best_price, current_high)
                            new_sl = pos.best_price - MOVE_SL_TRAIL * ea
                        else:
                            pos.best_price = min(pos.best_price, current_low)
                            new_sl = pos.best_price + MOVE_SL_TRAIL * ea
                        if (pos.direction == "LONG" and new_sl > pos.hard_sl) or \
                           (pos.direction == "SHORT" and new_sl < pos.hard_sl):
                            pos.hard_sl = new_sl
                            logger.debug(f"  TRAIL {pos.symbol} {pos.direction} → {_pfmt(new_sl)}")

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
                    exit_reason = "SL_MOVED" if pos.sl_step > 0 else "HARD_SL"
                    exit_price = pos.hard_sl
                elif hit_sl:
                    exit_reason = "SL_MOVED" if pos.sl_step > 0 else "HARD_SL"
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

                    self.positions.remove(pos)
                    self.cooldowns[pos.symbol] = pair_vb + COOLDOWN_BARS
                    self._log_trade_close(pos, exit_price, exit_reason, pnl_pct)

        self._save_state()
    def scan(self):
        """Run one scan cycle — each pair fully independent in its own thread."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        self.scan_count += 1
        ws_age = time.time() - self._ws_last_msg if hasattr(self, '_ws_last_msg') else 999
        kl_age = time.time() - self._ws_last_kline if hasattr(self, '_ws_last_kline') else 999
        ws_status = "OK" if ws_age < 60 else f"DEAD ({ws_age:.0f}s)"
        kl_status = "OK" if kl_age < 10 else f"DEAD ({kl_age:.0f}s)"
        t_count = getattr(self, '_trade_count', 0)
        logger.info(f"\n{'='*50}")
        logger.info(f"Scan #{self.scan_count} | Pos: {len(self.positions)} | WS: {ws_status} | Trades: {t_count}")

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

    def _log_hold_metrics(self, pos, price):
        """Log full 68 metrics every 5 sec while position is open."""
        now = time.time()
        if not hasattr(pos, '_last_hold_log'):
            pos._last_hold_log = 0
        if now - pos._last_hold_log < 5.0:
            return
        pos._last_hold_log = now

        try:
            if pos.direction == "LONG":
                pnl = (price - pos.avg_price) / pos.avg_price * 100
            else:
                pnl = (pos.avg_price - price) / pos.avg_price * 100

            sig = {"symbol": pos.symbol, "direction": pos.direction, "price": price, "atr": pos.entry_atr}
            # Temporarily replace SIGNAL_DATA prefix with HOLD_DATA
            self._hold_log_prefix = f"pnl={pnl:+.3f}% bars={pos.bars_held} sl_step={pos.sl_step} "
            self._log_signal_metrics(sig)
            self._hold_log_prefix = None
        except Exception:
            pass

    def _process_price(self, symbol: str, price: float):
        """Process a price tick for Move SL logic (called from WebSocket or polling)."""
        with self._lock:
            active = [p for p in self.positions if p.symbol == symbol]
        if not active:
            return

        for pos in active:
            self._log_hold_metrics(pos, price)
            ea = pos.entry_atr if pos.entry_atr > 0 else 0
            if ea <= 0:
                continue

            with self._lock:
                if pos not in self.positions:
                    continue

                pos.max_price = max(pos.max_price, price)
                pos.min_price = min(pos.min_price, price)

                if pos.sl_step < len(MOVE_SL_STEPS):
                    step_at, step_to = MOVE_SL_STEPS[pos.sl_step]
                    triggered = False
                    if pos.direction == "LONG" and price - pos.avg_price >= step_at * ea:
                        triggered = True
                    elif pos.direction == "SHORT" and pos.avg_price - price >= step_at * ea:
                        triggered = True
                    if triggered:
                        new_sl = pos.avg_price + step_to * ea if pos.direction == "LONG" else pos.avg_price - step_to * ea
                        if pos.direction == "LONG":
                            pos.hard_sl = max(pos.hard_sl, new_sl)
                        else:
                            pos.hard_sl = min(pos.hard_sl, new_sl)
                        pos.sl_step += 1
                        pos.sl_moved = True
                        logger.info(f"  SL_MOVED {pos.symbol} {pos.direction} step {pos.sl_step}/{len(MOVE_SL_STEPS)} → {_pfmt(new_sl)} price={_pfmt(price)} (ws)")
                        self._log_sl_moved(pos)
                elif MOVE_SL_TRAIL > 0:
                    if pos.direction == "LONG":
                        pos.best_price = max(pos.best_price, price)
                        new_sl = pos.best_price - MOVE_SL_TRAIL * ea
                    else:
                        pos.best_price = min(pos.best_price, price)
                        new_sl = pos.best_price + MOVE_SL_TRAIL * ea
                    if (pos.direction == "LONG" and new_sl > pos.hard_sl) or \
                       (pos.direction == "SHORT" and new_sl < pos.hard_sl):
                        pos.hard_sl = new_sl

                # Check all exits: SL_MOVED > HARD_SL > TP
                exit_reason = None
                exit_price = price

                if pos.sl_moved:
                    if pos.direction == "LONG" and price <= pos.hard_sl:
                        exit_reason = "SL_MOVED"
                    elif pos.direction == "SHORT" and price >= pos.hard_sl:
                        exit_reason = "SL_MOVED"

                if not exit_reason:
                    if pos.direction == "LONG" and price <= pos.hard_sl:
                        exit_reason = "HARD_SL"
                    elif pos.direction == "SHORT" and price >= pos.hard_sl:
                        exit_reason = "HARD_SL"

                if not exit_reason:
                    if pos.direction == "LONG" and price >= pos.tp:
                        exit_reason = "TP"
                    elif pos.direction == "SHORT" and price <= pos.tp:
                        exit_reason = "TP"

                if exit_reason:
                    if pos.direction == "LONG":
                        pnl_pct = (price - pos.avg_price) / pos.avg_price * 100 * pos.total_size
                    else:
                        pnl_pct = (pos.avg_price - price) / pos.avg_price * 100 * pos.total_size
                    logger.info(f"  {exit_reason} {pos.direction} {pos.symbol} @ {_pfmt(price)} | PnL {pnl_pct:+.2f}% (ws)")
                    self._log_trade_close(pos, price, exit_reason, pnl_pct)
                    self.positions.remove(pos)
                    pair_vb = self.vol_bar_counts.get(pos.symbol, 0)
                    self.cooldowns[pos.symbol] = pair_vb + COOLDOWN_BARS
                    pair_dir_key = f"{pos.symbol}_{pos.direction}"
                    if exit_reason == "HARD_SL":
                        self.pair_sl_streaks[pair_dir_key] = self.pair_sl_streaks.get(pair_dir_key, 0) + 1
                        if self.pair_sl_streaks[pair_dir_key] >= self.PAIR_COOLDOWN_SL:
                            self.pair_dir_cooldowns[pair_dir_key] = pair_vb + self.PAIR_COOLDOWN_BARS
                    else:
                        self.pair_sl_streaks[pair_dir_key] = 0
                    self._save_state()

    def _process_kline_1m(self, symbol: str, o: float, h: float, l: float, c: float, vol: float, kline_time):
        """Process a closed 1m kline: aggregate ATR + accumulate volume bar."""
        if symbol not in self.vol_buffers or symbol not in self.vol_bars:
            return

        # Aggregate 1m → 15m for ATR update
        if not hasattr(self, '_1m_buffers'):
            self._1m_buffers = {}
        mb = self._1m_buffers.get(symbol, {"count": 0, "high": 0, "low": float('inf'), "open": 0, "close": 0, "vol": 0})
        if mb["count"] == 0:
            mb["open"] = o; mb["high"] = h; mb["low"] = l
        else:
            mb["high"] = max(mb["high"], h); mb["low"] = min(mb["low"], l)
        mb["close"] = c; mb["vol"] += vol; mb["count"] += 1
        if mb["count"] >= 15:
            bar_15m = pd.DataFrame([{
                "open": mb["open"], "high": mb["high"], "low": mb["low"],
                "close": mb["close"], "volume": mb["vol"],
            }], index=[kline_time])
            with self._lock:
                if symbol in self.time_data:
                    base = self.time_data[symbol][["open", "high", "low", "close", "volume"]]
                    base = pd.concat([base, bar_15m])
                    base = base[~base.index.duplicated(keep='last')]
                    self.time_data[symbol] = self.indicators.calculate_all(base.tail(1000))
                    if symbol in self.vol_bars:
                        vb_sorted = self.vol_bars[symbol].sort_index()
                        time_atr = _safe_reindex(self.time_data[symbol]["atr"], vb_sorted.index)
                        self.vol_bars[symbol]["atr"] = time_atr.values
                if symbol in self.vol_history:
                    self.vol_history[symbol].append(mb["vol"])
                    if len(self.vol_history[symbol]) > 960:
                        self.vol_history[symbol] = self.vol_history[symbol][-960:]
                    self.time_bar_counts[symbol] = self.time_bar_counts.get(symbol, 0) + 1
                    if self.time_bar_counts[symbol] % 960 == 0:
                        new_thr = float(np.median(self.vol_history[symbol])) * 2
                        old_thr = self.vol_thresholds.get(symbol, 0)
                        self.vol_thresholds[symbol] = new_thr
                        if abs(new_thr - old_thr) / max(old_thr, 1) > 0.05:
                            logger.info(f"  {symbol} threshold: {old_thr:.0f} → {new_thr:.0f}")
            mb = {"count": 0, "high": 0, "low": float('inf'), "open": 0, "close": 0, "vol": 0}
        self._1m_buffers[symbol] = mb

        # Accumulate volume bar
        buf = self.vol_buffers[symbol]
        with self._lock:
            if buf["bar_open"] is None:
                buf["bar_open"] = o; buf["bar_high"] = h
                buf["bar_low"] = l; buf["bar_start"] = kline_time
            buf["bar_high"] = max(buf["bar_high"], h)
            buf["bar_low"] = min(buf["bar_low"], l)
            buf["cum_vol"] += vol
            threshold = self.vol_thresholds.get(symbol, 2000)
            if buf["cum_vol"] < threshold:
                return
            bar_data = (buf["bar_open"], buf["bar_high"], buf["bar_low"], c, buf["cum_vol"], buf["bar_start"])
            buf["cum_vol"] = 0
            buf["bar_open"] = None

        self._complete_volume_bar(symbol, *bar_data)


    def _sl_monitor_ws(self):
        """WebSocket: aggTrade for real-time price monitoring + volume bar construction."""
        try:
            import websocket
        except ImportError:
            logger.warning("websocket-client not installed, falling back to polling")
            self._sl_monitor_poll()
            return

        ws_to_pair = {}
        stream_parts = []
        for s in self.pairs:
            raw = s.replace("/", "").lower()
            stream_parts.append(f"{raw}@aggTrade")
            stream_parts.append(f"{raw}@bookTicker")
            stream_parts.append(f"{raw}@depth5@100ms")
            stream_parts.append(f"{raw}@markPrice")
            ws_to_pair[s.replace("/", "").upper()] = s
        streams = "/".join(stream_parts)
        url = f"wss://fstream.binance.com/stream?streams={streams}"

        self._ws_last_msg = time.time()
        self._ws_last_kline = time.time()
        self._ws_start_time = time.time()
        self._trade_count = 0
        _ws_ref = [None]

        def _bar_worker():
            """Separate thread for volume bar completion — never blocks WS callback."""
            while True:
                try:
                    item = self._msg_queue.get()
                    if item is None:
                        break
                    symbol, bar_open, bar_high, bar_low, bar_close, bar_vol, bar_start = item
                    self._complete_volume_bar(symbol, bar_open, bar_high, bar_low, bar_close, bar_vol, bar_start)
                except Exception as e:
                    logger.error(f"bar worker error: {e}", exc_info=True)

        bw = threading.Thread(target=_bar_worker, daemon=True)
        bw.start()

        def on_message(ws, message):
            self._ws_last_msg = time.time()
            try:
                msg = json.loads(message)
                data = msg.get("data", {})
                sym_raw = data.get("s", "")
                symbol = ws_to_pair.get(sym_raw)
                if not symbol:
                    return

                # Route by event type
                event_type = data.get("e", "")

                # 0a. Track depth5 (order book top 5)
                if event_type == "depthUpdate":
                    if not hasattr(self, '_depth'):
                        self._depth = {}
                    bids = data.get("b", [])
                    asks = data.get("a", [])
                    bid_vol = sum(float(b[1]) for b in bids[:5]) if bids else 0
                    ask_vol = sum(float(a[1]) for a in asks[:5]) if asks else 0
                    total = bid_vol + ask_vol
                    imbalance = (bid_vol - ask_vol) / total * 100 if total > 0 else 0
                    # Wall detection: any level with 3x+ avg volume
                    all_vols = [float(b[1]) for b in bids[:5]] + [float(a[1]) for a in asks[:5]]
                    avg_level_vol = sum(all_vols) / len(all_vols) if all_vols else 1
                    bid_wall = any(float(b[1]) > avg_level_vol * 3 for b in bids[:5])
                    ask_wall = any(float(a[1]) > avg_level_vol * 3 for a in asks[:5])
                    bids_raw = [float(b[1]) for b in bids[:5]]
                    asks_raw = [float(a[1]) for a in asks[:5]]
                    wall_price = 0
                    for b in bids[:5]:
                        if float(b[1]) > avg_level_vol * 3:
                            wall_price = float(b[0]); break
                    if wall_price == 0:
                        for a in asks[:5]:
                            if float(a[1]) > avg_level_vol * 3:
                                wall_price = float(a[0]); break
                    self._depth[symbol] = {
                        "bid_vol": bid_vol, "ask_vol": ask_vol,
                        "imbalance": imbalance,
                        "bid_wall": bid_wall, "ask_wall": ask_wall,
                        "bids_raw": bids_raw, "asks_raw": asks_raw,
                        "wall_price": wall_price,
                        "ts": time.time()
                    }
                    import collections
                    if not hasattr(self, '_depth_history'):
                        self._depth_history = {}
                    if symbol not in self._depth_history:
                        self._depth_history[symbol] = collections.deque(maxlen=20)
                    self._depth_history[symbol].append(bid_vol + ask_vol)
                    return

                # 0b2. Track markPrice (funding rate)
                if event_type == "markPriceUpdate":
                    if not hasattr(self, '_mark_data'):
                        self._mark_data = {}
                    self._mark_data[symbol] = {
                        "mark_price": float(data.get("p", 0)),
                        "index_price": float(data.get("i", 0)),
                        "funding_rate": float(data.get("r", 0)),
                        "next_funding": int(data.get("T", 0)),
                        "ts": time.time()
                    }
                    return

                # 0b. Track bookTicker (spread)
                if event_type == "bookTicker":
                    bid = float(data.get("b", 0))
                    ask = float(data.get("a", 0))
                    if bid > 0 and ask > 0:
                        if not hasattr(self, '_spreads'):
                            self._spreads = {}
                        bid_qty = float(data.get("B", 0))
                        ask_qty = float(data.get("A", 0))
                        spread_pct = (ask - bid) / ((ask + bid) / 2) * 100
                        self._spreads[symbol] = {
                            "bid": bid, "ask": ask, "bid_qty": bid_qty, "ask_qty": ask_qty,
                            "spread": ask - bid, "spread_pct": spread_pct,
                            "ts": time.time()
                        }
                        if not hasattr(self, '_spread_history'):
                            self._spread_history = {}
                        import collections
                        if symbol not in self._spread_history:
                            self._spread_history[symbol] = collections.deque(maxlen=20)
                        self._spread_history[symbol].append(spread_pct)
                    return

                # aggTrade only from here
                if event_type != "aggTrade" and event_type != "":
                    return  # not aggTrade, already handled above

                price = float(data.get("p", 0))
                qty = float(data.get("q", 0))
                trade_ts = pd.Timestamp(int(data.get("T", 0)), unit="ms")
                if price <= 0 or qty <= 0:
                    return

                self._trade_count += 1
                is_buyer_maker = data.get("m", False)
                if not hasattr(self, '_recent_trades'):
                    self._recent_trades = {}
                if symbol not in self._recent_trades:
                    import collections
                    self._recent_trades[symbol] = collections.deque(maxlen=100)
                self._recent_trades[symbol].append({
                    "price": price, "ts": time.time(),
                    "is_sell": is_buyer_maker, "qty": qty
                })
                self._ws_last_kline = time.time()

                # 1. SL/TP/Move SL check on every trade (real-time)
                self._process_price(symbol, price)

                # 2. Accumulate volume for volume bar
                if symbol in self.vol_buffers:
                    buf = self.vol_buffers[symbol]
                    with self._lock:
                        if buf["bar_open"] is None:
                            buf["bar_open"] = price
                            buf["bar_high"] = price
                            buf["bar_low"] = price
                            buf["bar_start"] = trade_ts
                        buf["bar_high"] = max(buf["bar_high"], price)
                        buf["bar_low"] = min(buf["bar_low"], price)
                        buf["cum_vol"] += qty
                        threshold = self.vol_thresholds.get(symbol, 2000)
                        if buf["cum_vol"] >= threshold:
                            bar_data = (buf["bar_open"], buf["bar_high"], buf["bar_low"],
                                       price, buf["cum_vol"], buf["bar_start"])
                            buf["cum_vol"] = 0
                            buf["bar_open"] = None
                            self._msg_queue.put((symbol, *bar_data))

                # 3. Aggregate for ATR (every ~15 min worth of volume)
                if not hasattr(self, '_1m_buffers'):
                    self._1m_buffers = {}
                mb = self._1m_buffers.get(symbol, {"count": 0, "high": 0, "low": float('inf'),
                                                    "open": 0, "close": 0, "vol": 0, "last_min": -1})
                cur_min = int(time.time() // 60)
                if cur_min != mb.get("last_min", -1):
                    if mb["count"] > 0:
                        mb["count"] += 1
                        mb["close"] = price
                        mb["high"] = max(mb["high"], price)
                        mb["low"] = min(mb["low"], price)
                        mb["vol"] += qty
                    if mb["count"] >= 15:
                        bar_15m = pd.DataFrame([{
                            "open": mb["open"], "high": mb["high"], "low": mb["low"],
                            "close": mb["close"], "volume": mb["vol"],
                        }], index=[trade_ts])
                        with self._lock:
                            if symbol in self.time_data:
                                base = self.time_data[symbol][["open", "high", "low", "close", "volume"]]
                                base = pd.concat([base, bar_15m])
                                base = base[~base.index.duplicated(keep='last')]
                                self.time_data[symbol] = self.indicators.calculate_all(base.tail(1000))
                                if symbol in self.vol_bars:
                                    vb_sorted = self.vol_bars[symbol].sort_index()
                                    time_atr = _safe_reindex(self.time_data[symbol]["atr"], vb_sorted.index)
                                    self.vol_bars[symbol]["atr"] = time_atr.values
                            if symbol in self.vol_history:
                                self.vol_history[symbol].append(mb["vol"])
                                if len(self.vol_history[symbol]) > 960:
                                    self.vol_history[symbol] = self.vol_history[symbol][-960:]
                                self.time_bar_counts[symbol] = self.time_bar_counts.get(symbol, 0) + 1
                                if self.time_bar_counts[symbol] % 960 == 0:
                                    new_thr = float(np.median(self.vol_history[symbol])) * 2
                                    old_thr = self.vol_thresholds.get(symbol, 0)
                                    self.vol_thresholds[symbol] = new_thr
                                    if abs(new_thr - old_thr) / max(old_thr, 1) > 0.05:
                                        logger.info(f"  {symbol} threshold: {old_thr:.0f} → {new_thr:.0f}")
                        mb = {"count": 0, "high": 0, "low": float('inf'), "open": 0, "close": 0, "vol": 0, "last_min": cur_min}
                    else:
                        mb["last_min"] = cur_min
                        if mb["count"] == 0:
                            mb["open"] = price
                            mb["high"] = price
                            mb["low"] = price
                        mb["count"] = max(mb["count"], 1)
                else:
                    mb["close"] = price
                    mb["high"] = max(mb["high"], price)
                    mb["low"] = min(mb["low"], price)
                    mb["vol"] += qty
                self._1m_buffers[symbol] = mb

            except Exception as e:
                logger.warning(f"WS parse error: {e}")

        def on_error(ws, error):
            logger.warning(f"WS error: {error}")

        def on_close(ws, close_status, close_msg):
            logger.warning(f"WS closed: {close_status} {close_msg}")

        def on_open(ws):
            self._ws_last_msg = time.time()
            self._ws_last_kline = time.time()
            self._ws_start_time = time.time()
            if hasattr(self, '_1m_buffers'):
                self._1m_buffers.clear()
            nonlocal _backoff
            _backoff = 3
            logger.info(f"WS connected: {len(self.pairs)} pairs (aggTrade)")

        def _watchdog():
            """Kill WS if no messages or no kline for too long."""
            while True:
                time.sleep(5)
                stale_msg = time.time() - self._ws_last_msg
                stale_kline = time.time() - self._ws_last_kline
                if stale_msg > 15 and _ws_ref[0]:
                    logger.warning(f"WS watchdog: no messages for {stale_msg:.0f}s, forcing reconnect")
                    try:
                        _ws_ref[0].close()
                    except Exception:
                        pass
                elif stale_kline > 10 and _ws_ref[0] and (time.time() - self._ws_start_time) > 60:
                    logger.warning(f"WS watchdog: no kline_1m for {stale_kline:.0f}s (miniTicker OK), forcing reconnect")
                    try:
                        _ws_ref[0].close()
                    except Exception:
                        pass

        wd = threading.Thread(target=_watchdog, daemon=True)
        wd.start()

        _backoff = 3
        while True:
            try:
                ws = websocket.WebSocketApp(
                    url, on_message=on_message, on_error=on_error,
                    on_close=on_close, on_open=on_open,
                )
                _ws_ref[0] = ws
                ws.run_forever(ping_interval=10, ping_timeout=5)
                _backoff = 3
            except Exception as e:
                logger.error(f"WS fatal: {e}", exc_info=True)
            logger.info(f"WS reconnecting in {_backoff}s...")
            time.sleep(_backoff)
            _backoff = min(_backoff * 2, 60)

    def _sl_monitor_poll(self):
        """Backup polling: runs when WS is dead, sleeps when WS is alive."""
        from concurrent.futures import ThreadPoolExecutor
        while True:
            try:
                time.sleep(MOVE_SL_CHECK)
                ws_alive = (time.time() - self._ws_last_msg) < 15 if hasattr(self, '_ws_last_msg') else False
                if ws_alive:
                    continue

                with self._lock:
                    active_positions = list(self.positions)
                if not active_positions:
                    continue

                def fetch_price(sym):
                    try:
                        t = self.fetcher.exchange.fetch_ticker(sym)
                        return sym, float(t["last"])
                    except Exception:
                        return sym, None

                symbols = list(set(p.symbol for p in active_positions))
                prices = {}
                with ThreadPoolExecutor(max_workers=len(symbols)) as pool:
                    for sym, price in pool.map(fetch_price, symbols):
                        if price is not None:
                            prices[sym] = price

                for pos in active_positions:
                    price = prices.get(pos.symbol)
                    if price is not None:
                        self._process_price(pos.symbol, price)
            except Exception as e:
                logger.error(f"SL poll error: {e}", exc_info=True)
                time.sleep(10)

    @staticmethod
    def _archive_logs():
        """Move old logs to archive folder on startup."""
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

        ws_thread = threading.Thread(target=self._sl_monitor_ws, daemon=True)
        ws_thread.start()
        poll_thread = threading.Thread(target=self._sl_monitor_poll, daemon=True)
        poll_thread.start()
        logger.info(f"WebSocket aggTrade + poll backup started ({len(MOVE_SL_STEPS)}-step Move SL + trail {MOVE_SL_TRAIL}x ATR)")

        logger.info(f"Starting backup scan loop (every 15m, position status only)...")
        while True:
            try:
                self.scan()
                self._save_state()
                time.sleep(900)
            except KeyboardInterrupt:
                logger.info("Stopping bot...")
                self._save_state()
                break
            except Exception as e:
                logger.error(f"Scan error: {e}", exc_info=True)
                self._save_state()
                time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Volume Bars Trading Bot")
    parser.add_argument("--once", action="store_true", help="Single scan then exit")
    parser.add_argument("--live", action="store_true", help="Real trading (default: paper)")
    args = parser.parse_args()

    bot = VolumeBarsBot(paper=not args.live)
    bot.run(once=args.once)
